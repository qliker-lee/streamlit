#
# streamlit를 이용한 Value Chain Diagram
# 2025. 11. 08.  Qliker
#

import streamlit as st
import pandas as pd
import yaml
import re
import sys
import os
import html
from graphviz import Digraph
from dataclasses import dataclass
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import time
import tempfile
import math
from pathlib import Path
from PIL import Image

APP_NAME = "Value Chain Diagram"
APP_DESC = """###### 
•	Value Chain → Process → Master Table 간의 연결 관계를 시각적으로 표현. \n
•	데이터 거버넌스 담당자가 프로세스별 관리 마스터 데이터 구조를 쉽게 이해하도록 지원. \n
•	Primary Process(메가 프로세스)와 Support Function(지원 기능)을 구분하여 표시. \n
"""

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Now import utils after adding to path
from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config, display_valuechain_sample_image

def html_escape(text: str) -> str:
    """Graphviz HTML-safe 텍스트 이스케이프"""
    if not isinstance(text, str):
        return ''
    return html.escape(text, quote=True)

def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def valuechain_color_definition():
    st.divider()
    st.markdown("**박스는 Process/Function, 원은 Master 입니다.**")
    # st.divider()
    st.markdown("##### Master Color Definition:")
    st.markdown("###### Green: Standard Master에도 있고, 우리 회사에도 관리하는 Master")
    st.markdown("###### Yellow: Standard Master에는 없지만, 우리 회사에는 관리하는 Master")
    st.markdown("###### Gray: Standard Master에는 있지만, 우리 회사에는 관리하지 않는 Master")

    return True

@dataclass
class FileConfig:
    """파일 설정 정보"""
    valuechain: str
    valuechain_standard_master: str
    valuechain_system: str

class FileLoader:
    """파일 로딩을 위한 클래스"""

    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.root_path = str(yaml_config.get("ROOT_PATH", str(PROJECT_ROOT)))
        self.files_config = self._setup_files_config()

    def _setup_files_config(self) -> FileConfig:
        """파일 설정 구성 (✅ ROOT_PATH 결합 문제 수정)"""
        files = self.yaml_config.get('files', {})

        def _full_path(path_str):
            p = Path(path_str)
            if not p.is_absolute():
                p = Path(self.root_path) / p
            return str(p.resolve())

        return FileConfig(
            valuechain=_full_path(files.get('valuechain', 'DataSense/DS_Meta/DataSense_ValueChain.csv')),
            valuechain_standard_master=_full_path(files.get('valuechain_standard_master', 'DataSense/DS_Meta/DataSense_ValueChain_Standard_Master.csv')),
            valuechain_system=_full_path(files.get('valuechain_system', 'DataSense/DS_Meta/DataSense_ValueChain_System.csv'))
        )

    def load_file(self, file_path: str, file_name: str) -> Optional[pd.DataFrame]:
        """개별 파일 로드"""
        if not os.path.exists(file_path):
            st.warning(f"{file_name} 파일이 존재하지 않습니다: {file_path}")
            return None
        
        try:
            extension = os.path.splitext(file_path)[1].lower()
            if extension == '.csv':
                return pd.read_csv(file_path)
            elif extension == '.xlsx':
                return pd.read_excel(file_path)
            elif extension == '.pkl':
                return pd.read_pickle(file_path)
            else:
                st.error(f"{file_name} 파일 형식을 지원하지 않습니다: {extension}")
                return None
        except Exception as e:
            st.error(f"{file_name} 파일 로드 실패: {str(e)}")
            return None
    
    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """필요한 모든 파일 로드"""
        files_to_load = {
            'valuechain': self.files_config.valuechain,
            'valuechain_standard_master': self.files_config.valuechain_standard_master,
            'valuechain_system': self.files_config.valuechain_system
        }
        
        loaded_data = {}
        for name, path in files_to_load.items():
            df = self.load_file(path, name)
            if df is None:
                st.warning(f"{name} 파일이 비어 있거나 존재하지 않습니다.")
                df = pd.DataFrame()
            else:
                df = df.fillna('')
            loaded_data[name] = df
        
        return loaded_data

# 전역 변수로 선택 상태 관리
selected_activities_global = set()
selected_activities_global_2nd = set()

class ValueChainDiagram:
    """Value Chain 다이어그램 생성 클래스"""
    
    def __init__(self, yaml_config: Dict[str, Any] = None):
        self.yaml_config = yaml_config
        self.valuechain_data = None
        self.valuechain_standard_master = None
        self.valuechain_system = None
    
    def load_all_valuechain_data(self, loaded_data: Dict[str, pd.DataFrame]) -> bool:
        """모든 ValueChain 관련 데이터 로드"""
        try:
            # 기본 파일 유무 확인
            self.valuechain_data = loaded_data.get('valuechain', pd.DataFrame())
            self.valuechain_standard_master = loaded_data.get('valuechain_standard_master', pd.DataFrame())
            self.valuechain_system = loaded_data.get('valuechain_system', pd.DataFrame())

            if self.valuechain_data.empty:
                st.warning("⚠️ ValueChain 데이터가 없습니다. 기본 예시를 표시합니다.")
                return False

            # 모든 Activity_Seq를 문자열로 통일
            for df in [self.valuechain_data, self.valuechain_standard_master, self.valuechain_system]:
                if 'Activity_Seq' in df.columns:
                    df['Activity_Seq'] = df['Activity_Seq'].astype(str)

            return True
        except Exception as e:
            st.error(f"ValueChain 데이터 로드 실패: {str(e)}")
            return False


    def valuechain_diagram(self, df):
        st.markdown("---")
        st.markdown("#### 📄 Value Chain Diagram")

        # 데이터 정렬 (Primary & Support)
        df['Activities_Type'] = df['Activities_Type'].str.strip()

        primary_df = df[df['Activities_Type'] == 'Primary'] \
            .sort_values('Activity_Seq').reset_index(drop=True)

        support_df = df[df['Activities_Type'] == 'Support'] \
            .sort_values('Activity_Seq').reset_index(drop=True)

        fig = go.Figure()

        # ✅ 상수 지정 (일관된 사이즈)
        x_gap = 1.1
        box_width = 0.9
        box_height = 0.8
        circle_diameter = 0.8
        text_font_size = 14
        font_color = "#000000"

        primary_y = 1.0
        support_y = 0

        # ✅ Primary Activities (사각형, 위쪽)
        for idx, row in primary_df.iterrows():
            x = idx * x_gap
            label = f"{row['Activities']}<br>({row['Activities_Kor']})"

            fig.add_shape(
                type="rect",
                x0=x, x1=x + box_width,
                y0=primary_y - box_height / 2, y1=primary_y + box_height / 2,
                line=dict(color="black"),
                fillcolor="lightblue"
            )

            fig.add_annotation(
                x=x + box_width / 2, y=primary_y,
                text=label,
                showarrow=False,
                font=dict(size=text_font_size, color=font_color),
                align="center"
            )

            # 화살표 (→ 방향)
            if idx < len(primary_df) - 1:
                fig.add_annotation(
                    x=x + x_gap, y=primary_y,
                    ax=x + box_width, ay=primary_y,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3
                )

        # ✅ Support Activities (원형, 아래쪽)
        for idx, row in support_df.iterrows():
            x = idx * x_gap
            label = f"{row['Activities']}<br>({row['Activities_Kor']})"

            fig.add_shape(
                type="circle",
                x0=x + (box_width - circle_diameter)/2,
                x1=x + (box_width + circle_diameter)/2,
                y0=support_y - circle_diameter / 2,
                y1=support_y + circle_diameter / 2,
                line=dict(color="black"),
                fillcolor="#d7fbc9"
            )

            fig.add_annotation(
                x=x + box_width / 2, y=support_y,
                text=label,
                showarrow=False,
                font=dict(size=text_font_size - 1, color=font_color),
                align="center"
            )

        # ✅ Layout
        total_width = (max(len(primary_df), len(support_df)) - 1) * x_gap + box_width
        fig.update_layout(
            height=400,
            width=total_width * 100, 
            # width=1000,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=10, t=0, b=0),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True, on_select="ignore", key="valuechain_diagram")
        st.markdown("\n")
        st.markdown("**박스는 Mega Process, 원은 Support Function 입니다.**")
        st.markdown("---")
        return fig

    def valuechain_summary(self, df):
        """Value Chain Summary 정보 표시"""
        if df is not None and not df.empty:
            st.markdown("### Value Chain Process/Function Summary")
            # 데이터 요약
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                industry = df['Industry'].unique().tolist()
                st.metric("Industry", industry[0] if industry else "N/A")
            with col2:
                st.metric("Total Process & Function", len(df))
            with col3:
                primary_count = len(df[
                    df['Activities_Type'].str.strip() == 'Primary'
                ]) if 'Activities_Type' in df.columns else 0
                st.metric("Mega Process", primary_count)
            with col4:
                support_count = len(df[
                    df['Activities_Type'].str.strip() == 'Support'
                ]) if 'Activities_Type' in df.columns else 0
                st.metric("Support Function", support_count)   
        else:
            st.warning("Value Chain 데이터가 없습니다.")
            return None
    
    def ValueChain_Master_Information(self, df):
        """ValueChain & Master Information 정보 표시"""
        try:
            # st.dataframe(df)
            required_columns = ["Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", "Systems",
                    "Master", "Master_Kor", "Our_Master", "Has_Flag"]
            df = df[required_columns]
            
            # st.markdown("#### Value Chain & Master Information")
            st.dataframe(df, use_container_width=False, hide_index=True, height=550, width=1200,
                         column_config={
                             "Activities_Type": st.column_config.TextColumn("구분", width=70),
                             "Activity_Seq": st.column_config.NumberColumn("Seq", format="%d", width=50),
                             "Activities": st.column_config.TextColumn("Process/Function", width=120),
                             "Activities_Kor": st.column_config.TextColumn("한글명", width=120),
                             "Systems": st.column_config.TextColumn("Systems", width=70),
                             "Master": st.column_config.TextColumn("Master", width=100),
                             "Master_Kor": st.column_config.TextColumn("Master Kor", width=150), 
                             "Our_Master": st.column_config.TextColumn("Our Master", width=150),
                             "Has_Flag": st.column_config.TextColumn("Has", width=50),
                         })

        except Exception as e:
            st.error(f"Value Chain & Master Information 표시 중 오류 발생: {str(e)}")

    def _Activities_Master_Diagram_Base(self, valuechain_df, master_df, master_column: str, title_suffix: str = "") -> Digraph:
        """
        통합 다이어그램 생성 함수
        
        Args:
            valuechain_df: Value Chain 데이터프레임
            master_df: Master 데이터프레임
            master_column: 사용할 Master 컬럼명 ('Master', 'Master2', 'Master_Kor')
            title_suffix: 제목에 추가할 suffix (예: "(Our Master)", "(Our Master Kor)")
        
        Returns:
            Digraph: 생성된 다이어그램 객체
        """
        try:
            # Master 컬럼 확인 및 생성
            if master_column not in master_df.columns:
                if master_column == 'Master2' and 'Master' in master_df.columns:
                    master_df[master_column] = master_df['Master'].copy()
                elif master_column == 'Master_Kor' and 'Master_Kor' in master_df.columns:
                    # 이미 존재하는 경우 (불필요한 복사 방지)
                    pass
                else:
                    st.warning(f"⚠️ {master_column} 컬럼이 없습니다.")
                    return None
            
            primary_activities = valuechain_df[valuechain_df['Activities_Type'].str.strip() == 'Primary']
            support_activities = valuechain_df[valuechain_df['Activities_Type'].str.strip() == 'Support']

            # ---------- 사전 매핑: 반복 필터 제거의 핵심 ----------
            # seq -> {masters} - 지정된 컬럼 기준으로 그룹화
            masters_by_seq = (
                master_df.groupby('Activity_Seq', group_keys=False)[master_column]
                .apply(lambda s: set(map(str, s.tolist())))
                .to_dict()
            )

            # master -> Has_Flag(최댓값 우선) - 지정된 컬럼 기준으로 그룹화
            has_flag_by_master = (
                master_df.assign(Master_temp=master_df[master_column].astype(str))
                        .groupby(master_column, group_keys=False)['Has_Flag']
                        .max()
                        .fillna(0).astype(int)
                        .to_dict()
            )

            _ensure_dir("tmp_image")
            # ---------- Primary Activities 다이어그램 ----------
            primary_dot = None
            if not primary_activities.empty:
                primary_title = f"#### Mega Process & Master Relationship Diagram{title_suffix}"
                st.markdown(primary_title)

                primary_dot = Digraph(format='png')
                primary_dot.attr(rankdir='LR', fontsize='10', fontname='Malgun Gothic')

                # Primary Activities 노드(박스)
                primary_sorted = primary_activities.sort_values('Activity_Seq')
                for _, row in primary_sorted.iterrows():
                    seq = row.get('Activity_Seq')
                    node_id = f"PA{seq}"
                    node_label = f"{row.get('Activities', 'Unknown')}\n({row.get('Activities_Kor', '')})"
                    primary_dot.attr('node', shape='box', style='filled',
                                     fillcolor='lightblue2', color='black', fontname='Malgun Gothic')
                    primary_dot.node(node_id, node_label)

                # 박스 간 연결
                seqs = list(primary_sorted['Activity_Seq'])
                for i in range(len(seqs) - 1):
                    primary_dot.edge(f"PA{seqs[i]}", f"PA{seqs[i+1]}", style='bold', color='red')

                # Master 노드(원)
                primary_activity_seqs = set(seqs)
                primary_masters = set().union(*(masters_by_seq.get(s, set()) for s in primary_activity_seqs))

                for master in primary_masters:
                    node_id = f"PM{master}"
                    node_label = str(master)
                    has_flag = has_flag_by_master.get(master, 0)
                    if has_flag == 1:
                        fill = '#cefec2'   # 초록
                    elif has_flag == 2:
                        fill = '#fdf476'   # 노랑
                    else:
                        fill = '#e5e5db'   # 회색
                    primary_dot.attr('node', shape='ellipse', style='filled',
                                     fillcolor=fill, color='black', fontname='Malgun Gothic')
                    primary_dot.node(node_id, node_label)

                # 엣지(박스→원)
                for seq in primary_activity_seqs:
                    for master in masters_by_seq.get(seq, []):
                        primary_dot.edge(f"PA{seq}", f"PM{master}", style='solid', color='blue')

                # 렌더링 & 표시
                with st.spinner("Rendering primary diagram..."):
                    timestamp = int(time.time() * 1000)
                    primary_filename = f"tmp_image/primary_process_{timestamp}"
                    primary_dot.render(primary_filename, cleanup=True)

                mcnt = len(primary_masters)
                
                
                # 이미지 크기 조건에 따라 설정
                if mcnt < 2:
                    width = 300
                elif mcnt < 4:
                    width = 500
                elif mcnt < 6:
                    width = 700
                elif mcnt < 8:
                    width = 700
                elif mcnt < 10:
                    width = 800
                elif mcnt < 12:
                    width = 900
                else:
                    width = 1000
                
                st.write(f"📊 Primary Diagram - Master 개수: {mcnt}, 너비: {width}")  # Debug
                st.image(f"{primary_filename}.png", width=width)

            # ---------- Support Activities 다이어그램 (그룹 페이징) ----------
            if not support_activities.empty:
                support_title = f"#### Support Function & Master Relationship Diagram{title_suffix}"
                st.markdown(support_title)

                support_sorted = support_activities.sort_values('Activity_Seq')
                current_activities_rows = []  # pandas Series 목록
                current_masters = set()
                current_master_count = 0
                graph_count = 1

                # 그룹 기준(마스터 수) — 필요시 조정
                GROUP_MASTER_LIMIT = 10

                for _, activity_row in support_sorted.iterrows():
                    seq = activity_row['Activity_Seq']
                    seq_masters = masters_by_seq.get(seq, set())
                    master_count = len(seq_masters)

                    # 현재 그룹에 추가 시 한도 초과 → 끊고 그리기
                    if current_master_count + master_count > GROUP_MASTER_LIMIT:
                        if current_activities_rows:
                            # System 컬럼인 경우 create_system_group_diagram 사용
                            if master_column == 'System':
                                self.create_system_group_diagram(
                                    activities=current_activities_rows,
                                    systems=current_masters,
                                    systems_by_seq=masters_by_seq,
                                    group_num=graph_count
                                )
                            else:
                                self.create_support_group_diagram(
                                    activities=current_activities_rows,
                                    masters=current_masters,
                                    has_flag_by_master=has_flag_by_master,
                                    masters_by_seq=masters_by_seq,
                                    group_num=graph_count
                                )
                            graph_count += 1

                        # 새 그룹 시작
                        current_activities_rows = [activity_row]
                        current_masters = set(seq_masters)
                        current_master_count = master_count
                    else:
                        current_activities_rows.append(activity_row)
                        current_masters.update(seq_masters)
                        current_master_count += master_count

                # 마지막 그룹 처리
                if current_activities_rows:
                    # System 컬럼인 경우 create_system_group_diagram 사용
                    if master_column == 'System':
                        self.create_system_group_diagram(
                            activities=current_activities_rows,
                            systems=current_masters,
                            systems_by_seq=masters_by_seq,
                            group_num=graph_count
                        )
                    else:
                        self.create_support_group_diagram(
                            activities=current_activities_rows,
                            masters=current_masters,
                            has_flag_by_master=has_flag_by_master,
                            masters_by_seq=masters_by_seq,
                            group_num=graph_count
                        )

            valuechain_color_definition()

            return primary_dot if (not primary_activities.empty) else None

        except Exception as e:
            st.info("Cloud 환경에서는 Diagram을 생성할 수 없습니다. Local 환경에서 실행해주세요. 샘플 이미지를 표시합니다.")
            display_valuechain_sample_image()
            valuechain_color_definition()
            return None

    def Activities_Master_Diagram_Our(self, valuechain_df, master_df, title: str = "") -> Digraph:
        """Our Master 다이어그램 (Master2 컬럼 사용)"""
        return self._Activities_Master_Diagram_Base(valuechain_df, master_df, 'Master2', ' (Our Master)')

    def Activities_Master_Diagram_Kor(self, valuechain_df, master_df, title: str = "") -> Digraph:
        """Our Master Kor 다이어그램 (Master_Kor 컬럼 사용)"""
        return self._Activities_Master_Diagram_Base(valuechain_df, master_df, 'Master_Kor', ' (Our Master Kor)')

    def Activities_Master_Diagram(self, valuechain_df, master_df, title: str = "") -> Digraph:
        """Standard Master 다이어그램 (Master 컬럼 사용)"""
        return self._Activities_Master_Diagram_Base(valuechain_df, master_df, 'Master', '')

    def System_Master_Diagram(self, valuechain_df, systems_df, master_df, title: str = "") -> Digraph:
        """
        Activities, System, Master 간의 관계 다이어그램 생성 (3단계: Activities → System → Master)
        
        Args:
            valuechain_df: Value Chain 데이터프레임 (Activity_Seq, Activities, Activities_Kor 컬럼 포함)
            systems_df: System 데이터프레임 (Activity_Seq, System 컬럼 포함)
            master_df: Master 데이터프레임 (Activity_Seq, Master 컬럼 포함)
            title: 다이어그램 제목
        
        Returns:
            Digraph: 생성된 다이어그램 객체
        """
        try:
            # 데이터 확인
            if valuechain_df is None or valuechain_df.empty:
                st.info("📝 Value Chain 데이터가 없습니다.")
                return None
            
            if systems_df is None or systems_df.empty:
                st.info("📝 System 데이터가 없습니다.")
                return None
            
            if master_df is None or master_df.empty:
                st.info("📝 Master 데이터가 없습니다.")
                return None
            
            # Activity_Seq를 기준으로 Activities, System, Master 매핑
            activity_map = {}  # {Activity_Seq: {'activity': str, 'activity_kor': str, 'systems': set(), 'masters': set()}}
            
            # Value Chain 데이터 처리 (Activities)
            for _, row in valuechain_df.iterrows():
                activity_seq = str(row.get('Activity_Seq', '')).strip()
                activity = str(row.get('Activities', '')).strip()
                activity_kor = str(row.get('Activities_Kor', '')).strip()
                if activity_seq:
                    if activity_seq not in activity_map:
                        activity_map[activity_seq] = {
                            'activity': activity,
                            'activity_kor': activity_kor,
                            'systems': set(),
                            'masters': set()
                        }
            
            # System 데이터 처리
            for _, row in systems_df.iterrows():
                activity_seq = str(row.get('Activity_Seq', '')).strip()
                system = str(row.get('System', '')).strip()
                if activity_seq and system and system.lower() != 'nan':
                    if activity_seq not in activity_map:
                        activity_map[activity_seq] = {
                            'activity': '',
                            'activity_kor': '',
                            'systems': set(),
                            'masters': set()
                        }
                    activity_map[activity_seq]['systems'].add(system)
            
            # Master 데이터 처리
            for _, row in master_df.iterrows():
                activity_seq = str(row.get('Activity_Seq', '')).strip()
                master = str(row.get('Master', '')).strip()
                if activity_seq and master and master.lower() != 'nan':
                    if activity_seq not in activity_map:
                        activity_map[activity_seq] = {
                            'activity': '',
                            'activity_kor': '',
                            'systems': set(),
                            'masters': set()
                        }
                    activity_map[activity_seq]['masters'].add(master)
            
            if not activity_map:
                st.info("📝 Activities, System, Master 간의 연결 데이터가 없습니다.")
                return None
            
            _ensure_dir("tmp_image")
            
            # 다이어그램 생성
            st.markdown(f"#### Activities & System & Master Relationship Diagram{title}")
            
            diagram_dot = Digraph(format='png')
            diagram_dot.attr(rankdir='TB', fontsize='10', fontname='Malgun Gothic')
            
            # 모든 Activities, System, Master 수집
            all_activities = {}  # {Activity_Seq: {'activity': str, 'activity_kor': str}}
            all_systems = set()
            all_masters = set()
            
            for activity_seq, data in activity_map.items():
                if data['activity'] or data['activity_kor']:
                    all_activities[activity_seq] = {
                        'activity': data['activity'],
                        'activity_kor': data['activity_kor']
                    }
                all_systems.update(data['systems'])
                all_masters.update(data['masters'])
            
            # Activities 노드 생성 (상단, 박스)
            for activity_seq, activity_info in sorted(all_activities.items()):
                node_id = f"ACT_{activity_seq}"
                activity_label = activity_info['activity']
                activity_kor = activity_info['activity_kor']
                if activity_kor:
                    label = f"{activity_label}\n({activity_kor})"
                else:
                    label = activity_label if activity_label else f"Seq:{activity_seq}"
                diagram_dot.attr('node', shape='box', style='filled',
                               fillcolor='#eef8d2', color='black', fontname='Malgun Gothic')
                diagram_dot.node(node_id, label)
            
            # System 노드 생성 (중간, 박스)
            for system in sorted(all_systems):
                node_id = f"SYS_{system}"
                diagram_dot.attr('node', shape='box', style='filled',
                               fillcolor='lightblue2', color='black', fontname='Malgun Gothic')
                diagram_dot.node(node_id, str(system))
            
            # Master 노드 생성 (하단, 원)
            # Has_Flag 정보가 있으면 색상 구분
            has_flag_by_master = {}
            if 'Has_Flag' in master_df.columns:
                has_flag_by_master = (
                    master_df.groupby('Master', group_keys=False)['Has_Flag']
                    .max()
                    .fillna(0).astype(int)
                    .to_dict()
                )
            
            for master in sorted(all_masters):
                node_id = f"MST_{master}"
                has_flag = has_flag_by_master.get(master, 0)
                if has_flag == 1:
                    fill = '#cefec2'   # 초록
                elif has_flag == 2:
                    fill = '#fdf476'   # 노랑
                else:
                    fill = '#e5e5db'   # 회색
                diagram_dot.attr('node', shape='ellipse', style='filled',
                               fillcolor=fill, color='black', fontname='Malgun Gothic')
                diagram_dot.node(node_id, str(master))
            
            # 연결: Activities → System → Master
            MAX_EDGES = 2000  # 안전장치
            edge_count = 0
            for activity_seq, data in activity_map.items():
                activity_node_id = f"ACT_{activity_seq}"
                systems = data['systems']
                masters = data['masters']
                
                # Activities → System 연결
                for system in systems:
                    system_node_id = f"SYS_{system}"
                    if activity_seq in all_activities:
                        diagram_dot.edge(activity_node_id, system_node_id, 
                                        style='solid', color='blue')
                        edge_count += 1
                        if edge_count >= MAX_EDGES:
                            break
                    
                    # System → Master 연결
                    for master in masters:
                        master_node_id = f"MST_{master}"
                        diagram_dot.edge(system_node_id, master_node_id, 
                                        style='solid', color='green')
                        edge_count += 1
                        if edge_count >= MAX_EDGES:
                            break
                    if edge_count >= MAX_EDGES:
                        break
                if edge_count >= MAX_EDGES:
                    break
            
            # 렌더링 & 표시
            actcnt = len(all_activities)
            syscnt = len(all_systems)
            mcnt = len(all_masters)
            st.write(f"📊 Activities 개수: {actcnt}, System 개수: {syscnt}, Master 개수: {mcnt}")  # Debug
            
            with st.spinner("Rendering Activities & System & Master diagram..."):
                timestamp = int(time.time() * 1000)
                filename = f"tmp_image/activities_system_master_{timestamp}"
                diagram_dot.render(filename, cleanup=True)
            
            # 이미지 크기 조건에 따라 설정
            total_nodes = actcnt + syscnt + mcnt
            if total_nodes < 5:
                width = 400
            elif total_nodes < 10:
                width = 600
            elif total_nodes < 15:
                width = 800
            elif total_nodes < 20:
                width = 1000
            else:
                width = 1200
            
            st.write(f"📊 총 노드 개수: {total_nodes}, 너비: {width}")  # Debug
            st.image(f"{filename}.png", width=width)
            
            valuechain_color_definition()
            
            return diagram_dot
            
        except Exception as e:
            st.error(f"Activities & System & Master 다이어그램 생성 중 오류 발생: {str(e)}")
            import traceback
            st.exception(e)
            return None

    def Activities_Systems_Diagram(self, valuechain_df, systems_df, title: str = "") -> Digraph:
        """
        System 다이어그램 (System 컬럼은 systems_df에 있음)
        
        Args:
            valuechain_df: Value Chain 데이터프레임
            systems_df: System 데이터프레임 (Activity_Seq, System 컬럼 포함)
        
        Returns:
            Digraph: 생성된 다이어그램 객체
        """
        try:
            # systems_df 확인
            if systems_df is None or systems_df.empty:
                st.info("📝 System 데이터가 없습니다.")
                return None
            
            # System 컬럼 확인
            if 'System' not in systems_df.columns:
                st.warning("⚠️ System 컬럼이 없습니다.")
                return None
            
            # Has_Flag 컬럼이 없으면 추가 (기본값 0)
            if 'Has_Flag' not in systems_df.columns:
                systems_df['Has_Flag'] = 0
            
            # _Activities_Master_Diagram_Base 함수를 사용하여 System 다이어그램 생성
            return self._Activities_Master_Diagram_Base(valuechain_df, systems_df, 'System', ' (System)')
            
        except Exception as e:
            st.error(f"System 다이어그램 생성 중 오류 발생: {str(e)}")
            return None

    def create_support_group_diagram(self, activities, masters,
                                     has_flag_by_master, masters_by_seq, group_num):
        """Support Activities 그룹 다이어그램 생성 - 별도 PNG 파일로 저장 (고속/안전)"""
        try:
            support_dot = Digraph(format='png')
            support_dot.attr(rankdir='TB', fontsize='12', size='12', dpi='100')

            # Activities 노드(박스)
            for activity_row in activities:
                seq = activity_row.get('Activity_Seq')
                node_id = f"SA{seq}"
                node_label = f"{activity_row.get('Activities','Unknown')}\n({activity_row.get('Activities_Kor','')})"
                support_dot.attr('node', shape='box', style='filled',
                                 fillcolor='#eef8d2', color='black', fontname='Malgun Gothic')
                support_dot.node(node_id, node_label)

            # Master 노드(원)
            for master in masters:
                node_id = f"SM{master}"
                node_label = str(master)
                has_flag = has_flag_by_master.get(master, 0)
                if has_flag == 1:
                    fill = '#cefec2'
                elif has_flag == 2:
                    fill = '#fdf476'
                else:
                    fill = '#e5e5db'
                support_dot.attr('node', shape='ellipse', style='filled',
                                 fillcolor=fill, color='black', fontname='Malgun Gothic')
                support_dot.node(node_id, node_label)

            # 엣지(박스→원) — 매핑 사용
            MAX_EDGES = 2000  # 안전장치
            edge_count = 0
            for activity_row in activities:
                seq = activity_row.get('Activity_Seq')
                for master in masters_by_seq.get(seq, []):
                    support_dot.edge(f"SA{seq}", f"SM{master}", style='solid', color='blue')
                    edge_count += 1
                    if edge_count >= MAX_EDGES:
                        break
                if edge_count >= MAX_EDGES:
                    break

            # 렌더링 & 표시
            mcnt = len(masters)
            
            with st.spinner(f"Rendering support group {group_num}..."):
                timestamp = int(time.time() * 1000)
                filename = f"tmp_image/support_group_{group_num}_{timestamp}"
                support_dot.render(filename, cleanup=True)

            # 이미지 크기 조건에 따라 설정
            if mcnt < 2:
                width = 200
            elif mcnt < 4:
                width = 400
            elif mcnt < 6:
                width = 500
            elif mcnt < 8:
                width = 600
            elif mcnt < 10:
                width = 1000
            elif mcnt < 12:
                width = 1200
            else:
                width = 1200

            st.write(f"📊 Support Group {group_num} - Master 개수: {mcnt}, 너비: {width}")  # Debug
            st.image(f"{filename}.png", width=width)

            return support_dot

        except Exception as e:
            st.error(f"Support 다이어그램 생성 중 오류: {str(e)}")
            return None

    def create_system_group_diagram(self, activities, systems,
                                     systems_by_seq, group_num):
        """System Activities 그룹 다이어그램 생성 - 별도 PNG 파일로 저장 (고속/안전)"""
        try:
            system_dot = Digraph(format='png')
            system_dot.attr(rankdir='TB', fontsize='12', size='12', dpi='100')

            # Activities 노드(박스)
            for activity_row in activities:
                seq = activity_row.get('Activity_Seq')
                node_id = f"SA{seq}"
                node_label = f"{activity_row.get('Activities','Unknown')}\n({activity_row.get('Activities_Kor','')})"
                system_dot.attr('node', shape='box', style='filled',
                                 fillcolor='#eef8d2', color='black', fontname='Malgun Gothic')
                system_dot.node(node_id, node_label)

            # System 노드(원) - Has_Flag 없이 단일 색상 사용
            for system in systems:
                node_id = f"SS{system}"
                node_label = str(system)
                # System은 Has_Flag가 없으므로 기본 색상 사용
                fill = '#cde3f1'  # 기본 파란색 계열
                system_dot.attr('node', shape='ellipse', style='filled',
                                 fillcolor=fill, color='black', fontname='Malgun Gothic')
                system_dot.node(node_id, node_label)

            # 엣지(박스→원) — 매핑 사용
            MAX_EDGES = 2000  # 안전장치
            edge_count = 0
            for activity_row in activities:
                seq = activity_row.get('Activity_Seq')
                for system in systems_by_seq.get(seq, []):
                    system_dot.edge(f"SA{seq}", f"SS{system}", style='solid', color='blue')
                    edge_count += 1
                    if edge_count >= MAX_EDGES:
                        break
                if edge_count >= MAX_EDGES:
                    break

            # 렌더링 & 표시
            mcnt = len(systems)
            
            with st.spinner(f"Rendering system group {group_num}..."):
                timestamp = int(time.time() * 1000)
                filename = f"tmp_image/system_group_{group_num}_{timestamp}"
                system_dot.render(filename, cleanup=True)

            # 이미지 크기 조건에 따라 설정
            if mcnt < 2:
                width = 300
            elif mcnt < 4:
                width = 500
            elif mcnt < 6:
                width = 600
            elif mcnt < 8:
                width = 800
            elif mcnt < 10:
                width = 900
            else:
                width = 1000

            st.write(f"📊 System Group {group_num} - System 개수: {mcnt}, 너비: {width}")  # Debug
            st.image(f"{filename}.png", width=width)

            return system_dot

        except Exception as e:
            st.error(f"System 다이어그램 생성 중 오류: {str(e)}")
            return None
        
    # def show_relationship_diagram_plotly2(self, selected_activity_seq):
    #     # Activity 및 Master 정보 필터링
    #     master_df = self.valuechain_standard_master.copy()
    #     master_df = master_df[master_df['Activity_Seq'].isin(selected_activity_seq)]

    #     if master_df.empty:
    #         st.warning("선택된 활동에 해당하는 Master 정보가 없습니다.")
    #         return

    #     selected_df = self.valuechain_data.copy()
    #     selected_df = selected_df[selected_df['Activity_Seq'].isin(selected_activity_seq)]

    #     # 활동 분리
    #     primary_df = selected_df[selected_df['Activities_Type'].str.strip() == 'Primary'].sort_values('Activity_Seq')
    #     support_df = selected_df[selected_df['Activities_Type'].str.strip() == 'Support'].sort_values('Activity_Seq')

    #     def draw_activity_master_plot(title, activity_df, activity_type):
    #         if activity_df.empty:
    #             return

    #         st.markdown(f"### {title}")
    #         fig = go.Figure()
    #         x_gap = 1.5
    #         y_activity = 1.0
    #         y_master = 0.0
    #         x_pos = {}

    #         # 활동 노드 (사각형 or 원)
    #         for idx, row in enumerate(activity_df.itertuples()):
    #             x = idx * x_gap
    #             x_pos[row.Activity_Seq] = x
    #             label = f"{row.Activities}<br>({row.Activities_Kor})"
    #             shape = "rect" if activity_type == "Primary" else "circle"
    #             fig.add_shape(
    #                 type=shape,
    #                 x0=x - 0.5, x1=x + 0.5,
    #                 y0=y_activity - 0.3, y1=y_activity + 0.3,
    #                 line=dict(color="black"),
    #                 fillcolor="lightblue" if activity_type == "Primary" else "#eef8d2"
    #             )
    #             fig.add_annotation(x=x, y=y_activity, text=label, showarrow=False, font=dict(size=14), align="center")

    #         # 마스터 노드 및 연결선
    #         used_masters = set()
    #         for idx, row in master_df.iterrows():
    #             seq = row.Activity_Seq
    #             master = row.Master
    #             if seq not in x_pos:
    #                 continue
    #             x = x_pos[seq]
    #             if master not in used_masters:
    #                 fig.add_shape(
    #                     type="rect",
    #                     x0=x - 0.3, x1=x + 0.3,
    #                     y0=y_master - 0.2, y1=y_master + 0.2,
    #                     line=dict(color="black"),
    #                     fillcolor="#f6e8c3"
    #                 )
    #                 fig.add_annotation(x=x, y=y_master, text=master, showarrow=False, font=dict(size=12), align="center")
    #                 used_masters.add(master)

    #             # 연결선
    #             fig.add_shape(
    #                 type="line",
    #                 x0=x, y0=y_activity - 0.3,
    #                 x1=x, y1=y_master + 0.2,
    #                 line=dict(color="gray", width=1)
    #             )

    #         fig.update_layout(
    #             height=500,
    #             width=max(len(activity_df), len(used_masters)) * 180,
    #             xaxis=dict(visible=False),
    #             yaxis=dict(visible=False),
    #             margin=dict(l=10, r=10, t=20, b=10),
    #             showlegend=False
    #         )
    #         st.plotly_chart(fig, use_container_width=False)
    #     # Primary Activities 표시
    #     draw_activity_master_plot("Primary Activities 다이어그램", primary_df, "Primary")

    #     # Support Activities 표시
    #     draw_activity_master_plot("Support Activities 다이어그램", support_df, "Support")

    def show_relationship_diagram_plotly(self, selected_activity_seq):
        import math
        title="활동-Master 연결도"
        valuechain_data = self.valuechain_data.copy()
        valuechain_standard_master = self.valuechain_standard_master.copy()

        # 활동 및 마스터 데이터 필터링
        activity_df = valuechain_data[valuechain_data['Activity_Seq'].isin(selected_activity_seq)]
        master_df = valuechain_standard_master[valuechain_standard_master['Activity_Seq'].isin(selected_activity_seq)]

        if activity_df.empty or master_df.empty:
            st.warning("활동 또는 Master 정보가 없습니다.")
            return None

        # 활동-Master 매핑 딕셔너리 생성
        activity_master_map = {}
        for _, row in activity_df.iterrows():
            seq = row['Activity_Seq']
            label = f"{row['Activities']}<br>({row['Activities_Kor']})"
            masters = master_df[master_df['Activity_Seq'] == seq]['Master'].tolist()
            activity_master_map[label] = masters

        # 노드 배치 및 크기 조정
        fig = go.Figure()
        x_gap = 2
        y_activity = 2.0
        y_master = 0.2

        num_activities = len(activity_master_map)
        num_masters = sum(len(m) for m in activity_master_map.values())
        width = max(800, (num_activities * x_gap + 1) * 100)
        height = 400 + (math.ceil(num_masters / max(num_activities, 1)) * 80)

        for i, (activity, masters) in enumerate(activity_master_map.items()):
            x = i * x_gap

            # 활동 박스
            fig.add_shape(
                type="rect",
                x0=x - 0.5, x1=x + 0.5,
                y0=y_activity - 0.3, y1=y_activity + 0.3,
                line=dict(color="black"),
                fillcolor="lightblue"
            )
            fig.add_annotation(
                x=x, y=y_activity,
                text=activity,
                showarrow=False,
                font=dict(size=14),
                align="center"
            )

            # 마스터 원 및 선 연결
            for j, master in enumerate(masters):
                mx = x - 0.6 + j * 0.6
                fig.add_shape(
                    type="circle",
                    x0=mx - 0.3, x1=mx + 0.3,
                    y0=y_master - 0.2, y1=y_master + 0.2,
                    line=dict(color="black"),
                    fillcolor="#efe5d1"
                )
                fig.add_annotation(
                    x=mx, y=y_master,
                    text=master,
                    showarrow=False,
                    font=dict(size=11),
                    align="center"
                )
                fig.add_shape(
                    type="line",
                    x0=x, y0=y_activity - 0.3,
                    x1=mx, y1=y_master + 0.2,
                    line=dict(color="gray", width=1)
                )

        # 전체 레이아웃 설정
        fig.update_layout(
            title=title,
            height=height,
            width=width,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=False)

    # # Master 기준 페이지 분할 다이어그램 (master_based_paged_diagram)
    # def Master_Detail_Diagram(self, df: pd.DataFrame, max_columns_per_master=10):
    #     st.markdown("#### 📄 Master & Attribute Relationship Diagram")

    #     required_cols = ['Master', 'Column_Name', 'Column_Kor', 'Mandatory']
    #     if not all(col in df.columns for col in required_cols):
    #         st.warning("필수 컬럼이 누락되어 있습니다: Master, Column_Name, Mandatory")
    #         return

    #     # ✅ Master 기준으로 정렬된 고유 마스터 리스트
    #     master_list = sorted(df['Master'].dropna().unique().tolist())
    #     total_pages = len(master_list)

    #     # ✅ Streamlit 페이지 선택 UI
    #     page = st.number_input("Select Page (Base on Master)", min_value=1, max_value=total_pages, value=1, step=1)
    #     current_master = master_list[page - 1]

    #     # ✅ 현재 Master에 해당하는 컬럼들 가져오기
    #     sub_df = df[df['Master'] == current_master].copy()
    #     sub_df = sub_df.head(max_columns_per_master)

    #     # ✅ Graphviz 그림 생성
    #     dot = Digraph(format='png')
    #     dot.attr(rankdir='TB', fontsize='12', size='14', dpi='100')

    #     master_node_id = f"MASTER_{page}_{current_master}"
        
    #     dot.attr('node', shape='box', style='filled', fillcolor='#cde3f1', fontname='Malgun Gothic')
    #     dot.node(master_node_id, current_master)

    #     for idx, row in sub_df.iterrows():
    #         col_name = str(row['Column_Name']).strip()
    #         col_kor = str(row['Column_Kor']).strip()
    #         if len(col_name) > 9:
    #             col_name = col_name[:9] + str(idx)
    #         if len(col_kor) > 9:
    #             col_kor = col_kor[:9] + str(idx)
    #         mandatory = str(row['Mandatory']).strip()
    #         col_id = f"COLUMN_{page}_{col_name}"

    #         label = f"{col_name}\n({col_kor})"

    #         color = '#aefc2f' if mandatory == '*' else '#fdf594'
            
    #         dot.attr('node', shape='ellipse', style='filled', fillcolor=color)
    #         dot.node(col_id, label)
    #         dot.edge(master_node_id, col_id, color='gray')

    #     # ✅ 이미지로 렌더링 후 출력
    #     # timestamp = int(time.time() * 1000)
    #     filename = f"tmp_image/master_page.png"

    #     dot.render(filename, cleanup=True)
        
    #     st.divider()

    #     if len(sub_df) < 2:
    #         st.image(f"{filename}.png", width=200)
    #     elif len(sub_df) < 4:
    #         st.image(f"{filename}.png", width=600)
    #     elif len(sub_df) < 6:
    #         st.image(f"{filename}.png", width=800)
    #     elif len(sub_df) < 8:
    #         st.image(f"{filename}.png", width=1000)
    #     else:
    #         st.image(f"{filename}.png", width=1000)

    #     st.divider()

    #     st.markdown("**박스는 Master, 원은 속성코드 입니다.**")
    #     st.write(f"Green: 필수 속성코드 입니다.")
    #     st.write(f"속성코드에 대한 상세 정보는 위의 탭을 이용하여 확인하세요.")

    #     return None

class DashboardManager:
    """대시보드 관리를 위한 클래스"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.file_loader = FileLoader(yaml_config)
        self.value_chain_diagram = ValueChainDiagram(yaml_config)
    
    def display_value_chain_dashboard(self) -> bool:
        """Value Chain 대시보드 표시"""
        try:
            loaded_data = self.file_loader.load_all_files() # 모든 파일 로드

            # load 된 모든 데이터프레임에서 Activity_Seq 컬럼을 문자열로 통일
            for name, df in loaded_data.items():
                if isinstance(df, pd.DataFrame) and 'Activity_Seq' in df.columns:
                    df['Activity_Seq'] = df['Activity_Seq'].astype(str).str.strip().str.replace('.0', '', regex=False)

            col1, col2 = st.columns([2, 8])
            with col1:
                selected_industry = st.selectbox("Industry를 선택하세요", loaded_data['valuechain']['Industry'].unique())
            
                vc_df = loaded_data['valuechain'][loaded_data['valuechain']['Industry'] == selected_industry].copy()
                sm_df = loaded_data['valuechain_standard_master'][loaded_data['valuechain_standard_master']['Industry'] == selected_industry].copy()
                systems_df = loaded_data['valuechain_system'][loaded_data['valuechain_system']['Industry'] == selected_industry].copy()
                sm_df = sm_df[['Activity_Seq', 'Master', 'Master_Kor', 'Our_Master']]
                systems_df = systems_df[['Activity_Seq', 'System']].copy()

                # Has_Flag 컬럼 생성
                def calculate_has_flag(row):
                    master_empty = pd.isna(row['Master']) or str(row['Master']).strip() == ''
                    our_master_exists = pd.notna(row['Our_Master']) and str(row['Our_Master']).strip() != ''
                    
                    if master_empty and our_master_exists:  # Master에 값이 없고, Our_Master에 값이 있으면 2
                        return 2
                    elif our_master_exists: # Our_Master에 값이 있으면 1
                        return 1
                    else: # 그 외: 0
                        return 0
                
                sm_df['Has_Flag'] = sm_df.apply(calculate_has_flag, axis=1)

                org_df = pd.merge(sm_df, vc_df, on='Activity_Seq', how='left') # 원본 데이터 복사
                # Master 컬럼 업데이트 로직:
                sm_df['Master'] = sm_df.apply(
                    lambda row: (
                        row['Our_Master'] if row['Has_Flag'] == 2 
                        and pd.notna(row['Our_Master']) 
                        and str(row['Our_Master']).strip() != ''
                        else row['Master']
                    ), 
                    axis=1
                )

                # Master_Kor에 값이 없으면: Master에 값이 있으면 Master, Master에 값이 없으면 Our_Master, Our_Master에 값이 없으면 Master_Kor 값 유지       
                sm_df['Master_Kor'] = sm_df.apply(
                    lambda row: (
                        row['Master_Kor'] if pd.notna(row['Master_Kor']) and str(row['Master_Kor']).strip() != ''
                        else (
                            row['Master'] if pd.notna(row['Master']) and str(row['Master']).strip() != ''
                            else (
                                row['Our_Master'] if pd.notna(row['Our_Master']) and str(row['Our_Master']).strip() != ''
                                else row['Master_Kor']  # 모두 없으면 원래 값 유지
                            )
                        )
                    ), 
                    axis=1
                )

                sm_df['Master2'] = sm_df.apply(
                    lambda row: (
                        row['Our_Master'] if row['Has_Flag'] != 0 
                        and pd.notna(row['Our_Master']) 
                        and str(row['Our_Master']).strip() != ''
                        else row['Master']
                    ), 
                    axis=1
                )
                df = pd.merge(sm_df, vc_df, on='Activity_Seq', how='left')
            

            # ValueChain 데이터 로드
            if not self.value_chain_diagram.load_all_valuechain_data(loaded_data):
                st.error("Value Chain 및 Master Table을 정의한 메타파일을 로드할 수 없습니다.")
                return False

            self.value_chain_diagram.valuechain_summary(vc_df) # Value Chain 데이터 요약
            
            fig = self.value_chain_diagram.valuechain_diagram(vc_df)

            st.dataframe(vc_df, use_container_width=False, hide_index=True, height=550, width=1200) # Debug

            tab1_title = "Value Chain & Standard Master Diagram"
            tab2_title = "Value Chain & Standard Master Kor Diagram"
            tab3_title = "Value Chain & Our Master Diagram"
            tab4_title = "Value Chain & System Diagram"
            # tab5_title = "System & Master Relationship Diagram"
            tab1, tab2, tab3, tab4 = st.tabs([tab1_title, tab2_title, tab3_title, tab4_title])
            with tab1:
                self.value_chain_diagram.Activities_Master_Diagram(vc_df, sm_df, tab1_title)
            with tab2:
                self.value_chain_diagram.Activities_Master_Diagram_Kor(vc_df, sm_df, tab2_title)
            with tab3:
                self.value_chain_diagram.Activities_Master_Diagram_Our(vc_df, sm_df, tab3_title)
            with tab4:
                self.value_chain_diagram.Activities_Systems_Diagram(vc_df, systems_df, tab4_title)
            # with tab5:
            #     self.value_chain_diagram.System_Master_Diagram(vc_df, systems_df, sm_df, tab5_title)

            st.divider()
            st.markdown(f"#### ({selected_industry}) Industry Value Chain & Master Information")
            self.value_chain_diagram.ValueChain_Master_Information(org_df)

            # tab1, tab2= st.tabs(["Process/Function & Master Diagram", "Value Chain & Master Information"])
            # with tab1:
            #     self.value_chain_diagram.Activities_Master_Diagram()
                
            # with tab2:
            #     self.value_chain_diagram.ValueChain_Master_Information()
                    
            # else:
            #     show_standard_master = False
            #     selected_activity_seq = []
            
            # # 2nd 데이터 정보 표시 (선택된 Activities 포함)
            # selected_activities_2nd = self.value_chain_diagram.display_valuechain_data_2nd()
            # if selected_activities_2nd is not None and not selected_activities_2nd.empty:
            #     master = loaded_data['valuechain_standard_master'];
            #     master = master[master['Activity_Seq'].isin(selected_activities_2nd['Activity_Seq'].tolist())]

            #     detail_master = loaded_data['valuechain_standard_master_detail'];

            #     df = pd.merge(master, detail_master, on='Master', how='left')

            #     df = df[ (df['Mandatory'].astype(str).str.len() > 0) | (df['Reference_Code'].astype(str).str.len() > 0)]

            #     tab1, tab2 = st.tabs([ "Master & Code Detail Diagram", "Code Attribute Detail Information"])
            #     with tab1:
            #         self.value_chain_diagram.Master_Detail_Diagram(df) # 다이어그램 생성 및 표시

            #     with tab2:
            #         df = df[ (df['Mandatory'].astype(str).str.len() > 0)]
            #         df = df[['Master', 'Master_Kor', 'Our_Master', 'Column_Name', 'Mandatory', 'Reference_Code']]
            #         st.markdown("#### Code Attribute Detail Information")
            #         st.dataframe(df, hide_index=True, height=500, use_container_width=True)

            #     st.divider()
            # else:
            #     st.write("상세 정보를 보기 위한 Activities를 선택하세요.")

            # return True
            
        except Exception as e:
            st.error(f"대시보드 표시 중 오류 발생: {str(e)}")
            return False

class FilesInformationApp:
    """Files Information 애플리케이션 메인 클래스"""
    
    def __init__(self):
        self.yaml_config = None
        self.dashboard_manager = None
    
    def initialize(self) -> bool:
        """애플리케이션 초기화"""
        try:
            self.yaml_config = load_yaml_datasense() # YAML 파일 로드
            if self.yaml_config is None:
                st.error("YAML 파일을 로드할 수 없습니다.")
                return False
            
            set_page_config(self.yaml_config) # 페이지 설정
            
            self.dashboard_manager = DashboardManager(self.yaml_config) # 대시보드 매니저 초기화
            
            return True
            
        except Exception as e:
            st.error(f"애플리케이션 초기화 중 오류 발생: {str(e)}")
            return False
    
    def run(self):
        """애플리케이션 실행"""
        st.title(APP_NAME)
        st.markdown(APP_DESC)
        st.divider()
        try:
            success = self.dashboard_manager.display_value_chain_dashboard()
        except Exception as e:
            st.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")

def main():
    """메인 함수"""

    app = FilesInformationApp()
    
    if app.initialize():
        app.run()
    else:
        st.error("애플리케이션 초기화 실패")

if __name__ == "__main__":
    main()
