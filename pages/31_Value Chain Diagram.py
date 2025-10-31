#
# streamlit를 이용한 Value Chain
# 2025. 7. 23.  Qliker
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
APP_DESC = "###### Value Chain의 Process/Function 과 Master Table 간의 Relationship Diagram 입니다."
APP_DESC2 = "###### 대상 사용자 : 데이터 거버넌스 관리자, 마스터데이터 담당자, 프로세스 아키텍트"

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Now import utils after adding to path
from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config

def html_escape(text: str) -> str:
    """Graphviz HTML-safe 텍스트 이스케이프"""
    if not isinstance(text, str):
        return ''
    return html.escape(text, quote=True)

def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def display_valuechain_sample_image():
    sample_image = "DataSense/DS_Output/valuechain_primary_sample.png"
    filepath = os.path.join(PROJECT_ROOT, sample_image)

    if not os.path.exists(filepath):
        st.error(f"Sample Image 파일이 존재하지 않습니다: {filepath}")
        return False
    st.markdown("#### Value Chain Primary Sample Image")
    image = Image.open(filepath)
    st.image(image, caption="Value Chain Primary Sample Image", width=1000)

    sample_image = "DataSense/DS_Output/valuechain_support_sample.png"
    filepath = os.path.join(PROJECT_ROOT, sample_image)

    if not os.path.exists(filepath):
        st.error(f"Sample Image 파일이 존재하지 않습니다: {filepath}")
        return False
    st.markdown("#### Value Chain Support Sample Image")
    image = Image.open(filepath)
    st.image(image, caption="Value Chain Support Sample Image", width=1000)
    return True

@dataclass
class FileConfig:
    """파일 설정 정보"""
    valuechain: str
    valuechain_standard_master: str
    valuechain_standard_master_detail: str

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
            valuechain_standard_master_detail=_full_path(files.get('valuechain_standard_master_detail', 'DataSense/DS_Meta/DataSense_ValueChain_Standard_Master_Detail.csv'))
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
            'valuechain_standard_master_detail': self.files_config.valuechain_standard_master_detail
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
        self.valuechain_standard_master_detail = None
    
    def load_all_valuechain_data(self, loaded_data: Dict[str, pd.DataFrame]) -> bool:
        """모든 ValueChain 관련 데이터 로드"""
        try:
            # 기본 파일 유무 확인
            self.valuechain_data = loaded_data.get('valuechain', pd.DataFrame())
            self.valuechain_standard_master = loaded_data.get('valuechain_standard_master', pd.DataFrame())
            self.valuechain_standard_master_detail = loaded_data.get('valuechain_standard_master_detail', pd.DataFrame())

            if self.valuechain_data.empty:
                st.warning("⚠️ ValueChain 데이터가 없습니다. 기본 예시를 표시합니다.")
                return False

            # 모든 Activity_Seq를 문자열로 통일
            for df in [self.valuechain_data, self.valuechain_standard_master, self.valuechain_standard_master_detail]:
                if 'Activity_Seq' in df.columns:
                    df['Activity_Seq'] = df['Activity_Seq'].astype(str)

            return True
        except Exception as e:
            st.error(f"ValueChain 데이터 로드 실패: {str(e)}")
            return False

    # def create_diagram_from_data(self) -> Digraph:
    #     import graphviz

    #     if self.valuechain_data is None or self.valuechain_data.empty:
    #         st.warning("ValueChain 데이터가 없습니다.")
    #         st.warning("다음은 예제 그림입니다. 실제 데이터를 입력하세요.")
    #         return self.create_default_diagram()

    #     dot = graphviz.Digraph(format='png')
    #     dot.graph_attr.update(rankdir='LR', fontsize='10', fontname='Malgun Gothic')
    #     dot.node_attr.update(fontname='Malgun Gothic')
    #     dot.edge_attr.update(fontname='Malgun Gothic')

    #     # Primary Activities
    #     primary_df = self.valuechain_data[
    #         self.valuechain_data['Activities_Type'].str.strip() == 'Primary'
    #     ].sort_values('Activity_Seq')

    #     primary_ids = []
    #     for row in primary_df.itertuples():
    #         pid = f"P{row.Activity_Seq}"
    #         en = html_escape(row.Activities)
    #         ko = html_escape(row.Activities_Kor)
    #         label = f"""<
    #         <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
    #             <TR><TD><B>{en}</B></TD></TR>
    #             <TR><TD>({ko})</TD></TR>
    #         </TABLE>
    #         >"""
    #         dot.node(pid, label=label, shape='box', style='filled', color='lightblue2')
    #         primary_ids.append(pid)

    #     for i in range(len(primary_ids) - 1):
    #         dot.edge(primary_ids[i], primary_ids[i + 1], style='bold', weight='10')

    #     # Support Activities
    #     support_df = self.valuechain_data[
    #         self.valuechain_data['Activities_Type'].str.strip() == 'Support'
    #     ].sort_values('Activity_Seq')

    #     support_ids = []
    #     with dot.subgraph(name='cluster_support') as s:
    #         s.attr(rank='same')
    #         for row in support_df.itertuples():
    #             sid = f"S{row.Activity_Seq}"
    #             en = html_escape(row.Activities)
    #             ko = html_escape(row.Activities_Kor)
    #             label = f"""<
    #             <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
    #                 <TR><TD><B>{en}</B></TD></TR>
    #                 <TR><TD>({ko})</TD></TR>
    #             </TABLE>
    #             >"""
    #             s.node(sid, label=label, shape='ellipse', style='filled', color='#eef8d2')
    #             support_ids.append(sid)

    #         if len(support_ids) > 1:
    #             s.node('support_left', '', width='0', height='0', style='invis')
    #             s.node('support_right', '', width='0', height='0', style='invis')
    #             s.edge('support_left', support_ids[0], style='invis', weight='100')
    #             s.edge(support_ids[-1], 'support_right', style='invis', weight='100')

    #     if primary_ids and support_ids:
    #         dot.edge(primary_ids[0], support_ids[0], style='invis', weight='1000')

    #     return dot

    def create_plotly_diagram(self):
        st.markdown("---")
        st.markdown("#### 📄 Value Chain Diagram")
        if self.valuechain_data is None or self.valuechain_data.empty:
            st.warning("ValueChain 데이터가 없습니다.")
            return None

        # 데이터 정렬 (Primary & Support)
        self.valuechain_data['Activities_Type'] = self.valuechain_data['Activities_Type'].str.strip()

        primary_df = self.valuechain_data[self.valuechain_data['Activities_Type'] == 'Primary'] \
            .sort_values('Activity_Seq').reset_index(drop=True)

        support_df = self.valuechain_data[self.valuechain_data['Activities_Type'] == 'Support'] \
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


    def create_default_diagram(self) -> Digraph:
        """기본 Value Chain 다이어그램 생성"""
        dot = Digraph(format='png')
        dot.attr(rankdir='LR', fontsize='10')

        # Primary Activities
        dot.attr('node', shape='box', style='filled', color='lightblue2')
        dot.node('P0', 'Inbound Logistics\n(수거/접수)\nKPI: 수거율\n시스템: TMS')
        dot.node('P1', 'Operations\n(허브 처리)\nKPI: 자동분류율\n시스템: WMS')
        dot.node('P2', 'Outbound Logistics\n(배송)\nKPI: 배송완료율\n시스템: PDA/모바일')
        dot.node('P3', 'Marketing & Sales\nKPI: 고객유치율\n시스템: CRM')
        dot.node('P4', 'Service\n(CS/반품)\nKPI: 클레임 처리율\n시스템: VOC 시스템')

        # Support Activities
        dot.attr('node', shape='ellipse', style='filled', color='#eef8d2', border='black')
        dot.node('Infra', 'Infrastructure\n(재무/법무)\n시스템: ERP')
        dot.node('HR', 'HR Management\n(인력 운영)\n시스템: HRIS')
        dot.node('Tech', 'Technology\n(TES, 자동화)\n시스템: AI/IoT')
        dot.node('Procure', 'Procurement\n(설비/차량)\n시스템: 자산관리')

        # Edges
        dot.edges([('P0', 'P1'), ('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4')])
        for support in ['Infra', 'HR', 'Tech', 'Procure']:
            for primary in ['P0', 'P1', 'P2', 'P3', 'P4']:
                dot.edge(support, primary, style='dashed', color='lightgrey')

        return dot
    
    def valuechain_summary(self):
        """Value Chain Summary 정보 표시"""
        if self.valuechain_data is not None and not self.valuechain_data.empty:
            st.markdown("### Value Chain Process/Function Summary")
            # 데이터 요약
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                industry = self.valuechain_data['Industry'].unique().tolist()
                st.metric("Industry", industry[0] if industry else "N/A")
            with col2:
                st.metric("Total Process & Function", len(self.valuechain_data))
            with col3:
                primary_count = len(self.valuechain_data[
                    self.valuechain_data['Activities_Type'].str.strip() == 'Primary'
                ]) if 'Activities_Type' in self.valuechain_data.columns else 0
                st.metric("Mega Process", primary_count)
            with col4:
                support_count = len(self.valuechain_data[
                    self.valuechain_data['Activities_Type'].str.strip() == 'Support'
                ]) if 'Activities_Type' in self.valuechain_data.columns else 0
                st.metric("Support Function", support_count)        
    
    def display_valuechain_data(self):
        """ValueChain 데이터 정보 표시"""
        global selected_activities_global
        
        if self.valuechain_data is not None and not self.valuechain_data.empty:
            st.markdown("##### Value Chain의 process/function을 체크하면 선택된 process/function과 Master 간의 관계도를 생성합니다.")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1: # Primary Activities
                
                primary_activities = self.valuechain_data[
                    self.valuechain_data['Activities_Type'].str.strip() == 'Primary'
                ]
                
                if not primary_activities.empty:
                    st.markdown("#### Mega Process")
                    for idx, row in primary_activities.iterrows():
                        activity_seq = str(row['Activity_Seq'])
                        activity_name = row['Activities']
                        activity_kor = row['Activities_Kor']
                        
                        # 안정적인 개별 선택 (체크박스 사용)
                        is_selected = st.checkbox(
                            f"{activity_name} ({activity_kor})",
                            value=activity_seq in selected_activities_global,
                            key=f"primary_{activity_seq}_{idx}"
                        )
                        
                        # 상태 업데이트
                        if is_selected:
                            selected_activities_global.add(activity_seq)
                        else:
                            selected_activities_global.discard(activity_seq)
            with col2: # Support Activities
                support_activities = self.valuechain_data[
                    self.valuechain_data['Activities_Type'].str.strip() == 'Support'
                ]
                
                if not support_activities.empty:
                    st.markdown("#### Support Function")
                    for idx, row in support_activities.iterrows():
                        activity_seq = str(row['Activity_Seq'])
                        activity_name = row['Activities']
                        activity_kor = row['Activities_Kor']
                        
                        # 안정적인 개별 선택 (체크박스 사용)
                        is_selected = st.checkbox(
                            f"{activity_name} ({activity_kor})",
                            value=activity_seq in selected_activities_global,
                            key=f"support_{activity_seq}_{idx}"
                        )
                        
                        # 상태 업데이트
                        if is_selected:
                            selected_activities_global.add(activity_seq)
                        else:
                            selected_activities_global.discard(activity_seq)
            
            st.markdown("---")           
            # 선택된 Activities 수집
            selected_activities = []
            for idx, row in self.valuechain_data.iterrows():
                activity_seq = str(row['Activity_Seq'])
                if activity_seq in selected_activities_global:
                    selected_activities.append(row)
            
            # 선택된 Activities를 DataFrame으로 반환
            if selected_activities:
                return pd.DataFrame(selected_activities)
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def display_valuechain_data_2nd(self):
        """ValueChain 데이터 정보 표시"""
        global selected_activities_global_2nd
        
        if self.valuechain_data is not None and not self.valuechain_data.empty:
            st.write("-"*100)
            st.markdown("### Detail Process/Function & Master Relationship")

            # Primary Activities와 Support Activities 모두 수집
            primary_activities = self.valuechain_data[
                self.valuechain_data['Activities_Type'].str.strip() == 'Primary'
            ]
            
            support_activities = self.valuechain_data[
                self.valuechain_data['Activities_Type'].str.strip() == 'Support'
            ]
            
            # 모든 활동 옵션과 라벨 수집
            all_activity_options = []
            all_activity_labels = []
            activity_types = []  # Primary인지 Support인지 구분
            
            # Primary Activities 추가
            for idx, row in primary_activities.iterrows():
                activity_seq = str(row['Activity_Seq'])
                activity_name = row['Activities']
                activity_kor = row['Activities_Kor']
                all_activity_options.append(activity_seq)
                all_activity_labels.append(f"[Primary] {activity_name} ({activity_kor})")
                activity_types.append("Primary")
            
            # Support Activities 추가
            for idx, row in support_activities.iterrows():
                activity_seq = str(row['Activity_Seq'])
                activity_name = row['Activities']
                activity_kor = row['Activities_Kor']
                all_activity_options.append(activity_seq)
                all_activity_labels.append(f"[Support] {activity_name} ({activity_kor})")
                activity_types.append("Support")
            
            # 현재 선택된 값 찾기
            current_selection = None
            if selected_activities_global_2nd:
                current_selection = list(selected_activities_global_2nd)[0]
            
            # 통합 라디오 버튼으로 선택 (Primary + Support 모두 포함)
            if all_activity_options:
                selected_activity = st.radio(
                    "Select Process/Function (Primary + Support):",
                    options=all_activity_options,
                    index=all_activity_options.index(current_selection) if current_selection in all_activity_options else 0,
                    format_func=lambda x: all_activity_labels[all_activity_options.index(x)],
                    key="unified_activity_radio"
                )
                
                # 선택된 활동만 유지하고 나머지는 제거
                selected_activities_global_2nd.clear()
                if selected_activity:
                    selected_activities_global_2nd.add(selected_activity)
            
    
            # 선택된 Activities 수집
            selected_activities = []
            for idx, row in self.valuechain_data.iterrows():
                activity_seq = str(row['Activity_Seq'])
                if activity_seq in selected_activities_global_2nd:
                    selected_activities.append(row)
        
            # 선택된 Activities를 DataFrame으로 반환
            if selected_activities:
                return pd.DataFrame(selected_activities)
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def display_standard_master_info(self, selected_activity_seq):
        """ValueChain Standard Master 정보 표시"""
        try:
            df = self.valuechain_standard_master.copy()
                
            # 선택된 활동에 해당하는 Master 찾기
            master_df = df[df['Activity_Seq'].isin(selected_activity_seq)]
            
            st.markdown("#### Value Chain Standard Master")
            st.markdown(f"**선택된 활동의 Master Code ({len(master_df)}개)**")
            # st.table(master_df)
            st.dataframe(master_df, use_container_width=False, hide_index=True, height=700, width=1400,
                         column_config={
                             "Activities_Type": st.column_config.TextColumn(width=100),
                             "Activity_Seq": st.column_config.NumberColumn(format="%d", width=70),
                             "Activities": st.column_config.TextColumn(width=200),
                             "Master": st.column_config.TextColumn(width=200),
                             "Master_Kor": st.column_config.TextColumn(width=200), 
                             "Master Description": st.column_config.TextColumn(width=500),
                         })

        except Exception as e:
            st.error(f"Standard Master 표시 중 오류 발생: {str(e)}")

    
    def display_standard_master_detail_info(self, selected_activity_seq):
        """ValueChain Standard Master Detail 정보 표시"""
        try:
            df = self.valuechain_standard_master_detail.copy()
            
            master_df = self.valuechain_standard_master.copy()
            master_df = master_df[master_df['Activity_Seq'].isin(selected_activity_seq)]
            
            # st.dataframe(master_df)
            selected_masters = master_df['Master'].str.strip().tolist()
            df = df[df['Master'].str.strip().isin(selected_masters)]
            df = df.merge(master_df, on='Master', how='left')
            df = df.fillna('')
            df = df[['Activities_Type', 'Activities', 'Master', 'Master_Kor', 'Column_Name', 'Column_Kor', 'Mandatory', 'Reference_Code', 'Remark']]
            
            st.markdown("### Value Chain Standard Master Detail")
            st.markdown(f"**선택된 활동의 상세 정보 ({len(df)}개) 속성 정보**")
            st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Standard Master Detail 표시 중 오류 발생: {str(e)}")


    def Activities_Master_Diagram(self, selected_activity_seq) -> Digraph:
        try:
            # ✅ 선택된 seq도 문자열로 통일
            selected_activity_seq = [str(x).strip().replace('.0', '') for x in selected_activity_seq if pd.notna(x)]

            master_df = self.valuechain_standard_master.copy()
            master_df['Activity_Seq'] = master_df['Activity_Seq'].astype(str).str.strip().str.replace('.0', '', regex=False)
            master_df = master_df[master_df['Activity_Seq'].isin(selected_activity_seq)]

            if master_df.empty:
                st.warning("선택된 Process/Function에 해당하는 Master 정보가 없습니다.")
                return None

            activities_df = self.valuechain_data.copy()
            selected_activities_df = activities_df[activities_df['Activity_Seq'].isin(selected_activity_seq)]

            primary_activities = selected_activities_df[selected_activities_df['Activities_Type'].str.strip() == 'Primary']
            support_activities = selected_activities_df[selected_activities_df['Activities_Type'].str.strip() == 'Support']

            # ---------- 사전 매핑: 반복 필터 제거의 핵심 ----------
            # seq -> {masters}
            masters_by_seq: dict[int, set] = (
                master_df.groupby('Activity_Seq')['Master']
                .apply(lambda s: set(map(str, s.tolist())))
                .to_dict()
            )

            # master -> Has_Flag(최댓값 우선)
            has_flag_by_master: dict[str, int] = (
                master_df.assign(Master=master_df['Master'].astype(str))
                         .groupby('Master')['Has_Flag']
                         .max()
                         .fillna(0).astype(int)
                         .to_dict()
            )

            _ensure_dir("tmp_image")
            # ---------- Primary Activities 다이어그램 ----------
            primary_dot = None
            if not primary_activities.empty:
                st.markdown("#### Mega Process & Master Relationship Diagram")

                primary_dot = Digraph(format='png')
                primary_dot.attr(rankdir='LR', fontsize='10', fontname='NanumGothic')

                # Primary Activities 노드(박스)
                primary_sorted = primary_activities.sort_values('Activity_Seq')
                for _, row in primary_sorted.iterrows():
                    seq = row.get('Activity_Seq')
                    node_id = f"PA{seq}"
                    node_label = f"{row.get('Activities', 'Unknown')}\n({row.get('Activities_Kor', '')})"
                    primary_dot.attr('node', shape='box', style='filled',
                                     fillcolor='lightblue2', color='black', fontname='NanumGothic')
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
                                     fillcolor=fill, color='black', fontname='NanumGothic')
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
                width = 100 if mcnt < 2 else 400 if mcnt < 4 else 600 if mcnt < 6 else 800 if mcnt < 8 else 900
                st.image(f"{primary_filename}.png", width=width)

            # ---------- Support Activities 다이어그램 (그룹 페이징) ----------
            if not support_activities.empty:
                st.markdown("#### Support Function & Master Relationship Diagram")

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
                    self.create_support_group_diagram(
                        activities=current_activities_rows,
                        masters=current_masters,
                        has_flag_by_master=has_flag_by_master,
                        masters_by_seq=masters_by_seq,
                        group_num=graph_count
                    )

            st.divider()
            st.markdown("**박스는 Process/Function, 원은 Master 입니다.**")
            st.divider()
            st.markdown("##### Master Color Definition:")
            st.markdown("###### Green: Standard Master에도 있고, 우리 회사에도 관리하는 Master")
            st.markdown("###### Yellow: Standard Master에는 없지만, 우리 회사에는 관리하는 Master")
            st.markdown("###### Gray: Standard Master에는 있지만, 우리 회사에는 관리하지 않는 Master")

        
            return primary_dot if (not primary_activities.empty) else None

        except Exception as e:
            # st.error(f"다이어그램 생성 중 오류 발생: {str(e)}")
            st.info("Cloud 환경에서는 Diagram을 생성할 수 없습니다. Local 환경에서 실행해주세요. 샘플 이미지를 표시합니다.")
            display_valuechain_sample_image()
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
                                 fillcolor='#eef8d2', color='black', fontname='NanumGothic')
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
                                 fillcolor=fill, color='black', fontname='NanumGothic')
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
            with st.spinner(f"Rendering support group {group_num}..."):
                timestamp = int(time.time() * 1000)
                filename = f"tmp_image/support_group_{group_num}_{timestamp}"
                support_dot.render(filename, cleanup=True)

            mcnt = len(masters)
            width = 100 if mcnt < 2 else 400 if mcnt < 4 else 500 if mcnt < 6 else 800 if mcnt < 8 else 900
            st.image(f"{filename}.png", width=width)

            return support_dot

        except Exception as e:
            st.error(f"Support 다이어그램 생성 중 오류: {str(e)}")
            return None
        
    def show_relationship_diagram_plotly(self, selected_activity_seq):
        # Activity 및 Master 정보 필터링
        master_df = self.valuechain_standard_master.copy()
        master_df = master_df[master_df['Activity_Seq'].isin(selected_activity_seq)]

        if master_df.empty:
            st.warning("선택된 활동에 해당하는 Master 정보가 없습니다.")
            return

        selected_df = self.valuechain_data.copy()
        selected_df = selected_df[selected_df['Activity_Seq'].isin(selected_activity_seq)]

        # 활동 분리
        primary_df = selected_df[selected_df['Activities_Type'].str.strip() == 'Primary'].sort_values('Activity_Seq')
        support_df = selected_df[selected_df['Activities_Type'].str.strip() == 'Support'].sort_values('Activity_Seq')

        def draw_activity_master_plot(title, activity_df, activity_type):
            if activity_df.empty:
                return

            st.markdown(f"### {title}")
            fig = go.Figure()
            x_gap = 1.5
            y_activity = 1.0
            y_master = 0.0
            x_pos = {}

            # 활동 노드 (사각형 or 원)
            for idx, row in enumerate(activity_df.itertuples()):
                x = idx * x_gap
                x_pos[row.Activity_Seq] = x
                label = f"{row.Activities}<br>({row.Activities_Kor})"
                shape = "rect" if activity_type == "Primary" else "circle"
                fig.add_shape(
                    type=shape,
                    x0=x - 0.5, x1=x + 0.5,
                    y0=y_activity - 0.3, y1=y_activity + 0.3,
                    line=dict(color="black"),
                    fillcolor="lightblue" if activity_type == "Primary" else "#eef8d2"
                )
                fig.add_annotation(x=x, y=y_activity, text=label, showarrow=False, font=dict(size=14), align="center")

            # 마스터 노드 및 연결선
            used_masters = set()
            for idx, row in master_df.iterrows():
                seq = row.Activity_Seq
                master = row.Master
                if seq not in x_pos:
                    continue
                x = x_pos[seq]
                if master not in used_masters:
                    fig.add_shape(
                        type="rect",
                        x0=x - 0.3, x1=x + 0.3,
                        y0=y_master - 0.2, y1=y_master + 0.2,
                        line=dict(color="black"),
                        fillcolor="#f6e8c3"
                    )
                    fig.add_annotation(x=x, y=y_master, text=master, showarrow=False, font=dict(size=12), align="center")
                    used_masters.add(master)

                # 연결선
                fig.add_shape(
                    type="line",
                    x0=x, y0=y_activity - 0.3,
                    x1=x, y1=y_master + 0.2,
                    line=dict(color="gray", width=1)
                )

            fig.update_layout(
                height=500,
                width=max(len(activity_df), len(used_masters)) * 180,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=False)
                   # Primary Activities 표시
        draw_activity_master_plot("Primary Activities 다이어그램", primary_df, "Primary")

        # Support Activities 표시
        draw_activity_master_plot("Support Activities 다이어그램", support_df, "Support")

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


    # Master 기준 페이지 분할 다이어그램 (master_based_paged_diagram)
    def Master_Detail_Diagram(self, df: pd.DataFrame, max_columns_per_master=10):
        st.markdown("#### 📄 Master & Attribute Relationship Diagram")

        required_cols = ['Master', 'Column_Name', 'Column_Kor', 'Mandatory']
        if not all(col in df.columns for col in required_cols):
            st.warning("필수 컬럼이 누락되어 있습니다: Master, Column_Name, Mandatory")
            return

        # ✅ Master 기준으로 정렬된 고유 마스터 리스트
        master_list = sorted(df['Master'].dropna().unique().tolist())
        total_pages = len(master_list)

        # ✅ Streamlit 페이지 선택 UI
        page = st.number_input("Select Page (Base on Master)", min_value=1, max_value=total_pages, value=1, step=1)
        current_master = master_list[page - 1]

        # ✅ 현재 Master에 해당하는 컬럼들 가져오기
        sub_df = df[df['Master'] == current_master].copy()
        sub_df = sub_df.head(max_columns_per_master)

        # ✅ Graphviz 그림 생성
        dot = Digraph(format='png')
        dot.attr(rankdir='TB', fontsize='12', size='14', dpi='100')

        master_node_id = f"MASTER_{page}_{current_master}"
        
        dot.attr('node', shape='box', style='filled', fillcolor='#cde3f1', fontname='Malgun Gothic')
        dot.node(master_node_id, current_master)

        for idx, row in sub_df.iterrows():
            col_name = str(row['Column_Name']).strip()
            col_kor = str(row['Column_Kor']).strip()
            if len(col_name) > 9:
                col_name = col_name[:9] + str(idx)
            if len(col_kor) > 9:
                col_kor = col_kor[:9] + str(idx)
            mandatory = str(row['Mandatory']).strip()
            col_id = f"COLUMN_{page}_{col_name}"

            label = f"{col_name}\n({col_kor})"

            color = '#aefc2f' if mandatory == '*' else '#fdf594'
            
            dot.attr('node', shape='ellipse', style='filled', fillcolor=color)
            dot.node(col_id, label)
            dot.edge(master_node_id, col_id, color='gray')

        # ✅ 이미지로 렌더링 후 출력
        # timestamp = int(time.time() * 1000)
        filename = f"tmp_image/master_page.png"

        dot.render(filename, cleanup=True)
        
        st.divider()

        if len(sub_df) < 2:
            st.image(f"{filename}.png", width=200)
        elif len(sub_df) < 4:
            st.image(f"{filename}.png", width=600)
        elif len(sub_df) < 6:
            st.image(f"{filename}.png", width=800)
        elif len(sub_df) < 8:
            st.image(f"{filename}.png", width=1000)
        else:
            st.image(f"{filename}.png", width=1000)

        st.divider()

        st.markdown("**박스는 Master, 원은 속성코드 입니다.**")
        st.write(f"Green: 필수 속성코드 입니다.")
        st.write(f"속성코드에 대한 상세 정보는 위의 탭을 이용하여 확인하세요.")

        return None

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

            # ✅ Activity_Seq 문자열로 통일
            for name, df in loaded_data.items():
                if isinstance(df, pd.DataFrame) and 'Activity_Seq' in df.columns:
                    df['Activity_Seq'] = df['Activity_Seq'].astype(str).str.strip().str.replace('.0', '', regex=False)

            col1, col2 = st.columns([2, 8])
            with col1:
                selected_industry = st.selectbox("Industry를 선택하세요", loaded_data['valuechain']['Industry'].unique())
            
            loaded_data['valuechain'] = loaded_data['valuechain'][loaded_data['valuechain']['Industry'] == selected_industry]
            loaded_data['valuechain_standard_master'] = loaded_data['valuechain_standard_master'][loaded_data['valuechain_standard_master']['Industry'] == selected_industry]
            # loaded_data['valuechain_standard_master_detail'] = loaded_data['valuechain_standard_master_detail'][loaded_data['valuechain_standard_master_detail']['Industry'] == selected_industry]
           
            df = loaded_data['valuechain_standard_master']

            # ValueChain 데이터 로드
            if not self.value_chain_diagram.load_all_valuechain_data(loaded_data):
                st.error("Value Chain 및 Master Table을 정의한 메타파일을 로드할 수 없습니다.")
                return False

            self.value_chain_diagram.valuechain_summary() # Value Chain 데이터 요약
            
            fig = self.value_chain_diagram.create_plotly_diagram()

            # 데이터 정보 표시 (선택된 Activities 포함)
            selected_activities = self.value_chain_diagram.display_valuechain_data()
            
            # 선택된 Activities 처리
            if selected_activities is not None and not selected_activities.empty:
                selected_activity_seq = selected_activities['Activity_Seq'].tolist()

                show_standard_master = True
                if show_standard_master:
                    tab1, tab2= st.tabs(["Process/Function & Master Diagram", "Show Master List"])
                    with tab1:
                        self.value_chain_diagram.Activities_Master_Diagram(selected_activity_seq)
                        
                    with tab2:
                        self.value_chain_diagram.display_standard_master_info(selected_activity_seq)
                    
            else:
                show_standard_master = False
                selected_activity_seq = []
            
            # 2nd 데이터 정보 표시 (선택된 Activities 포함)
            selected_activities_2nd = self.value_chain_diagram.display_valuechain_data_2nd()
            if selected_activities_2nd is not None and not selected_activities_2nd.empty:
                master = loaded_data['valuechain_standard_master'];
                master = master[master['Activity_Seq'].isin(selected_activities_2nd['Activity_Seq'].tolist())]

                detail_master = loaded_data['valuechain_standard_master_detail'];

                df = pd.merge(master, detail_master, on='Master', how='left')

                df = df[ (df['Mandatory'].astype(str).str.len() > 0) | (df['Reference_Code'].astype(str).str.len() > 0)]

                tab1, tab2 = st.tabs([ "Master & Code Detail Diagram", "Code Attribute Detail Information"])
                with tab1:
                    self.value_chain_diagram.Master_Detail_Diagram(df) # 다이어그램 생성 및 표시

                with tab2:
                    df = df[ (df['Mandatory'].astype(str).str.len() > 0)]
                    df = df[['Master', 'Master_Kor', 'Our_Master', 'Column_Name', 'Mandatory', 'Reference_Code']]
                    st.markdown("#### Code Attribute Detail Information")
                    st.dataframe(df, hide_index=True, height=500, use_container_width=True)

                st.divider()
            else:
                st.write("상세 정보를 보기 위한 Activities를 선택하세요.")

            return True
            
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
        st.markdown(APP_DESC2)
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
