#
# streamlit를 이용한 QDQM Analyzer : Master Information
# 2024. 11. 9.  Qliker
#

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import re
from datetime import datetime
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import PIL.Image as Image

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    st.warning("Graphviz를 사용할 수 없습니다. Code Relationship Diagram을 생성할 수 없습니다.")
    
from dataclasses import dataclass
from typing import Dict, Any, Optional
import plotly.graph_objects as go

SOLUTION_NAME = "Data Sense System"
SOLUTION_KOR_NAME = "데이터 센스 시스템"
APP_NAME = "Code Relationship Diagram (CRD)"
APP_KOR_NAME = "###### 코드들 간의 관계도를 생성합니다. Code Relationship Analyzer 결과를 이용하여 생성합니다."

# # 현재 파일의 상위 디렉토리를 path에 추가

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
app_name = re.sub(r'\s*', '', re.sub(r'^\d+_|\.py$', '', os.path.basename(__file__)))
QDQM_ver = '2.0'

# Now import utils after adding to path
from DataSense.util.Files_FunctionV20 import load_yaml, load_yaml_datasense, set_page_config
from DataSense.util.Display import (
    create_metric_card,
    display_kpi_metrics
)

from DataSense.util.erd_from_mapping import Display_ERD
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

def Display_ERD_sample():
    image = Image.open("DataSense/DS_Output/CRD_Sample.png")
    st.image(image, caption="CodeMapping 기반 Code Relationship Diagram", width=480)
    return True

def Display_ERD_Safe(df, img_width=480, view_mode="All"):
    if not GRAPHVIZ_AVAILABLE:
        st.warning("⚠️ Graphviz 실행 환경이 없어 다이어그램을 표시할 수 없습니다.")
        return
    try:
        Display_ERD(df, img_width=img_width, view_mode=view_mode)
    except FileNotFoundError as e:
        if "dot" in str(e):
            st.error("⚠️ Graphviz 실행기가 설치되어 있지 않습니다. Cloud에서는 `apt.txt`에 graphviz를 추가하세요.")
        else:
            st.error(f"ERD 생성 중 오류: {e}")

#-----------------------------------------------------------------------------------------
# Master KPI 
def Display_Master_KPIs(loaded_data):
    """ Master Statistics KPIs """
    def calculate_master_type_counts(df):
        """Code Type별 파일 수 계산"""
        if 'MasterType' not in df.columns or 'FileName' not in df.columns:
            return {}
        try:
            master_type_counts = df.groupby('MasterType')['FileName'].nunique()
            expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
            
            result = {}
            for master_type in expected_types:
                count = master_type_counts.get(master_type, 0)
                result[master_type] = f"{count:,}"
            return result
        except Exception as e:
            st.error(f"MasterType 계산 중 오류 발생: {str(e)}")
            return {}

    df = loaded_data['filestats']
    if df is None or df.empty:
        st.warning(f"마스터 코드 통계 정보 파일을 로드할 수 없습니다. {df}")
        return
    
    st.markdown("### Code File Statistics")
    # KPI 계산
    total_files = len(df['FileName'].unique()) if 'FileName' in df.columns else 0
    total_records = df['RecordCnt'].sum() if 'RecordCnt' in df.columns else 0
    total_filesize = df['FileSize'].sum() if 'FileSize' in df.columns else 0
    total_master_types = len(df['MasterType'].unique()) if 'MasterType' in df.columns else 0
    work_date = df['WorkDate'].max() if 'WorkDate' in df.columns else ''

    if total_records < 1000:
        total_records_unit = '건'
    else:
        total_records = total_records / 10000
        total_records_unit = '만건'

    if total_filesize < 1000:
        total_filesize = total_filesize 
        total_filesize_unit = 'Bytes'
    elif total_filesize < 1000000:
        total_filesize = total_filesize / 1000
        total_filesize_unit = 'KB'
    elif total_filesize < 1000000000:
        total_filesize = total_filesize / 1000000
        total_filesize_unit = 'MB'
    else:
        total_filesize = total_filesize / 1000000000
        total_filesize_unit = 'GB'        

    summary = {
        "Code File #": f"{total_files:,}",
        "Total Record #": f"{total_records:,.0f} {total_records_unit}",
        "Total File Size": f"{total_filesize:,.0f} {total_filesize_unit}",
        "Code Type #": f"{total_master_types:,}",
        "Work Date": f"{work_date}"
    }

    # 각 메트릭에 대한 색상 정의
    metric_colors = {
        "Code File #":      "#1f77b4",
        "Total Record #":   "#2ca02c", 
        "Total File Size":  "#ff7f0e",
        "Code Type #":      "#d62728",    # 빨간색
        "Work Date":        "#9467bd"     # 보라색
    }

    # 메트릭 표시
    cols = st.columns(len(summary))
    for col, (key, value) in zip(cols, summary.items()):
        color = metric_colors.get(key, "#0072B2")
        col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

    # MasterType별 파일 수
    st.markdown("### Statistics by Code Type")
    master_type_counts = calculate_master_type_counts(df)
    
    if master_type_counts:
        metric_colors = {
            "Master":   "#ff7f0e",     # 주황색
            "Operation": "#2ca02c",    # 초록색
            "Reference": "#84994f",    # 녹색
            "Attribute": "#d62728",    # 빨간색
            "Common":     "#9467bd",   # 보라색
            "Validation": "#a7aae1"    # 블루
        }

        cols = st.columns(len(master_type_counts))
        for col, (key, value) in zip(cols, master_type_counts.items()):
            color = metric_colors.get(key, "#0072B2")
            col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

    return True

#-----------------------------------------------------------------------------------------
def Display_MasterFile_List(loaded_data):
    """Master File List 화면 출력"""
    st.markdown("### Master File List")
    
    # 데이터 로드
    filestats_df = loaded_data.get('filestats')
    if filestats_df is None or filestats_df.empty:
        st.warning(f"마스터 코드 통계 정보 파일을 로드할 수 없습니다. {filestats_df}")
        return []

    # 필요한 컬럼만 추출
    required_columns = [
        'FileName', 'MasterType', 'FileSize', 'RecordCnt', 
        'ColumnCnt', 'SamplingRows', 'Sampling(%)'
    ]
    missing_cols = [c for c in required_columns if c not in filestats_df.columns]
    if missing_cols:
        st.error(f"⚠️ 필수 컬럼이 누락되었습니다: {missing_cols}")
        return []

    filestats_df = filestats_df[required_columns].copy()

    # MasterType별 탭 생성
    master_types = filestats_df['MasterType'].unique().tolist()
    tabs = st.tabs(master_types)

    selected_files = []

    for mtype, tab in zip(master_types, tabs):
        with tab:
            df = filestats_df[filestats_df['MasterType'] == mtype].copy()

            # 선택 컬럼 추가
            df.insert(0, 'selected', False)

            # Streamlit Data Editor 표시
            edited_df = st.data_editor(
                df,
                width=1400, height=600, hide_index=True,
                column_config={
                    'selected': st.column_config.CheckboxColumn(
                        "선택", help="파일 선택 여부", width=100
                    )
                },
                disabled=[
                    'FileName', 'MasterType', 'FileSize', 'RecordCnt', 
                    'ColumnCnt', 'SamplingRows', 'Sampling(%)'
                ]
            )

            # 선택된 파일 수집
            selected_files.extend(
                edited_df.loc[edited_df['selected'], 'FileName'].tolist()
            )

    return selected_files

#-----------------------------------------------------------------------------------------
def Display_MasterFile_Matched_List(loaded_data):
    """Master File List 화면 출력"""

    def Matched_Statistics(df):
        """Master File Matched Statistics"""

        codemapping_df = df.copy()
        # 🔹 컬럼명 정리 (공백, BOM 제거)
        codemapping_df.columns = codemapping_df.columns.str.replace('\ufeff', '').str.strip()

        # 🔹 필수 컬럼 체크
        required_cols = ['FileName', 'MasterType', 'ColumnName', 'CodeType_1']
        missing = [c for c in required_cols if c not in codemapping_df.columns]
        if missing:
            st.error(f"⚠️ 필수 컬럼 누락: {missing}")
            st.write("현재 컬럼명 목록:", codemapping_df.columns.tolist())
            return False

        df = codemapping_df.copy()

        # 🔹 CodeType_1 정리 (비어있는 값 제외)
        df['CodeType_1'] = df['CodeType_1'].astype(str).str.strip()
        df = df[df['CodeType_1'] != ""]

        # 🔹 CodeType_1별 매칭 컬럼 수
        matched_counts = (
            df.groupby(['FileName', 'MasterType', 'CodeType_1'])['ColumnName']
            .count()
            .reset_index(name='MatchedCols')
        )

        # 🔹 피벗 변환: 각 CodeType_1을 컬럼으로
        pivot_df = matched_counts.pivot_table(
            index=['FileName', 'MasterType'],
            columns='CodeType_1',
            values='MatchedCols',
            fill_value=0
        ).reset_index()

        # 🔹 NaN → 0
        num_cols = pivot_df.select_dtypes(include=['number']).columns
        pivot_df[num_cols] = pivot_df[num_cols].fillna(0).astype(int)

        return pivot_df

    st.markdown("### Master File Matched List")
    st.markdown("###### 마스터 코드 통계 정보와 속성코드들(Operation, Attribute,  Reference, Rule)과의 매칭된 컬럼 수를 분석합니다.")
    
    # 데이터 로드
    filestats_df = loaded_data.get('filestats')
    codemapping_df = loaded_data.get('codemapping')

    if filestats_df is None or filestats_df.empty or codemapping_df is None or codemapping_df.empty:
        st.warning("매핑 통계 정보 파일을 로드할 수 없습니다. ")
        return []

    # ---------------------- Matched Column Count 계산 ----------------------
    matched_df = codemapping_df[codemapping_df['Matched(%)_1'] > 0]
    matched_count = (
        matched_df.groupby(['FileName','MasterType'])['Matched(%)_1']
        .count()
        .to_dict()   # dict 변환
    )

    filestats_df['Matched Col #'] = filestats_df.apply(
        lambda r: matched_count.get((r['FileName'], r['MasterType']), 0),
        axis=1
    )
    filestats_df['Matched(%)'] = (
        filestats_df['Matched Col #'].astype(int) / filestats_df['ColumnCnt'].astype(int) * 100
    ).round(2)

    # ---------------------- 필수 컬럼 체크 ----------------------
    required_columns = [
        'FileName', 'MasterType', 'FileSize', 'RecordCnt', 
        'ColumnCnt', 'Matched Col #', 'Matched(%)'
    ]
    missing_cols = [c for c in required_columns if c not in filestats_df.columns]
    if missing_cols:
        st.error(f"⚠️ 필수 컬럼이 누락되었습니다: {missing_cols}")
        return []

    filestats_df = filestats_df[required_columns].copy()

    detail_df = Matched_Statistics(codemapping_df)
    filestats_df = filestats_df.merge(detail_df, on=['FileName', 'MasterType'], how='left')

    # ---------------------- MasterType별 탭 생성 ----------------------
    master_types = filestats_df['MasterType'].unique().tolist()
    tabs = st.tabs(master_types)

    selected_files = []

    for mtype, tab in zip(master_types, tabs):
        with tab:
            df = filestats_df[filestats_df['MasterType'] == mtype].copy()

            # 선택 컬럼 추가
            df.insert(0, 'selected', False)

            edited_df = st.data_editor(
                df,
                width=1400, height=600, hide_index=True,
                column_config={
                    'selected': st.column_config.CheckboxColumn(
                        "선택", help="파일 선택 여부", width=100
                    )
                },
                disabled=[
                    'FileName', 'MasterType', 'FileSize', 'RecordCnt', 
                    'ColumnCnt','Matched Col #', 'Matched(%)'      
                ]
            )

            # ✅ FileName + MasterType 튜플로 저장
            selected_files.extend(
                list(
                    edited_df.loc[edited_df['selected'], ['FileName', 'MasterType']]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
            )

    return selected_files

def Display_MasterMapping_Detail(loaded_data, selected_files):
    """Master Mapping Detail 화면 출력"""
    VIEW_COLUMNS = {
        "Mapping Information": [
            'FileName', 'ColumnName', 'PK', 'FK', 'ValueCnt', 'FormatCnt', 'Format', 'Format(%)',
            'CodeColumn_1', 'CodeType_1', 'CodeFile_1', 'Matched_1', 'Matched(%)_1',
        ],
        "Value Information": [
            'FileName', 'ColumnName', 'OracleType', 'PK', 'ValueCnt', 'Null(%)', 'Unique(%)',
            'FormatCnt', 'Format', 'Format(%)', 'Top10', 'Top10(%)',
            'MinString', 'MaxString', 'ModeString', 'MedianString',
        ],
    }

    def render_table(df, title, cols):
        st.markdown(f"###### {title}")
        st.dataframe(df[cols].reset_index(drop=True), width=1400, height=600, hide_index=True)

    st.markdown("###### 상세 분석을 하고자 하는 File Name을 체크를 하세요.")
    st.divider()
    st.markdown("### Code File Mapping Information")
    st.markdown("###### 각 코드의 데이터 값을 분석하여 Mapping 정보를 표시합니다.")

    codemapping_df = loaded_data.get('codemapping', pd.DataFrame())
    if codemapping_df.empty:
        st.warning("마스터 매핑 정의 파일을 로드할 수 없습니다.")
        return False

    # ✅ FileName + MasterType 기준 필터링
    mapping_df = codemapping_df.merge(
        pd.DataFrame(selected_files, columns=['FileName','MasterType']),
        on=['FileName','MasterType'],
        how='inner'
    )

    if mapping_df.empty:
        st.info("선택된 파일에 해당하는 매핑 데이터가 없습니다.")
        return False

    required_columns = [
        'FileName', 'ColumnName', 'MasterType', 'OracleType', 'PK', 'FK', 'Rule',
        'ValueCnt', 'Null(%)', 'Unique(%)',    'FormatCnt', 'Format', 'Format(%)',
        'MinString', 'MaxString', 'ModeString', 'MedianString', 'Top10', 'Top10(%)',
        'CodeColumn_1', 'CodeType_1', 'CodeFile_1', 'Matched_1', 'Matched(%)_1',
    ]
    mapping_df = mapping_df[required_columns].copy()

    if 'Matched(%)_1' in mapping_df.columns:
        mapping_df['Matched(%)_1'] = (
            mapping_df['Matched(%)_1'].replace(['', 'nan', 'None'], '0').astype(float)
        )
    if 'Matched_1' in mapping_df.columns:
        # 문자열 정리 후 float로 변환한 다음 int로 변환
        mapping_df['Matched_1'] = (
            mapping_df['Matched_1']
            .astype(str)
            .replace(['nan', 'None', ''], '0')
            .astype(float)
            .astype(int)
        )

    if "selected_view" not in st.session_state:
        st.session_state.selected_view = list(VIEW_COLUMNS.keys())[0]

    selected_view = st.radio(
        "보기 유형 선택:",
        list(VIEW_COLUMNS.keys()),
        horizontal=True,
        index=list(VIEW_COLUMNS.keys()).index(st.session_state.selected_view),
        key="selected_view_radio"
    )

    st.session_state.selected_view = selected_view
    render_table(mapping_df, selected_view, VIEW_COLUMNS[selected_view])

    return True
#-----------------------------------------------------------------------------------------
@dataclass
class FileConfig:
    """파일 설정 정보"""
    fileformat: str
    filestats: str
    codemapping: str

class FileLoader:
    """파일 로딩을 위한 클래스"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.root_path = yaml_config['ROOT_PATH']
        self.files_config = self._setup_files_config()
    
    def _setup_files_config(self) -> FileConfig:
        """파일 설정 구성"""
        files = self.yaml_config['files']
        return FileConfig(
            fileformat=f"{self.root_path}/{files['fileformat']}",
            filestats=f"{self.root_path}/{files['filestats']}",
            codemapping=f"{self.root_path}/{files['codemapping']}",
        )
    
    def load_file(self, file_path: str, file_name: str) -> Optional[pd.DataFrame]:
        """단일 파일 로드"""
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
        """모든 파일 로드"""
        files_to_load = {
            'fileformat': self.files_config.fileformat,
            'filestats': self.files_config.filestats,
            'codemapping': self.files_config.codemapping,
        }
        
        loaded_data = {}
        for name, path in files_to_load.items():
            # 확장자가 없는 경우 .csv 추가
            if not os.path.splitext(path)[1]:
                path = path + ".csv"

            df = self.load_file(path, name)
            if df is not None:
                df = df.fillna('')
                
                # Arrow 직렬화 오류 방지를 위한 데이터 타입 정리
                if name == 'codemapping':
                    # 퍼센트 컬럼들 처리
                    percentage_columns = ['Matched(%)_1']
                    for col in percentage_columns:
                        if col in df.columns:
                            # 빈 문자열을 0으로 변환하고 숫자형으로 변환
                            df[col] = df[col].replace('', '0')
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # 매칭 결과 컬럼들 처리 (문자열로 통일)
                    match_columns = ['Matched_1']
                    for col in match_columns:
                        if col in df.columns:
                            # 모든 값을 문자열로 변환
                            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
                
                loaded_data[name] = df

        return loaded_data

class DashboardManager:
    """대시보드 관리를 위한 클래스"""
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.file_loader = FileLoader(yaml_config)
        # self.value_chain_diagram = ValueChainDiagram(yaml_config)
    
    def display_dashboard(self) -> bool:
        """Value Chain 대시보드 표시"""
        try:
            st.title(APP_NAME)
            st.markdown(APP_KOR_NAME)
            
            loaded_data = self.file_loader.load_all_files() # 모든 파일 로드
           
            # 마스터 통계 표시
            Display_Master_KPIs(loaded_data)

            # 마스터 파일 리스트 & Matched 리스트
            selected_file = Display_MasterFile_Matched_List(loaded_data)

            Display_MasterMapping_Detail(loaded_data, selected_file)
            st.divider()
            st.subheader("Code Relationship Diagram")
            st.markdown("###### 선택한 코드 파일들로 PK/FK를 추정하여 Code Relationship Diagram를 생성합니다.")

            codemapping_df = loaded_data['codemapping'].copy()
            erd_df = codemapping_df.merge(
                pd.DataFrame(selected_file, columns=['FileName','MasterType']),
                on=['FileName','MasterType'],
                how='inner'
            )

            # 👇 라디오를 먼저 노출
            view_mode = st.radio(
                "View",
                options=["All", "Operation", "Reference"],
                horizontal=True,
                index=0,
                key="erd_view_mode"
            )

            # 이후 버튼 클릭 시 생성
            if st.button("선택 파일로 Code Relationship Diagram 생성", key="btn_open_erd_panel"):
                if erd_df.empty:
                    st.info("선택된 파일에 해당하는 매핑 데이터가 없습니다.")
                else:  # 👇 선택값 전달
                    try:
                        if GRAPHVIZ_AVAILABLE:
                            Display_ERD_Safe(erd_df, img_width=480, view_mode=view_mode)
                        else:
                            st.info("Cloud 환경에서는 Diagram을 생성할 수 없습니다. Local 환경에서 실행해주세요.")

                        st.write("색상은 code file type 기준으로 표시됩니다.")
                    except FileNotFoundError as e:
                        if "PosixPath('dot')" in str(e) or "Graphviz executables" in str(e):
                            st.error("Graphviz가 설치되지 않았습니다. Code Relationship Diagram을 생성할 수 없습니다.")
                            st.info("로컬 환경에서는 `pip install graphviz` 및 Graphviz 바이너리를 설치해야 합니다.")
                        else:
                            st.error(f"파일 오류: {str(e)}")
                    except Exception as e:
                        # st.error(f"Code Relationship Diagram 생성 중 오류 발생: {str(e)}")
                        st.info("Cloud 환경에서는 Diagram을 생성할 수 없습니다. Local 환경에서 실행해주세요. 샘플 이미지를 표시합니다.")
                        Display_ERD_sample()

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
        try:
            success = self.dashboard_manager.display_dashboard()
                
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
