# -------------------------------------------------
# 25_Value Chain & System Definition.py 에서 입력된 내용을 분석
# 2025.12.26 Qliker 
# -------------------------------------------------
# -------------------------------------------------------------------
# 1. 경로 설정 (Streamlit warnings import 전에 필요)
# -------------------------------------------------------------------
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# -------------------------------------------------------------------
# 2. Streamlit 경고 억제 설정 (Streamlit import 전에 호출)
# -------------------------------------------------------------------
from DataSense.util.streamlit_warnings import setup_streamlit_warnings
setup_streamlit_warnings()
# -------------------------------------------------------------------
# 3. 필수 라이브러리 import
# -------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path


SOLUTION_NAME = "Value Chain & System Analysis"
SOLUTION_KOR_NAME = "Value Chain & System Analysis"
APP_NAME = "Value Chain & System Analysis"
APP_DESC = "###### Value Chain & System를 기반으로 각 파일들에 대한 통계 정보입니다.  "

from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config
set_page_config(APP_NAME)

# -------------------------------------------------
# 3. Streamlit 페이지 설정
# ------------------------------------------------- 
# 경로 설정 (Pathlib 활용)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "DataSense"
OUTPUT_DIR = BASE_PATH / "DS_Output"
VC_FILE = OUTPUT_DIR / "DS_ValueChain_System_File.csv"
MAPPING_FILE = OUTPUT_DIR / "CodeMapping.csv"

# -------------------------------------------------
# 4. 데이터 로드
# ------------------------------------------------- 
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def get_file_summary(file_names, df_mapping):
    """선택된 파일 리스트에 대해 FileName, ColumnCnt, PK List를 추출"""
    if df_mapping is None or len(file_names) == 0:
        return pd.DataFrame(columns=['FileName', 'ColumnCnt', 'PK_List'])
    
    relevant_mapping = df_mapping[df_mapping['FileName'].isin(file_names)]
    summary = []
    for f_name in file_names:
        f_data = relevant_mapping[relevant_mapping['FileName'] == f_name]
        col_cnt = len(f_data)
        
        # PK 컬럼 추출 (PK 값이 1인 컬럼들)
        pk_str = "-"
        if 'PK' in f_data.columns:
            pk_cols = f_data[f_data['PK'].astype(str).str.contains('1', na=False)]['ColumnName'].tolist()
            if pk_cols:
                pk_str = ", ".join(pk_cols)
            
        summary.append({
            'FileName': f_name,
            'ColumnCnt': col_cnt,
            'PK_List': pk_str
        })
    
    return pd.DataFrame(summary)

def main():
    st.title(APP_NAME)
    st.markdown(APP_DESC)

    df_vc = load_data(VC_FILE)
    df_mapping = load_data(load_mapping_path := MAPPING_FILE)

    if df_vc is None:
        st.error(f"데이터 파일을 찾을 수 없습니다: {VC_FILE}")
        return

    # --- [전처리] Unknown 제외 ---
    df_vc = df_vc.dropna(subset=['Activity', 'System'])
    df_vc = df_vc[(df_vc['Activity'] != 'Unknown') & (df_vc['System'] != 'Unknown')]

    # 1. Industry 선택 (대시보드 공통 필터)
    st.header("🏢 1. Industry Selection")
    industries = sorted(df_vc['Industry'].unique())
    selected_industry = st.selectbox("분석할 산업군을 선택하세요", industries)
    
    # 해당 산업군 데이터 (Activity와 System 섹션의 독립적 소스)
    df_ind = df_vc[df_vc['Industry'] == selected_industry]
    st.divider()

    # ---------------------------------------------------------
    # 2. Activity 섹션 (파이 차트 + 독립 정보)
    # ---------------------------------------------------------
    st.header(f"⚙️ Activity Analysis")
    all_activities = sorted(df_ind['Activity'].unique())
    
    act_col1, act_col2 = st.columns([3, 3])
    
    with act_col1:
        act_tab1, act_tab2 = st.tabs(["Activity별 파일 분포(파이 차트)", "Activity별 파일 수(막대 차트)"])
        with act_tab1:  
            # 파이 차트 생성 (도넛 형태)
            act_counts = df_ind.groupby('Activity')['FileName'].count().reset_index()
            fig_act = px.pie(act_counts, names='Activity', values='FileName', 
                            title=f"Activity별 파일 분포",
                            hole=0.4, # 도넛 형태
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_act.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_act, width="stretch")

        with act_tab2:
            # 막대 차트 생성 (Activity별 파일 수)
            act_counts = df_ind.groupby('Activity')['FileName'].count().reset_index()
            fig_act = px.bar(act_counts, x='Activity', y='FileName', 
                            title=f"Activity별 파일 수",
                            color='Activity', height=400)
            fig_act.update_layout(bargap=0.2, showlegend=False)
            st.plotly_chart(fig_act, width="stretch")

    with act_col2:
        selected_act = st.selectbox("Activity를 선택하세요", all_activities, key="sel_act")
        # st.subheader(f"📄 '{selected_act}' Activity에 속한 파일 요약")
        act_files = df_ind[df_ind['Activity'] == selected_act]['FileName'].unique()
        act_summary = get_file_summary(act_files, df_mapping)
        st.dataframe(act_summary, width="stretch", height=400, hide_index=True)

    st.divider()

    # ---------------------------------------------------------
    # 3. System 섹션 (막대 차트 + 독립 정보)
    # ---------------------------------------------------------
    st.header(f"💻 System Analysis")
    all_systems = sorted(df_ind['System'].unique())
    
    sys_col1, sys_col2 = st.columns([3, 3])
    
    with sys_col1:
        sys_tab1, sys_tab2 = st.tabs(["System별 파일 분포(파이 차트)", "System별 파일 수(막대 차트)"])
        with sys_tab1:
            # 파이 차트 생성
            sys_counts = df_ind.groupby('System')['FileName'].count().reset_index()
            fig_sys = px.pie(sys_counts, names='System', values='FileName', 
                            title=f"System별 파일 분포",
                            hole=0.4, # 도넛 형태
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_sys.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sys, width="stretch")
        with sys_tab2:
            # 막대 차트 생성 (System별 파일 수)
            sys_counts = df_ind.groupby('System')['FileName'].count().reset_index()
            fig_sys = px.bar(sys_counts, x='System', y='FileName', 
                            title=f"System별 파일 수",
                            color='System', height=400)
            st.plotly_chart(fig_sys, width="stretch")

    with sys_col2:
        selected_sys = st.selectbox("System을 선택하세요", all_systems, key="sel_sys")
        sys_files = df_ind[df_ind['System'] == selected_sys]['FileName'].unique()
        sys_summary = get_file_summary(sys_files, df_mapping)
        st.dataframe(sys_summary, width="stretch", height=400, hide_index=True)


    # ---------------------------------------------------------
    # STEP 4: 파일 선택 및 상세 속성 (CodeMapping)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 📑 STEP 4: [{selected_sys}] 내 파일 상세 정보")
    
    final_files = sorted(sys_files)
    selected_file = st.selectbox("조회할 파일을 최종 선택하세요", final_files)

    if selected_file and df_mapping is not None:
        detail_df = df_mapping[df_mapping['FileName'] == selected_file]
        
        if not detail_df.empty:
            st.success(f"✅ '{selected_file}' 상세 속성 조회 결과")
            
            # 메트릭 표시
            m1, m2, m3, m4 = st.columns(4)
            
            # 총 레코드 수 (여러 컬럼명 시도)
            total_records = "N/A"
            if 'TotalRecords' in detail_df.columns:
                total_records = f"{int(detail_df['TotalRecords'].iloc[0]):,}"
            elif 'RecordCnt' in detail_df.columns:
                total_records = f"{int(detail_df['RecordCnt'].iloc[0]):,}"
            elif 'ValueCnt' in detail_df.columns:
                # ValueCnt의 최대값 사용 (일반적으로 파일의 총 레코드 수와 유사)
                total_records = f"{int(detail_df['ValueCnt'].max()):,}"
            m1.metric("총 레코드", total_records)
            
            # Null(%) 평균
            null_pct = "N/A"
            if 'Null_pct' in detail_df.columns:
                null_pct = f"{detail_df['Null_pct'].mean():.1f}%"
            elif 'Null(%)' in detail_df.columns:
                null_pct = f"{detail_df['Null(%)'].mean():.1f}%"
            m2.metric("평균 Null(%)", null_pct)
            
            # 중복(%) 평균
            dup_pct = "N/A"
            if 'Duplicate_pct' in detail_df.columns:
                dup_pct = f"{detail_df['Duplicate_pct'].mean():.1f}%"
            m3.metric("중복(%)", dup_pct)
            
            m4.metric("컬럼 수", len(detail_df))

            # 테이블 표시
            st.dataframe(detail_df, width="stretch", height=600, hide_index=True)
        else:
            st.warning("상세 매핑 데이터가 없습니다.")
    else:
        st.warning("System 파일이 없습니다.")

    # 드릴다운: 특정 컬럼의 Format 분포 등을 더 보고 싶을 때를 위한 확장
    with st.expander("Raw Data (CodeMapping) 전체 보기", expanded=False):
        st.dataframe(df_mapping, width="stretch", height=600, hide_index=True)
    st.markdown("---")
if __name__ == "__main__":
    main()