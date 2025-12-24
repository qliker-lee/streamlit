import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# 페이지 설정
st.set_page_config(page_title="DataSense Independent Analyzer", layout="wide")

# 파일 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "DataSense"
OUTPUT_DIR = BASE_PATH / "DS_Output"
VC_FILE = OUTPUT_DIR / "DS_ValueChain_System_File.csv"
MAPPING_FILE = OUTPUT_DIR / "CodeMapping.csv"

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def get_file_summary(file_names, df_mapping):
    """선택된 파일들의 FileName, ColumnCnt, PK List를 추출하는 함수"""
    if df_mapping is None or len(file_names) == 0:
        return pd.DataFrame(columns=['FileName', 'ColumnCnt', 'PK_List'])
    
    # 해당 파일들만 필터링
    relevant_mapping = df_mapping[df_mapping['FileName'].isin(file_names)]
    
    summary = []
    for f_name in file_names:
        f_data = relevant_mapping[relevant_mapping['FileName'] == f_name]
        col_cnt = len(f_data)
        # PK가 1인 컬럼들 추출
        pk_cols = f_data[f_data['PK'].astype(str) == '1']['ColumnName'].tolist()
        pk_str = ", ".join(pk_cols) if pk_cols else "-"
        
        summary.append({
            'FileName': f_name,
            'ColumnCnt': col_cnt,
            'PK_List': pk_str
        })
    
    return pd.DataFrame(summary)

def main():
    st.title("📊 DataSense Independent Analysis Dashboard")
    
    df_vc = load_data(VC_FILE)
    df_mapping = load_data(MAPPING_FILE)

    if df_vc is None:
        st.error(f"기초 데이터를 찾을 수 없습니다. (경로: {VC_FILE})")
        return

    # 전처리: Unknown 제외
    df_vc = df_vc.dropna(subset=['Activity', 'System'])
    df_vc = df_vc[(df_vc['Activity'] != 'Unknown') & (df_vc['System'] != 'Unknown')]

    # [STEP 1] Industry 선택 (전체 데이터 필터 기준)
    st.header("🏢 Industry Selection")
    industries = sorted(df_vc['Industry'].unique())
    selected_industry = st.selectbox("분석할 산업군을 선택하세요", industries)
    df_ind = df_vc[df_vc['Industry'] == selected_industry]
    
    st.markdown("---")

    # [STEP 2] Activity 별 독립 정보 출력
    st.header(f"⚙️ Activity Analysis ({selected_industry})")
    act_list = sorted(df_ind['Activity'].unique())
    
    col1, col2 = st.columns([1, 2])
    with col1:       
        # 차트: 해당 산업 내 전체 Activity 분포
        act_counts = df_ind.groupby('Activity')['FileName'].count().reset_index()
        fig_act = px.pie(act_counts, names='Activity', values='FileName', title="Activity 분포", hole=0.4)
        st.plotly_chart(fig_act, use_container_width=True)

    with col2:
        selected_act = st.selectbox("Activity 선택", act_list, key="sb_act")
        # 선택된 Activity에 속한 파일들
        act_files = df_ind[df_ind['Activity'] == selected_act]['FileName'].unique()

        st.subheader(f"📄 '{selected_act}' 소속 파일 요약")
        act_summary = get_file_summary(act_files, df_mapping)
        st.dataframe(act_summary, use_container_width=True, hide_index=True)

    st.markdown("---")

    # [STEP 3] System 별 독립 정보 출력
    st.header(f"💻 System Analysis ({selected_industry})")
    sys_list = sorted(df_ind['System'].unique())
    
    col3, col4 = st.columns([1, 2])
    with col3:
        # 차트: 해당 산업 내 전체 System 분포
        sys_counts = df_ind.groupby('System')['FileName'].count().reset_index()
        # # 막대차트 생성 (use_container_width로 자동 크기 조절)
        # fig_sys = px.bar(sys_counts, x='System', y='FileName', title="System별 파일 수", color='System', 
        # height=300, width=600, bar_width=0.5)
        # st.plotly_chart(fig_sys, use_container_width=True)

        # 사용자님 요청사항 반영: 막대 너비를 크게 조정
        fig_sys = px.bar(sys_counts, x='System', y='FileName', color='System', height=400)
        
        # [핵심] bargap=0.1~0.3 정도로 설정하면 막대가 훨씬 듬직하게 보입니다.
        fig_sys.update_layout(
            bargap=0.15, 
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sys, use_container_width=True)

    with col4:
        selected_sys = st.selectbox("System 선택", sys_list, key="sb_sys")
        # 선택된 System에 속한 파일들
        sys_files = df_ind[df_ind['System'] == selected_sys]['FileName'].unique()

        st.subheader(f"📋 '{selected_sys}' 소속 파일 요약")
        sys_summary = get_file_summary(sys_files, df_mapping)
        st.dataframe(sys_summary, use_container_width=True, hide_index=True)

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
            m1.metric("총 레코드", f"{int(detail_df['TotalRecords'].iloc[0]):,}")
            m2.metric("평균 Null(%)", f"{detail_df['Null_pct'].mean():.1f}%")
            m3.metric("중복(%)", f"{detail_df['Duplicate_pct'].mean():.1f}%")
            m4.metric("컬럼 수", len(detail_df))

            # 테이블 표시
            st.dataframe(detail_df, use_container_width=True)
        else:
            st.warning("상세 매핑 데이터가 없습니다.")
    else:
        st.warning("System 파일이 없습니다.")

    # 드릴다운: 특정 컬럼의 Format 분포 등을 더 보고 싶을 때를 위한 확장
    with st.expander("Raw Data (CodeMapping) 전체 보기", expanded=False):
        st.dataframe(df_mapping, use_container_width=True)
    st.markdown("---")
if __name__ == "__main__":
    main()