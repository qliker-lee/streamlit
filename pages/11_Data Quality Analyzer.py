import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
import pandas as pd

# --- 프로젝트 루트 기준으로 util 경로 추가 ---
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# from DataSense.util.io import Load_Yaml_File

st.title("📊 Data Quality Analyzer")
st.markdown("#### 모든 파일의 각 컬럼들에 대한 프로파일링을 수행하여 품질분석을 위한 통계를 생성합니다. ")
st.markdown("#### 통계 상세 내역")
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
with col1:
    st.markdown("#####  속성 정보 ")
    st.markdown("###### 데이터 타입")
    st.markdown("###### 오라클 타입")
    st.markdown("###### 룰 기반 타입")
with col2:
    st.markdown("#####  Value 정보 ")
    st.markdown("###### Primary Key 여부")
    st.markdown("###### 데이터 값의 열 개수")
    st.markdown("###### Uniqueness 비율")
    st.markdown("###### Null 비율")
    st.markdown("###### 최소/최대/평균/중앙 값")
with col3:
    st.markdown("#####  Length 정보 ")
    st.markdown("###### Length 종류 ")
    st.markdown("###### Length 최소")
    st.markdown("###### Length 최대")
    st.markdown("###### Length 다빈도")
    st.markdown("###### Length 평균/중앙값")
with col4:
    st.markdown("#####  Value 구성(패턴)")
    st.markdown("###### 영문, 한글, 숫자 등으로 패턴 구성")
    st.markdown("###### 패턴의 종류 수")
    st.markdown("###### 다빈도 패턴 구성")
    st.markdown("###### 다빈도 패턴 및 비율")
    st.markdown("###### 2nd/3rd 패턴 및 비율")
with col5:
    st.markdown("#####  Value Top 10")
    st.markdown("###### Top 10 값")
    st.markdown("###### Top 10 비율")
with col6:
    st.markdown("#####  데이터 문자 통계")
    st.markdown("###### 영문 대소문자 열 수")
    st.markdown("###### 한글 포함 열 수")
    st.markdown("###### 숫자 포함 열 수")
    st.markdown("###### 특수문자 열 수")
    st.markdown("###### 혼합 문자 열 수")

st.divider()
st.markdown("##### 통계를 기반으로 데이터 품질 분석을 수행하고, 다음 단계로 코드간 관계도를 작성합니다. ")

# --- 실행 버튼 ---
st.divider()
st.markdown("##### 전체 파일의 수 및 크기에 따라 시간이 많이 소요될 수 있습니다.")
if st.button("🔍 Data Quality Analyzer 실행"):
    with st.spinner("분석 실행 중... 잠시만 기다려주세요."):
        
        script_path = PROJECT_ROOT / "DataSense" / "util" / "DS_11_MasterCodeFormat.py"
        cmd = [sys.executable, str(script_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("분석이 완료되었습니다 ✅")
            st.text_area("📜 실행 로그", result.stdout, height=300)
        except subprocess.CalledProcessError as e:
            st.error("❌ 실행 중 오류가 발생했습니다.")
            st.text_area("⚠️ 오류 로그", e.stderr, height=300)

st.divider()
st.caption("실행 후 결과 파일은 DataSense/DS_Output 하위에 저장됩니다.")
st.markdown("##### Data Quality Analyzer의 결과 입니다. 스크롤하여 전체 내용을 분석하세요. ")

fileformat_df = pd.read_csv('DataSense/DS_Output/FileFormat.csv')
st.dataframe(fileformat_df, width=1400, height=800, hide_index=True)
st.markdown("##### Data Quality Information Menu 에서 상세 분석을 수행합니다. ")
st.divider()
