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

st.title("📊 Code Relationship Analyzer")
st.divider()
st.markdown("##### Data Quality Analyzer 결과를 기반으로 모든 파일의 컬럼들에 대한 관계도를 작성합니다.\n")

# --- 실행 버튼 ---
st.divider()
st.markdown("##### 전체 파일의 수 및 크기에 따라 시간이 많이 소요될 수 있습니다. (약 10분 소요)")
if st.button("🔍 Code Relationship 분석 실행"):
    with st.spinner("분석 실행 중... 잠시만 기다려주세요."):
        # fs_13_CodeRelationship.py 실행
        script_path = PROJECT_ROOT / "DataSense" / "util" / "DS_13_Code Relationship Analyzer.py"
        cmd = [sys.executable, str(script_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("분석이 완료되었습니다 ✅")
            st.text_area("📜 실행 로그", result.stdout, height=300)
        except subprocess.CalledProcessError as e:
            st.error("❌ 실행 중 오류가 발생했습니다.")
            st.text_area("⚠️ 오류 로그", e.stderr, height=300)

st.divider()
st.caption("결과 파일은 DataSense/DS_Output 하위에 저장됩니다.")
st.markdown("##### Data Quality Information Menu에서 상세 분석을 수행합니다.")

code_mapping_df = pd.read_csv('DataSense/DS_Output/CodeMapping.csv')

st.dataframe(code_mapping_df, width=1400, height=600, hide_index=True)