# -*- coding: utf-8 -*-
"""
2025.11.05  Qliker 
📊 Data Analyzer (통합)
- Data Quality Analyzer: 모든 파일의 각 컬럼들에 대한 프로파일링을 수행하여 품질분석을 위한 통계를 생성
- Data Type & Rule Analyzer: Data Quality Analyzer 결과를 기반으로 각 컬럼에 대한 Rule 프로파일링 수행
- Code Relationship Analyzer: Data Quality Analyzer 결과를 기반으로 모든 파일의 컬럼들에 대한 관계도 작성
Class-based Version (Tab Integration)
"""
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
import subprocess
import os
import sys
# import warnings
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml

# -------------------------------------------------------------------
# 기본 앱 정보
# -------------------------------------------------------------------
APP_NAME = "Data Analyzer (Data Profile)"
APP_DESC = "##### 데이터 품질 분석, 데이터 타입 및 룰 분석, 데이터 관계도 분석을 위한 기초 작업입니다."
APP_DESC2 = """
- Data Quality Analyzer: 모든 데이터에 대한 프로파일링을 수행하여 품질분석을 위한 통계를 생성
- Data Type & Rule Analyzer: 모든 데이터의 데이터 타입 및 사전 정의된 Rule 기반 프로파일링 수행
- Data Relationship Analyzer: 데이터 간의 관계도를 작성
###### 아래의 탭들을 단계적으로 수행합니다. 
"""


from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config

set_page_config(APP_NAME)

# -------------------------------------------------------------------
# YAML CONFIG 로더
# -------------------------------------------------------------------
def _fallback_load_yaml_datasense() -> Dict[str, Any]:
    """YAML 로드 실패 시 기본 설정 반환"""
    guessed_root = str(PROJECT_ROOT)
    cfg = {
        "ROOT_PATH": guessed_root,
        "files": {
            "fileformat_output": "DataSense/DS_Output/FileFormat.csv",
            "ruledatatype_output": "DataSense/DS_Output/RuleDataType.csv",
            "codemapping_output": "DataSense/DS_Output/CodeMapping.csv",
        },
        "DataSense_Password": "tkfkdgo",  # 기본 패스워드
    }
    path = Path(guessed_root) / "DataSense" / "util" / "DS_Master.yaml"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
                y.setdefault("ROOT_PATH", guessed_root)
                y.setdefault("files", cfg["files"])
                return y
        except Exception:
            pass
    return cfg

try:
    from DataSense.util.Files_FunctionV20 import load_yaml_datasense  # type: ignore
except Exception:
    load_yaml_datasense = _fallback_load_yaml_datasense

# -------------------------------------------------------------------
# 유틸 함수
# -------------------------------------------------------------------
def normalize_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame을 Streamlit 표시용으로 정규화
    - 숫자형 컬럼: NaN을 0으로 변환 (None 표시 방지)
    - object 타입 컬럼: 숫자로 변환 가능하면 변환, 불가능하면 None을 빈 문자열로
    - 문자열 컬럼: None을 빈 문자열로
    Args:
        df: 처리할 DataFrame
    Returns:
        정규화된 DataFrame
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # 숫자형 컬럼: NaN을 0으로 변환 (None 표시 방지)
            df[col] = df[col].fillna(0)
        elif df[col].dtype == 'object': # object 타입인 경우 숫자로 변환 가능한지 확인 
            try:       
                numeric_series = pd.to_numeric(df[col], errors='coerce') # 숫자로 변환 시도
                if not numeric_series.isna().all():
                    # 숫자 값이 있으면 숫자형으로 변환하고 NaN은 0으로
                    df[col] = numeric_series.fillna(0)
                else:
                    # 숫자 값이 없으면 문자열로 처리
                    df[col] = df[col].fillna("")
            except Exception:
                # 변환 실패 시 문자열로 처리
                df[col] = df[col].fillna("")
        else:
            # 문자열 컬럼은 None을 빈 문자열로
            df[col] = df[col].fillna("")
    
    return df

# -------------------------------------------------------------------
# FILE LOADER
# -------------------------------------------------------------------
@dataclass
class FileConfig:
    """파일 설정 정보"""
    fileformat_output: str
    ruledatatype_output: str
    codemapping_output: str
    analyzer_script_quality: str
    analyzer_script_rule: str
    analyzer_script_relationship: str

class FileLoader:
    """파일 로딩을 위한 클래스"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.root_path = str(yaml_config.get("ROOT_PATH", str(PROJECT_ROOT)))
        self.files_config = self._setup_files_config()
    
    def _setup_files_config(self) -> FileConfig:
        """파일 설정 구성"""
        files = self.yaml_config.get('files', {})
        
        def _full_path(path_str):
            p = Path(path_str)
            if not p.is_absolute():
                p = Path(self.root_path) / p
            return str(p.resolve())
        
        return FileConfig(
            fileformat_output=_full_path(files.get('fileformat_output', 'DataSense/DS_Output/FileFormat.csv')),
            ruledatatype_output=_full_path(files.get('ruledatatype_output', 'DataSense/DS_Output/RuleDataType.csv')),
            codemapping_output=_full_path(files.get('codemapping_output', 'DataSense/DS_Output/CodeMapping.csv')),
            analyzer_script_quality=_full_path(files.get('analyzer_script_quality', 'DataSense/util/DS_11_MasterCodeFormat.py')),
            analyzer_script_rule=_full_path(files.get('analyzer_script_rule', 'DataSense/util/DS_12_MasterRuleDataType.py')),
            analyzer_script_relationship=_full_path(files.get('analyzer_script_relationship', 'DataSense/util/DS_13_Code Relationship Analyzer.py'))
        )
    
    def load_file(self, file_path: str, file_name: str) -> Optional[pd.DataFrame]:
        """CSV 파일 로드"""
        path = Path(file_path)
        if not path.exists():
            return None
        
        try:
            for enc in ("utf-8-sig", "utf-8", "cp949"):
                try:
                    df = pd.read_csv(path, encoding=enc)
                    return df
                except Exception:
                    continue
            return None
        except Exception as e:
            st.error(f"{file_name} 로드 실패: {str(e)}")
            return None

# -------------------------------------------------------------------
# DATA QUALITY ANALYZER
# -------------------------------------------------------------------
class DataQualityAnalyzer:
    """Data Quality Analyzer 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any], loader: FileLoader):
        self.yaml_config = yaml_config
        self.loader = loader
        self.script_path = Path(loader.files_config.analyzer_script_quality)
        self.output_path = Path(loader.files_config.fileformat_output)
        self.password = yaml_config.get("DataSense_Password", "tkfkdgo")
    
    def display_statistics_info(self):
        """통계 상세 내역 정보 표시"""
        st.divider()
        st.markdown("###### 통계 상세 내역은 다음과 같은 내용들이 포함되어 있습니다.")
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        
        with col1:
            st.markdown("##### 속성 정보")
            st.write("데이터 타입")
            st.write("오라클 타입")
            st.write("룰 기반 타입")
        
        with col2:
            st.markdown("##### Value 정보")
            st.write("Primary Key 여부")
            st.write("데이터 값의 열 개수")
            st.write("Uniqueness 비율")
            st.write("Null 비율")
            st.write("최소/최대/평균/중앙 값")
        
        with col3:
            st.markdown("##### Length 정보")
            st.write("Length 종류")
            st.write("Length 최소")
            st.write("Length 최대")
            st.write("Length 다빈도")
            st.write("Length 평균/중앙값")
        
        with col4:
            st.markdown("##### Value 구성")
            st.write("영문, 한글, 숫자 등 패턴 구성")
            st.write("패턴의 종류 수")
            st.write("다빈도 패턴 구성")
            st.write("다빈도 패턴 및 비율")
            st.write("2nd/3rd 패턴 및 비율")
        
        with col5:
            st.markdown("##### Value Top 10")
            st.write("Top 10 값")
            st.write("Top 10 비율")
        
        with col6:
            st.markdown("##### 문자 통계")
            st.write("영문 대소문자 열 수")
            st.write("한글 포함 열 수")
            st.write("숫자 포함 열 수")
            st.write("특수문자 열 수")
            st.write("혼합 문자 열 수")
    
    def run_analyzer(self) -> bool:
        """Data Quality Analyzer 스크립트 실행"""
        if not self.script_path.exists():
            st.error(f"❌ 분석 스크립트를 찾을 수 없습니다: {self.script_path}")
            return False
        
        cmd = [sys.executable, str(self.script_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("분석이 완료되었습니다 ✅")
            st.text_area("📜 실행 로그", result.stdout, height=300)
            return True
        except subprocess.CalledProcessError as e:
            st.error("❌ 실행 중 오류가 발생했습니다.")
            st.text_area("⚠️ 오류 로그", e.stderr, height=300)
            return False
    
    def display_results(self):
        """분석 결과 표시"""
        df = self.loader.load_file(self.loader.files_config.fileformat_output, "FileFormat")
        
        if df is None:
            st.warning(f"⚠️ 결과 파일을 찾을 수 없습니다: {self.output_path}")
            st.info("📝 Data Quality Analyzer를 실행하여 결과를 생성하세요.")
            return
        
        # DataFrame 정규화 (None 값 처리 및 Arrow 호환성)
        df = normalize_dataframe_for_display(df)
        
        df = df.drop(columns=['FilePath'])
        st.dataframe(df, width='stretch', height=550, hide_index=True)
    
    def display(self):
        """메인 UI 표시"""
        st.markdown("##### 모든 파일의 각 컬럼들에 대한 프로파일링을 수행하여 품질분석을 위한 통계를 생성합니다.")
        
        # 통계 상세 내역 표시
        self.display_statistics_info()
        
        st.divider()
        st.markdown("##### 생성된 통계 정보를 기반으로 데이터 품질 분석을 수행하고, 코드간 관계도를 작성합니다.")
        
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            with st.expander("🔐 실행 패스워드 입력", expanded=True):
                password_input = st.text_input(
                    "패스워드를 입력하세요",
                    type="password",
                    key="quality_analyzer_password_input",
                    help="Data Quality Analyzer 실행을 위한 패스워드가 필요합니다."
                )
        with col2:
            st.markdown("###### 전체 파일의 수 및 크기에 따라 시간이 많이 소요될 수 있습니다.")
            if st.button("🔍 Data Quality Analyzer 실행", key="btn_quality_analyzer"):
                if not password_input:
                    st.error("❌ 패스워드를 입력하세요.")
                elif password_input != self.password:
                    st.error("❌ 패스워드가 올바르지 않습니다.")
                else:
                    with st.spinner("분석 실행 중... 잠시만 기다려주세요."):
                        self.run_analyzer()
        
        st.divider()
        st.caption(f"실행 후 결과 파일은 {self.output_path.parent} 하위에 저장됩니다.")
        st.markdown("##### Data Quality Analyzer의 결과 입니다. 스크롤하여 전체 내용을 분석하세요.")
        st.write("생성된 결과는 데이터 프레임에 커서를 위치하면 다운로드 버튼이 생성됩니다.")
        
        self.display_results()
        
        # st.markdown("##### Data Quality Information Menu 에서 상세 분석을 수행합니다.")

# -------------------------------------------------------------------
# DATA TYPE & RULE ANALYZER
# -------------------------------------------------------------------
class DataTypeRuleAnalyzer:
    """Data Type & Rule Analyzer 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any], loader: FileLoader):
        self.yaml_config = yaml_config
        self.loader = loader
        self.script_path = Path(loader.files_config.analyzer_script_rule)
        self.output_path = Path(loader.files_config.ruledatatype_output)
        self.password = yaml_config.get("DataSense_Password", "tkfkdgo")
        self.required_columns = [
            "FileName", "MasterType", "ColumnName", "DataType", "OracleType",
            "Rule", "RuleType", "MatchedRule", "MatchedScoreList", 
            "MatchScoreAvg", "MatchScoreMax"
        ]
    
    def run_analyzer(self) -> bool:
        """Data Type & Rule Analyzer 스크립트 실행"""
        if not self.script_path.exists():
            st.error(f"❌ 분석 스크립트를 찾을 수 없습니다: {self.script_path}")
            return False
        
        cmd = [sys.executable, str(self.script_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("분석이 완료되었습니다 ✅")
            st.text_area("📜 실행 로그", result.stdout, height=300)
            return True
        except subprocess.CalledProcessError as e:
            st.error("❌ 실행 중 오류가 발생했습니다.")
            st.text_area("⚠️ 오류 로그", e.stderr, height=300)
            return False
    
    def display_results(self):
        """분석 결과 표시"""
        df = self.loader.load_file(self.loader.files_config.ruledatatype_output, "RuleDataType")
        
        if df is None:
            st.warning(f"⚠️ 결과 파일을 찾을 수 없습니다: {self.output_path}")
            st.info("📝 Data Type & Rule Analyzer를 실행하여 결과를 생성하세요.")
            return
        
        # 필수 컬럼 필터링
        available_columns = [col for col in self.required_columns if col in df.columns]
        if available_columns:
            df = df[available_columns]
        
        # DataFrame 정규화 (None 값 처리 및 Arrow 호환성)
        df = normalize_dataframe_for_display(df)
        
        # df = df.drop(columns=['FilePath'])

        st.dataframe(df, width='stretch', height=600, hide_index=True)
    
    def display(self):
        """메인 UI 표시"""
        st.markdown("##### Data Quality Analyzer 결과를 기반으로 각 컬럼에 대한 Rule 프로파일링을 수행합니다.")
        st.markdown("##### Value 구성(패턴) 정보를 통해 각 컬럼에 대한 기본적인 속성을 정의합니다.")
        
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            with st.expander("🔐 실행 패스워드 입력", expanded=True):
                password_input = st.text_input(
                    "패스워드를 입력하세요",
                    type="password",
                    key="rule_analyzer_password_input",
                    help="Data Type & Rule Analyzer 실행을 위한 패스워드가 필요합니다."
                )
        with col2:
            st.markdown("###### 전체 파일의 수 및 크기에 따라 시간이 많이 소요될 수 있습니다.")
            if st.button("🔍 Data Type & Rule 분석 실행", key="btn_rule_analyzer"):
                if not password_input:
                    st.error("❌ 패스워드를 입력하세요.")
                elif password_input != self.password:
                    st.error("❌ 패스워드가 올바르지 않습니다.")
                else:
                    with st.spinner("분석 실행 중... 잠시만 기다려주세요."):
                        self.run_analyzer()
        
        st.divider()
        st.caption(f"결과 파일은 {self.output_path.parent} 하위에 저장됩니다.")
        st.markdown("##### Data Quality Information Menu에서 상세 분석을 수행합니다.")
        
        self.display_results()

# -------------------------------------------------------------------
# CODE RELATIONSHIP ANALYZER
# -------------------------------------------------------------------
class CodeRelationshipAnalyzer:
    """Code Relationship Analyzer 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any], loader: FileLoader):
        self.yaml_config = yaml_config
        self.loader = loader
        self.script_path = Path(loader.files_config.analyzer_script_relationship)
        self.output_path = Path(loader.files_config.codemapping_output)
        self.password = yaml_config.get("DataSense_Password", "tkfkdgo")
    
    def run_analyzer(self) -> bool:
        """Code Relationship Analyzer 스크립트 실행"""
        if not self.script_path.exists():
            st.error(f"❌ 분석 스크립트를 찾을 수 없습니다: {self.script_path}")
            return False
        
        cmd = [sys.executable, str(self.script_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("분석이 완료되었습니다 ✅")
            st.text_area("📜 실행 로그", result.stdout, height=300)
            return True
        except subprocess.CalledProcessError as e:
            st.error("❌ 실행 중 오류가 발생했습니다.")
            st.text_area("⚠️ 오류 로그", e.stderr, height=300)
            return False
    
    def display_results(self):
        """분석 결과 표시"""
        df = self.loader.load_file(self.loader.files_config.codemapping_output, "CodeMapping")
        
        if df is None:
            st.warning(f"⚠️ 결과 파일을 찾을 수 없습니다: {self.output_path}")
            st.info("📝 Code Relationship Analyzer를 실행하여 결과를 생성하세요.")
            return
        
        # DataFrame 정규화 (None 값 처리 및 Arrow 호환성)
        df = normalize_dataframe_for_display(df)
        
        df = df.drop(columns=['FilePath'])

        st.dataframe(df, width='stretch', height=600, hide_index=True)
    
    def display(self):
        """메인 UI 표시"""
        st.markdown("##### Data Quality Analyzer 결과를 기반으로 모든 파일의 컬럼들에 대한 관계도를 작성합니다.")
        
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            with st.expander("🔐 실행 패스워드 입력", expanded=True):
                password_input = st.text_input(
                    "패스워드를 입력하세요",
                    type="password",
                    key="code_relationship_password_input",
                    help="Code Relationship Analyzer 실행을 위한 패스워드가 필요합니다."
                )
        with col2:
            st.markdown("###### 전체 파일의 수 및 크기에 따라 시간이 많이 소요될 수 있습니다. (약 10분 이상 소요)")
            if st.button("🔍 Code Relationship 분석 실행", key="btn_relationship_analyzer"):
                if not password_input:
                    st.error("❌ 패스워드를 입력하세요.")
                elif password_input != self.password:
                    st.error("❌ 패스워드가 올바르지 않습니다.")
                else:
                    with st.spinner("분석 실행 중... 잠시만 기다려주세요."):
                        self.run_analyzer()
        
        st.divider()
        st.caption(f"결과 파일은 {self.output_path.parent} 하위에 저장됩니다.")
        st.markdown("##### Data Quality Information Menu에서 상세 분석을 수행합니다.")
        
        self.display_results()

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
class DataAnalyzerApp:
    """Data Analyzer 통합 애플리케이션"""
    
    def __init__(self):
        self.yaml_config = None
        self.loader = None
        self.quality_analyzer = None
        self.rule_analyzer = None
        self.relationship_analyzer = None
    
    def initialize(self) -> bool:
        """초기화"""
        try:
            self.yaml_config = load_yaml_datasense()
            self.loader = FileLoader(self.yaml_config)
            self.quality_analyzer = DataQualityAnalyzer(self.yaml_config, self.loader)
            self.rule_analyzer = DataTypeRuleAnalyzer(self.yaml_config, self.loader)
            self.relationship_analyzer = CodeRelationshipAnalyzer(self.yaml_config, self.loader)
            return True
        except Exception as e:
            st.error(f"초기화 오류: {e}")
            return False
    
    def display(self):
        """메인 UI 표시"""
        st.title(f"📊 {APP_NAME}")
        st.markdown(APP_DESC)
        st.markdown(APP_DESC2)
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Data Quality Analyzer", 
            "📋 Data Type & Rule Analyzer", 
            "🔗 Data Relationship Analyzer"
        ])
        
        with tab1:
            self.quality_analyzer.display()
        
        with tab2:
            self.rule_analyzer.display()
        
        with tab3:
            self.relationship_analyzer.display()

        st.markdown("##### Data Quality Information Menu 에서 상세 분석을 수행합니다.")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    try:
        app = DataAnalyzerApp()
        if app.initialize():
            app.display()
        else:
            st.error("DataAnalyzerApp 초기화 실패")
    except Exception as e:
        st.error(f"애플리케이션 오류: {e}")
        import traceback
        st.exception(e)

if __name__ == "__main__":
    main()

