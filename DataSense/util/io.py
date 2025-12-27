# -------------------------------------------------------------------

import os
import shutil
import yaml
import logging
from os.path import basename
from pathlib import Path
from datetime import datetime
import pandas as pd

# ---------------------- 로깅 설정 ----------------------
def setup_logger(app_name: str, debug_mode: bool = False) -> logging.Logger:
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{app_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# ---------------------- 파일 읽기 ----------------------
def read_csv_any(path: str) -> pd.DataFrame:
    """다양한 인코딩으로 CSV 파일 읽기"""
    path = os.path.expanduser(os.path.expandvars(str(path)))
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise FileNotFoundError(path)

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """헤더 정리"""
    out = df.copy()
    out.columns = [str(c).replace('\ufeff', '').strip() for c in out.columns]
    return out

    #str(c): 컬럼명이 숫자로 되어 있는 경우를 대비해 모두 문자열로 변환합니다.
    # .replace("\ufeff", ""): BOM(Byte Order Mark) 제거.
    # 윈도우 메모장이나 엑셀에서 'UTF-8(BOM)' 형식으로 저장된 CSV를 읽으면 첫 번째 컬럼명 앞에 
    # 눈에 보이지 않는 \ufeff 문자가 붙어 df['ID']로 호출해도 찾지 못하는 경우가 생기는데, 이를 완벽히 방지합니다.
    # .strip(): 앞뒤 공백 제거.
    # " 이름 " 처럼 공백이 포함된 헤더를 "이름"으로 정리합니다.

# ================================================================
# 도우미 함수
# ================================================================
def normalize_str(s: str) -> str:
    """일반적인 문자열 정규화"""
    import unicodedata  # 한글 자모 결합 정규화를 위해 임포트
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\u3000", " ")
    return " ".join(s.split())

    # 🔍 함수 설명
    # 1. 한글 자모 결합 정규화 (NFC)
    # s = unicodedata.normalize("NFC", str(s))
    # 현상: Mac에서 작성한 파일명이나 텍스트를 Windows에서 보면 'ㄱㅏ'처럼 자모가 분리되어 보이는 현상(NFD 방식)이 있습니다. 
    # 혹은 눈에는 똑같이 '가'로 보이지만 컴퓨터는 서로 다른 문자로 인식하는 경우가 발생합니다.
    # 해결: NFC(Normalization Form Canonical Composition) 방식은 분리된 자음과 모음을 하나의 완성된 글자로 합쳐줍니다.
    # 효과: 데이터 그룹화(groupby)나 조인(merge)을 할 때, **"눈에는 같아 보이는데 데이터상으로는 다르게 처리되는 에러"**를 완벽히 방지합니다.

    # 2. 전각 공백 처리 (\u3000)
    # s = s.replace("\u3000", " ")
    # 현상: 일본어나 한국어 입력기 사용 중 실수로 들어가는 **전각 공백(Ideographic Space)**은 일반적인 공백( )과 다르게 인식됩니다.
    # 해결: 이를 표준 반각 공백으로 변환하여 데이터 형식을 통일합니다.

    # 3. 중복 공백 제거 및 트리밍 (split & join)
    # return " ".join(s.split())
    # 작동 원리:
    # s.split()은 문자열 사이의 모든 공백(탭, 줄바꿈, 여러 개의 연속된 스페이스)을 기준으로 단어를 나눕니다.
    # " ".join(...)은 나눠진 단어들을 딱 하나의 공백으로만 연결합니다.
    # 효과: 문자열 앞뒤의 불필요한 공백을 제거(Trim)함과 동시에, 문자열 중간에 실수로 들어간 이중 공백을 단일 공백으로 깔끔하게 정리합니다.

    # 💡 왜 DataSense(DQ)에 이 기능이 필수적인가요?
    # 데이터 품질 분석(Data Quality Analysis)에서 문자열 정규화는 '데이터 클렌징'의 핵심입니다.
    # 중복 제거의 정확도: " 삼성전자"와 "삼성 전자"를 동일한 업체로 인식하게 해줍니다.
    # 패턴 분석의 일관성: 이전에 만드신 Get_String_Pattern 함수가 작동하기 전에 이 함수를 먼저 거치면, 훨씬 정확한 문자열 패턴을 추출할 수 있습니다.
    # 검색 성능: 데이터베이스에 적재하기 전 이 과정을 거치면 검색 엔진이나 인덱스가 훨씬 효율적으로 작동합니다.

    # ✅ 참고 사항
    # 이 함수를 사용하려면 파일 상단에 반드시 아래 임포트 문이 포함되어야 합니다.
    # import unicodedata    
#------------------------------------------------------------------
# YAML 파일 로드 함수
def Load_Yaml_File(config_path: str | None = None):
    """
    YAML 파일을 읽어 dict 로 반환한다.

    Parameters
    ----------
    config_path : str, optional
        YAML 파일 경로. 지정하지 않으면
        프로젝트 기본 경로(DataSense/util/DS_Master.yaml)를 사용한다.
    """

    # ① 기본 경로 계산 (모듈 기준 상대경로)
    if config_path is None:
        raise FileNotFoundError(f"YAML 설정 파일을 지정하지 않았습니다: {config_path}")
        # base_dir = os.path.dirname(os.path.dirname(__file__))   # DataSense/
        # config_path = os.path.join(base_dir, "DataSense/util", "DS_Master.yaml")

    # ② 파일 존재 여부 확인
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML 설정 파일을 찾을 수 없습니다: {config_path}")

    # ③ YAML 로드
    with open(config_path, encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    
    # ④ ROOT_PATH가 없거나 절대경로인 경우 자동 감지
    if config and ('ROOT_PATH' not in config or Path(config.get('ROOT_PATH', '')).is_absolute()):
        # config_path 기준으로 프로젝트 루트 찾기
        yaml_path = Path(config_path)
        # DataSense/util/DS_Master.yaml -> DataSense/util -> DataSense -> QDQM
        project_root = yaml_path.parent.parent.parent  # DataSense/util -> DataSense -> QDQM
        config['ROOT_PATH'] = str(project_root)
    
    return config

# YAML 파일 로드 함수
def load_yaml_datasense():
    import yaml
    import sys
    # 현재 파일의 상위 디렉토리를 path에 추가
    CURRENT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(CURRENT_DIR_PATH)
    yaml_path = 'C:/projects/myproject/QDQM/DataSense/util'
    yaml_file_name = 'DS_Master.yaml'

    file_path = os.path.join(yaml_path, yaml_file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:  
        st.error(f"QDQM의 기본 YAML 파일을 찾을 수 없습니다: {file_path}")
        return None


def set_page_config(yaml_file):
    st.set_page_config(
        page_title="QDQM Analyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.header('Quick Data Quality Management')
    st.sidebar.markdown("""
    <div style='background-color: #F0F8FF; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <p style='font-size: 20px; color: #333; line-height: 1.6;'>
            모든 데이터(Data)를 <span style='font-size: 20px; color: #0066cc; font-weight: bold;'> 쉽고(Easy)</span>, 
            <span style='font-size: 20px; color: #cc3300; font-weight: bold;'> 빠르며(Fast)</span>, 
            <span style='color: #006633; font-weight: bold;'> 정확하게(Accurate)</span> 분석합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("")
    st.sidebar.markdown("<h4>Powered by tifisoft</h4>", unsafe_allow_html=True)
    st.sidebar.markdown("<h4>qdqm@tifisoft.com</h4>", unsafe_allow_html=True)

    return None

def Backup_File(OUTPUT_DIR, FileName, extension): # 기존 파일 백업
    try:
        Backup_Dir = f"{OUTPUT_DIR}/Backup"
        if not os.path.exists(Backup_Dir):
            os.makedirs(Backup_Dir)

        file_path = os.path.join(OUTPUT_DIR, f'{FileName}.{extension}')
        file_path_old = os.path.join(Backup_Dir, f'{FileName}.{extension}')
        if os.path.exists(file_path):
            shutil.copy(file_path, file_path_old)

    except Exception as e:
        print(f"{OUTPUT_DIR}\\{FileName}.{extension} 파일 백업 중 오류: {e}")
        return False
    return True

def Directory_Recreate(Directory):
    """ 관련 디렉토리 확인 및 생성 """
    print(f"{Directory} 폴더 확인 및 재생성")
    try:
        # MASTER_DIR 폴더가 없으면 생성하고, 기존 파일 삭제
        if not os.path.exists(Directory):  # 폴더가 없으면
            os.makedirs(Directory)  # 폴더 생성
        else:  # 폴더가 이미 존재하면
            shutil.rmtree(Directory)  # 기존 파일 삭제
            os.makedirs(Directory)  # 폴더 재생성
    except Exception as e:
        print(f"{Directory} 폴더 재생성 중 오류: {e}")
        return False
    return True

