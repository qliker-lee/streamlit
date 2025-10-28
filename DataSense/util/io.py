
import os
import shutil
import yaml
from pathlib import Path

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

