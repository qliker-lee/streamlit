###################################################
# 2025. 12. 27.  Qliker
# 데이터 센스 솔루션 Main 
###################################################
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
# 2. 컴파일 발생하는 Streamlit 경고 메시지 억제 설정 (Streamlit import 전에 호출)
# -------------------------------------------------------------------
from DataSense.util.streamlit_warnings import setup_streamlit_warnings
setup_streamlit_warnings()

# -------------------------------------------------------------------
# 3. 필수 라이브러리 import
# -------------------------------------------------------------------
import streamlit as st
import os
from PIL import Image

SOLUTION_NAME = "Data Sense"
SOLUTION_KOR_NAME = "데이터 센스"

# 페이지 기본 설정
st.set_page_config(
    page_title="Data Sense for AI ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main_page():
    """메인 페이지"""
    st.title(SOLUTION_NAME)
    
    # 주요 기능 소개
    st.write("---")
    st.subheader(SOLUTION_KOR_NAME + "는 ")
    st.markdown("""
    <div style='background-color: #F0F8FF; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <p style='font-size: 22px; color: #333; line-height: 1.6;'>
        <span style='font-size: 22px; color: #0033ff; font-weight: bold;'> 데이터 센스는 </span> 원천 데이터 프로파일링부터 비즈니스 가치 사슬(Value Chain)까지 연결하여 
        <span style='font-size: 22px; color: #0033ff; font-weight: bold;'><br>데이터의 생성-흐름-품질</span>을 통합 관리하는 지능형 솔루션입니다.
        <br><br>
        핵심 철학은 <span style='font-size: 22px; color: #0033ff; font-weight: bold;'> "데이터는 비즈니스의 언어다." (Data as a Business Language) </span>
        </p> 
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #F0F8FF; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <p style='font-size: 22px; color: #333; line-height: 1.6;'>
        * 기업의 모든 데이터를  
        <span style='font-size: 22px; color: #0033ff; font-weight: bold;'> 정합성 확보와 정제된 데이터 기반 구축을 </span> 위한 지능형 도구 입니다. 
<br>
<span style='font-size: 22px; color: #333; line-height: 1.6;'> * 이를 활용하여 모든 데이터를 
<span style='font-size: 22px; color: #00cc33; font-weight: bold;'> 고품질의 데이터 활용 기반 및 AI 전환을 위한 기반 구축을 확보</span> 할 수 있습니다. 
</span>
<br>
    """, unsafe_allow_html=True)
       
   
    # PDF 파일 경로
    pdf_path = PROJECT_ROOT / "QDQM"/ "DataSense" / "DS_Output" / "images" / "Data Sense 소개서_01.pdf"
    
    # 파일이 존재하는 경우 다운로드 버튼 표시
    if pdf_path.exists():
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            st.download_button(
                label="📄 Data Sense 소개자료 다운로드 (PDF)",
                data=pdf_bytes,
                file_name="Data Sense 소개서_01.pdf",
                mime="application/pdf",
                type="primary"
            )
    else:
        st.warning(f"소개자료 파일을 찾을 수 없습니다: {pdf_path}")


    st.subheader("\n" + SOLUTION_KOR_NAME + "의 핵심 목표") 
    st.markdown("""
<span style='font-size: 20px; '> 
    - 데이터 품질 향상으로 <span style='color: #0033ff; font-weight: bold;'> 정확한 의사결정 지원 <br></span>
    - 프로세스 자동화와 시스템 간 연계 정확성으로  <span style='color: #0033ff; font-weight: bold;'>운용 효율화 <br></span>
    - 품질 문제를 사후가 아닌 사전 대응으로  <span style='color: #0033ff; font-weight: bold;'>비용 절감 <br></span>
    - 정확한 고객정보/거래정보 및 마스터 데이터 관리로  <span style='color: #0033ff; font-weight: bold;'>고객 신뢰도 향상  <br> </span>
    - 감사 추적성 확보 및 데이터 통제 기반 마련으로  <span style='color: #0033ff; font-weight: bold;'>법적/규제 준수 지원  <br></span>
    - <span style='color: #0033ff; font-weight: bold;'> AI, BI, 데이터 분석의 정확도와 신뢰성 확보  <span style='color: #0033ff; font-weight: bold;'>  <br> </span>
    - <span style='color: #0033ff; font-weight: bold;'> 데이터 통합과 거버넌스 기반 마련  <br></span>
    - <span style='color: #0033ff; font-weight: bold;'> 신속하고 안전한 데이터 전환 및 마이그레이션  <br><br> </span>
    """, unsafe_allow_html=True)

    tab1, tab3, tab4, tab5 = st.tabs(["주요 기능",  "수행 방법", "주요 특징",  "솔루션 구성도"])
    with tab1:
        st.subheader("\n" + SOLUTION_KOR_NAME + "주요 기능")
        st.markdown("""
        <span style='font-size: 20px; font-weight: bold;'>
        1. 기업의 내부 코드와 외부 참조코드(Reference Code)와 비교하여 데이터 정합성 검증   <br></span>
            - ISO, 우편번호, 사업자등록번호, 법인코드, 행정동코드 등 표준 레퍼런스 코드와 내부 코드 비교 검증  <br></span>
            - 데이터 구성에 룰 비교가 아닌 데이터 값에 의한 비교로 정합성 확인  <br></span>
        <span style='font-size: 20px; font-weight: bold;'>
        <br>2.	내부 코드 명칭에 의미 중복성/중의성 탐지 및 참조 무결성 확인 <br></span>
            - 동일 또는 유사 의미를 가진 유사 코드 존재, 활용, 연계 여부 탐지 <br></span>
            - Levenshtein, fastText + 군집화 유사도 분석등을 통한 의미 추정 자동 진단 <br></span>
        <span style='font-size: 20px; font-weight: bold;'>
        <br>3. 계층 구조 코드의 논리적 타당성 검증             <br></span>
            - 상위-하위 관계(예: 지역, 조직, 제품, 서비스 등) 구조의 논리적 타당성 확인 <br></span>
            - 순환 참조, 고아 노드, 중복 참조, 최대 깊이 초과 여부 등 자동 진단     <br></span>
        <span style='font-size: 20px; font-weight: bold;'>
    <br> 4.	데이터 파일에서 Primary Key를 찾고, 각 데이터 간의 연관관계 작성 . <br></span>
            - PK 데이터 기반의 물리적 ERD를 작성. <br></span>
            - ERD를 기반으로 상위수준 데이터 아키텍쳐 정립 <br></span>
        <span style='font-size: 20px; font-weight: bold;'>
        <br>5.	시스템 간 코드 연계 정확도 확인 <br></span>
            - ERP, 및 관계 주요 시스템 간 마스터 코드 의미 일치 여부 검토 <br></span>
            - 비즈니스 공통코드의 관계를 고려한 개별 속성 코드화 <br></span>
        <span style='font-size: 20px; font-weight: bold;'>
        <br>6. 마이그레이션을 위한 데이터 정합성 검증 및 매핑 테이블 생성 <br></span>
            - 소스와 변환된 데이터에 대한 전체 건수, 숫자 필드에 대한 다양한 통계 값 검증 <br></span>
            - Check Sum 기능을 활용한 1대1, 1대다, 다대1 변환 검증 <br></span>
        <span style='font-size: 20px; font-weight: bold;'>
        <br>7.	이상 값 탐지 및 클렌징 <br></span>
            - NULL, TEMP, 유니코드, 미완성 한글 및 특수문자 등 데이터 이상 값 탐지 <br></span>
            - 이상 값 유형별 정제 방식 정의 및 마이그레이션 데이터셋 구성 방안 제시 <br></span>
        <span style='font-size: 20px; font-weight: bold;'>
        <br>8. 코드 라이프사이클 추적 <br></span>
            - 코드 생성/변경/미사용 이력을 타임스탬프 기반으로 분석 <br></span>
            - 연계/통합/변환 전략 수립을 위한 이력 추적 및 영향 분석 <br></span>
            - 타 테이블에서 코드의 참조 이력을 집계하여 코드 값 별로 사용 여부 검증 <br><br></span>
        """, unsafe_allow_html=True)


        st.subheader("\n" + SOLUTION_KOR_NAME + "의 수행 방법")
        
        st.subheader("데이터 속성 분석")
        st.markdown("""
        - 모든 데이터에 대한 프로파일링
        - 데이터 값의 타입 및 포맷 분석
        - 고유성(Unique) 및 유효성 검증
        - 특수문자 (유니코드, 미완성한글, 한자) 검증
        """)
            

        st.subheader("데이터 통계 분석 및 참조 무결성 검증")
        st.markdown("""
        - 데이터 값에 대한 다양한 통계 분석을 통한 품질 측정
        - 숫자 컬럼에 대한 통계 분석 및 이상치 탐지 
        - 코드 값 비교에 의한 참조 무결성 검증 
        - 데이터의 길이 검증 (최대, 최소, 정밀도 등)
        """)


        st.subheader("자동으로 매핑")
        st.markdown("""
        - 데이터 속성 분석 및 통계 분석을 활용한 자동 매핑
        - Rule 규칙에 따라 자동 매핑
        - 사용자가 코드 매핑 관계 및 SQL 쿼리를 지정하지 않음 
        - 마스터 데이터와 코드간에 자동으로 릴레이션 생성하여 ERD 작성
        """)
    st.write("---")
    with tab4:
        st.subheader("\n" + SOLUTION_KOR_NAME + "주요 특징")
        st.markdown("""
        <div style='background-color: #F0F8FF; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <p style='font-size: 22px; color: #333; line-height: 1.6;'>
                모든 데이터(Data)를 <span style='font-size: 24px; color: #0066cc; font-weight: bold;'>쉽고(Easy)</span>, 
                <span style='font-size: 24px; color: #cc3300; font-weight: bold;'>빠르며(Fast)</span>, 
                <span style='color: #006633; font-weight: bold;'>정확하게(Accurate)</span> 분석합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.3, 8])
        with col1:
            st.markdown("""
        <span style='font-size: 24px; color: #0066cc; font-weight: bold;'>쉽고(Easy)</span> 
        """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
        <div style='border-radius: 5px;'>
            <span style='font-size: 24px; color: #0066cc; font-weight: bold;'>아주 쉽게 작업을 수행합니다.</span>
            <p style='font-family: "Noto Sans KR", sans-serif; font-size: 16px; line-height: 2;'>
            • 검사할 테이블 지정으로 모든 작업이 자동으로 수행됩니다.<br>
            • 전문기술(데이터베이스, SQL)을 보유하지 않아도 사용할 수 있습니다.<br>
            • 모든 컬럼을 자동으로 마스터 코드와 비교 검사합니다. (사용자가 지정 필요없음)
            </p>
        </div>
        """, unsafe_allow_html=True)
            
        st.write("---")               
        col1, col2 = st.columns([1.3, 8])
        with col1:
            st.markdown("""
                <span style='font-size: 24px; color: #cc3300; font-weight: bold;'>빨리(Fast)</span>
        """, unsafe_allow_html=True)                   
        with col2:
            st.markdown("""
        <div style='border-radius: 5px;'>
            <span style='font-size: 24px; color: #cc3300; font-weight: bold;'>무지 빠르게 수행합니다.</span>
            <p style='font-family: "Noto Sans KR", sans-serif; font-size: 16px; line-height: 2;'>
            • 대용량 데이터는 랜덤으로 샘플링하여 검사합니다. 필요에 따라 전체 데이터를 검사할 수 있습니다.<br>
            • 모든 데이터를 작업할 공간으로 이동 후 검사합니다. 운영계 데이터를 직접 검사할 수 있습니다.<br>
            • 검사할 테이블 단위 및 마스터 코드 데이터를 메모리에 상주시켜 빠르게 검사합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
            
        st.write("---")  
        col1, col2 = st.columns([1.3, 8])
        with col1:
            st.markdown("""
                <span style='font-size: 24px; color: #006633; font-weight: bold;'>정확하게(Accurate)</span> 
        """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
        <div style='border-radius: 5px;'>
            <span style='font-size: 24px; color: #006633; font-weight: bold;'>매우 정확하게 수행합니다.</span></span>
            <p style='font-family: "Noto Sans KR", sans-serif; font-size: 16px; line-height: 2;'></span>
            • 코드 값과 직접 비교하기 때문에 정확합니다. <br>
            • 코드별로의 우선순위, 가중치 및 허용값을 정의하여 판단합니다. <br>
            • 유효 코드가 복수개일 경우 후보 유효 코드도 제공됩니다. <br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.subheader("\n" + SOLUTION_KOR_NAME + " 솔루션 구성도")

        # 이미지 파일 경로 설정
        image_path = "c:/projects/myproject/QDQM/Image/qdqm configrration.png"

        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(
                    image,
                    caption="QDQM Overview",
                    width=600,
                )
            except Exception as e:
                st.error(f"이미지를 불러오는 중 오류가 발생했습니다: {str(e)}")
        
        st.markdown("""
        <div style= padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h2 style='color: #1565C0;'>Connector</h2>
            <p style='font-size: 16px; margin: 10px 0;'>모든 상용데이터베이스(Oracle, MySQL, PostgreSQL, MariaDB, Amazon RDS, EC2, SingleStore), 어플리케이션(Apach, Cloudera, SFDC), 파일시스템(Excel, CSV, Text, Drop Box) 및 REST 액세스 제공합니다.</p>
            <p style='font-size: 16px; margin: 10px 0;'>새로운 커넥터를 신속하게 개발할 수 있는 기능을 통해 신규 데이터에 액세스할 수 있습니다.</p>
        </div>

        <div style=padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h2 style='color: #7B1FA2;'>Extractor</h2>
            <p style='font-size: 16px; margin: 10px 0;'>Connector로 연결된 검사할 시스템에서 검사 대상 테이블들을 스케쥴러에 의해 추출하여 별도의 시스템에 저장합니다.</p>
        </div>

        <div style=padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h2 style='color: #2E7D32;'>Master Code Builder</h2>
            <p style='font-size: 16px; margin: 10px 0;'>검사의 기준이 되는 마스터 코드 데이터를 포맷분석, 길이정보 및 문자속성을 파악하여 마스터테이블을 생성합니다.</p>
            <p style='font-size: 16px; margin: 10px 0;'>외부의 마스터 코드 데이터를 기본으로 제공합니다. (ex : 사업자등록번호, 우편번호, 국가코드 등)</p>
            <p style='font-size: 16px; margin: 10px 0;'>마스터 코드가 없는 경우 Rule 분석 필드를 생성합니다. (eMail, URL, IP Address, 일자 등)</p>
        </div>

        <div style=padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h2 style='color: #E65100;'>Profiler</h2>
            <p style='font-size: 16px; margin: 10px 0;'>데이터 탐색, 프로파일링을 통하여 데이터 타입, 포맷분석, 길이속성, 문자속성, 논리속성 및 통계분석을 분석을 수행합니다.</p>
            <p style='font-size: 16px; margin: 10px 0;'>검사할 각 필드의 값을 마스터 코드 데이터와 비교하여 유효성을 검사합니다. 각 마스터별 매칭 허용율을 관리하여 매칭의 유효율을 제공합니다.</p>
        </div>

        <div style=padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h2 style='color: #006064;'>Report Builder</h2>
            <p style='font-size: 16px; margin: 10px 0;'>모든 결과물을 해석하여 리포트 및 파일을 생성합니다.</p>
        </div>
        """, unsafe_allow_html=True)

def login_page():
    """로그인 페이지"""
    st.title("Welcome to Code Management System")
    st.markdown("\n### Login")
    
    # 로그인 폼
    with st.form("login_form"):
        username = st.text_input("User Name")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username == "qliker" and password == "love1004":
                st.success("Login Success!")
                # URL 파라미터로 로그인 상태 설정
                st.query_params["logged_in"] = True
                st.rerun()
            else:
                st.error("User Name and Password are incorrect. contact: 010-3716-2863")

def sidebar():
    """사이드바"""
    st.sidebar.header(SOLUTION_NAME)
    st.sidebar.markdown("""
        <div style='background-color: #F0F8FF; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <p style='font-size: 20px; color: #333; line-height: 1.6;'>
            Data has <span style='font-size: 20px; color: #cc3300; font-weight: bold;'> a value.</span><br>
            Data is<span style='font-size: 20px; color: #cc3300; font-weight: bold;'> an asset.</span><br>
            Data shapes <span style='color: #cc3300; font-weight: bold;'> our future.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("")
    st.sidebar.markdown("<h4>qliker@kakao.com</h4>", unsafe_allow_html=True)

def main():
    """메인 함수"""
    # 사이드바 표시
    sidebar()
    
    main_page()

if __name__ == "__main__":
    main() 
