# -*- coding: utf-8 -*-
"""
📘 Value Chain Master's vs Our's Master Mapping
@gist-36 Value Chain Master 와 Our's Master 간의 매핑을 정의합니다.
2025.11.05 Qliker (Class-based Version)
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import yaml
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import sys

# -------------------------------------------------------------------
# 기본 앱 정보
# -------------------------------------------------------------------
APP_NAME = "Value Chain Master's vs Our's Master Mapping"
APP_DESC = "##### Value Chain Master 와 Our's Master 간의 매핑을 정의합니다."

# -------------------------------------------------------------------
# 경로 설정
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config
set_page_config(APP_NAME)

# -------------------------------------------------------------------
# YAML CONFIG 로더
# -------------------------------------------------------------------
def _fallback_load_yaml_datasense() -> Dict[str, Any]:
    guessed_root = str(PROJECT_ROOT)
    cfg = {
        "ROOT_PATH": guessed_root,
        "files": {
            "valuechain_master_column_list": "DataSense/DS_Meta/DataSense_ValueChain_Master_ColumnList.csv",
            "valuechain_our_master": "DataSense/DS_Meta/DataSense_ValueChain_Our_Master.csv",
        },
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
# UTILITY FUNCTIONS
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
    valuechain_master_column_list: str
    valuechain_our_master: str
    codemapping: str

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
            # 확장자가 없으면 .csv 추가 (출력 파일의 경우)
            if not p.suffix:
                p = p.with_suffix('.csv')
            if not p.is_absolute():
                p = Path(self.root_path) / p
            return str(p.resolve())
        
        return FileConfig(
            valuechain_master_column_list=_full_path(files.get('valuechain_master_column_list', 'DataSense/DS_Meta/DataSense_ValueChain_Master_ColumnList.csv')),
            valuechain_our_master=_full_path(files.get('valuechain_our_master', 'DataSense/DS_Meta/DataSense_ValueChain_Our_Master.csv')),
            codemapping=_full_path(files.get('codemapping', 'DataSense/DS_Output/CodeMapping.csv'))
        )
    
    def load_file(self, file_path: str, file_name: str, fill_na: bool = True) -> pd.DataFrame:
        """
        개별 파일 로드 (CSV)
        
        Args:
            file_path: 로드할 파일 경로
            file_name: 파일 이름 (에러 메시지용)
            fill_na: NaN 값을 빈 문자열로 채울지 여부 (기본값: True)
        
        Returns:
            pd.DataFrame: 로드된 데이터프레임 (파일이 없거나 로드 실패 시 빈 DataFrame 반환)
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        
        for enc in ("utf-8-sig", "utf-8", "cp949"):
            try:
                df = pd.read_csv(file_path, encoding=enc)
                if fill_na:
                    df = df.fillna("")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"❌ {file_name} 파일 로드 실패 ({enc}): {str(e)}")
                return pd.DataFrame()
        
        st.error(f"❌ {file_name} 파일 인코딩을 확인할 수 없습니다: {file_path}")
        return pd.DataFrame()
    
    def load_valuechain_master_columnlist(self) -> pd.DataFrame:
        """ValueChain Master Column List 파일 로드"""
        required_columns = ["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", 
                            "Master", "Master_Kor"]
        path = self.files_config.valuechain_master_column_list
        df = self.load_file(path, "DataSense_ValueChain_Master_Columnlist")
        if df.empty:
            st.warning(f"⚠️ 파일이 존재하지 않습니다: {path}")
        if not all(col in df.columns for col in required_columns):
            st.warning(f"⚠️ 필요한 컬럼이 없습니다: {path}")
            return pd.DataFrame()
        df = df[required_columns]
        return df

    
    def load_valuechain_our_master(self) -> pd.DataFrame:
        """ValueChain Our Master 파일 로드"""
        path = self.files_config.valuechain_our_master
        df = self.load_file(path, "DataSense_ValueChain_Our_Master")
        if df.empty:
            mc_df = self.load_valuechain_master_columnlist()
            if not mc_df.empty:
                st.info(f"⚠️ 파일이 존재하지 않습니다: {path}. 새롭게 생성합니다.")
                df = mc_df.copy()
                # df["Our_Master"] = ""
                return df
        return df

    def load_codemapping(self) -> pd.DataFrame:
        """Code Mapping 파일 로드"""
        path = self.files_config.codemapping
        df = self.load_file(path, "CodeMapping")    
        if df.empty:
            st.warning(f"⚠️ Our Master Mapping 파일이 존재하지 않습니다: {path}")
            return pd.DataFrame()
        return df

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
class MasterColumnDefinition:
    """Value Chain Master Column Definition 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.loader = FileLoader(yaml_config)
        self.valuechain_master_column_list_path = self.loader.files_config.valuechain_master_column_list
        self.valuechain_our_master_path = self.loader.files_config.valuechain_our_master
        self.codemapping_path = self.loader.files_config.codemapping
    
    def save_data(self, df: pd.DataFrame) -> bool:
        """CSV 파일 저장"""
        try:
            df.to_csv(self.valuechain_our_master_path, index=False, encoding="utf-8-sig")
            st.success(f"✅ 저장 완료: {self.valuechain_our_master_path}")
            return True
        except Exception as e:
            st.error(f"❌ {self.valuechain_our_master_path} 저장 실패: {e}")
            return False
    
    def display(self):
        st.title(APP_NAME)
        st.markdown(APP_DESC)
        st.divider()
        
        # 1️⃣ 데이터 로드
        df = self.loader.load_valuechain_our_master()
        if df.empty:
            st.info("📝 데이터가 없습니다. 새롭게 생성합니다. (로직 점검하세요)")
            return
  
        codemapping_df = self.loader.load_codemapping()
        if codemapping_df.empty:
            st.info("📝 Code Mapping 데이터가 없습니다. Code Mapping 파일을 생성 후 작업을 수행 하세요.")
            return
        # DataFrame 정규화 (None 값 처리 및 Arrow 호환성)
        codemapping_df = codemapping_df[codemapping_df["MasterType"] == "Master"]
        codemapping_df = normalize_dataframe_for_display(codemapping_df)
        # st.write(codemapping_df) # Debug

        industries = ["전체"] + sorted(df["Industry"].unique().tolist())

        # 2️⃣ Industry 선택
        col0, col1, col2 = st.columns([1, 2, 4])
        with col0:
            st.markdown("##### 📊 현재 데이터")

        with col1:
            selected_industry = st.selectbox("Industry 선택", options=industries, key="vc_col_industry")
        
        with col2:
            if selected_industry == "전체":
                filtered_df = df.copy()
                st.info(f"📊 전체 데이터 ({len(filtered_df)}개 행)")
            else:
                filtered_df = df[df["Industry"] == selected_industry].copy()
                filtered_df = filtered_df.reset_index(drop=True)
                display_df = df.copy()
                display_df = display_df[["Activities_Type", "Activity_Seq", "Activities", "Activities_Kor",
                    "Master", "Master_Kor"]]
                st.info(f"📊 Industry : {selected_industry}  ({len(display_df)}개 행)")
            
            if "Our_Master" not in filtered_df.columns:
                filtered_df["Our_Master"] = ""
            if "Our_Master_Kor" not in filtered_df.columns:
                filtered_df["Our_Master_Kor"] = ""

        # ✅ Industry 선택된 경우에만 이후 기능 수행
        if selected_industry != "전체":
            # Our Master 목록 추출 (codemapping_df의 FileName 컬럼에서 유니크한 값만)
            selected_our_master = []
            if not codemapping_df.empty and "FileName" in codemapping_df.columns:
                selected_our_master = codemapping_df["FileName"].dropna().unique().tolist()
            
            # 빈 값 제거 및 정렬
            selected_our_master = [str(s).strip() for s in selected_our_master if s and str(s).strip()]
            selected_our_master = sorted(selected_our_master)
            
            # 빈 값 옵션을 맨 앞에 추가 (값이 없는 것을 선택할 수 있도록)
            selected_our_master = [""] + selected_our_master
            
            # 세션에서 편집 중인 데이터 관리
            try:
                if "edited_filtered_df" not in st.session_state:
                    st.session_state.edited_filtered_df = filtered_df.copy()
                
                if "last_selected_industry" not in st.session_state or st.session_state.last_selected_industry != selected_industry:
                    st.session_state.edited_filtered_df = filtered_df.copy()
                    st.session_state.last_selected_industry = selected_industry
                
                edited_df = st.session_state.edited_filtered_df
            except Exception:
                edited_df = filtered_df.copy()

            col01, col02 = st.columns([8, 1])
            with col01:
                display_df = filtered_df.copy()
                display_df = display_df[["Activities_Type", "Activity_Seq", "Activities", "Activities_Kor",
                    "Master", "Master_Kor", "Our_Master", "Our_Master_Kor"]]
                # DataFrame 정규화 (None 값 처리 및 Arrow 호환성)
                display_df = normalize_dataframe_for_display(display_df)

                st.dataframe(display_df, use_container_width=True, height=500, hide_index=False, column_config={
                    "Activities_Type": st.column_config.TextColumn("Type", width=50),
                    "Activity_Seq": st.column_config.NumberColumn("Seq", width=20),
                    "Activities": st.column_config.TextColumn("Activities"),
                    "Activities_Kor": st.column_config.TextColumn("Activities Kor"),
                    "Master": st.column_config.TextColumn("Master"),
                    "Master_Kor": st.column_config.TextColumn("Master Kor"),
                    "Our_Master": st.column_config.TextColumn("Our Master"),
                    "Our_Master_Kor": st.column_config.TextColumn("Our Master Kor"),
                })

            # ----------------------------
            # 행 편집 UI 및 저장 기능
            # ----------------------------
            with col02:
                st.write("편집할 행 번호")
            
                available_indices = list(edited_df.index.tolist())

                row_index = st.number_input(
                    f"행 번호 : ({min(available_indices)} ~ {max(available_indices)})", 
                    min_value=min(available_indices),
                    max_value=max(available_indices),
                    value=min(available_indices),
                    key="row_index_input"
                )
                actual_idx = int(row_index)

            if actual_idx in edited_df.index:
                selected_row = edited_df.loc[actual_idx]
                

            if actual_idx in edited_df.index:
                with st.form(f"edit_form_{actual_idx}", clear_on_submit=False):
                    row_data = edited_df.loc[actual_idx]

                    cola1, cola2 = st.columns([3, 4])
                    with cola1:
                        st.info(f"**Value Chain Information**")
                    with cola2:
                        st.info(f"**Value Chain Master vs Our Master의 Mapping을 선택하세요.**")

                    col1, col2, col3 = st.columns([1, 2, 4])
                    with col1:
                        # st.text_input("Industry", value=str(row_data.get("Industry", "")), disabled=True)
                        st.text_input("Activities Type", value=str(row_data.get("Activities_Type", "")), disabled=True)
                        st.text_input("Activity Seq", value=str(row_data.get("Activity_Seq", "")), disabled=True)
                        st.text_input("Activities", value=str(row_data.get("Activities", "")), disabled=True)

                    with col2:
                        st.text_input("Activities Kor", value=str(row_data.get("Activities_Kor", "")), disabled=True)
                        st.text_input("Master", value=str(row_data.get("Master", "")), disabled=True)
                        st.text_input("Master Kor", value=str(row_data.get("Master_Kor", "")), disabled=True)

                    with col3:
                        # our_master selectbox 안전 처리
                        current_our_master = str(row_data.get("Our_Master", "")).strip()
                        if current_our_master and current_our_master in selected_our_master:
                            our_master_index = selected_our_master.index(current_our_master)
                        else:
                            our_master_index = 0  # 기본값
                        
                        our_master_val = st.selectbox(
                            "Our Master", 
                            options=selected_our_master, 
                            index=our_master_index, 
                            disabled=False, 
                            key=f"our_master_selectbox_{actual_idx}"
                        )

                        our_master_kor_val = st.text_input("Our Master Kor", value=str(row_data.get("Our_Master_Kor", "")), disabled=False)
                    with col3:
                        submitted = st.form_submit_button("✅ 적용")                     
                        
                        if submitted:
                            edited_df.loc[actual_idx, "Our_Master"] = our_master_val
                            edited_df.loc[actual_idx, "Our_Master_Kor"] = our_master_kor_val
                            st.session_state.edited_filtered_df = edited_df
                            st.success(f"✅ 행 Index {actual_idx} 업데이트 완료")
                            st.rerun()

            st.divider()
            st.write(f"💾 {self.valuechain_our_master_path} 에 저장됩니다.")
            col21, col22 = st.columns([2, 8])
            with col21:
                if st.button("💾 파일 저장", use_container_width=True):
                    # 전체 데이터에서 선택된 Industry 외의 데이터 가져오기
                    all_df = self.loader.load_valuechain_our_master()
                    if selected_industry == "전체":
                        save_df = edited_df.copy()
                    else:
                        other_df = all_df[all_df["Industry"] != selected_industry].copy()
                        save_df = pd.concat([other_df, edited_df], ignore_index=True)
                    
                    if self.save_data(save_df):
                        st.success(f"✅ 파일 저장 완료: {len(edited_df)}개 행 업데이트")
                        st.session_state.edited_filtered_df = None
                        st.session_state.last_selected_industry = None
                        st.rerun()
        else:
            st.divider()
            st.markdown("##### 편집을 하기 위해서는 Industry를 선택하세요.")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    try:
        cfg = load_yaml_datasense()
        app = MasterColumnDefinition(cfg)
        app.display()
    except Exception as e:
        st.error(f"애플리케이션 오류: {e}")
        import traceback
        st.exception(e)

if __name__ == "__main__":
    main()
