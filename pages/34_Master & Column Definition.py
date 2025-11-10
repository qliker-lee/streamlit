# -*- coding: utf-8 -*-
"""
📘 Value Chain Master Column Definition
@gist-36 Value Chain Master에 Column List를 정의합니다.
2025.11.02 Qliker (Class-based Version)
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
APP_NAME = "Value Chain Master's Column Definition"
APP_DESC = "##### Value Chain Master에 관리해야 할 필수 컬럼 리스트를 정의하고 관리합니다."

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
            "valuechain_system": "DataSense/DS_Meta/DataSense_ValueChain_System.csv",   
            "valuechain_master": "DataSense/DS_Meta/DataSense_ValueChain_Master.csv",
            "valuechain_master_column_list": "DataSense/DS_Meta/DataSense_ValueChain_Master_ColumnList.csv",
            "valuechain_master_column": "DataSense/DS_Meta/DataSense_ValueChain_Master_Column.csv",
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
# FILE LOADER
# -------------------------------------------------------------------
@dataclass
class FileConfig:
    """파일 설정 정보"""
    valuechain_system: str
    valuechain_master: str
    valuechain_master_column_list: str
    valuechain_master_column: str

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
            valuechain_system=_full_path(files.get('valuechain_system', 'DataSense/DS_Meta/DataSense_ValueChain_System.csv')),
            valuechain_master=_full_path(files.get('valuechain_master', 'DataSense/DS_Meta/DataSense_ValueChain_Master.csv')),
            valuechain_master_column_list=_full_path(files.get('valuechain_master_column_list', 'DataSense/DS_Meta/DataSense_ValueChain_Master_ColumnList.csv')),
            valuechain_master_column=_full_path(files.get('valuechain_master_column', 'DataSense/DS_Meta/DataSense_ValueChain_Master_Column.csv'))
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
    
    def load_valuechain_master(self) -> pd.DataFrame:
        """ValueChain Master 파일 로드"""
        path = self.files_config.valuechain_master
        df = self.load_file(path, "ValueChain Master")
        if df.empty:
            st.warning(f"⚠️ 파일이 존재하지 않습니다: {path}")
        return df

    def load_valuechain_system(self) -> pd.DataFrame:
        """System 파일 로드"""
        path = self.files_config.valuechain_system
        df = self.load_file(path, "ValueChain System Definition")
        if df.empty:
            st.warning(f"⚠️ ValueChain System 정의 파일이 존재하지 않습니다. ValueChain System Definition 페이지에서 데이터를 생성 및 입력하세요.")
        return df
    
    def load_valuechain_master_column_list(self) -> pd.DataFrame:
        """ValueChain Master Column List 파일 로드"""
        path = self.files_config.valuechain_master_column_list
        master_column_list_df = self.load_file(path, "ValueChain Master Column List")
        master_df = self.load_valuechain_master()
        
        # 파일이 없으면 Master 파일에서 기본 구조 생성
        if master_column_list_df.empty:
            if not master_df.empty:
                st.info(f"⚠️ 파일이 존재하지 않습니다: {path}. 새롭게 생성합니다.")
                mc_df = master_df.copy()
                # 필요한 컬럼 추가
                for col in ["Master_Kor",  "ColumnList"]:
                    if col not in mc_df.columns:
                        mc_df[col] = ""
                

        else: # 파일이 존재하는 경우에도 master_df 가 변경되었을 경우 master_column_list_df 를 변경함 
            key_columns = ["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", "Master"]
            master_df = master_df[key_columns]
            mc_df = pd.merge(master_df, master_column_list_df, on=key_columns, how="left")
            mc_df.fillna("", inplace=True)

        system_df = self.load_valuechain_system()
        
        # System 컬럼이 없으면 추가
        if "System" not in mc_df.columns:
            mc_df["System"] = ""
        
        # System 값이 없는 행만 필터링하여 merge 수행
        system_empty_mask = (mc_df["System"].isna()) | (mc_df["System"].astype(str).str.strip() == "")
        
        if system_empty_mask.any() and not system_df.empty and "System" in system_df.columns:
            merge_keys = ["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor"]
            # merge key가 system_df에 모두 있는지 확인
            available_keys = [key for key in merge_keys if key in system_df.columns]
            if available_keys:
                # 각 키 조합에 대해 첫 번째 System만 선택
                system_df_first = system_df.groupby(available_keys, as_index=False).first()
                # System 값이 없는 행만 merge
                mc_df_empty = mc_df[system_empty_mask].copy()
                if not mc_df_empty.empty:
                    mc_df_merged = pd.merge(mc_df_empty, system_df_first[available_keys + ["System"]], on=available_keys, how="left")
                    # merge 결과를 원본 mc_df에 반영 (System 컬럼이 있는지 확인)
                    if "System" in mc_df_merged.columns:
                        mc_df.loc[system_empty_mask, "System"] = mc_df_merged["System"].fillna("")
                    else:
                        # System 컬럼이 없으면 빈 문자열로 유지
                        mc_df.loc[system_empty_mask, "System"] = ""
        
        return mc_df
# -------------------------------------------------------------------
# 유틸 함수
# -------------------------------------------------------------------
def split_master_column_list_to_rows(
    df: pd.DataFrame,
    columnlist_col: str = "ColumnList",
    drop_empty: bool = True,
    strip_space: bool = True,
    rename_to_singular: bool = True,
) -> pd.DataFrame:
    """Master Column List를 행으로 분리"""
    if columnlist_col not in df.columns:
        raise KeyError(f"'{columnlist_col}' 컬럼을 찾을 수 없습니다.")

    out = df.copy()

    s = out[columnlist_col].fillna("").astype(str)
    if strip_space:
        splitted = s.str.split(",").apply(lambda items: [x.strip() for x in items])
    else:
        splitted = s.str.split(",")

    out[columnlist_col] = splitted
    out = out.explode(columnlist_col, ignore_index=True)

    if drop_empty:
        col = out[columnlist_col].astype(str)
        mask = col.notna() & (col.str.strip() != "") & (col.str.lower().str.strip() != "nan")
        out = out.loc[mask].copy()

    if rename_to_singular:
        out = out.rename(columns={columnlist_col: "Column"})

    out = out[["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor",
        "Master", "Master_Kor", "Column"]]
    return out

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
class MasterColumnDefinition:
    """Value Chain Master Column Definition 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.loader = FileLoader(yaml_config)
        self.master_path = self.loader.files_config.valuechain_master
        self.column_list_path = self.loader.files_config.valuechain_master_column_list
        self.column_path = self.loader.files_config.valuechain_master_column
    
    def save_data(self, df: pd.DataFrame) -> bool:
        """CSV 파일 저장"""
        try:
            df.to_csv(self.column_list_path, index=False, encoding="utf-8-sig")
            
            split_df = split_master_column_list_to_rows(df)
            if not split_df.empty:
                split_df.to_csv(self.column_path, index=False, encoding="utf-8-sig")
            else:
                st.warning(f"⚠️ {self.column_path} 생성이 실패했습니다.")
                return False
            return True
        except Exception as e:
            st.error(f"❌ {self.column_list_path} 저장 실패: {e}")
            return False
    
    def display(self):
        st.title(APP_NAME)
        st.markdown(APP_DESC)
        st.divider()
        
        # 1️⃣ 데이터 로드
        # System 데이터 로드
        valuechain_system_df = self.loader.load_valuechain_system()
        if valuechain_system_df.empty:
            st.info("📝 ValueChain System 정의 데이터가 없습니다. ValueChain Definition 페이지에서 데이터를 생성 및 입력하세요.")
            return
        # Value Chain Master Column List 데이터 로드
        column_list_path = Path(self.loader.files_config.valuechain_master_column_list)
        file_existed = column_list_path.exists()
        
        df = self.loader.load_valuechain_master_column_list()

        if df.empty:
            st.info("📝 데이터가 없습니다. 로직을 점검하세요.")
            return
        
        # 파일이 새로 생성된 경우 자동 저장 (파일이 없었다가 생성된 경우)
        if not file_existed and "System" in df.columns and not df.empty:
            if self.save_data(df):
                st.success("✅ 새로 생성된 파일이 자동 저장되었습니다.")
                st.rerun()  # 저장 후 페이지 새로고침
        # Value Chain Master 데이터 로드
        vc_master_df = self.loader.load_valuechain_master()

        if vc_master_df.empty:
            st.info("📝 Value Chain Master 데이터가 없습니다. Value Chain Definition에서 데이터를 생성 및 입력하세요.")
            return
       
        industries = ["전체"] + sorted(df["Industry"].unique().tolist())
        
        # 2️⃣ Industry 선택
        col0, col1, col2 = st.columns([1, 2, 4])
        with col0:
            st.markdown("##### 📊 현재 데이터")

        with col1:
            selected_industry = st.selectbox("Industry 선택", options=industries, key="vc_col_industry")
        
        # System 데이터 준비
        selected_system = []
        if selected_industry != "전체":
            system_df_filtered = valuechain_system_df[valuechain_system_df["Industry"] == selected_industry].copy()
            if not system_df_filtered.empty and "System" in system_df_filtered.columns:
                selected_system = system_df_filtered["System"].unique().tolist()
                selected_system = [s for s in selected_system if s and str(s).strip()]  # 빈 값 제거

        # System 리스트가 비어있으면 기본값 추가
        if not selected_system:
            selected_system = [""]
        
        with col2:
            if selected_industry == "전체":
                filtered_df = df.copy()
                st.info(f"📊 전체 데이터 ({len(filtered_df)}개 행)")
            else:
                filtered_df = df[df["Industry"] == selected_industry].copy()
                st.info(f"📊 Industry : {selected_industry}  ({len(filtered_df)}개 행)")
            
            if filtered_df.empty:
                st.warning("선택된 Industry에 데이터가 없습니다.")
                return
            
        filtered_df = filtered_df.reset_index(drop=True)

        if "System" not in filtered_df.columns:
            filtered_df["System"] = ""
        if "Master_Kor" not in filtered_df.columns:
            filtered_df["Master_Kor"] = ""
        if "ColumnList" not in filtered_df.columns:
            filtered_df["ColumnList"] = ""

        # ✅ Industry 선택된 경우에만 이후 기능 수행
        if selected_industry != "전체":
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

            edited_df = edited_df.reset_index(drop=True)

            col01, col02 = st.columns([8, 1])
            with col01:
                display_df = filtered_df.copy()
                display_df = display_df[["Activities_Type", "Activity_Seq", "Activities", "Activities_Kor",
                    "Master", "Master_Kor", "System", "ColumnList"]]

                st.dataframe(display_df, use_container_width=True, height=500, hide_index=False, column_config={
                    "Activities_Type": st.column_config.TextColumn("Type", width=50),
                    "Activity_Seq": st.column_config.NumberColumn("Seq", width=20),
                    "Activities": st.column_config.TextColumn("Activities"),
                    "Activities_Kor": st.column_config.TextColumn("Activities Kor"),
                    "Master": st.column_config.TextColumn("Master"),
                    "Master_Kor": st.column_config.TextColumn("Master Kor"),
                    "System": st.column_config.TextColumn("System"),
                    "ColumnList": st.column_config.TextColumn("Column List"),
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
                        st.info(f"**Value Chain Master Information**")
                    with cola2:
                        st.info(f"**Master의 한글정보, System 정보, Column List를 입력하세요.**")

                    col1, col2, col3 = st.columns([1, 2, 4])
                    with col1:
                        st.text_input("Industry", value=str(row_data.get("Industry", "")), disabled=True)
                        st.text_input("Activities Type", value=str(row_data.get("Activities_Type", "")), disabled=True)
                        st.text_input("Activity Seq", value=str(row_data.get("Activity_Seq", "")), disabled=True)

                    with col2:
                        st.text_input("Activities", value=str(row_data.get("Activities", "")), disabled=True)
                        st.text_input("Activities Kor", value=str(row_data.get("Activities_Kor", "")), disabled=True)
                        st.text_input("Master", value=str(row_data.get("Master", "")), disabled=True)
                    with col3:
                        master_kor_val = st.text_input("Master Kor", value=str(row_data.get("Master_Kor", "")), disabled=False, key=f"master_kor_{actual_idx}")
                        
                        # System selectbox 안전 처리
                        current_system = str(row_data.get("System", "")).strip()
                        if current_system and current_system in selected_system:
                            system_index = selected_system.index(current_system)
                        else:
                            system_index = 0  # 기본값
                        
                        system_val = st.selectbox(
                            "System", 
                            options=selected_system, 
                            index=system_index, 
                            disabled=False, 
                            key=f"system_{actual_idx}"
                        )

                        columns_val = st.text_area("Column List (컴마로 구분)", value=str(row_data.get("ColumnList", "")), height=100)
    
                    with col2:
                        submitted = st.form_submit_button("✅ 적용")                     
                        
                    if submitted:
                        edited_df.loc[actual_idx, "Master_Kor"] = master_kor_val
                        edited_df.loc[actual_idx, "System"] = system_val
                        edited_df.loc[actual_idx, "ColumnList"] = columns_val
                        st.session_state.edited_filtered_df = edited_df
                        st.success(f"✅ 행 Index {actual_idx} 업데이트 완료")
                        st.rerun()

            st.divider()
            st.write(f"💾 {self.column_list_path} 에 저장됩니다.")
            st.write(f"💾 {self.column_path} 에 저장됩니다.")
            col21, col22 = st.columns([2, 8])
            with col21:
                if st.button("💾 파일 저장", use_container_width=True):
                    # 전체 데이터에서 선택된 Industry 외의 데이터 가져오기
                    all_df = self.loader.load_valuechain_master_column_list()
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
