# -*- coding: utf-8 -*-
"""
📘 Value & Master Definition
Value Chain Definition, Master Column Definition, Master Mapping을 통합한 프로그램
2025.11.10 Qliker (Integrated Version)
"""

import os
import sys
import warnings
import datetime
import shutil
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import yaml
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

APP_NAME = "Value Chain & Master Definition"
APP_DESC = "##### Value Chain Definition, Master Column Definition, Master & Our's Master Mapping을 통합 관리합니다."
APP_DESC2 = "###### Value Chain Definition : Value Chain 의 Primary Process & Support Function 들과 Master을 정의합니다."
APP_DESC3 = "###### Master Column Definition : Master의 한글명칭, System 및 필수 컬럼 리스트를 정의하고 관리합니다."
APP_DESC4 = "###### Master & Our's Master Mapping : Master 와 Our's Master 간의 매핑을 정의하고 관리합니다."
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
            "valuechain": "DataSense/DS_Meta/DataSense_ValueChain.csv",
            "valuechain_system": "DataSense/DS_Meta/DataSense_ValueChain_System.csv",
            "valuechain_master": "DataSense/DS_Meta/DataSense_ValueChain_Master.csv",
            "valuechain_master_column_list": "DataSense/DS_Meta/DataSense_ValueChain_Master_ColumnList.csv",
            "valuechain_master_column": "DataSense/DS_Meta/DataSense_ValueChain_Master_Column.csv",
            "valuechain_our_master": "DataSense/DS_Meta/DataSense_ValueChain_Our_Master.csv",
            "codemapping": "DataSense/DS_Output/CodeMapping.csv",
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
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            df[col] = df[col].fillna(0)
        elif df[col].dtype == 'object':
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    df[col] = numeric_series.fillna(0)
                else:
                    df[col] = df[col].fillna("")
            except Exception:
                df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna("")
    
    return df

BACKUP_DIR = PROJECT_ROOT / "DataSense" / "__backup"
BACKUP_DIR.mkdir(exist_ok=True)

def backup_file(file_path: Path):
    """현재 파일 백업 (timestamp 포함)"""
    try:
        if file_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = BACKUP_DIR / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            shutil.copy(file_path, backup_path)
            st.info(f"🗄️ 백업 완료: {backup_path.name}")
    except Exception as e:
        st.warning(f"⚠️ 백업 중 오류 발생: {e}")

def split_master_list_to_rows(df):
    """Masters 컬럼을 행으로 분리"""
    if "Masters" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["Masters"] = out["Masters"].fillna("").astype(str)
    out = out.assign(Masters=out["Masters"].str.split(",")).explode("Masters")
    out["Masters"] = out["Masters"].str.strip()
    out = out[out["Masters"] != ""]
    out = out.rename(columns={"Masters": "Master"})
    return out[["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", "Master"]]

def split_system_list_to_rows(df):
    """Systems 컬럼을 행으로 분리"""
    if "Systems" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["Systems"] = out["Systems"].fillna("").astype(str)
    out = out.assign(Systems=out["Systems"].str.split(",")).explode("Systems")
    out["Systems"] = out["Systems"].str.strip()
    out = out[out["Systems"] != ""]
    out = out.rename(columns={"Systems": "System"})
    return out[["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", "System"]]

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
# FILE LOADER
# -------------------------------------------------------------------
@dataclass
class FileConfig:
    """파일 설정 정보"""
    valuechain: str
    valuechain_system: str
    valuechain_master: str
    valuechain_master_column_list: str
    valuechain_master_column: str
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
            if not p.suffix:
                p = p.with_suffix('.csv')
            if not p.is_absolute():
                p = Path(self.root_path) / p
            return str(p.resolve())
        
        return FileConfig(
            valuechain=_full_path(files.get('valuechain', 'DataSense/DS_Meta/DataSense_ValueChain.csv')),
            valuechain_system=_full_path(files.get('valuechain_system', 'DataSense/DS_Meta/DataSense_ValueChain_System.csv')),
            valuechain_master=_full_path(files.get('valuechain_master', 'DataSense/DS_Meta/DataSense_ValueChain_Master.csv')),
            valuechain_master_column_list=_full_path(files.get('valuechain_master_column_list', 'DataSense/DS_Meta/DataSense_ValueChain_Master_ColumnList.csv')),
            valuechain_master_column=_full_path(files.get('valuechain_master_column', 'DataSense/DS_Meta/DataSense_ValueChain_Master_Column.csv')),
            valuechain_our_master=_full_path(files.get('valuechain_our_master', 'DataSense/DS_Meta/DataSense_ValueChain_Our_Master.csv')),
            codemapping=_full_path(files.get('codemapping', 'DataSense/DS_Output/CodeMapping.csv'))
        )
    
    def load_file(self, file_path: str, file_name: str, fill_na: bool = True) -> pd.DataFrame:
        """개별 파일 로드 (CSV)"""
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
    
    def load_valuechain(self) -> pd.DataFrame:
        """ValueChain 파일 로드"""
        cols = [
            "Industry", "Activities_Type", "Activity_Seq", "Activities",
            "Activities_Kor", "Masters", "Systems", "Activity_Detail"
        ]
        path = self.files_config.valuechain
        if os.path.exists(path):
            df = self.load_file(path, "ValueChain")
            for col in cols:
                if col not in df.columns:
                    df[col] = ""
            return df[cols]
        return pd.DataFrame(columns=cols)
    
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
            st.warning(f"⚠️ ValueChain System 정의 파일이 존재하지 않습니다.")
        return df
    
    def load_valuechain_master_column_list(self) -> pd.DataFrame:
        """ValueChain Master Column List 파일 로드"""
        path = self.files_config.valuechain_master_column_list
        master_column_list_df = self.load_file(path, "ValueChain Master Column List")
        master_df = self.load_valuechain_master()
        
        if master_column_list_df.empty:
            if not master_df.empty:
                st.info(f"⚠️ 파일이 존재하지 않습니다: {path}. 새롭게 생성합니다.")
                mc_df = master_df.copy()
                for col in ["Master_Kor", "ColumnList"]:
                    if col not in mc_df.columns:
                        mc_df[col] = ""
            else:
                return pd.DataFrame()
        else:
            key_columns = ["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", "Master"]
            master_df = master_df[key_columns]
            mc_df = pd.merge(master_df, master_column_list_df, on=key_columns, how="left")
            mc_df.fillna("", inplace=True)

        system_df = self.load_valuechain_system()
        
        if "System" not in mc_df.columns:
            mc_df["System"] = ""
        
        system_empty_mask = (mc_df["System"].isna()) | (mc_df["System"].astype(str).str.strip() == "")
        
        if system_empty_mask.any() and not system_df.empty and "System" in system_df.columns:
            merge_keys = ["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor"]
            available_keys = [key for key in merge_keys if key in system_df.columns]
            if available_keys:
                system_df_first = system_df.groupby(available_keys, as_index=False).first()
                mc_df_empty = mc_df[system_empty_mask].copy()
                if not mc_df_empty.empty:
                    mc_df_merged = pd.merge(mc_df_empty, system_df_first[available_keys + ["System"]], on=available_keys, how="left")
                    if "System" in mc_df_merged.columns:
                        mc_df.loc[system_empty_mask, "System"] = mc_df_merged["System"].fillna("")
                    else:
                        mc_df.loc[system_empty_mask, "System"] = ""
        
        return mc_df
    
    def load_valuechain_our_master(self) -> pd.DataFrame:
        """ValueChain Our Master 파일 로드"""
        path = self.files_config.valuechain_our_master
        df = self.load_file(path, "DataSense_ValueChain_Our_Master")
        if df.empty:
            mc_df = self.load_valuechain_master_column_list()
            if not mc_df.empty:
                st.info(f"⚠️ 파일이 존재하지 않습니다: {path}. 새롭게 생성합니다.")
                df = mc_df.copy()
        return df

    def load_codemapping(self) -> pd.DataFrame:
        """Code Mapping 파일 로드"""
        path = self.files_config.codemapping
        df = self.load_file(path, "CodeMapping")
        if df.empty:
            st.warning(f"⚠️ Our Master Mapping 파일이 존재하지 않습니다: {path}")
        return df

# -------------------------------------------------------------------
# VALUE CHAIN DEFINITION
# -------------------------------------------------------------------
class ValueChainDefinition:
    """Value Chain Definition 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.loader = FileLoader(yaml_config)
        self.vc_path = Path(self.loader.files_config.valuechain)
        self.vc_system_path = Path(self.loader.files_config.valuechain_system)
        self.vc_master_path = Path(self.loader.files_config.valuechain_master)
    
    def save_data(self, new_df: pd.DataFrame, existing_df: pd.DataFrame = None, mode: str = "merge"):
        """Value Chain 데이터를 저장"""
        try:
            backup_file(self.vc_path)

            if existing_df is None:
                existing_df = pd.DataFrame()
                if self.vc_path.exists():
                    existing_df = self.loader.load_valuechain()

            new_df = new_df.fillna("")
            new_df["Activity_Seq"] = new_df["Activity_Seq"].astype(str).str.strip()

            if mode == "overwrite":
                final_df = new_df.copy()
            elif mode == "merge":
                if existing_df.empty:
                    final_df = new_df.copy()
                else:
                    existing_df["Activity_Seq"] = existing_df["Activity_Seq"].astype(str).str.strip()
                    new_industries = new_df["Industry"].unique() if "Industry" in new_df.columns else []
                    other_industries_df = existing_df[~existing_df["Industry"].isin(new_industries)].copy()
                    same_industry_df = existing_df[existing_df["Industry"].isin(new_industries)].copy()
                    
                    if same_industry_df.empty:
                        final_df = pd.concat([other_industries_df, new_df], ignore_index=True)
                    else:
                        merged = same_industry_df.merge(
                            new_df,
                            on=["Industry", "Activity_Seq"],
                            how="outer",
                            suffixes=("_old", "_new"),
                            indicator=True
                        )
                        
                        for col in new_df.columns:
                            if col == "Industry" or col == "Activity_Seq":
                                continue
                            old_col = f"{col}_old"
                            new_col = f"{col}_new"
                            
                            if old_col in merged.columns and new_col in merged.columns:
                                merged[col] = merged[new_col].combine_first(merged[old_col])
                                merged.drop(columns=[old_col, new_col], inplace=True, errors="ignore")
                            elif new_col in merged.columns:
                                merged[col] = merged[new_col]
                                merged.drop(columns=[new_col], inplace=True, errors="ignore")
                        
                        merged = merged.drop(columns=["_merge"], errors="ignore")
                        final_df = pd.concat([other_industries_df, merged], ignore_index=True)
            elif mode == "delete":
                final_df = new_df.copy()
            else:
                raise ValueError("mode must be one of: merge | overwrite | delete")

            final_df["Activity_Seq_num"] = pd.to_numeric(final_df["Activity_Seq"], errors="coerce")
            final_df = final_df.sort_values(
                by=["Industry", "Activities_Type", "Activity_Seq_num"],
                ascending=[True, True, True]
            ).drop(columns=["Activity_Seq_num"], errors="ignore")

            final_df.to_csv(self.vc_path, index=False, encoding="utf-8-sig")

            split_df = split_master_list_to_rows(final_df)
            if not split_df.empty:
                split_df.to_csv(self.vc_master_path, index=False, encoding="utf-8-sig")

            split_df = split_system_list_to_rows(final_df)
            if not split_df.empty:
                split_df.to_csv(self.vc_system_path, index=False, encoding="utf-8-sig")

            st.success(f"✅ 저장 완료 ({mode.upper()}) — {len(final_df)}개 행")
            return final_df
        except Exception as e:
            st.error(f"❌ 저장 실패: {e}")
            return existing_df if existing_df is not None else pd.DataFrame()
    
    def display(self):
        st.markdown("#### 📘 Value Chain Definition")
        st.markdown("Value Chain을 입력, 수정, 삭제하고 관리합니다.")
        st.divider()

        df = self.loader.load_valuechain()

        industries = sorted(df["Industry"].unique().tolist()) if not df.empty else []
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_industry = st.selectbox("📊 기존 Industry 선택", ["(New)"] + industries)
        with col2:
            new_industry = st.text_input("또는 New Industry 입력", "")

        industry = new_industry.strip() if new_industry else (
            selected_industry if selected_industry != "(New)" else None
        )

        if not industry:
            st.warning("⚠️ Industry를 선택하거나 입력하세요.")
            return

        col1, col2 = st.columns([7, 1])
        with col1:
            filtered_df = df[df["Industry"] == industry].copy()
            display_df = df[df["Industry"] == industry].copy()
            st.markdown(f"#### 📋 Industry: `{industry}` 의 Value Chain 정의")
            display_df = display_df[["Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", 
                "Masters", "Systems", "Activity_Detail"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400,
                column_config={
                    "Activities_Type": st.column_config.SelectboxColumn("Type", options=["Primary", "Support"]),
                    "Activity_Seq": st.column_config.NumberColumn("Seq", min_value=1, step=1),
                    "Activities": st.column_config.TextColumn("Activities"),
                    "Activities_Kor": st.column_config.TextColumn("Activities Kor"),
                    "Masters": st.column_config.TextColumn("Masters"),
                    "Systems": st.column_config.TextColumn("Systems"),
                    "Activity_Detail": st.column_config.TextColumn("Activity Detail"),
                }
            )
        with col2:
            seq_list = sorted(filtered_df["Activity_Seq"].astype(str).unique().tolist())
            st.write("Value Chain Seq 선택")
            seq_choice = st.selectbox("선택 or New", ["(New)"] + seq_list)

            if seq_choice == "(New)":
                activity_seq = st.text_input("New Seq", key="new_seq_input")
                if not activity_seq.strip():
                    st.info("신규 Seq를 입력하세요.(Unique 값)")
                    st.stop()
            else:
                activity_seq = seq_choice

        row_data = (
            filtered_df[filtered_df["Activity_Seq"].astype(str) == activity_seq].iloc[0]
            if activity_seq in seq_list else pd.Series({}, dtype=object)
        )

        with st.form("edit_activity_form"):
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                activity_type = st.selectbox("Activities Type", ["Primary", "Support"],
                    index=0 if row_data.get("Activities_Type", "Primary") == "Primary" else 1)
                activities = st.text_input("Activities", row_data.get("Activities", ""))
            with col2:
                activities_kor = st.text_input("Activities Kor", row_data.get("Activities_Kor", ""))
                masters = st.text_input("Masters (콤마로 구분)", row_data.get("Masters", ""))
                systems = st.text_input("Systems (콤마로 구분)", row_data.get("Systems", ""))
            with col3:
                activity_detail = st.text_area("Activity Detail", row_data.get("Activity_Detail", ""), height=150)
                submitted = st.form_submit_button("✅ 추가/수정 적용")

            if submitted:
                new_row = {
                    "Industry": industry,
                    "Activities_Type": activity_type.strip(),
                    "Activity_Seq": str(activity_seq).strip(),
                    "Activities": activities.strip(),
                    "Activities_Kor": activities_kor.strip(),
                    "Masters": masters.strip(),
                    "Systems": systems.strip(),
                    "Activity_Detail": activity_detail.strip(),
                }
                key_check = (
                    (df["Industry"] == industry) &
                    (df["Activity_Seq"].astype(str).str.strip() == new_row["Activity_Seq"]) &
                    (df["Activities"].astype(str).str.strip() == new_row["Activities"]) &
                    (df["Activities_Kor"].astype(str).str.strip() == new_row["Activities_Kor"])
                )
                if key_check.any():
                    st.warning("⚠️ 동일 Activity가 이미 존재합니다. 덮어쓰기합니다.")
                    df = df[~key_check]

                st.session_state["pending_update"] = pd.DataFrame([new_row])
                st.success("✅ 변경/추가 데이터가 준비되었습니다.")

        if seq_choice != "(New)" and activity_seq in seq_list:
            st.warning(f"🗑️ 선택한 Activity_Seq : `{activity_seq}` 를 삭제하시겠습니까?")
            confirm = st.checkbox("⚠️ 삭제를 확인합니다", key="delete_confirm")

            if confirm and st.button("❌ 삭제 실행", type="primary"):
                try:
                    mask_delete = (
                        (df["Industry"] == industry) &
                        (df["Activity_Seq"].astype(str).str.strip() == str(activity_seq).strip())
                    )
                    if not mask_delete.any():
                        st.warning("⚠️ 삭제 대상 레코드가 존재하지 않습니다.")
                    else:
                        new_df = df[~mask_delete].copy()
                        self.save_data(new_df, mode="delete")
                        st.success(f"✅ Activity_Seq '{activity_seq}' 삭제 완료")
                        st.rerun()
                except Exception as e:
                    st.error(f"삭제 실패: {e}")

        if "pending_update" in st.session_state:
            st.divider()
            st.markdown("### 💾 저장 미리보기")
            st.dataframe(st.session_state["pending_update"], use_container_width=True, hide_index=True)
            
            if st.button("📁 파일 저장", type="primary"):
                df = self.save_data(st.session_state["pending_update"], existing_df=df, mode="merge")
                st.session_state.pop("pending_update")
                st.rerun()

# -------------------------------------------------------------------
# MASTER COLUMN DEFINITION
# -------------------------------------------------------------------
class MasterColumnDefinition:
    """Value Chain Master Column Definition 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.loader = FileLoader(yaml_config)
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
        st.markdown("#### 📘 Master & Column Definition")
        st.markdown("Value Chain Master에 관리해야 할 필수 컬럼 리스트를 정의하고 관리합니다.")
        st.divider()
        
        valuechain_system_df = self.loader.load_valuechain_system()
        if valuechain_system_df.empty:
            st.info("📝 ValueChain System 정의 데이터가 없습니다. ValueChain Definition 탭에서 데이터를 생성 및 입력하세요.")
            return
        
        column_list_path = Path(self.column_list_path)
        file_existed = column_list_path.exists()
        
        df = self.loader.load_valuechain_master_column_list()

        if df.empty:
            st.info("📝 데이터가 없습니다. 로직을 점검하세요.")
            return
        
        if not file_existed and "System" in df.columns and not df.empty:
            if self.save_data(df):
                st.success("✅ 새로 생성된 파일이 자동 저장되었습니다.")
                st.rerun()
        
        vc_master_df = self.loader.load_valuechain_master()
        if vc_master_df.empty:
            st.info("📝 Value Chain Master 데이터가 없습니다. Value Chain Definition 탭에서 데이터를 생성 및 입력하세요.")
            return
       
        industries = ["전체"] + sorted(df["Industry"].unique().tolist())
        
        col0, col1, col2 = st.columns([1, 2, 4])
        with col0:
            st.markdown("##### 📊 현재 데이터")
        with col1:
            selected_industry = st.selectbox("Industry 선택", options=industries, key="vc_col_industry")
        
        selected_system = []
        if selected_industry != "전체":
            system_df_filtered = valuechain_system_df[valuechain_system_df["Industry"] == selected_industry].copy()
            if not system_df_filtered.empty and "System" in system_df_filtered.columns:
                selected_system = system_df_filtered["System"].unique().tolist()
                selected_system = [s for s in selected_system if s and str(s).strip()]

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

        if selected_industry != "전체":
            try:
                if "edited_filtered_df_mc" not in st.session_state:
                    st.session_state.edited_filtered_df_mc = filtered_df.copy()
                
                if "last_selected_industry_mc" not in st.session_state or st.session_state.last_selected_industry_mc != selected_industry:
                    st.session_state.edited_filtered_df_mc = filtered_df.copy()
                    st.session_state.last_selected_industry_mc = selected_industry
                
                edited_df = st.session_state.edited_filtered_df_mc
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

            with col02:
                st.write("편집할 행 번호")
            
                available_indices = list(edited_df.index.tolist())

                row_index = st.number_input(
                    f"행 번호 : ({min(available_indices)} ~ {max(available_indices)})", 
                    min_value=min(available_indices),
                    max_value=max(available_indices),
                    value=min(available_indices),
                    key="row_index_input_mc"
                )
                actual_idx = int(row_index)

            if actual_idx in edited_df.index:
                with st.form(f"edit_form_mc_{actual_idx}", clear_on_submit=False):
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
                        
                        current_system = str(row_data.get("System", "")).strip()
                        if current_system and current_system in selected_system:
                            system_index = selected_system.index(current_system)
                        else:
                            system_index = 0
                        
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
                        st.session_state.edited_filtered_df_mc = edited_df
                        st.success(f"✅ 행 Index {actual_idx} 업데이트 완료")
                        st.rerun()

            st.divider()
            st.write(f"💾 {self.column_list_path} 에 저장됩니다.")
            st.write(f"💾 {self.column_path} 에 저장됩니다.")
            col21, col22 = st.columns([2, 8])
            with col21:
                if st.button("💾 파일 저장", use_container_width=True, key="save_mc"):
                    all_df = self.loader.load_valuechain_master_column_list()
                    if selected_industry == "전체":
                        save_df = edited_df.copy()
                    else:
                        other_df = all_df[all_df["Industry"] != selected_industry].copy()
                        save_df = pd.concat([other_df, edited_df], ignore_index=True)
                    
                    if self.save_data(save_df):
                        st.success(f"✅ 파일 저장 완료: {len(edited_df)}개 행 업데이트")
                        st.session_state.edited_filtered_df_mc = None
                        st.session_state.last_selected_industry_mc = None
                        st.rerun()
        else:
            st.divider()
            st.markdown("##### 편집을 하기 위해서는 Industry를 선택하세요.")

# -------------------------------------------------------------------
# MASTER MAPPING
# -------------------------------------------------------------------
class MasterMapping:
    """Value Chain Master's vs Our's Master Mapping 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.loader = FileLoader(yaml_config)
        self.valuechain_our_master_path = self.loader.files_config.valuechain_our_master
    
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
        st.markdown("#### 📘 Master & Our's Master Mapping")
        st.markdown("Value Chain Master 와 Our's Master 간의 매핑을 정의합니다.")
        st.divider()
        
        df = self.loader.load_valuechain_our_master()
        if df.empty:
            st.info("📝 데이터가 없습니다. 새롭게 생성합니다. (로직 점검하세요)")
            return
  
        codemapping_df = self.loader.load_codemapping()
        if codemapping_df.empty:
            st.info("📝 Code Mapping 데이터가 없습니다. Code Mapping 파일을 생성 후 작업을 수행 하세요.")
            return
        
        codemapping_df = codemapping_df[codemapping_df["MasterType"] == "Master"]
        codemapping_df = normalize_dataframe_for_display(codemapping_df)

        industries = ["전체"] + sorted(df["Industry"].unique().tolist())

        col0, col1, col2 = st.columns([1, 2, 4])
        with col0:
            st.markdown("##### 📊 현재 데이터")

        with col1:
            selected_industry = st.selectbox("Industry 선택", options=industries, key="vc_mapping_industry")
        
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

        if selected_industry != "전체":
            selected_our_master = []
            if not codemapping_df.empty and "FileName" in codemapping_df.columns:
                selected_our_master = codemapping_df["FileName"].dropna().unique().tolist()
            
            selected_our_master = [str(s).strip() for s in selected_our_master if s and str(s).strip()]
            selected_our_master = sorted(selected_our_master)
            selected_our_master = [""] + selected_our_master
            
            try:
                if "edited_filtered_df_mm" not in st.session_state:
                    st.session_state.edited_filtered_df_mm = filtered_df.copy()
                
                if "last_selected_industry_mm" not in st.session_state or st.session_state.last_selected_industry_mm != selected_industry:
                    st.session_state.edited_filtered_df_mm = filtered_df.copy()
                    st.session_state.last_selected_industry_mm = selected_industry
                
                edited_df = st.session_state.edited_filtered_df_mm
            except Exception:
                edited_df = filtered_df.copy()

            col01, col02 = st.columns([8, 1])
            with col01:
                display_df = filtered_df.copy()
                display_df = display_df[["Activities_Type", "Activity_Seq", "Activities", "Activities_Kor",
                    "Master", "Master_Kor", "Our_Master", "Our_Master_Kor"]]
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

            with col02:
                st.write("편집할 행 번호")
            
                available_indices = list(edited_df.index.tolist())

                row_index = st.number_input(
                    f"행 번호 : ({min(available_indices)} ~ {max(available_indices)})", 
                    min_value=min(available_indices),
                    max_value=max(available_indices),
                    value=min(available_indices),
                    key="row_index_input_mm"
                )
                actual_idx = int(row_index)

            if actual_idx in edited_df.index:
                with st.form(f"edit_form_mm_{actual_idx}", clear_on_submit=False):
                    row_data = edited_df.loc[actual_idx]

                    cola1, cola2 = st.columns([3, 4])
                    with cola1:
                        st.info(f"**Value Chain Information**")
                    with cola2:
                        st.info(f"**Value Chain Master vs Our Master의 Mapping을 선택하세요.**")

                    col1, col2, col3 = st.columns([1, 2, 4])
                    with col1:
                        st.text_input("Activities Type", value=str(row_data.get("Activities_Type", "")), disabled=True)
                        st.text_input("Activity Seq", value=str(row_data.get("Activity_Seq", "")), disabled=True)
                        st.text_input("Activities", value=str(row_data.get("Activities", "")), disabled=True)

                    with col2:
                        st.text_input("Activities Kor", value=str(row_data.get("Activities_Kor", "")), disabled=True)
                        st.text_input("Master", value=str(row_data.get("Master", "")), disabled=True)
                        st.text_input("Master Kor", value=str(row_data.get("Master_Kor", "")), disabled=True)

                    with col3:
                        current_our_master = str(row_data.get("Our_Master", "")).strip()
                        if current_our_master and current_our_master in selected_our_master:
                            our_master_index = selected_our_master.index(current_our_master)
                        else:
                            our_master_index = 0
                        
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
                            st.session_state.edited_filtered_df_mm = edited_df
                            st.success(f"✅ 행 Index {actual_idx} 업데이트 완료")
                            st.rerun()

            st.divider()
            st.write(f"💾 {self.valuechain_our_master_path} 에 저장됩니다.")
            col21, col22 = st.columns([2, 8])
            with col21:
                if st.button("💾 파일 저장", use_container_width=True, key="save_mm"):
                    all_df = self.loader.load_valuechain_our_master()
                    if selected_industry == "전체":
                        save_df = edited_df.copy()
                    else:
                        other_df = all_df[all_df["Industry"] != selected_industry].copy()
                        save_df = pd.concat([other_df, edited_df], ignore_index=True)
                    
                    if self.save_data(save_df):
                        st.success(f"✅ 파일 저장 완료: {len(edited_df)}개 행 업데이트")
                        st.session_state.edited_filtered_df_mm = None
                        st.session_state.last_selected_industry_mm = None
                        st.rerun()
        else:
            st.divider()
            st.markdown("##### 편집을 하기 위해서는 Industry를 선택하세요.")

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
class ValueMasterDefinitionApp:
    """통합 애플리케이션"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.value_chain_def = ValueChainDefinition(yaml_config)
        self.master_column_def = MasterColumnDefinition(yaml_config)
        self.master_mapping = MasterMapping(yaml_config)
    
    def run(self):
        st.title(APP_NAME)
        st.markdown(APP_DESC)
        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("###### Value Chain Definition")
            st.markdown("###### Master Column Definition")
            st.markdown("###### Master & Our's Master Mapping")
        with col2:
            st.markdown("###### Value Chain 의 Primary Process & Support Function 들과 Master을 정의합니다.")
            st.markdown("###### Master의 한글명칭, System 및 필수 컬럼 리스트를 정의하고 관리합니다.")
            st.markdown("###### Master 와 Our's Master 간의 매핑을 정의하고 관리합니다.")
        
        tab1, tab2, tab3 = st.tabs([
            "📘 Value Chain Definition",
            "📘 Master & Column Definition",
            "📘 Master & Our's Master Mapping"
        ])
        
        with tab1:
            self.value_chain_def.display()
        
        with tab2:
            self.master_column_def.display()
        
        with tab3:
            self.master_mapping.display()

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    try:
        cfg = load_yaml_datasense()
        app = ValueMasterDefinitionApp(cfg)
        app.run()
    except Exception as e:
        st.error(f"애플리케이션 오류: {e}")
        import traceback
        st.exception(e)

if __name__ == "__main__":
    main()

