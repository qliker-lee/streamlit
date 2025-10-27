# -*- coding: utf-8 -*-
"""
📘 Value Chain Definition
2025.10.26 Qliker (Stable Version)
서브메뉴 + 독립실행 모두 호환 버전
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

# -------------------------------------------------------------------
# 기본 앱 정보
# -------------------------------------------------------------------
APP_NAME = "Value Chain Definition"
APP_DESC = "##### 📘 Industry 별 Value Chain 의 Activities(Process/Function) & Master List를 정의합니다."
st.set_page_config(page_title=APP_NAME, layout="wide")

# -------------------------------------------------------------------
# 안전한 세션 접근 유틸
# -------------------------------------------------------------------
def safe_get_state(key, default=None):
    """Streamlit 세션 상태 안전 접근자"""
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default

def safe_set_state(key, value):
    """Streamlit 세션 상태 안전 설정자"""
    try:
        st.session_state[key] = value
    except Exception:
        pass

# -------------------------------------------------------------------
# 경로 설정
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

# -------------------------------------------------------------------
# YAML CONFIG 로더
# -------------------------------------------------------------------
def _fallback_load_yaml_datasense() -> Dict[str, Any]:
    guessed_root = str(PROJECT_ROOT)
    cfg = {
        "ROOT_PATH": guessed_root,
        "files": {
            "valuechain": "DataSense/ValueChain/valuechain.csv",
            "valuechain_master": "DataSense/ValueChain/valuechain_master.csv",
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
    valuechain: str
    valuechain_master: str

class FileLoader:
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        root = yaml_config.get("ROOT_PATH", str(PROJECT_ROOT))
        files = yaml_config.get("files", {})
        self.files_config = FileConfig(
            valuechain=os.path.join(root, files.get("valuechain", "DataSense/ValueChain/valuechain.csv")),
            valuechain_master=os.path.join(root, files.get("valuechain_master", "DataSense/ValueChain/valuechain_master.csv")),
        )

    def load_valuechain(self) -> pd.DataFrame:
        path = self.files_config.valuechain
        if not os.path.exists(path):
            return pd.DataFrame(columns=[
                "Industry", "Activities_Type", "Activity_Seq", "Activities",
                "Activities_Kor", "Masters", "Systems", "KPIs", "Activity_Detail"
            ])
        for enc in ("utf-8-sig", "utf-8", "cp949"):
            try:
                df = pd.read_csv(path, encoding=enc)
                return df.fillna("")
            except Exception:
                continue
        return pd.DataFrame()

# -------------------------------------------------------------------
# 유틸: Master 분리 함수
# -------------------------------------------------------------------
def split_masters(df: pd.DataFrame) -> pd.DataFrame:
    if "Masters" not in df.columns:
        return pd.DataFrame()
    temp = df.copy()
    temp["Masters"] = temp["Masters"].fillna("").astype(str)
    temp = temp.assign(Master=temp["Masters"].str.split(","))
    temp = temp.explode("Master")
    temp["Master"] = temp["Master"].astype(str).str.strip()
    temp = temp[temp["Master"] != ""]
    return temp[["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", "Master"]]

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
class ValueChainApp:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.loader = FileLoader(cfg)
        self.valuechain_path = self.loader.files_config.valuechain
        self.master_path = self.loader.files_config.valuechain_master

    def display(self):
        st.title(APP_NAME)
        st.markdown(APP_DESC)

        # 데이터 로드 or 세션에서 복원
        df = safe_get_state("valuechain_df")
        if df is None or df.empty:
            df = self.loader.load_valuechain()
            safe_set_state("valuechain_df", df)

        # self.display_help()

        # Industry 선택
        industry_list = ["전체"] + sorted(df["Industry"].dropna().unique().tolist()) if not df.empty else ["전체"]
        selected = st.selectbox("🏭 Industry 선택", industry_list, key="vc_industry_selector")

        # 필터링
        filtered = df if selected == "전체" else df[df["Industry"] == selected].copy()
        safe_set_state("filtered_df", filtered)

        # 편집기 표시
        st.subheader("🧩 Value Chain Editor")
        edited_df = st.data_editor(
            filtered,
            use_container_width=True,
            num_rows="dynamic",
            key=f"vc_editor_{selected}",
            hide_index=True,
            disabled=False,
            column_config={
                "Industry": st.column_config.TextColumn("Industry", help="산업명"),
                "Activities_Type": st.column_config.SelectboxColumn("Type", help="Primary/Support 선택", options=["Primary", "Support"]),
                "Activity_Seq": st.column_config.NumberColumn("Seq", help="전체 일련번호 지정", min_value=1, step=1),
                "Activities": st.column_config.TextColumn("Activities", help="활동명 (영문)"),
                "Activities_Kor": st.column_config.TextColumn("Activities Kor", help="활동명 (한글)"),
                "Masters": st.column_config.TextColumn("Masters", help="Master List (컴마로 구분)"),
                "Systems": st.column_config.TextColumn("Systems", help="시스템명 (컴마로 구분)"),
                "KPIs": st.column_config.TextColumn("KPIs", help="측정 지표 (컴마로 구분)"),
                "Activity_Detail": st.column_config.TextColumn("Activity Detail", help="해당 활동의 상세한 설명"),
            },
        )

        # 저장 버튼
        if st.button("💾 Save (저장)", use_container_width=True):
            try:
                if selected == "전체":
                    merged = edited_df.copy()
                else:
                    rest = df[df["Industry"] != selected].copy()
                    merged = pd.concat([rest, edited_df], ignore_index=True)

                merged.to_csv(self.valuechain_path, index=False, encoding="utf-8-sig")
                # st.success(f"✅ ValueChain 저장 완료: {self.valuechain_path}")

                # Master 분리
                master_df = split_masters(merged)
                master_df.to_csv(self.master_path, index=False, encoding="utf-8-sig")
                
                # st.success(f"✅ Master 분리 완료: {self.master_path}")
                # st.dataframe(master_df, use_container_width=True, hide_index=True, height=500)
                # safe_set_state("valuechain_df", merged)

            except Exception as e:
                st.error(f"❌ 저장 실패: {e}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    try:
        cfg = load_yaml_datasense()
        app = ValueChainApp(cfg)
        app.display()
    except Exception as e:
        st.error(f"애플리케이션 오류: {e}")

if __name__ == "__main__":
    main()
