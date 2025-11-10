# -*- coding: utf-8 -*-
"""
📘 Value Chain Definition (수정 + 추가 + 삭제 + 정렬 저장)
2025.11.04 Qliker (Stable Version)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
from pathlib import Path

APP_NAME = "Value Chain & Master Definition"
APP_DESC = "##### Value Chain & Master를 입력, 수정, 삭제하고 관리합니다."

# -------------------------------------------------------------------
# 경로 설정
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config
set_page_config(APP_NAME)

VC_PATH = PROJECT_ROOT / "DataSense" / "DS_Meta" / "DataSense_ValueChain.csv"
VC_SYSTEM_PATH = PROJECT_ROOT / "DataSense" / "DS_Meta" / "DataSense_ValueChain_System.csv"
VC_MASTER_PATH = PROJECT_ROOT / "DataSense" / "DS_Meta" / "DataSense_ValueChain_Master.csv"
# -----------------------------------------------------------

def load_data():
    cols = [
        "Industry", "Activities_Type", "Activity_Seq", "Activities",
        "Activities_Kor", "Masters", "Systems", "Activity_Detail"
    ]
    if VC_PATH.exists():
        for enc in ("utf-8-sig", "utf-8", "cp949"):
            try:
                df = pd.read_csv(VC_PATH, encoding=enc).fillna("")
                for col in cols:
                    if col not in df.columns:
                        df[col] = ""
                return df[cols]
            except Exception:
                continue
    return pd.DataFrame(columns=cols)

def split_master_list_to_rows(df):
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
    if "Systems" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["Systems"] = out["Systems"].fillna("").astype(str)
    out = out.assign(Systems=out["Systems"].str.split(",")).explode("Systems")
    out["Systems"] = out["Systems"].str.strip()
    out = out[out["Systems"] != ""]
    out = out.rename(columns={"Systems": "System"})
    return out[["Industry", "Activities_Type", "Activity_Seq", "Activities", "Activities_Kor", "System"]]

import datetime
import shutil

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

def save_data(new_df: pd.DataFrame, existing_df: pd.DataFrame = None, mode: str = "merge"):
    """
    Value Chain 데이터를 저장 (백업 + 정렬 + Master 자동 갱신)
    new_df: 새로 추가/수정/삭제된 데이터
    existing_df: 기존 전체 데이터 (선택적)
    mode: 'merge' (추가/수정), 'overwrite' (전체 덮어쓰기), 'delete' (삭제 반영)
    """
    try:
        backup_file(VC_PATH)

        # ✅ 기존 데이터 로드 (필요 시)
        if existing_df is None:
            existing_df = pd.DataFrame()
            if VC_PATH.exists():
                for enc in ("utf-8-sig", "utf-8", "cp949"):
                    try:
                        existing_df = pd.read_csv(VC_PATH, encoding=enc)
                        break
                    except Exception:
                        continue

        # ✅ 데이터 전처리
        new_df = new_df.fillna("")
        new_df["Activity_Seq"] = new_df["Activity_Seq"].astype(str).str.strip()

        if mode == "overwrite":
            final_df = new_df.copy()

        elif mode == "merge":
            if existing_df.empty:
                final_df = new_df.copy()
            else:
                existing_df["Activity_Seq"] = existing_df["Activity_Seq"].astype(str).str.strip()
                
                # new_df에서 Industry 추출 (Industry별로 처리)
                new_industries = new_df["Industry"].unique() if "Industry" in new_df.columns else []
                
                # 기존 데이터에서 new_df의 Industry와 다른 Industry 데이터는 그대로 유지
                other_industries_df = existing_df[~existing_df["Industry"].isin(new_industries)].copy()
                
                # 같은 Industry에 대한 기존 데이터
                same_industry_df = existing_df[existing_df["Industry"].isin(new_industries)].copy()
                
                if same_industry_df.empty:
                    # 같은 Industry 데이터가 없으면 그냥 추가
                    final_df = pd.concat([other_industries_df, new_df], ignore_index=True)
                else:
                    # 같은 Industry 데이터와 merge
                    merged = same_industry_df.merge(
                        new_df,
                        on=["Industry", "Activity_Seq"],
                        how="outer",
                        suffixes=("_old", "_new"),
                        indicator=True
                    )
                    
                    # merge 결과 처리: _new 값이 있으면 _new 사용, 없으면 _old 사용
                    for col in new_df.columns:
                        if col == "Industry" or col == "Activity_Seq":
                            # Industry와 Activity_Seq는 이미 merge key이므로 그대로 사용
                            continue
                        old_col = f"{col}_old"
                        new_col = f"{col}_new"
                        
                        if old_col in merged.columns and new_col in merged.columns:
                            # _new 값이 있으면 _new 사용, 없으면 _old 사용
                            merged[col] = merged[new_col].combine_first(merged[old_col])
                            merged.drop(columns=[old_col, new_col], inplace=True, errors="ignore")
                        elif new_col in merged.columns:
                            merged[col] = merged[new_col]
                            merged.drop(columns=[new_col], inplace=True, errors="ignore")
                    
                    # _merge 컬럼 제거
                    merged = merged.drop(columns=["_merge"], errors="ignore")
                    
                    # 다른 Industry 데이터와 합치기
                    final_df = pd.concat([other_industries_df, merged], ignore_index=True)

        elif mode == "delete":
            final_df = new_df.copy()
        else:
            raise ValueError("mode must be one of: merge | overwrite | delete")

        # ✅ 정렬
        final_df["Activity_Seq_num"] = pd.to_numeric(final_df["Activity_Seq"], errors="coerce")
        final_df = final_df.sort_values(
            by=["Industry", "Activities_Type", "Activity_Seq_num"],
            ascending=[True, True, True]
        ).drop(columns=["Activity_Seq_num"], errors="ignore")

        # ✅ 저장
        final_df.to_csv(VC_PATH, index=False, encoding="utf-8-sig")

        # ✅ Master 파일 갱신
        split_df = split_master_list_to_rows(final_df)
        st.write(split_df)
        if not split_df.empty:
            split_df.to_csv(VC_MASTER_PATH, index=False, encoding="utf-8-sig")

        # ✅ Syste 파일 갱신
        split_df = split_system_list_to_rows(final_df)
        if not split_df.empty:
            split_df.to_csv(VC_SYSTEM_PATH, index=False, encoding="utf-8-sig")

        st.success(f"✅ 저장 완료 ({mode.upper()}) — {len(final_df)}개 행")
        return final_df

    except Exception as e:
        st.error(f"❌ 저장 실패: {e}")
        return existing_df


# -----------------------------------------------------------
def main():
    st.title(APP_NAME)
    st.markdown(APP_DESC)
    st.divider()

    df = load_data()

    # Industry 선택
    industries = sorted(df["Industry"].unique().tolist())
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
        # 데이터 표시
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
        # Activity 선택
        seq_list = sorted(filtered_df["Activity_Seq"].astype(str).unique().tolist())
        st.write("Value Chain Seq 선택")
        seq_choice = st.selectbox("선택 or New", ["(New)"] + seq_list)

        if seq_choice == "(New)":
            activity_seq = st.text_input("New Seq", key="new_seq_input")
            if not activity_seq.strip():
                st.info("신규 Seq를 입력하세요.(Unique 값)")
                st.stop()   # ✅ 안전하게 UI 렌더링은 유지하면서 실행 중단
        else:
            activity_seq = seq_choice

    # 기존 데이터 가져오기
    row_data = (
        filtered_df[filtered_df["Activity_Seq"].astype(str) == activity_seq].iloc[0]
        if activity_seq in seq_list else pd.Series({})
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

    # 삭제 기능 (폼 밖)
    if seq_choice != "(New)" and activity_seq in seq_list:
        st.warning(f"🗑️ 선택한 Activity_Seq : `{activity_seq}` 를 삭제하시겠습니까?")
        confirm = st.checkbox("⚠️ 삭제를 확인합니다", key="delete_confirm")

        # ✅ 삭제 처리 (자동 백업 + 정렬 + 즉시 저장)
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
                    save_data(new_df, mode="delete")
                    st.success(f"✅ Activity_Seq '{activity_seq}' 삭제 완료")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"삭제 실패: {e}")

    # 저장
    if "pending_update" in st.session_state:
        st.divider()
        st.markdown("### 💾 저장 미리보기")
        st.dataframe(st.session_state["pending_update"], use_container_width=True, hide_index=True)
        
        if st.button("📁 파일 저장", type="primary"):
            df = save_data(st.session_state["pending_update"], existing_df=df, mode="merge")
            st.session_state.pop("pending_update")
            st.rerun()

if __name__ == "__main__":
    main()
