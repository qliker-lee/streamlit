# -*- coding: utf-8 -*-
"""
🔗 DataSense ERD 컬럼 단위 관계 제어
Author: Qliker 2026-01-07
"""
# -------------------------------------------------
# 1. Path / Warning setup (Streamlit import 전)
# -------------------------------------------------
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util.streamlit_warnings import setup_streamlit_warnings
setup_streamlit_warnings()

# -------------------------------------------------
# 2. Standard / Third-party imports
# -------------------------------------------------
import streamlit as st
import pandas as pd
from collections import defaultdict
from graphviz import Digraph
from PIL import Image
from pathlib import Path
from datetime import datetime

# 1. 환경 설정
Image.MAX_IMAGE_PIXELS = None

# 경로 설정
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'DS_Output'
IMAGE_DIR = PROJECT_ROOT / 'images'
EXCLUSIVE_FILE = OUTPUT_DIR / "ERD_exclusive.csv"

# --- 1. 블랙리스트 파일(ERD_exclusive.csv) 관리 로직 ---
def manage_exclusive_config(df_cm):
    """
    ERD_exclusive.csv 관리 및 데이터 일관성(Type/Format) 체크 기능
    """
    # --- [내부 함수] 최신 통계 및 타입 정보 생성 ---
    def generate_current_stats(df):
        col_to_tables = defaultdict(set)
        col_types = {}
        for _, row in df.iterrows():
            c_name = str(row['ColumnName']).strip()
            f_name = str(row['FileName']).strip()
            o_type = str(row.get('OracleType', '')).strip().upper() if pd.notna(row.get('OracleType')) else ""
            if c_name and f_name:
                col_to_tables[c_name].add(f_name)
                if c_name not in col_types or o_type in ['DATE', 'TIMESTAMP', 'DATETIME']:
                    col_types[c_name] = o_type
        
        stats = []
        for col, tables in col_to_tables.items():
            curr_type = col_types.get(col, "")
            stats.append({
                "ColumnName": col,
                "OracleType": curr_type,
                "ConnectionCount": len(tables),
                "exclusive": 1 if curr_type in ['DATE', 'TIMESTAMP', 'DATETIME'] else 0
            })
        return pd.DataFrame(stats)

    # 1. 최신 정보 생성 및 병합 (이전 로직과 동일)
    current_stats_df = generate_current_stats(df_cm)
    if not EXCLUSIVE_FILE.exists():
        final_df = current_stats_df.sort_values(by="ConnectionCount", ascending=False)
    else:
        old_df = pd.read_csv(EXCLUSIVE_FILE)
        final_df = pd.merge(current_stats_df, old_df[['ColumnName', 'exclusive']], on='ColumnName', how='left')
        final_df['exclusive'] = final_df['exclusive'].fillna(0).astype(int)
        final_df = final_df.sort_values(by="ConnectionCount", ascending=False)

    st.subheader("ERD 생성시 제외할 컬럼 설정")
    final_df['exclusive_bool'] = final_df['exclusive'].astype(bool)
    edited_df = st.data_editor(
        final_df,
        column_config={
            "exclusive_bool": st.column_config.CheckboxColumn("제외", width="small"),
            "exclusive": None 
        },
        disabled=["ColumnName", "OracleType", "ConnectionCount"],
        hide_index=True, width='stretch', key="ex_editor_v3"
    )

    if st.button("설정 저장하기", type="primary"):
        save_df = edited_df.copy()
        save_df['exclusive'] = save_df['exclusive_bool'].astype(int)
        save_df.drop(columns=['exclusive_bool']).to_csv(EXCLUSIVE_FILE, index=False, encoding='utf-8-sig')
        st.toast("설정이 저장되었습니다!", icon="✅")
        st.rerun()

    st.write("---")

    # --- [신규 추가] 데이터 일관성 체크 섹션 ---
    st.subheader("🧪 2. 데이터 모델 일관성 분석")
    
    tab1, tab2 = st.tabs(["⚠️ OracleType 불일치", "📝 Format 불일치 (FormatCnt ≤ 3)"])

    with tab1:
        # 동일 ColumnName인데 OracleType이 다른 경우 추출
        # 1. 컬럼별 Unique한 Type 개수 계산
        type_diff = df_cm.groupby('ColumnName')['OracleType'].nunique()
        diff_cols = type_diff[type_diff > 1].index.tolist()
        
        if diff_cols:
            st.warning(f"동일한 컬럼명에 대해 OracleType이 다르게 정의된 항목이 {len(diff_cols)}건 발견되었습니다.")
            diff_df = df_cm[df_cm['ColumnName'].isin(diff_cols)][['FileName', 'ColumnName', 'OracleType']]
            st.dataframe(diff_df.sort_values(['ColumnName', 'FileName']), width='stretch', hide_index=True)
        else:
            st.success("모든 동일 컬럼의 OracleType이 일치합니다.")

    with tab2:
        # FormatCnt가 3이하인 컬럼 중 동일 ColumnName인데 Format이 다른 경우 추출
        if 'FormatCnt' in df_cm.columns and 'Format' in df_cm.columns:
            # 1. FormatCnt <= 3 조건 필터링
            f_df = df_cm[df_cm['FormatCnt'] <= 3].copy()
            # 2. 컬럼별 Unique한 Format 개수 계산
            format_diff = f_df.groupby('ColumnName')['Format'].nunique()
            diff_f_cols = format_diff[format_diff > 1].index.tolist()
            
            if diff_f_cols:
                st.warning(f"FormatCnt 3이하인 컬럼 중 Format이 불일치하는 항목이 {len(diff_f_cols)}건 발견되었습니다.")
                diff_f_df = f_df[f_df['ColumnName'].isin(diff_f_cols)][['FileName', 'ColumnName', 'Format', 'FormatCnt']]
                st.dataframe(diff_f_df.sort_values(['ColumnName', 'FileName']), width='stretch', hide_index=True)
            else:
                st.success("조건에 해당하는 모든 컬럼의 Format이 일치합니다.")
        else:
            st.info("데이터프레임에 'FormatCnt' 또는 'Format' 컬럼이 존재하지 않습니다.")

    blacklist = edited_df[edited_df['exclusive_bool'] == True]['ColumnName'].tolist()
    return blacklist

def run_column_control_erd(df_cm):
    # 1. 마스터 데이터만 필터링 (사용자 기존 로직 유지)
    df_cm = df_cm[df_cm['MasterType'] == 'Master'].copy()

    # 2. 블랙리스트 관리 UI 실행 (OracleType 포함 및 저장 버튼 로직)
    blacklist = manage_exclusive_config(df_cm)
    
    # 3. 결과 표시
    if blacklist:
        st.caption(f"현재 제외된 컬럼 수: {len(blacklist)}개")
    
    st.write("---")

def manage_exclusive_config(df_cm):
    """
    ERD_exclusive.csv 관리 및 데이터 모델 불일치 상세 분석 기능
    """
    # 1. 최신 정보(통계) 생성
    def get_fresh_stats(df):
        col_to_tables = defaultdict(set)
        col_types = {}
        for _, row in df.iterrows():
            c = str(row['ColumnName']).strip()
            f = str(row['FileName']).strip()
            t = str(row.get('OracleType', '')).strip().upper() if pd.notna(row.get('OracleType')) else ""
            if c and f:
                col_to_tables[c].add(f)
                if c not in col_types or t in ['DATE', 'TIMESTAMP', 'DATETIME']:
                    col_types[c] = t
        
        data = []
        for col, tables in col_to_tables.items():
            t_type = col_types.get(col, "")
            data.append({
                "ColumnName": col,
                "OracleType": t_type,
                "ConnectionCount": len(tables),
                "exclusive": 1 if t_type in ['DATE', 'TIMESTAMP', 'DATETIME'] else 0
            })
        return pd.DataFrame(data)

    current_df = get_fresh_stats(df_cm)

    # 2. 파일 로드 및 병합 (exclusive_x/y 방지 로직)
    if EXCLUSIVE_FILE.exists():
        try:
            old_df = pd.read_csv(EXCLUSIVE_FILE)
            if 'exclusive' in old_df.columns:
                # 필요한 컬럼만 추출하여 merge 시 충돌 방지
                old_settings = old_df[['ColumnName', 'exclusive']].drop_duplicates('ColumnName')
                final_df = pd.merge(current_df, old_settings, on='ColumnName', how='left', suffixes=('_init', ''))
                # 기존 설정이 있으면 쓰고, 없으면 초기값(_init) 사용
                final_df['exclusive'] = final_df['exclusive'].fillna(final_df['exclusive_init']).astype(int)
                final_df = final_df.drop(columns=['exclusive_init'])
            else:
                final_df = current_df
        except:
            final_df = current_df
    else:
        final_df = current_df

    final_df = final_df.sort_values(by="ConnectionCount", ascending=False)

    # --- UI: 블랙리스트 설정 ---
    st.subheader("ERD 생성시 제외할 컬럼 설정")
    final_df['exclusive_bool'] = final_df['exclusive'].astype(bool)
    
    edited_df = st.data_editor(
        final_df,
        column_config={"exclusive_bool": st.column_config.CheckboxColumn("제외"), "exclusive": None},
        disabled=["ColumnName", "OracleType", "ConnectionCount"],
        hide_index=True, width='stretch', key="ex_editor_v4"
    )

    if st.button("💾 블랙리스트 설정 저장하기", type="primary"):
        save_df = edited_df.copy()
        save_df['exclusive'] = save_df['exclusive_bool'].astype(int)
        save_df.drop(columns=['exclusive_bool']).to_csv(EXCLUSIVE_FILE, index=False, encoding='utf-8-sig')
        st.toast("설정이 파일에 저장되었습니다!", icon="✅")
        st.rerun()

    return edited_df[edited_df['exclusive_bool'] == True]['ColumnName'].tolist()

def render_consistency_checks2(df_cm):

    st.write("---")

    # --- UI: 데이터 모델 일관성 분석 (Group By 방식) ---
    st.subheader("🧪 2. 데이터 모델 일관성 분석 (Group By)")
    tab1, tab2 = st.tabs(["⚠️ OracleType 불일치", "📝 Format 불일치 (FormatCnt ≤ 3)"])

    with tab1:
        # Group By: ColumnName, OracleType 별 건수 및 파일 리스트
        type_group = df_cm.groupby(['ColumnName', 'OracleType']).agg(
            Count=('FileName', 'count'),
            FileList=('FileName', lambda x: ", ".join(sorted(x.unique())))
        ).reset_index()
        
        # 2개 이상의 타입을 가진 컬럼명 추출
        diff_type_cols = type_group.groupby('ColumnName').filter(lambda x: len(x) > 1)['ColumnName'].unique()
        
        if len(diff_type_cols) > 0:
            st.warning(f"동일 컬럼명 내 OracleType이 다른 사례: {len(diff_type_cols)}건")
            res_type = type_group[type_group['ColumnName'].isin(diff_type_cols)]
            st.dataframe(res_type.sort_values('ColumnName'), width='stretch', hide_index=True)
        else:
            st.success("모든 동일 컬럼의 OracleType이 일치합니다.")

    with tab2:
        if 'Format' in df_cm.columns and 'FormatCnt' in df_cm.columns:
            f_base = df_cm[df_cm['FormatCnt'] <= 3].copy()
            # Group By: ColumnName, Format 별 건수 및 파일 리스트
            format_group = f_base.groupby(['ColumnName', 'Format']).agg(
                Count=('FileName', 'count'),
                FileList=('FileName', lambda x: ", ".join(sorted(x.unique())))
            ).reset_index()
            
            diff_format_cols = format_group.groupby('ColumnName').filter(lambda x: len(x) > 1)['ColumnName'].unique()
            
            if len(diff_format_cols) > 0:
                st.warning(f"FormatCnt 3이하 컬럼 중 Format 불일치: {len(diff_format_cols)}건")
                res_format = format_group[format_group['ColumnName'].isin(diff_format_cols)]
                st.dataframe(res_format.sort_values('ColumnName'), width='stretch', hide_index=True)
            else:
                st.success("조건에 해당하는 모든 컬럼의 Format이 일치합니다.")
        else:
            st.info("Format 정보가 데이터프레임에 없습니다.")

def render_consistency_checks(df_cm):
    st.write("---")
    st.subheader("🧪 2. 데이터 모델 일관성 분석")
    tab1, tab2 = st.tabs(["⚠️ OracleType 불일치", "📝 Format 불일치 (FormatCnt ≤ 3)"])
    
    with tab1:
        type_diff = df_cm.groupby('ColumnName')['OracleType'].nunique()
        diff_cols = type_diff[type_diff > 1].index.tolist()
        if diff_cols:
            st.warning(f"OracleType 불일치: {len(diff_cols)}건")
            st.dataframe(df_cm[df_cm['ColumnName'].isin(diff_cols)][['FileName', 'ColumnName', 'OracleType']].sort_values('ColumnName'), hide_index=True)
        else: st.success("OracleType 일치")

    with tab2:
        if 'FormatCnt' in df_cm.columns:
            f_df = df_cm[df_cm['FormatCnt'] <= 3].copy()
            format_diff = f_df.groupby('ColumnName')['Format'].nunique()
            diff_f_cols = format_diff[format_diff > 1].index.tolist()
            if diff_f_cols:
                st.warning(f"Format 불일치: {len(diff_f_cols)}건")
                st.dataframe(f_df[f_df['ColumnName'].isin(diff_f_cols)][['FileName', 'ColumnName', 'Format', 'FormatCnt']].sort_values('ColumnName'), hide_index=True)
def main():
    st.set_page_config(layout="wide")

    APP_TITLE = "ERD 컬럼 단위 관계 제어"
    APP_DESCRIPTION = "#### 물리적 ERD 생성시 제외할 컬럼을 설정하고, 최적화된 ERD를 생성합니다."
    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)
    
    path = OUTPUT_DIR / "CodeMapping.csv"
    if path.exists():
        df_cm = pd.read_csv(path)
        run_column_control_erd(df_cm)
        render_consistency_checks2(df_cm)
    else:
        st.error(f"'{path}' 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main() 