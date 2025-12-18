# -*- coding: utf-8 -*-
"""
📘 Master ERD 에서 사용하는 함수들  
2025.12.02 Qliker (New Version)
"""

import pandas as pd
import numpy as np
import os
import re
import sys
import streamlit as st
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
import itertools
from itertools import combinations
import graphviz
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import plotly.graph_objects as go
import traceback
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning)


def fill_na_zero_empty_string(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """숫자 컬럼은 NaN을 0으로 채우고, 나머지는 빈 문자열로 채우기"""
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = df[col].replace('', 0) # 빈 문자열을 0으로 변환
            df[col] = df[col].astype(int)
    
    # 나머지 컬럼은 빈 문자열로 채우기
    other_cols = [col for col in df.columns if col not in numeric_cols]
    df[other_cols] = df[other_cols].fillna('')
    return df
    
from typing import List, Tuple, Dict

def parse_relations(level_rel: str, self_file: str = None, self_col: str = None) -> List[Tuple[str, str]]:
    """
    'A.a -> B.b -> C.c' 또는 'A.a->B.b->C.c' 등 다양한 포맷을 처리하여
    자기 자신(A.a)은 제외한 [('B','b'), ('C','c')] 형태로 반환.
    - level_rel: 원본 문자열
    - self_file, self_col: 자기 자신을 명확히 제외하려면 제공 (선택)
    """
    relations = []
    if level_rel is None:
        return relations

    s = str(level_rel).strip()
    if s == '' or s.lower() in ['nan', 'none']:
        return relations

    # '->' 또는 '->' 주변 공백 무시, 혹은 단순 '>' 등 비정상 기호 허용(정규식 약간 유연)
    # 기본적으로 '->' 를 구분자로 사용. 만약 '->'가 없고 '.'만 여러개 있는 경우 한 덩어리로 취급.
    parts = [p.strip() for p in re.split(r'\s*->\s*', s) if p.strip()]

    # 각 part는 보통 'File.Column' 형식. 만약 'csv.File.Column' 같이 접두사가 있으면 마지막 두 토큰으로 처리
    for part in parts:
        # normalize: comma or semicolon separated (보험)
        part = part.strip().rstrip(';,')
        if '.' not in part:
            continue

        file_part, col_part = part.rsplit('.', 1)
        file_part = file_part.strip()
        col_part = col_part.strip()

        if file_part.lower().startswith('csv.'):
            file_part = file_part[len('csv.'):]

        # 필터: nan/none/빈 문자열 제외
        if not file_part or not col_part:
            continue
        if file_part.lower() in ['nan', 'none'] or col_part.lower() in ['nan', 'none']:
            continue

        # 자기 자신 제외(옵션)
        if self_file and self_col:
            if file_part == self_file and col_part == self_col:
                continue

        relations.append((file_part, col_part))

    # 보통 첫 부분이 자기 자신(A.a)로 들어온다면 parts[0]을 제거했을 때 이미 제외되므로 추가 조치는 필요 없음.
    # 하지만 만약 parts[0]이 자기자신으로 들어오지 않고 다른 포맷이라면 상위에서 자기자신 체크 가능.
    return relations


def expand_rel_rows(df: pd.DataFrame, filter_predicate=None, include_level_cols: bool = True) -> pd.DataFrame:
    """
    반복되는 확장 로직을 하나로 합친 함수.
    - df: 원본 DataFrame (FileName, ColumnName, MasterType, PK, FK, Attribute, Level_Depth, Level_Relationship 등 포함)
    - filter_predicate: 각 row에 대해 확장 대상인지 판단하는 함수(row) -> bool, None이면 모든 row 대상
    - include_level_cols: 결과에 Level_Depth, Level_Relationship 컬럼을 포함할지 여부
    """
    rows = []
    columns_needed = ['FileName', 'ColumnName', 'MasterType', 'PK', 'FK', 'Attribute', 'Level_Depth', 'Level_Relationship']

    for _, row in df.iterrows():
        if filter_predicate and not filter_predicate(row):
            continue

        # parse_relations에 자기 자신 정보 전달하여 자기 참조 제거
        relations = parse_relations(row.get('Level_Relationship', ''), self_file=row.get('FileName'), self_col=row.get('ColumnName'))

        if not relations:
            r = {
                'FileName': row.get('FileName', ''),
                'ColumnName': row.get('ColumnName', ''),
                'MasterType': row.get('MasterType', ''),
                'PK': row.get('PK', ''),
                'FK': row.get('FK', ''),
                'Attribute': row.get('Attribute', ''),
                'Level': 0,
                'To FileName': '',
                'To ColumnName': ''
            }
            if include_level_cols:
                r['Level_Depth'] = row.get('Level_Depth', '')
                r['Level_Relationship'] = row.get('Level_Relationship', '')
            rows.append(r)
            continue

        for idx, (to_file, to_col) in enumerate(relations, start=1):
            r = {
                'FileName': row.get('FileName', ''),
                'ColumnName': row.get('ColumnName', ''),
                'MasterType': row.get('MasterType', ''),
                'PK': row.get('PK', ''),
                'FK': row.get('FK', ''),
                'Attribute': row.get('Attribute', ''),
                'Level': idx,
                'To FileName': to_file,
                'To ColumnName': to_col
            }
            if include_level_cols:
                r['Level_Depth'] = row.get('Level_Depth', '')
                r['Level_Relationship'] = row.get('Level_Relationship', '')
            rows.append(r)

    # 결과가 비어 있어도 컬럼구조 유지
    if rows:
        out = pd.DataFrame(rows)
    else:
        base_cols = ['FileName','ColumnName','MasterType','PK','FK','Attribute','Level','To FileName','To ColumnName']
        if include_level_cols:
            base_cols += ['Level_Depth','Level_Relationship']
        out = pd.DataFrame(columns=base_cols)
    return out

    
def display_erd_kpis(df: pd.DataFrame):  # 첫번째 Main KPI 출력 
    """초기 Files Information 표시 """
    st.markdown("### Files & Columns KPI ")
    
    table_info = [] 
    
    table_info = df.groupby('FileName').agg({
        'MasterType': 'first',
        'ColumnName': lambda x: ', '.join(x)
    }).reset_index().to_dict(orient='records')
            
    table_list = pd.DataFrame(table_info) # DataFrame 생성
    
    summary = {
        "Files Cnt #": f"{len(table_list):,}",
        "Column Cnt #": f"{table_list['ColumnName'].apply(len).sum():,}",
    }
    
    # 각 메트릭에 대한 색상 정의
    metric_colors = {
        "Files Cnt #": "#1f77b4",
        "Column Cnt #": "#2ca02c",
    }
    
    # 메트릭 표시
    cols = st.columns(len(summary))
    for col, (key, value) in zip(cols, summary.items()):
        color = metric_colors.get(key, "#0072B2")
        col.markdown(f"""
            <div style="text-align: center; padding: 1.2rem; background-color: #FFFFFF; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="color: {color}; font-size: 3em; font-weight: bold;">{value}</div>
                <div style="color: #404040; font-size: 1.5em; margin-top: 0.5rem;">{key}</div>
            </div>
        """, unsafe_allow_html=True)

def display_erd_table_list(df: pd.DataFrame): 
    """2nd Step: Files & Column List Information """

    st.divider()
    st.markdown("##### Files & Columns Information")

    summary_df = summary_erd_info_tables(df)

    summary_df['선택'] = False  # 체크박스를 위한 컬럼 추가
    
    # '선택' 컬럼을 첫 번째 위치로 이동
    cols = summary_df.columns.tolist()
    cols.remove('선택')
    cols.insert(0, '선택')
    summary_df = summary_df[cols]
    
    # 세션 상태 초기화
    if 'selected_tables' not in st.session_state:
        st.session_state.selected_tables = []
    
    # 체크박스와 함께 DataFrame 표시
    edited_table_list = st.data_editor(
        summary_df,
        column_config={
            '선택': st.column_config.CheckboxColumn('선택', width=50),
            'FileName': st.column_config.TextColumn('FileName', width=200),
            'Type': st.column_config.TextColumn('Type', width=100),
            'Cols': st.column_config.NumberColumn('Cols', width=50),
            'Column List': st.column_config.TextColumn('Column List', width=500, disabled=True),
        },
        hide_index=True,
        height=500,
        width=1000,
        key="table_list_editor"
    )
    
    # 선택된 테이블 추출
    selected_tables = edited_table_list[edited_table_list['선택'] == True]['FileName'].tolist()
    st.session_state.selected_tables = selected_tables

# -------------------------------------------------------------------
# Summary Table & Column Information
# -------------------------------------------------------------------
def summary_erd_info_tables(df:pd.DataFrame) -> pd.DataFrame:
    table_df = df.groupby('FileName').agg({
        'MasterType': 'first',
        'ColumnName': ['count', lambda x: ', '.join(x.astype(str))]
    }).reset_index()
    
    # 컬럼명 정리 (MultiIndex를 단일 컬럼명으로 변경)
    table_df.columns = ['FileName', 'Type', 'Cols', 'Column List']
    summary_df = pd.DataFrame(table_df)

    # PK 정보 집계
    pk_df = pd.DataFrame()
    if 'PK' in df.columns:
        pk_filtered = df[(df['PK'] == 1) | (df['PK'] == '1')]
        if not pk_filtered.empty:
            pk_df = pk_filtered.groupby('FileName').agg({
                'ColumnName': ['count', lambda x: ', '.join(x.astype(str))]
            }).reset_index()

            pk_df.columns = ['FileName', 'PK Cols', 'PK Column List']
        else: # 빈 DataFrame이지만 컬럼은 생성
            pk_df = pd.DataFrame(columns=['FileName', 'PK Cols', 'PK Column List'])
        summary_df = pd.merge(summary_df, pk_df, on='FileName', how='left')

    # FK 정보 집계
    fk_df = pd.DataFrame()
    if 'FK' in df.columns:
        fk_filtered = df[(df['FK'] == 'FK') | (df['FK'] == '1')]
        if not fk_filtered.empty:
            fk_df = fk_filtered.groupby('FileName').agg({
                'ColumnName': ['count', lambda x: ', '.join(x.astype(str))]
            }).reset_index()

            fk_df.columns = ['FileName', 'FK Cols', 'FK Column List']
        else: # 빈 DataFrame이지만 컬럼은 생성
            fk_df = pd.DataFrame(columns=['FileName', 'FK Cols', 'FK Column List'])
        summary_df = pd.merge(summary_df, fk_df, on='FileName', how='left')

    if not summary_df.empty:
        numeric_cols = ['FK Cols', 'PK Cols', 'Cols']
        summary_df = fill_na_zero_empty_string(summary_df, numeric_cols) # 공통함수

    return summary_df

def get_erd_ref_info(codemapping_df:pd.DataFrame, selected_df:pd.DataFrame, concat_df:pd.DataFrame):
    """참조 테이블 정보 표시"""

    cols = ['FileName','ColumnName','MasterType','PK','FK','Attribute','Level_Depth','Level_Relationship']
    ref_candidates_df = selected_df[cols].copy()
    predicate_ref = lambda r: (r.get('Level_Depth') is not None) and (r.get('Level_Depth') != '') and (float(r.get('Level_Depth') or 0) > 0)
    ref_df = expand_rel_rows(ref_candidates_df, filter_predicate=predicate_ref, include_level_cols=True)

    master_type_by_table = codemapping_df.groupby('FileName')['MasterType'].first().to_dict()
    ref_df['To Type'] = ref_df['To FileName'].map(master_type_by_table)

    ref_df = ref_df.drop(columns=['Level_Depth', 'Level_Relationship'])
    ref_df = ref_df.fillna('')

    concat_cols = ['FileName','ColumnName','MasterType', 'CodeFile', 'CodeColumn', 'CodeType', 'Matched', 'Matched(%)']
    concat_df = concat_df[concat_cols]

    concat_df = concat_df.rename(columns={
        'CodeFile': 'To FileName', 'CodeColumn': 'To ColumnName', 'CodeType': 'To Type'})
    ref_df = ref_df.merge(concat_df, on=['FileName','ColumnName','MasterType', 'To FileName', 'To ColumnName', 'To Type'], how='left')
    return ref_df

def selected_tables_info(codemapping_df:pd.DataFrame, selected_tables: list, concat_df:pd.DataFrame):
    """선택한 테이블들의 정보를 테이블 형태로 출력"""
    
    selected_df = codemapping_df[codemapping_df['FileName'].isin(selected_tables)].copy()
    selected_df = selected_df.fillna('')

    ref_df = get_erd_ref_info(codemapping_df, selected_df, concat_df)

    # 탭 생성
    SUB_TITLE1 = "테이블별 컬럼리스트, PK, FK 정보"
    SUB_TITLE2 = "컬럼간의 관계 정보"
    SUB_TITLE3 = "Foreign Key별 관계 정보 상세"
    SUB_TITLE4 = "Reference 컬럼별 관계 정보 상세"
    SUB_TITLE5 = "전체 컬럼 관계 정보 상세"
    tab1, tab2, tab3, tab4, tab5 = st.tabs([SUB_TITLE1, SUB_TITLE2, SUB_TITLE3, SUB_TITLE4, SUB_TITLE5])       
    # 1. 테이블 및 Primary Key 정보 (통합)
    with tab1:
        st.markdown(f"### {SUB_TITLE1}")

        selected_table_info_df = summary_erd_info_tables(selected_df)

        st.dataframe(selected_table_info_df, hide_index=True, width=1400, height=300)
    
    # 2. Level_relationship 정보
    with tab2:
        st.markdown(f"### {SUB_TITLE2}")
        df = selected_df[['FileName', 'ColumnName', 'MasterType', 'PK', 'FK', 'Attribute', 'Level_Depth', 'Level_Relationship']]
        df = df.fillna('')
        st.dataframe(df, hide_index=True, width=1400, height=500)
    # 3. FK 정보
    with tab3:
        st.markdown(f"### {SUB_TITLE3}")

        fk_df = ref_df[(ref_df['PK'] == 1) | (ref_df['FK'] == 'FK')]
        st.dataframe(fk_df, hide_index=True, width=1000, height=500)

    # 4. 참조 테이블 정보
    with tab4:
        st.markdown(f"### {SUB_TITLE4}")
        st.dataframe(ref_df, hide_index=True, width=1000, height=500)

    # 5. 참조 테이블 정보
    with tab5:
        st.markdown(f"### {SUB_TITLE5}")
        all_df = selected_df[['FileName', 'ColumnName', 'MasterType', 'PK', 'FK', 'Attribute']]
        all_df = all_df.merge(ref_df, on=['FileName', 'ColumnName', 'MasterType', 'PK', 'FK', 'Attribute',], how='left')
        all_df = all_df.fillna('')
        st.dataframe(all_df, hide_index=True, width=1000, height=500)

def get_max_depth(codemapping_df:pd.DataFrame, key:str): # 논리관계 요약/상세 ERD 생성시 Depth 입력
    """Level 관계의 최대 깊이 추출"""
    level_cols = [col for col in codemapping_df.columns if col.startswith('Level') and '_File' in col]
    if level_cols:
        max_available_depth = max([int(col.replace('Level', '').replace('_File', '')) for col in level_cols]) + 1
        
        st.markdown("Depth는 Level 관계의 깊이를 의미합니다. 예: Depth=1이면 Level0->Level1 관계만 표시, Depth=2이면 Level0->Level1, Level1->Level2까지 표시")
        col1, col2 = st.columns([1, 2])
        with col1:
            max_depth_input = st.number_input(
                "Depth (모든 Level을 표시하려면 0 또는 비워두세요)",
                min_value=0,
                max_value=max_available_depth,
                value=0,
                step=1,
                key=key
            )
        max_depth = None if max_depth_input == 0 else max_depth_input

        with col2:
            st.markdown(f"**Depth 설정** (최대 사용 가능: {max_available_depth})")
    else:
        max_depth = None

    return max_depth