# -*- coding: utf-8 -*-
"""
📘 🔗 데이터 관계 (ERD) 시각화 (CodeMapping_relationship.csv 기반)
2025.12.17 Qliker
초기 import 시 경로설정, streamlit warnings 억제 설정 순서 중요
"""
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
# 2. Streamlit 경고 억제 설정 (Streamlit import 전에 호출)
# -------------------------------------------------------------------
from DataSense.util.streamlit_warnings import setup_streamlit_warnings
setup_streamlit_warnings()

# -------------------------------------------------------------------
# 3. 필수 라이브러리 import
# -------------------------------------------------------------------
# 표준 라이브러리
import os
from collections import defaultdict
from datetime import datetime

# 서드파티 라이브러리
import streamlit as st
import pandas as pd
import graphviz
import streamlit.components.v1 as components
from PIL import Image

#----------------------------------------------------------------------------
# 4. 로컬 모듈 import
#----------------------------------------------------------------------------
from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config

from DataSense.util.Display import (
    create_metric_card,
    display_kpi_metrics     # df, colors, title
)

APP_NAME = "Developing 2"
APP_KOR_NAME = "개발 2"
APP_TITLE = "🔗 데이터 관계 (ERD) 분석"
APP_DESCRIPTION = "데이터 값에 의한 논리적 ERD를 생성합니다. CodeMapping 기반으로 생성합니다."
# -------------------------------------------------------------------
# 상수 설정
# -------------------------------------------------------------------
MAPPING_FILE = "CodeMapping_relationship.csv"
MAPPING_ORG_FILE = "CodeMapping.csv"
OUTPUT_DIR = PROJECT_ROOT / 'DataSense' / 'DS_Output'

set_page_config(APP_NAME)

MAX_RELATED_TABLE_COUNT = 100
#----------------------------------------------------------------------------
# 5. 함수 정의
#----------------------------------------------------------------------------
def parse_relationship(relationship_str):
    """Level_Relationship 문자열에서 모든 관계를 추출합니다."""
    if not isinstance(relationship_str, str) or '->' not in relationship_str:
        return []
    
    segments = relationship_str.split(' -> ')
    relationships = []
    
    for i in range(len(segments) - 1):
        parent_segment = segments[i].strip()
        child_segment = segments[i+1].strip()
        
        try:
            # 파일명과 컬럼명 분리 (마지막 '.' 기준)
            parent_file, parent_col = parent_segment.rsplit('.', 1)
            child_file, child_col = child_segment.rsplit('.', 1)
            
            # Level_Relationship 순서를 반대로 해석하여 FK 관계 생성
            relationships.append({
                'Child_Table': child_file,
                'Child_Column': child_col,
                'Parent_Table': parent_file,
                'Parent_Column': parent_col
            })
            
        except ValueError:
            continue
            
    return relationships

def _extract_and_load_erd_data_impl(input_file_path: Path):
    """원본 파일을 로드하고 관계를 추출하여 DataFrame으로 반환합니다."""
    try:
        df_raw = pd.read_csv(input_file_path, encoding='utf-8-sig') # CodeMapping_relationship.csv 로드
    except Exception as e:
        st.error(f"원본 파일 로드 중 오류 발생: {e}")
        return None, None, None

    required_columns = ['FileName', 'ColumnName', 'PK']
    missing_columns = [col for col in required_columns if col not in df_raw.columns]
    if missing_columns:
        st.error(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        return None, None, None

    if 'Level_Relationship' not in df_raw.columns:
        st.warning("⚠️ 'Level_Relationship' 컬럼이 없습니다. FK 관계를 추출할 수 없습니다.")
        df_raw['Level_Relationship'] = ''

    # --- 2. ERD 정보 추출 메인 로직 ---
    tables_data = {}
    df_raw = df_raw.fillna('')

    # 2.1. 모든 테이블 및 컬럼 정보 추출 (벡터화된 연산 사용)
    df_raw['FileName'] = df_raw['FileName'].astype(str).str.strip()
    df_raw['ColumnName'] = df_raw['ColumnName'].astype(str).str.strip()
    df_valid = df_raw[(df_raw['FileName'] != '') & (df_raw['ColumnName'] != '')].copy()
    
    for file_name, group in df_valid.groupby('FileName'):
        if file_name not in tables_data:
            tables_data[file_name] = defaultdict(lambda: {'PK': '', 'FK': '', 'Parent_Table': ''})
        
        for _, row in group.iterrows():
            col_name = row['ColumnName']
            pk_status = 'PK' if str(row.get('PK', '')).strip() == '1' else ''
            tables_data[file_name][col_name]['PK'] = pk_status

    # 2.2. 관계 정보 추출 및 FK 업데이트 (필터링된 데이터만 처리)
    all_relationships = []
    df_with_rel = df_valid[df_valid.get('Level_Relationship', '').astype(str).str.strip() != ''].copy()
    
    for _, row in df_with_rel.iterrows():
        rel_str = str(row.get('Level_Relationship', '')).strip()
        parsed_rels = parse_relationship(rel_str)
        
        for rel in parsed_rels:
            all_relationships.append(rel)
            
            child_table = str(rel['Child_Table']).strip()
            child_col = str(rel['Child_Column']).strip()
            parent_table = str(rel['Parent_Table']).strip()
            
            if not child_table or not child_col or not parent_table:
                continue
            
            if child_table in tables_data and child_col in tables_data[child_table]:
                tables_data[child_table][child_col]['FK'] = 'FK'
                
                current_parents = str(tables_data[child_table][child_col]['Parent_Table']).strip()
                if current_parents:
                    parent_list = [p.strip() for p in current_parents.split(',') if p.strip()]
                    if parent_table not in parent_list:
                        parent_list.append(parent_table)
                        tables_data[child_table][child_col]['Parent_Table'] = ', '.join(parent_list)
                else:
                    tables_data[child_table][child_col]['Parent_Table'] = parent_table


    # 2.3. 최종 통합 DataFrame 생성 (벡터화된 연산 사용)
    df_raw['FileName'] = df_raw['FileName'].astype(str).str.strip()
    df_raw['ColumnName'] = df_raw['ColumnName'].astype(str).str.strip()
    df_raw = df_raw[(df_raw['FileName'] != '') & (df_raw['ColumnName'] != '')]
    
    # Level_Depth 처리
    if 'Level_Depth' in df_raw.columns:
        df_raw['Level_Depth'] = pd.to_numeric(df_raw['Level_Depth'], errors='coerce').fillna(0).astype(int)
    else:
        df_raw['Level_Depth'] = 0
    
    # FilePath 처리
    if 'FilePath' in df_raw.columns:
        df_raw['FilePath'] = df_raw['FilePath'].astype(str).str.strip()
    else:
        df_raw['FilePath'] = ''
    
    # Level_Relationship 처리
    if 'Level_Relationship' in df_raw.columns:
        df_raw['Level_Relationship'] = df_raw['Level_Relationship'].astype(str).str.strip()
    else:
        df_raw['Level_Relationship'] = ''
    
    # tables_data와 병합하여 PK/FK 정보 추가
    erd_data_list = []
    for _, row in df_raw.iterrows():
        file_name = row['FileName']
        col_name = row['ColumnName']
        
        if file_name in tables_data and col_name in tables_data[file_name]:
            info = tables_data[file_name][col_name]
            erd_data_list.append({
                'FileName': file_name,
                'ColumnName': col_name,
                'PK': 1 if info['PK'] == 'PK' else 0,
                'FK': 1 if info['FK'] == 'FK' else 0,
                'Parent_Table': str(info['Parent_Table']).strip(),
                'Level_Relationship': row['Level_Relationship'],
                'Level_Depth': int(row['Level_Depth']),
                'FilePath': row['FilePath']
            })
    
    if not erd_data_list:
        st.error("ERD 데이터를 생성할 수 없습니다. 입력 파일의 데이터를 확인해주세요.")
        return None, None, None
    
    df_erd_attributes = pd.DataFrame(erd_data_list)

    unique_relationships = {}
    for rel in all_relationships:
        key = (rel['Child_Table'], rel['Parent_Table'])
        
        if key not in unique_relationships:
            unique_relationships[key] = {
                'Child Table': rel['Child_Table'],
                'Parent Table': rel['Parent_Table'],
                'FK Columns': set(),
                'PK Columns': set()
            }
        
        unique_relationships[key]['FK Columns'].add(rel['Child_Column'])
        unique_relationships[key]['PK Columns'].add(rel['Parent_Column'])

    df_erd_relationships = pd.DataFrame([
        {
            'Child Table': rel['Child Table'],
            'Parent Table': rel['Parent Table'],
            'FK Columns': ', '.join(sorted(rel['FK Columns'])),
            'PK Columns': ', '.join(sorted(rel['PK Columns']))
        }
        for rel in unique_relationships.values()
    ])
    
    pk_map = df_erd_attributes[df_erd_attributes['PK'] == 1].groupby('FileName')['ColumnName'].apply(
        lambda x: list(x.astype(str))
    ).to_dict()
    
    return pk_map, df_erd_relationships, df_erd_attributes

try:
    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is not None:
        extract_and_load_erd_data = st.cache_data(_extract_and_load_erd_data_impl)
    else:
        extract_and_load_erd_data = _extract_and_load_erd_data_impl
except:
    extract_and_load_erd_data = _extract_and_load_erd_data_impl

def _extract_relationships_from_erd_logic(selected_tables: list, all_tables: set, it_df: pd.DataFrame):
    """
    generate_erd_graph와 동일한 로직으로 관계를 추출합니다.
    반환: relationships_list = [(from_file, from_col, to_file, to_col), ...]
    """
    relationships_list = []
    
    if 'Level_Relationship' not in it_df.columns:
        return relationships_list
    
    # selected_tables가 None이거나 빈 리스트인 경우 전체 데이터 기준으로 처리
    if selected_tables is None:
        selected_tables = []
    use_all_data = (len(selected_tables) == 0)
    
    # 선택된 테이블의 컬럼 정보 수집
    selected_table_columns = {}
    if not use_all_data:
        selected_df = it_df[it_df['FileName'].isin(selected_tables)]
        for table_name, group in selected_df.groupby('FileName'):
            selected_table_columns[table_name] = set(group['ColumnName'].dropna().astype(str).str.strip())
    
    # Level_Relationship이 있는 행만 필터링
    df_with_rel = it_df[
        (it_df['Level_Relationship'].notna()) & 
        (it_df['Level_Relationship'].astype(str).str.strip() != '')
    ].copy()
    
    # all_tables가 None이거나 빈 set인 경우 모든 테이블 허용
    if all_tables is None:
        all_tables = set()
    use_all_tables = (len(all_tables) == 0)
    
    for _, row in df_with_rel.iterrows():
        file_name = str(row['FileName']).strip()
        col_name = str(row['ColumnName']).strip()
        rel_str = str(row['Level_Relationship']).strip()
        
        is_selected_column = False
        if not use_all_data:
            is_selected_column = (file_name in selected_tables and 
                                 col_name in selected_table_columns.get(file_name, set()))
        
        segments = rel_str.split(' -> ')
        parsed_segments = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            try:
                file_part, col_part = segment.rsplit('.', 1)
                parsed_segments.append((file_part.strip(), col_part.strip()))
            except ValueError:
                continue
        
        # 전체 데이터 모드이거나 선택된 컬럼/테이블이 포함된 경우
        should_process = use_all_data or is_selected_column or any(seg_file in selected_tables for seg_file, _ in parsed_segments)
        
        if should_process:
            for i in range(len(parsed_segments) - 1):
                from_file, from_col = parsed_segments[i]
                to_file, to_col = parsed_segments[i+1]
                
                # all_tables 필터링 (전체 모드가 아닐 때만)
                if not use_all_tables:
                    if from_file not in all_tables or to_file not in all_tables:
                        continue
                
                relationships_list.append((from_file, from_col, to_file, to_col))
                
                if is_selected_column and i == 0 and file_name != from_file:
                    if use_all_tables or (file_name in all_tables and from_file in all_tables):
                        relationships_list.append((file_name, col_name, from_file, from_col))
    
    return relationships_list

def _extract_edge_groups_from_relationships(it_df: pd.DataFrame, selected_tables: list = None, all_tables: set = None):
    """Level_Relationship에서 엣지 그룹을 추출합니다. ERD와 동일한 로직 사용."""
    if selected_tables is None:
        selected_tables = []
    if all_tables is None:
        all_tables = set()
    
    # ERD와 동일한 로직으로 관계 추출
    relationships_list = _extract_relationships_from_erd_logic(selected_tables, all_tables, it_df)
    
    # 엣지 그룹으로 집계 (ERD와 동일하게 all_tables 필터링)
    edge_groups = {}  # {(from_file, to_file): set()}
    edge_groups_by_file = {}  # {file_name: set of (from_file, to_file)}
    
    for from_file, from_col, to_file, to_col in relationships_list:
        # ERD와 동일하게 all_tables 필터링
        if all_tables and (from_file not in all_tables or to_file not in all_tables):
            continue
            
        key = (from_file, to_file)
        if key not in edge_groups:
            edge_groups[key] = set()
        edge_groups[key].add((from_col, to_col))
        
        # 각 파일별로 엣지 그룹 수집
        if from_file not in edge_groups_by_file:
            edge_groups_by_file[from_file] = set()
        edge_groups_by_file[from_file].add(key)
        
        if to_file not in edge_groups_by_file:
            edge_groups_by_file[to_file] = set()
        edge_groups_by_file[to_file].add(key)
    
    return edge_groups_by_file

def export_summary_result(integrated_df: pd.DataFrame, selected_tables: list = None, all_tables: set = None):
    """FileName 기준으로 집계하여 요약 정보를 생성합니다."""
    # ERD와 동일한 로직으로 엣지 그룹 추출
    edge_groups_by_file = _extract_edge_groups_from_relationships(integrated_df, selected_tables, all_tables)
    
    grouped = integrated_df.groupby('FileName', sort=False)
    
    summary_df = grouped.agg({
        'FilePath': 'first',
        'ColumnName': 'nunique',
        'Level_Depth': lambda x: int(x.max()) if x.notna().any() else 0
    }).reset_index()
    
    summary_df.columns = ['FileName', 'FilePath', 'Column #', 'Max_Level']
    
    # Rel Table #: ERD에 그려지는 엣지 그룹 개수 (동일한 기준)
    summary_df['Rel Table #'] = summary_df['FileName'].apply(
        lambda x: len(edge_groups_by_file.get(str(x).strip(), set()))
    )
    
    summary_df = summary_df.sort_values(by='FileName')
    return summary_df

#----------------------------------------------------------------------------
def export_summary_result_new(integrated_df: pd.DataFrame):
    """
    FileName 기준으로 집계하여 요약 정보를 저장합니다. (Level Rel Table # 계산 추가)
    """
    
    # Level_Relationship 문자열에서 모든 고유 파일 이름을 추출하는 유틸리티 함수
    def extract_unique_files_from_chain(relationship_str):
        if not isinstance(relationship_str, str) or not relationship_str.strip():
            return set()
        files = set()
        segments = relationship_str.split(' -> ')
        for segment in segments:
            segment = segment.strip()
            if not segment: continue
            try:
                # FileName.Column에서 FileName만 추출
                file_part, _ = segment.rsplit('.', 1)
                files.add(file_part.strip())
            except ValueError:
                continue
        return files

    # 1. 파일별 기본 통계 계산 (Column #, FilePath, Max_Level_Depth)
    def get_max_level_depth(series):
        if 'Level_Depth' not in integrated_df.columns: return 0
        non_na = series.dropna()
        if non_na.empty: return 0
        try:
            return int(pd.to_numeric(non_na, errors='coerce').max())
        except (ValueError, TypeError):
            return 0

    table_stats = integrated_df.groupby('FileName').agg(
        {'ColumnName': 'nunique',
         'FilePath': lambda x: x.iloc[0] if not x.empty and 'FilePath' in integrated_df.columns else '',
         'Level_Depth': get_max_level_depth}
    ).reset_index()
    table_stats.rename(columns={'ColumnName': 'Column #', 'Level_Depth': 'Max_Level'}, inplace=True)
    
    
    # 2. Level_Relationship 기반 총 관련 파일 개수 계산 (★★★ 사용자 요청 지표 최적화 ★★★)
    temp_df = integrated_df[integrated_df['Level_Relationship'].astype(bool)].copy()
    # Level_Relationship 문자열에 함수를 적용하여 각 행의 고유 관련 파일 목록을 추출
    temp_df['Related_Files'] = temp_df['Level_Relationship'].apply(extract_unique_files_from_chain)
    
    # FileName별로 Related_Files set을 union하여 총 고유 파일 개수 계산
    total_rel_files_map = temp_df.groupby('FileName')['Related_Files'].apply(
        lambda x: len(set.union(*x)) if x.any() else 0
    ).to_dict()

    # 4. 결과 DataFrame에 병합
    summary_df = table_stats.copy()

    # **사용자 요청 필드 추가**
    summary_df['Rel Table #'] = summary_df['FileName'].apply(
        lambda x: total_rel_files_map.get(x, 0)
    )
    
    summary_df = summary_df.sort_values(by='FileName').fillna(0)
    
    return summary_df

def get_related_tables(selected_tables: list, it_df: pd.DataFrame):
    """선택된 테이블과 관련된 모든 테이블을 찾습니다."""
    if it_df is None or 'Level_Relationship' not in it_df.columns:
        return set(selected_tables)
    
    # 선택된 테이블의 컬럼 정보 수집 (벡터화된 연산)
    selected_table_columns = {}
    selected_df = it_df[it_df['FileName'].isin(selected_tables)]
    for table_name, group in selected_df.groupby('FileName'):
        selected_table_columns[table_name] = set(group['ColumnName'].dropna().astype(str).str.strip())
    
    # Level_Relationship이 있는 행만 필터링
    df_with_rel = it_df[
        (it_df['Level_Relationship'].notna()) & 
        (it_df['Level_Relationship'].astype(str).str.strip() != '')
    ].copy()
    
    all_relations = []
    for _, row in df_with_rel.iterrows():
        file_name = str(row['FileName']).strip()
        col_name = str(row['ColumnName']).strip()
        rel_str = str(row['Level_Relationship']).strip()
        
        is_selected_column = (file_name in selected_tables and 
                             col_name in selected_table_columns.get(file_name, set()))
        
        segments = rel_str.split(' -> ')
        parsed_segments = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            try:
                file_part, _ = segment.rsplit('.', 1)
                parsed_segments.append((file_part.strip(), ''))
            except ValueError:
                continue
        
        if is_selected_column or any(seg_file in selected_tables for seg_file, _ in parsed_segments):
            for i in range(len(parsed_segments) - 1):
                from_file, _ = parsed_segments[i]
                to_file, _ = parsed_segments[i+1]
                all_relations.append((from_file, to_file))
                
                if is_selected_column and i == 0 and file_name != from_file:
                    all_relations.append((file_name, from_file))
    
    related_tables = set(selected_tables)
    tables_to_check = set(selected_tables)
    
    for _ in range(5):
        newly_added = set()
        for from_table, to_table in all_relations:
            if from_table in tables_to_check:
                newly_added.add(to_table)
            if to_table in tables_to_check:
                newly_added.add(from_table)
        
        if not newly_added:
            break
        
        newly_added -= related_tables
        related_tables.update(newly_added)
        tables_to_check = newly_added
    
    return related_tables

def generate_erd_graph(selected_tables: list, all_tables: set, pk_map: dict, it_df: pd.DataFrame):
    """Graphviz 객체를 생성하고 ERD 관계를 추가합니다."""

    table_count = len(all_tables)
    graph_size = max(20, min(20 + table_count * 3, 150))
    
    dot = graphviz.Digraph(comment='Dynamic ERD', engine='dot', graph_attr={
        'rankdir': 'LR', 
        'splines': 'curved', 
        'concentrate': 'true',
        'nodesep': '0.25',
        'ranksep': '1',
        'size': f'{graph_size},{graph_size}'
    })
    dot.attr('node', shape='none', fontname='Malgun Gothic', fontsize='10')
    dot.attr('edge', fontname='Malgun Gothic', fontsize='10', penwidth='1.0')
    
    # ERD와 동일한 로직으로 관계 추출
    relationships_list = _extract_relationships_from_erd_logic(selected_tables, all_tables, it_df)
    # 연결된 컬럼 수집
    connected_columns = {}
    for from_file, from_col, to_file, to_col in relationships_list:
        if from_file not in connected_columns:
            connected_columns[from_file] = set()
        connected_columns[from_file].add(from_col)
        
        if to_file not in connected_columns:
            connected_columns[to_file] = set()
        connected_columns[to_file].add(to_col)
    
    # 2. 각 테이블별로 표시할 컬럼 결정
    display_columns = {}
    for table_name in all_tables:
        pk_cols_ordered = pk_map.get(table_name, [])
        pk_cols_set = set(pk_cols_ordered)
        connected_cols = connected_columns.get(table_name, set())
        pk_to_display = [col for col in pk_cols_ordered if col in connected_cols]
        other_to_display = sorted(list(connected_cols - pk_cols_set))
        display_columns[table_name] = pk_to_display + other_to_display

    # 3. 테이블 노드 생성
    def escape_html(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    for table_name in sorted(all_tables):
        pk_cols_ordered = pk_map.get(table_name, [])
        pk_cols_set = set(pk_cols_ordered)
        table_cols = display_columns.get(table_name, [])
        
        is_selected = table_name in selected_tables
        title_bgcolor = '#FFA500' if is_selected else '#FFF8DC'
        
        table_label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        table_label += f'<TR><TD COLSPAN="2" PORT="title" BGCOLOR="{title_bgcolor}"><B>{escape_html(table_name)}</B></TD></TR>'
        
        pk_to_display = [col for col in table_cols if col in pk_cols_set]
        other_to_display = [col for col in table_cols if col not in pk_cols_set]
        
        for col in pk_to_display:
            safe_col = escape_html(col)
            table_label += f'<TR><TD ALIGN="LEFT" BGCOLOR="#E6E6FA" PORT="{safe_col}"><B>🔑 {safe_col}</B></TD></TR>'
        
        for col in other_to_display:
            safe_col = escape_html(col)
            table_label += f'<TR><TD ALIGN="LEFT" PORT="{safe_col}"><B>🔗 {safe_col}</B></TD></TR>'
        
        table_label += '</TABLE>>'
        dot.node(table_name, table_label, shape='none')

    # 4. FK 관계 (Edge) 추가
    edge_groups = {}
    for from_file, from_col, to_file, to_col in relationships_list:
        key = (from_file, to_file)
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append((from_col, to_col))
    
    edge_count = 0
    for (from_file, to_file), cols_list in edge_groups.items():
        if from_file not in all_tables or to_file not in all_tables:
            continue
        
        from_col, to_col = cols_list[0]
        safe_from_col = escape_html(from_col)
        safe_to_col = escape_html(to_col)
        
        dot.edge(f'{from_file}:{safe_from_col}', 
                f'{to_file}:{safe_to_col}',
                dir='both',
                arrowtail='crow',
                arrowhead='none',
                constraint='true')
        edge_count += 1
    
    return dot, edge_count

def create_erd_result_dataframe(selected_tables: list, all_tables: set, pk_map: dict, it_df: pd.DataFrame):
    """ERD 생성 결과를 데이터프레임으로 정리합니다. ERD와 동일한 필터링 로직 사용."""
    # ERD와 동일한 로직으로 관계 추출
    relationships_list = _extract_relationships_from_erd_logic(selected_tables, all_tables, it_df)
    
    # 엣지 그룹으로 집계 (ERD와 동일)
    edge_groups = {}  # {(from_file, to_file): list of (from_col, to_col)}
    from_edge_groups = {}  # {table_name: set of (from_file, to_file)}
    to_edge_groups = {}  # {table_name: set of (from_file, to_file)}
    
    for from_file, from_col, to_file, to_col in relationships_list:
        # ERD와 동일하게 all_tables 필터링
        if from_file not in all_tables or to_file not in all_tables:
            continue
            
        key = (from_file, to_file)
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append((from_col, to_col))
        
        # From 관계 (이 테이블이 참조하는 테이블)
        if from_file not in from_edge_groups:
            from_edge_groups[from_file] = set()
        from_edge_groups[from_file].add(key)
        
        # To 관계 (이 테이블을 참조하는 테이블)
        if to_file not in to_edge_groups:
            to_edge_groups[to_file] = set()
        to_edge_groups[to_file].add(key)
    
    # 컬럼 정보 수집
    from_relations = {}  # {table_name: [(col, to_table, to_col), ...]}
    to_relations = {}  # {table_name: [(from_table, from_col, col), ...]}
    
    for from_file, from_col, to_file, to_col in relationships_list:
        if from_file not in from_relations:
            from_relations[from_file] = []
        from_relations[from_file].append((from_col, to_file, to_col))
        
        if to_file not in to_relations:
            to_relations[to_file] = []
        to_relations[to_file].append((from_file, from_col, to_col))
    
    result_data = []
    for table_name in sorted(all_tables):
        is_selected = table_name in selected_tables
        pk_cols_ordered = pk_map.get(table_name, [])
        pk_cols_str = ', '.join(pk_cols_ordered) if pk_cols_ordered else ''
        
        all_fk_cols = set()
        parent_tables_set = set()
        child_tables_set = set()
        
        if table_name in from_relations:
            for from_col, to_table, _ in from_relations[table_name]:
                all_fk_cols.add(from_col)
                if to_table:
                    parent_tables_set.add(to_table)
        
        if table_name in to_relations:
            for from_table, _, to_col in to_relations[table_name]:
                all_fk_cols.add(to_col)
                if from_table:
                    child_tables_set.add(from_table)
        
        # 관계 수: ERD와 동일하게 엣지 그룹 개수로 계산
        from_edge_count = len(from_edge_groups.get(table_name, set()))
        to_edge_count = len(to_edge_groups.get(table_name, set()))
        
        result_data.append({
            '테이블명': table_name,
            '선택여부': '✓' if is_selected else '',
            'PK 컬럼': pk_cols_str,
            'FK 컬럼': ', '.join(sorted(all_fk_cols)) if all_fk_cols else '',
            'Parent 테이블': ', '.join(sorted(parent_tables_set)) if parent_tables_set else '',
            'Child 테이블': ', '.join(sorted(child_tables_set)) if child_tables_set else '',
            '관계 수': from_edge_count + to_edge_count
        })
    
    return pd.DataFrame(result_data)

#-----------------------------------------------------------------------------------------
# Master KPI 
def Display_File_Statistics(filestats_df):
    """ Master Statistics KPIs """
    # def calculate_master_type_counts(df):
    #     """Code Type별 파일 수 계산"""
    #     if 'MasterType' not in df.columns or 'FileName' not in df.columns:
    #         return {}
    #     try:
    #         master_type_counts = df.groupby('MasterType')['FileName'].nunique()
    #         expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
            
    #         result = {}
    #         for master_type in expected_types:
    #             count = master_type_counts.get(master_type, 0)
    #             result[master_type] = f"{count:,}"
    #         return result
    #     except Exception as e:
    #         st.error(f"MasterType 계산 중 오류 발생: {str(e)}")
    #         return {}

    df = filestats_df.copy()
    df = df[(df['MasterType'] != 'Common') & (df['MasterType'] != 'Reference') & (df['MasterType'] != 'Validation')]
    
    # KPI 계산
    total_files = len(df['FileName'].unique()) if 'FileName' in df.columns else 0
    total_records = df['RecordCnt'].sum() if 'RecordCnt' in df.columns else 0
    total_filesize = df['FileSize'].sum() if 'FileSize' in df.columns else 0
    total_master_types = len(df['MasterType'].unique()) if 'MasterType' in df.columns else 0
    work_date = df['WorkDate'].max() if 'WorkDate' in df.columns else ''

    if total_records < 1000:
        total_records_unit = '건'
    else:
        total_records = total_records / 10000
        total_records_unit = '만건'

    if total_filesize < 1000:
        total_filesize = total_filesize 
        total_filesize_unit = 'Bytes'
    elif total_filesize < 1000000:
        total_filesize = total_filesize / 1000
        total_filesize_unit = 'KB'
    elif total_filesize < 1000000000:
        total_filesize = total_filesize / 1000000
        total_filesize_unit = 'MB'
    else:
        total_filesize = total_filesize / 1000000000
        total_filesize_unit = 'GB'        

    summary = {
        "Code File #": f"{total_files:,}",
        "Total Record #": f"{total_records:,.0f} {total_records_unit}",
        "Total File Size": f"{total_filesize:,.0f} {total_filesize_unit}",
        # "Code Type #": f"{total_master_types:,}",
        "Work Date": f"{work_date}"
    }

    # 각 메트릭에 대한 색상 정의
    metric_colors = {
        "Code File #":      "#1f77b4",
        "Total Record #":   "#2ca02c", 
        "Total File Size":  "#ff7f0e",
        "Work Date":        "#9467bd"     # 보라색
    }

    # 메트릭 표시
    display_kpi_metrics(summary, metric_colors, 'File Statistics')


    return True
#---------------------------------------------------------------------------
def load_data_mapping():
    """ 
    1st Step: 데이터 추출 및 로드
    """
    mapping_file_path = OUTPUT_DIR / MAPPING_FILE
    
    if not mapping_file_path.exists():
        st.error(f"⚠️ 원본 파일 '{MAPPING_FILE}'을 찾을 수 없습니다.")
        return None, None, None

    # 1. 데이터 추출 및 로드
    with st.spinner(f"'{MAPPING_FILE}' 파일에서 관계 정보 추출 중..."):
        pk_map, fk_df, it_df = extract_and_load_erd_data(mapping_file_path)
    
    if pk_map is None or fk_df is None or it_df is None:
        st.error("ERD 데이터를 생성할 수 없습니다. 입력 파일의 데이터를 확인해주세요.")
        return None, None, None

    return pk_map, fk_df, it_df

def load_data_org():
    """ 
    1.1st Step: CodeMapping.csv 기반 데이터 추출 및 로드
    """
    try:
        file_path = OUTPUT_DIR / MAPPING_ORG_FILE
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        return df
    except Exception as e:  
        st.error(f"원본 파일 로드 중 오류 발생: {e}")
        return None

def load_data_filestats():
    """ 
    1.2nd Step: filestats.csv 기반 데이터 추출 및 로드
    """
    try:
        file_path = OUTPUT_DIR / "FileStats.csv"
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:  
        st.error(f"원본 파일 로드 중 오류 발생: {e}")
        return None

    return df

def select_tables(it_df, it_org_df) -> list:
    """ 
    2nd Step: 테이블 선택
    """
    st.subheader("1. 테이블 선택")
    #-----------------------------------------------
    # CodeMapping.csv 기반 데이터 가공 및 병합
    #-----------------------------------------------
    it_org_cols = ['FileName', 'ColumnName', 'FilePath', 'Attribute', 'ValueCnt', 'Unique(%)', 
    'Format', 'Format(%)', 'CodeColumn_1', 'Matched(%)_1', 'CodeColumn_2', 'Matched(%)_2',
        'CodeColumn_3', 'Matched(%)_3',
    ]
    
    # 존재하는 컬럼만 선택
    available_cols = [col for col in it_org_cols if col in it_org_df.columns]
    if not available_cols:
        st.warning("CodeMapping.csv에서 필요한 컬럼을 찾을 수 없습니다.")
        return
    
    it_org_df = it_org_df[available_cols].copy()
    
    # 숫자 컬럼과 문자열 컬럼을 구분하여 fillna 적용
    numeric_cols = ['ValueCnt', 'Unique(%)', 'Format(%)', 'Matched(%)_1', 'Matched(%)_2', 'Matched(%)_3']
    string_cols = ['FileName', 'ColumnName', 'FilePath', 'Attribute', 'Format', 'CodeColumn_1', 'CodeColumn_2', 'CodeColumn_3']
    
    # 존재하는 숫자 컬럼에만 fillna(0) 적용
    numeric_cols_exist = [col for col in numeric_cols if col in it_org_df.columns]
    if numeric_cols_exist:
        it_org_df[numeric_cols_exist] = it_org_df[numeric_cols_exist].fillna(0)
    
    # 존재하는 문자열 컬럼에만 fillna('') 적용
    string_cols_exist = [col for col in string_cols if col in it_org_df.columns]
    if string_cols_exist:
        it_org_df[string_cols_exist] = it_org_df[string_cols_exist].fillna('')
    
    merged_df = pd.merge(it_df, it_org_df, on=['FileName', 'ColumnName', 'FilePath'], how='left')

    #-----------------------------------------------
    all_tables = sorted(list(merged_df['FileName'].unique()))

    # 초기 요약 정보 생성 (선택 전이므로 전체 데이터 기준, selected_tables=None)
    summary_df = export_summary_result_new(it_df)

    pk_cols = merged_df[merged_df['PK'] == 1].groupby('FileName')['ColumnName'].apply(
        lambda x: ', '.join([str(item) for item in x if pd.notna(item) and str(item).strip()])
    ).reset_index()
    pk_cols.columns = ['FileName', 'PK Columns']
    it_sum_df = pk_cols

    edited_df = pd.merge(it_sum_df, summary_df,  on='FileName', how='left')
    edited_df = edited_df[(edited_df['Column #'] > 1)]
    edited_df = edited_df.sort_values(by='FileName')

    # 선택 체크박스 컬럼 추가 (기본값 False)
    if '선택' not in edited_df.columns:
        edited_df['선택'] = False
    
    # cols_order = ['선택'] + [col for col in edited_df.columns if col != '선택']
    cols_order = ['선택','FileName', 'PK Columns', 'Column #', 'Max_Level', 'Rel Table #', 'FilePath']
    edited_df = edited_df[cols_order]
    
    edited_df = st.data_editor(edited_df, hide_index=True, width=1000, height=500, column_config={
        '선택': st.column_config.CheckboxColumn('선택', width='small'),
        'FileName': st.column_config.TextColumn(help='파일 이름', width=150),
        'PK Columns': st.column_config.TextColumn(help='PK 컬럼', width=150),
        'Column #': st.column_config.NumberColumn(help='컬럼 수', width=100),
        'Max_Level': st.column_config.NumberColumn(help='최대 레벨', width=100),
        'Rel Table #': st.column_config.NumberColumn(help='엣지 그룹 개수', width=100),
        'FilePath': st.column_config.TextColumn(help='파일 경로', width='large')
    })

    # 선택된 테이블 추출 (선택 컬럼이 True인 행의 FileName)
    selected_tables = edited_df[edited_df['선택'] == True]['FileName'].tolist()
    selected_tables = [t for t in selected_tables if t in all_tables]
    
    if not selected_tables:
        st.info(f"최대 related_tables 수는 합계 {MAX_RELATED_TABLE_COUNT}개까지 가능합니다.")
        return None

    return selected_tables

def generate_erd(selected_tables, pk_map, it_df):
    """ 
    3rd Step: Logical ERD 생성
    """
    
    st.subheader("2. Logical ERD 분석 결과")
    related_tables = get_related_tables(selected_tables, it_df)

    related_table_count = len(related_tables) # 연결된 테이블 수    

    st.caption(f"**선택된 테이블:** {selected_tables}")
    st.caption(f"**연결된 총 테이블 수:** {related_table_count}개")
    
    if related_table_count > MAX_RELATED_TABLE_COUNT:
        st.error(f"연결된 테이블 수가 {MAX_RELATED_TABLE_COUNT}개를 초과했습니다.")
        return False

    # # Graphviz 설치 확인
    # try:
    #     import graphviz
    #     # Graphviz 실행 파일 확인
    #     try:
    #         graphviz.version()
    #     except Exception:
    #         st.info("""
    #         **ERD 생성이 불가능합니다.**
    #         Cloud 환경에서는 Graphviz 설치가 제한될 수 있어서 ERD 생성이 불가능합니다.
    #         로컬 환경에서 실행하세요. 
            
    #         **예제 ERD를 표시합니다.**
    #         """)
    #         image = Image.open(OUTPUT_DIR / "DataSense_Logical_COMPANY.png")
    #         st.image(image, caption="단순한 예제 ERD", width=480)
    #         st.divider()
    #         image = Image.open(OUTPUT_DIR / "DataSense_Logical_ERD_복잡한예.png")
    #         st.image(image, caption="복잡한 예제 ERD", width=480)
    #         return False
    # except ImportError:
    #     st.error("❌ Graphviz 라이브러리를 import할 수 없습니다.")
    #     return False

    try:
        graph, erd_edge_count = generate_erd_graph(selected_tables, related_tables, pk_map, it_df)
        
        if graph is None:
            st.error("❌ ERD 그래프 객체를 생성할 수 없습니다.")
            return False
        
        file_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_filename = f"DataSense_Logical_ERD_{file_time}.png"
        png_filepath = OUTPUT_DIR / png_filename
       
        # PNG 저장 시도
        png_success = False
        actual_png_filepath = None
        try:
            graph.attr(dpi='300')
            erd_path = png_filepath.with_suffix('')
            graph.render(str(erd_path), format='png', cleanup=True)
            actual_png_filepath = OUTPUT_DIR / f"{erd_path.name}.png"
            
            if actual_png_filepath.exists():
                png_success = True
                st.caption(f"📁 저장 경로: `{actual_png_filepath}`")
            else:
                st.warning("⚠️ PNG 파일이 생성되었지만 파일을 찾을 수 없습니다.")
        except Exception as e:
            error_msg = str(e)
            if 'ExecutableNotFound' in error_msg or 'not found' in error_msg.lower():
                st.error("❌ Graphviz 실행 파일을 찾을 수 없습니다.")
                st.info("""
                **ERD 생성 실패:**
                
                Streamlit Cloud 환경에서는 Graphviz 실행 파일이 설치되어 있지 않을 수 있습니다.
                
                **대안:**
                - SVG 형식으로 ERD를 표시하려고 시도합니다.
                """)
            else:
                st.warning(f"⚠️ PNG 파일 저장 실패: {error_msg}")
        
        # PNG 파일이 성공적으로 생성된 경우
        if png_success and actual_png_filepath:
            try:
                with open(actual_png_filepath, 'rb') as f:
                    png_data = f.read()
                if png_data:
                    st.download_button(
                        label="📥 PNG 파일 다운로드",
                        data=png_data,
                        file_name=actual_png_filepath.name,
                        mime="image/png"
                    )

                image = Image.open(actual_png_filepath)
                caption = f"ERD: {', '.join(selected_tables[:5])}{'...' if len(selected_tables) > 5 else ''}"
                st.image(image, caption=caption, width=1000)
                return related_tables
            except Exception as e:
                st.warning(f"⚠️ PNG 이미지 로드 실패: {e}")
        
        # PNG 실패 시 SVG로 대체 시도
        try:
            st.info("🔄 SVG 형식으로 ERD를 표시합니다...")
            svg_data = graph.pipe(format='svg').decode('utf-8')
            if svg_data and len(svg_data) > 0:
                components.html(svg_data, height=800, scrolling=True)
                st.success("✅ ERD가 SVG 형식으로 표시되었습니다.")
                return related_tables
            else:
                st.error("❌ SVG 데이터가 비어있습니다.")
                return False
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ ERD 생성에 실패했습니다: {error_msg}")
            st.info("""
            **ERD 생성이 불가능한 상황입니다.**
            
            **가능한 원인:**
            1. Graphviz가 설치되어 있지 않음
            2. Graphviz 실행 파일 경로 문제
            3. Streamlit Cloud 환경 제한
            
            **해결 방법:**
            - 로컬 환경에서 실행하거나
            - 시스템 관리자에게 Graphviz 설치를 요청하세요.
            """)
            return False

    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ ERD 생성 중 오류가 발생했습니다: {error_msg}")
        
        # Graphviz 관련 오류인지 확인
        if 'graphviz' in error_msg.lower() or 'ExecutableNotFound' in error_msg:
            st.info("""
            **Graphviz 관련 오류입니다.**
            
            Streamlit Cloud에서는 Graphviz 실행 파일이 설치되어 있지 않을 수 있습니다.
            로컬 환경에서 실행하거나, 시스템 관리자에게 문의하세요.
            """)
        else:
            st.info("예상치 못한 오류가 발생했습니다. 오류 메시지를 확인하세요.")
        
        return False

def display_erd_result(selected_tables, related_tables, pk_map, it_df):
    """ 
    4th Step: ERD 결과 요약
    """
    st.divider()
    st.subheader("3. ERD 결과 요약")
    tab1, tab2, tab3 = st.tabs(["ERD 결과 요약", "선택된 테이블 상세 정보", "관계된 테이블 상세 정보"])
    with tab1:
        erd_result_df = create_erd_result_dataframe(selected_tables, related_tables, pk_map, it_df)
        st.dataframe(
            erd_result_df,
            hide_index=True,
            width='stretch',    
            height=400,
            column_config={
                '테이블명': st.column_config.TextColumn('테이블명', width=150),
                '선택여부': st.column_config.TextColumn('선택', width=50),
                'PK 컬럼': st.column_config.TextColumn('PK 컬럼', width=150),
                'FK 컬럼': st.column_config.TextColumn('FK 컬럼', width=150),
                'Parent 테이블': st.column_config.TextColumn('Parent 테이블', width=200),
                'Child 테이블': st.column_config.TextColumn('Child 테이블', width=200),
                '관계 수': st.column_config.NumberColumn('관계 수', width=50)
            }
        )

    with tab2:
        selected_tables_df = it_df[it_df['FileName'].isin(selected_tables)]
        selected_tables_df = selected_tables_df.drop(columns=['FilePath'])
        st.dataframe(selected_tables_df, hide_index=True, width=1000, height=400)

    with tab3:
        related_tables_df = it_df[it_df['FileName'].isin(related_tables)]
        related_tables_df = related_tables_df.drop(columns=['FilePath'])
        st.dataframe(related_tables_df, hide_index=True, width=1000, height=400)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 테이블 수", len(erd_result_df))
    with col2:
        st.metric("선택된 테이블 수", len(erd_result_df[erd_result_df['선택여부'] == '✓']))
    with col3:
        st.metric("총 관계 수", erd_result_df['관계 수'].sum())
    with col4:
        st.metric("PK 보유 테이블", len(erd_result_df[erd_result_df['PK 컬럼'] != '']))

    return True

#---------------------------------------------------------------------------
# 6. main 함수
#---------------------------------------------------------------------------
def main():
    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)

    try:
        # 1. 데이터 추출 및 로드
        pk_map, fk_df, it_df = load_data_mapping() # CodeMapping_relationship.csv 기반

        it_org_df = load_data_org() # CodeMapping.csv 기반
        filestats_df = load_data_filestats() # filestats.csv 기반
       
        if it_org_df is None or filestats_df is None:
            st.error("CodeMapping.csv 또는 filestats.csv 파일을 로드할 수 없습니다.")
            return

        # 1.1 KPI 표시
        Display_File_Statistics(filestats_df)

        # 2. 테이블 선택     
        selected_tables = select_tables(it_df, it_org_df)
        if selected_tables is None:
            return

        # ERD 생성 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            erd_button = st.button("🔗 ERD 생성", type="primary", use_container_width=True)
        
        if not erd_button:
            st.info(f"최대 related_tables 수는 합계 {MAX_RELATED_TABLE_COUNT}개까지 가능합니다.")
            return

        # 3. ERD 생성
        related_tables = generate_erd(selected_tables, pk_map, it_df)
        if not related_tables:
            return

        erd_success = display_erd_result(selected_tables, related_tables, pk_map, it_df)
        if not erd_success:
            return
        return 

    except Exception as e:
        st.error(f"ERD 생성 중 치명적인 오류가 발생했습니다: {e}")
        return

if __name__ == '__main__':
    main()
