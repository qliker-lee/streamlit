#
# streamlit를 이용한 Data Sense System : Data Quality Information
# 2024. 11. 10.  Qliker
#

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import re
from datetime import datetime
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
from graphviz import Digraph
from dataclasses import dataclass
from typing import Dict, Any, Optional
import plotly.graph_objects as go
warnings.filterwarnings("ignore", category=UserWarning)

SOLUTION_NAME = "Data Sense System"
SOLUTION_KOR_NAME = "데이터 센스 시스템"
APP_NAME = "Value Chain Data Analysis"
APP_DESC = "#### Value Chain & Sysyem 별 Data 현황 분석"

# -------------------------------------------------------------------
# 경로 설정
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from DataSense.util.Files_FunctionV20 import load_yaml_datasense, set_page_config

set_page_config(APP_NAME)

from DataSense.util.Display import create_metric_card

#-----------------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#-----------------------------------------------------------------------------------------
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
            # 숫자형 컬럼: NaN을 0으로 변환 (None 표시 방지)
            df[col] = df[col].fillna(0)
        elif df[col].dtype == 'object':
            # object 타입인 경우 숫자로 변환 가능한지 확인
            try:
                # 숫자로 변환 시도
                numeric_series = pd.to_numeric(df[col], errors='coerce')
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

#-----------------------------------------------------------------------------------------
def Display_File_Stats_by_Systems(df):
    """System별 FileName 통계 표시"""
    if df is None or df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # System 컬럼이 있는지 확인
    if 'System' not in df.columns:
        st.warning("System 컬럼이 없습니다.")
        return
    
    # FileName 컬럼이 있는지 확인
    if 'FileName' not in df.columns:
        st.warning("FileName 컬럼이 없습니다.")
        return
    
    try:
        # 데이터프레임 복사 및 전처리
        df_processed = df.copy()
        
        # System 컬럼 처리
        df_processed['System'] = df_processed['System'].fillna('')
        df_processed['System'] = df_processed['System'].astype(str).str.strip()
        # 빈 문자열이거나 'nan'인 경우 'Not Defined'로 변경
        df_processed.loc[
            (df_processed['System'] == '') | 
            (df_processed['System'].str.lower() == 'nan'),
            'System'
        ] = 'Not Defined'
        
        # System별 FileName의 유니크한 수 계산
        system_file_counts = df_processed.groupby('System')['FileName'].nunique()
        
        # 'Not Defined'를 제외하고 정렬
        defined_systems = system_file_counts[system_file_counts.index != 'Not Defined'].sort_values(ascending=False)
        systems_order = defined_systems.index.tolist()
        
        # 'Not Defined'가 있으면 마지막에 추가
        if 'Not Defined' in system_file_counts.index:
            systems_order.append('Not Defined')
        
        # System 순서대로 재정렬 (Not Defined는 마지막)
        system_file_counts = system_file_counts.reindex(systems_order)
        
        if system_file_counts.empty:
            st.info("System별 파일 통계 데이터가 없습니다.")
            return
        
        st.markdown("### Statistics by System")
        
        # System별 파일 수를 메트릭 카드로 표시
        systems_list = system_file_counts.index.tolist()
        file_counts_list = system_file_counts.values.tolist()
        
        # 색상 정의
        system_color = "#1f77b4"  # 파란색
        not_defined_color = "#7f7f7f"  # 회색
        
        cols_per_row = 5
        num_rows = (len(systems_list) + cols_per_row - 1) // cols_per_row
        
        for row_idx in range(num_rows):
            start_idx = row_idx * cols_per_row
            end_idx = min(start_idx + cols_per_row, len(systems_list))
            row_systems = systems_list[start_idx:end_idx]
            row_counts = file_counts_list[start_idx:end_idx]
            
            cols = st.columns(len(row_systems))
            for col, system, count in zip(cols, row_systems, row_counts):
                # Not Defined는 회색, 나머지는 파란색
                color = not_defined_color if system == 'Not Defined' else system_color
                value = f"{count:,}"
                col.markdown(create_metric_card(value, system, color), unsafe_allow_html=True)
        
        # 상세 통계 테이블 표시
        st.markdown("#### System별 파일 통계 상세")
        stats_df = pd.DataFrame({
            'System': system_file_counts.index,
            'File Count': system_file_counts.values
        }).reset_index(drop=True)
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # System, Activities_Type, Activities, FileName 상세 DataFrame
        st.markdown("#### System, Activities_Type, Activities, FileName 상세 정보")
        # 필요한 컬럼이 모두 있는지 확인
        required_cols = ['System', 'Activities_Type', 'Activities', 'FileName']
        available_cols = [col for col in required_cols if col in df_processed.columns]
        
        if len(available_cols) == len(required_cols):
            detail_df = df_processed[required_cols].copy()
            # FileName이 'Not Defined'가 아니고, Activities가 'Not Defined'이거나 System이 'Not Defined'인 행 제외
            detail_df = detail_df[
                (detail_df['FileName'] != 'Not Defined') &
                (detail_df['Activities'] != 'Not Defined') &
                (detail_df['System'] != 'Not Defined')
            ].copy()
            # 중복 제거
            detail_df = detail_df.drop_duplicates()
            # System, Activities_Type, Activities 순서대로 정렬
            detail_df = detail_df.sort_values(['System', 'Activities_Type', 'Activities', 'FileName'])
            detail_df = normalize_dataframe_for_display(detail_df)
            st.dataframe(detail_df, use_container_width=True, hide_index=True, height=400)
        else:
            missing_cols = [col for col in required_cols if col not in df_processed.columns]
            st.warning(f"⚠️ 다음 컬럼이 없어 상세 정보를 표시할 수 없습니다: {', '.join(missing_cols)}")
        
        return True
        
    except Exception as e:
        st.error(f"System별 파일 통계 계산 중 오류 발생: {str(e)}")
        import traceback
        st.exception(e)
        return False

def Display_File_Stats_by_Activities(df):
    """Activities_Type, Activities별 FileName 통계 표시"""
    if df is None or df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # Activities 컬럼이 있는지 확인
    if 'Activities' not in df.columns:
        st.warning("Activities 컬럼이 없습니다.")
        return
    
    # FileName 컬럼이 있는지 확인
    if 'FileName' not in df.columns:
        st.warning("FileName 컬럼이 없습니다.")
        return
    
    try:
        # 데이터프레임 복사 및 전처리
        df_processed = df.copy()
        
        # Activities_Type 컬럼 처리
        if 'Activities_Type' in df_processed.columns:
            df_processed['Activities_Type'] = df_processed['Activities_Type'].fillna('')
            df_processed['Activities_Type'] = df_processed['Activities_Type'].astype(str).str.strip()
            df_processed.loc[
                (df_processed['Activities_Type'] == '') | 
                (df_processed['Activities_Type'].str.lower() == 'nan'),
                'Activities_Type'
            ] = 'Not Defined'
        else:
            df_processed['Activities_Type'] = 'Not Defined'
        
        # Activities 컬럼 복사 및 빈 값 처리
        df_processed['Activities'] = df_processed['Activities'].fillna('')
        df_processed['Activities'] = df_processed['Activities'].astype(str).str.strip()
        # 빈 문자열이거나 'nan'인 경우 'Not Defined'로 변경
        df_processed.loc[
            (df_processed['Activities'] == '') | 
            (df_processed['Activities'].str.lower() == 'nan'),
            'Activities'
        ] = 'Not Defined'
        
        # Activities_Type과 Activities를 결합한 키 생성
        df_processed['Activities_Key'] = df_processed['Activities_Type'] + ' - ' + df_processed['Activities']
        
        # Activity_Seq 컬럼이 있으면 숫자로 변환하여 정렬에 사용
        if 'Activity_Seq' in df_processed.columns:
            # Activity_Seq를 숫자로 변환 (문자열인 경우 처리)
            df_processed['Activity_Seq'] = pd.to_numeric(df_processed['Activity_Seq'], errors='coerce')
            # Activities_Key별 첫 번째 Activity_Seq 값을 가져와서 정렬 기준으로 사용
            activities_seq_map = df_processed.groupby('Activities_Key')['Activity_Seq'].first()
            # 'Not Defined'를 포함한 항목은 제외하고 정렬
            defined_activities = activities_seq_map[
                ~activities_seq_map.index.str.contains('Not Defined', na=False)
            ].sort_values()
            # Activity_Seq 순서대로 Activities_Key 정렬 (Not Defined 제외)
            activities_order = defined_activities.index.tolist()
            # 'Not Defined'를 포함한 항목을 마지막에 추가
            not_defined_activities = activities_seq_map[
                activities_seq_map.index.str.contains('Not Defined', na=False)
            ].index.tolist()
            activities_order.extend(not_defined_activities)
        else:
            # Activity_Seq가 없으면 Activities_Key 이름 순서로 정렬
            all_activities = sorted(df_processed['Activities_Key'].unique())
            # 'Not Defined'를 포함한 항목을 마지막으로 이동
            not_defined = [a for a in all_activities if 'Not Defined' in a]
            defined = [a for a in all_activities if 'Not Defined' not in a]
            activities_order = defined + not_defined
        
        # Activities_Key별 FileName의 유니크한 수 계산
        activities_file_counts = df_processed.groupby('Activities_Key')['FileName'].nunique()
        
        # Activity_Seq 순서대로 재정렬 (Not Defined는 마지막)
        activities_file_counts = activities_file_counts.reindex(activities_order)
        
        if activities_file_counts.empty:
            st.info("Activities별 파일 통계 데이터가 없습니다.")
            return
        
        st.markdown("### Statistics by Activities_Type & Activities")
        
        # Activities_Key별 파일 수를 메트릭 카드로 표시
        activities_list = activities_file_counts.index.tolist()
        file_counts_list = activities_file_counts.values.tolist()
        
        # Primary와 Support로 분리
        primary_activities = []
        primary_counts = []
        support_activities = []
        support_counts = []
        not_defined_activities = []
        not_defined_counts = []
        
        for activity, count in zip(activities_list, file_counts_list):
            if 'Not Defined' in activity:
                not_defined_activities.append(activity)
                not_defined_counts.append(count)
            elif activity.startswith('Primary'):
                primary_activities.append(activity)
                primary_counts.append(count)
            elif activity.startswith('Support'):
                support_activities.append(activity)
                support_counts.append(count)
            else:
                # Activities_Type이 명시되지 않은 경우 Support로 분류
                support_activities.append(activity)
                support_counts.append(count)
        
        # 색상 정의
        primary_color = "#1f77b4"  # 파란색
        support_color = "#2ca02c"  # 초록색
        not_defined_color = "#7f7f7f"  # 회색
        
        cols_per_row = 5
        
        # Primary Activities 표시
        if primary_activities:
            st.markdown("#### Primary Activities")
            num_rows_primary = (len(primary_activities) + cols_per_row - 1) // cols_per_row
            for row_idx in range(num_rows_primary):
                start_idx = row_idx * cols_per_row
                end_idx = min(start_idx + cols_per_row, len(primary_activities))
                row_activities = primary_activities[start_idx:end_idx]
                row_counts = primary_counts[start_idx:end_idx]
                
                cols = st.columns(len(row_activities))
                for col, activity, count in zip(cols, row_activities, row_counts):
                    value = f"{count:,}"
                    col.markdown(create_metric_card(value, activity, primary_color), unsafe_allow_html=True)
        
        # Support Activities 표시 (Not Defined 포함)
        if support_activities or not_defined_activities:
            st.markdown("#### Support Activities")
            # Support와 Not Defined를 합침
            all_support = support_activities + not_defined_activities
            all_support_counts = support_counts + not_defined_counts
            
            num_rows_support = (len(all_support) + cols_per_row - 1) // cols_per_row
            for row_idx in range(num_rows_support):
                start_idx = row_idx * cols_per_row
                end_idx = min(start_idx + cols_per_row, len(all_support))
                row_activities = all_support[start_idx:end_idx]
                row_counts = all_support_counts[start_idx:end_idx]
                
                cols = st.columns(len(row_activities))
                for col, activity, count in zip(cols, row_activities, row_counts):
                    # Not Defined는 회색, Support는 초록색
                    color = not_defined_color if 'Not Defined' in activity else support_color
                    value = f"{count:,}"
                    col.markdown(create_metric_card(value, activity, color), unsafe_allow_html=True)
        
        # 상세 통계 테이블 표시
        st.markdown("#### Activities_Type & Activities별 파일 통계 상세")
        # Activities_Key를 Activities_Type과 Activities로 분리
        stats_data = []
        for key, count in zip(activities_file_counts.index, activities_file_counts.values):
            if ' - ' in key:
                activities_type, activities = key.split(' - ', 1)
            else:
                activities_type = 'Not Defined'
                activities = key
            stats_data.append({
                'Activities_Type': activities_type,
                'Activities': activities,
                'File Count': count
            })
        stats_df = pd.DataFrame(stats_data)
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Activities_Type, Activities, System, FileName 상세 DataFrame
        st.markdown("#### Activities_Type, Activities, System, FileName 상세 정보")
        detail_df = df_processed[['Activities_Type', 'Activities', 'System', 'FileName']].copy()
        # FileName이 'Not Defined'가 아니고, Activities가 'Not Defined'이거나 System이 'Not Defined'인 행 제외
        detail_df = detail_df[
            (detail_df['FileName'] != 'Not Defined') &
            (detail_df['Activities'] != 'Not Defined') &
            (detail_df['System'] != 'Not Defined')
        ].copy()
        # 중복 제거
        detail_df = detail_df.drop_duplicates()
        # Activities_Type과 Activities 순서대로 정렬
        detail_df = detail_df.sort_values(['Activities_Type', 'Activities', 'System', 'FileName'])
        detail_df = normalize_dataframe_for_display(detail_df)
        st.dataframe(detail_df, use_container_width=True, hide_index=True, height=400)
        
        return True
        
    except Exception as e:
        st.error(f"Activities별 파일 통계 계산 중 오류 발생: {str(e)}")
        import traceback
        st.exception(e)
        return False

# # Master KPI 
# def Display_File_Stats(df):
#     """ File Statistics """
#     def calculate_file_stats(df):
#         """MasterType별 파일 수 계산"""
#         try:
#             activities_type = df['Activities'].unique()
#             activity_file_counts = df.groupby('Activities')['FileName'].nunique()
            
#             result = {}
#             for master_type in expected_types:
#                 count = master_type_counts.get(master_type, 0)
#                 result[master_type] = f"{count:,}"
#             return result
#         except Exception as e:
#             st.error(f"MasterType 계산 중 오류 발생: {str(e)}")
#             return {}

#     df = loaded_data['filestats']
#     if df is None or df.empty:
#         st.warning("데이터가 없습니다.")
#         return
    
#     st.markdown("### File Statistics")
#     # KPI 계산
#     total_files = len(df['FileName'].unique()) if 'FileName' in df.columns else 0
#     total_records = df['RecordCnt'].sum() if 'RecordCnt' in df.columns else 0
#     total_filesize = df['FileSize'].sum() if 'FileSize' in df.columns else 0
#     total_master_types = len(df['MasterType'].unique()) if 'MasterType' in df.columns else 0
#     work_date = df['WorkDate'].max() if 'WorkDate' in df.columns else ''

#     if total_records < 1000:
#         total_records_unit = '건'
#     else:
#         total_records = total_records / 10000
#         total_records_unit = '만건'

#     if total_filesize < 1000:
#         total_filesize = total_filesize 
#         total_filesize_unit = 'Bytes'
#     elif total_filesize < 1000000:
#         total_filesize = total_filesize / 1000
#         total_filesize_unit = 'KB'
#     elif total_filesize < 1000000000:
#         total_filesize = total_filesize / 1000000
#         total_filesize_unit = 'MB'
#     else:
#         total_filesize = total_filesize / 1000000000
#         total_filesize_unit = 'GB'        

#     summary = {
#         "Code File #": f"{total_files:,}",
#         "Total Record #": f"{total_records:,.0f} {total_records_unit}",
#         "Total File Size": f"{total_filesize:,.0f} {total_filesize_unit}",
#         "Code Type #": f"{total_master_types:,}",
#         "Work Date": f"{work_date}"
#     }

#     # # 각 메트릭에 대한 색상 정의
#     # metric_colors = {
#     #     "Code File #": "#1f77b4",
#     #     "Total Record #": "#2ca02c", 
#     #     "Total File Size": "#ff7f0e",
#     #     "Code Type #": "#d62728",
#     #     "Work Date": "#9467bd"
#     # }

#     # 메트릭 표시
#     cols = st.columns(len(summary))
#     for col, (key, value) in zip(cols, summary.items()):
#         color = metric_colors.get(key, "#0072B2")
#         col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

#     # MasterType별 파일 수
#     st.markdown("### Statistics by Code Type")
#     master_type_counts = calculate_master_type_counts(df)
    
#     if master_type_counts:
#         metric_colors = {
#             "Master": "#1f77b4",      # 파란색
#             "Operation": "#2ca02c",    # 초록색
#             "Reference": "#ff7f0e",    # 주황색
#             "Attribute": "#d62728",    # 빨간색
#             "Common": "#9467bd",       # 보라색
#             "Validation": "#8c564b"    # 갈색
#         }

#         cols = st.columns(len(master_type_counts))
#         for col, (key, value) in zip(cols, master_type_counts.items()):
#             color = metric_colors.get(key, "#0072B2")
#             col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

#     return True

#-----------------------------------------------------------------------------------------
def Display_MasterFormat_Detail_old(loaded_data):
    """ Master Format Detail """

    def value_info(df):
        st.markdown("###### Value Information")

        df= df[['FileName', 'ColumnName', 'OracleType', 'PK',  'ValueCnt', 
                'Null(%)', 'UniqueCnt', 'Unique(%)', 
                'MinString', 'MaxString', 'ModeString', 'MedianString', 'ModeCnt', 'Mode(%)']]

        df = df.reset_index(drop=True)
        st.dataframe(data=df, width=1400, height=600,hide_index=True)

    def value_type_info(df):
        st.markdown("###### Value Pattern Information")

        df= df[['FileName', 'ColumnName', 'ValueCnt', 'FormatCnt',
                'Format', 'Format(%)', 'FormatMin', 'FormatMax', 'FormatMode', 'FormatMedian',
                'Format2nd', 'Format2nd(%)', 'Format2ndMin', 'Format2ndMax', 'Format2ndMode', 'Format2ndMedian',
                'Format3rd', 'Format3rd(%)', 
                ]]
        st.dataframe(data=df, width=1400, height=600,hide_index=True)

    def top10_info(df):
        st.markdown("###### Value Top 10 Information")
        df= df[['FileName', 'ColumnName', 'ValueCnt', 'ModeString', 'ModeCnt', 'Mode(%)', 'Top10', 'Top10(%)']]

        st.dataframe(data=df, width=1400, height=600,hide_index=True)   

    def length_info(df):
        st.markdown("###### Data Length Information")

        df= df[['FileName', 'ColumnName', 'OracleType', 'PK', 'DetailDataType', 'LenCnt', 'LenMin', 'LenMax', 'LenAvg',
                    'LenMode', 'RecordCnt', 'SampleRows', 'ValueCnt', 'NullCnt', 'Null(%)', 'UniqueCnt', 'Unique(%)']]

        st.dataframe(data=df, width=1400, height=600,hide_index=True)

    def character_info(df):
        st.markdown("###### Character Information")

        df= df[['FileName', 'ColumnName', 'ValueCnt', 'HasBlank', 'HasDash', 'HasDot', 'HasAt', 'HasAlpha', 'HasKor', 'HasNum', 
                'HasBracket', 'HasMinus', 'HasOnlyAlpha', 'HasOnlyNum', 'HasOnlyKor', 'HasOnlyAlphanum', 
                'FirstChrKor', 'FirstChrNum', 'FirstChrAlpha', 'FirstChrSpecial']]

        st.dataframe(data=df, width=1400, height=600,hide_index=True)

    def dq_score_info(df):
        st.markdown("###### Data Quality Score Information")
        st.write("DQ Score 현재 개발중임. (기업의 상황에 따라 기준이 다를 수 있습니다. 컨설팅 후 확정합니다. ")

        df= df[['FileName', 'ColumnName', 'ValueCnt', 'Null_pct', 'TypeMixed_pct', 'LengthVol_pct', 'Duplicate_pct',
                    'DQ_Score', 'DQ_Issues', 'Issue_Count', 'Weighted_DQ_Score']]

        st.dataframe(data=df, width=1400, height=600,hide_index=True)          

    #---------------------------------
    st.write("\n")
    st.markdown("### Data Quality Information")
    st.markdown("###### 모든 파일의 데이터 값을 분석하여 Value Pattern별로 통계를 작성합니다.")
    fileformat_df = loaded_data['fileformat'].copy()

    if fileformat_df is not None and not fileformat_df.empty:
        # MasterType들
        expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(expected_types)

        for master_type, tab in zip(expected_types, [tab1, tab2, tab3, tab4, tab5, tab6]):
            with tab:
                Value, value_type,Length,  top10, character, dq_score = st.columns(6)

                master_type_df = fileformat_df[fileformat_df['MasterType'] == master_type]
                if Value.button("Value Information", key=f"Value_{master_type}"):
                    display_df = value_info(master_type_df)
                if value_type.button("Value Type Information", key=f"value_type_{master_type}"):
                    display_df = value_type_info(master_type_df)
                if Length.button("Length Information", key=f"Length_{master_type}"):
                    display_df = length_info(master_type_df)
                if top10.button("Top10 Information", key=f"top10_{master_type}"):
                    display_df = top10_info(master_type_df)
                if character.button("Character Information", key=f"character_{master_type}"):
                    display_df = character_info(master_type_df)
                if dq_score.button("DQ Score Information", key=f"dq_score_{master_type}"):
                    display_df = dq_score_info(master_type_df)

    else:
        st.warning("Data Quality 분석 파일을 로드할 수 없습니다.")
        return False
    return True

#-----------------------------------------------------------------------------------------
def Display_MasterFormat_Detail(loaded_data):
    """Master Format Detail 화면 출력"""

    # 각 뷰별 컬럼 정의
    VIEW_COLUMNS = {
        "Value Info": [
            'FileName', 'ColumnName', 'OracleType', 'PK', 'ValueCnt',
            'Null(%)', 'UniqueCnt', 'Unique(%)',
            'MinString', 'MaxString', 'ModeString', 'MedianString', 'ModeCnt', 'Mode(%)'
        ],
        "Value Type Info": [
            'FileName', 'ColumnName', 'ValueCnt', 'FormatCnt',
            'Format', 'Format(%)', 'FormatMin', 'FormatMax', 'FormatMode', 'FormatMedian',
            'Format2nd', 'Format2nd(%)', 'Format2ndMin', 'Format2ndMax', 'Format2ndMode', 'Format2ndMedian',
            'Format3rd', 'Format3rd(%)'
        ],

        "Top10 Info": [
            'FileName', 'ColumnName', 'ValueCnt', 'ModeString', 'ModeCnt', 'Mode(%)',
            'Top10', 'Top10(%)'
        ],
        "Length Info": [
            'FileName', 'ColumnName', 'OracleType', 'PK', 'DetailDataType',
            'LenCnt', 'LenMin', 'LenMax', 'LenAvg', 'LenMode',
            'RecordCnt', 'SampleRows', 'ValueCnt', 'NullCnt', 'Null(%)',
            'UniqueCnt', 'Unique(%)'
        ],
        "Character Info": [
            'FileName', 'ColumnName', 'ValueCnt', 'HasBlank', 'HasDash', 'HasDot', 'HasAt', 'HasAlpha',
            'HasKor', 'HasNum', 'HasBracket', 'HasMinus', 'HasOnlyAlpha', 'HasOnlyNum',
            'HasOnlyKor', 'HasOnlyAlphanum',
            'FirstChrKor', 'FirstChrNum', 'FirstChrAlpha', 'FirstChrSpecial'
        ],
        "DQ Score Info": [
            'FileName', 'ColumnName', 'ValueCnt', 'Null_pct', 'TypeMixed_pct', 'LengthVol_pct', 'Duplicate_pct',
            'DQ_Score', 'DQ_Issues', 'Issue_Count', 'Weighted_DQ_Score'
        ]
    }

    def render_table(df, title, cols):
        """공통 테이블 렌더링 함수"""
        st.markdown(f"###### {title}")
        if title == "DQ Score Information":
            st.write("DQ Score 기업의 상황에 따라 기준이 다를 수 있습니다. 컨설팅 후 확정합니다. ")

        df = df[cols].reset_index(drop=True)
        st.dataframe(data=df, width=1400, height=600, hide_index=True)

    # ---------------------------
    st.markdown("### Data Quality Information")
    st.markdown("###### 모든 파일의 데이터 값을 분석하여 Value Pattern별로 통계를 작성합니다.")
    fileformat_df = loaded_data.get('fileformat', pd.DataFrame())

    if fileformat_df.empty:
        st.warning("Data Quality 분석 파일을 로드할 수 없습니다.")
        return False

    expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
    tabs = st.tabs(expected_types)

    for master_type, tab in zip(expected_types, tabs):
        with tab:
            cols = st.columns(len(VIEW_COLUMNS))
            
            master_type_df = fileformat_df[fileformat_df['MasterType'] == master_type]

            for col, (title, cols_to_show) in zip(cols, VIEW_COLUMNS.items()):
                if col.button(title, key=f"{title}_{master_type}"):
                    render_table(master_type_df, title, cols_to_show)
    return True


@dataclass
class FileConfig:
    """파일 설정 정보"""
    valuechain_our_master: str
    codemapping: str

class FileLoader:
    """파일 로딩을 위한 클래스"""
    
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.root_path = yaml_config['ROOT_PATH']
        self.files_config = self._setup_files_config()
    
    def _setup_files_config(self) -> FileConfig:
        """파일 설정 구성"""
        files = self.yaml_config['files']
        return FileConfig(
            valuechain_our_master=f"{self.root_path}/{files['valuechain_our_master']}",
            codemapping=f"{self.root_path}/{files['codemapping']}",
        )
    
    def load_file(self, file_path: str, file_name: str) -> Optional[pd.DataFrame]:
        """단일 파일 로드"""
        if not os.path.exists(file_path):
            st.warning(f"{file_name} 파일이 존재하지 않습니다: {file_path}")
            return None
        
        try:
            extension = os.path.splitext(file_path)[1].lower()
            if extension == '.csv':
                return pd.read_csv(file_path)
            elif extension == '.xlsx':
                return pd.read_excel(file_path)
            elif extension == '.pkl':
                return pd.read_pickle(file_path)
            else:
                st.error(f"{file_name} 파일 형식을 지원하지 않습니다: {extension}")
                return None
        except Exception as e:
            st.error(f"{file_name} 파일 로드 실패: {str(e)}")
            return None
    
    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """모든 파일 로드"""
        files_to_load = {
            'valuechain_our_master': self.files_config.valuechain_our_master,
            'codemapping': self.files_config.codemapping,
        }
        
        loaded_data = {}
        for name, path in files_to_load.items():
            # 확장자가 없는 경우 .csv 추가
            if not os.path.splitext(path)[1]:
                path = path + ".csv"

            df = self.load_file(path, name)
            if df is not None:
                df = df.fillna('')
                loaded_data[name] = df

        return loaded_data

class DashboardManager:
    """대시보드 관리를 위한 클래스"""
    def __init__(self, yaml_config: Dict[str, Any]):
        self.yaml_config = yaml_config
        self.file_loader = FileLoader(yaml_config)
        # self.value_chain_diagram = ValueChainDiagram(yaml_config)
    
    def display_dashboard(self) -> bool:
        """Value Chain 대시보드 표시"""
        try:
            st.title(APP_NAME)
            st.markdown(APP_DESC)
            
            loaded_data = self.file_loader.load_all_files() # 모든 파일 로드

            # load 된 모든 데이터프레임에서 Activity_Seq 컬럼을 문자열로 통일
            for name, df in loaded_data.items():
                if isinstance(df, pd.DataFrame) and 'Activity_Seq' in df.columns:
                    df['Activity_Seq'] = df['Activity_Seq'].astype(str).str.strip().str.replace('.0', '', regex=False)

            our_df = loaded_data['valuechain_our_master']
            cm_df = loaded_data['codemapping']
            cm_df = cm_df[cm_df['MasterType'] == 'Master']

            # DataFrame 정규화 (None 값 처리 및 Arrow 호환성)
            our_df = normalize_dataframe_for_display(our_df)
            cm_df = normalize_dataframe_for_display(cm_df)

            col1, col2 = st.columns([2, 8])
            with col1:
                selected_industry = st.selectbox("Industry를 선택하세요", our_df['Industry'].unique())
            
            our_df = our_df[our_df['Industry'] == selected_industry].copy()

            our_df = our_df.rename(columns={'Our_Master': 'FileName'})
            cm_df = pd.merge(cm_df, our_df, on='FileName', how='left')
            cm_df = cm_df.fillna('')
            df = normalize_dataframe_for_display(cm_df)

            # Activities_Type & Activities별 통계 표시
            Display_File_Stats_by_Activities(df)

            # System별 통계 표시
            Display_File_Stats_by_Systems(df)

            # 마스터 포맷 상세 표시
            st.markdown("#### File Information Detail")
            # st.dataframe(df, use_container_width=False, hide_index=True, height=550, width=1200)

            display_df = df[['Industry', 'Activities_Type', 'Activity_Seq', 'Activities', 'Activities_Kor',
                'Master', 'Master_Kor', 'System', 'FileName']].drop_duplicates()
            display_df = display_df.sort_values(['Industry', 'Activities_Type', 'Activity_Seq', 'FileName']
                , ascending=[False, True, True, True])
            display_df = display_df.rename(columns={'Activity_Seq': 'Seq'})
            st.dataframe(display_df, use_container_width=False, hide_index=True, height=550, width=1200)

            return True
            
        except Exception as e:
            st.error(f"대시보드 표시 중 오류 발생: {str(e)}")
            return False

class FilesInformationApp:
    """Files Information 애플리케이션 메인 클래스"""
    
    def __init__(self):
        self.yaml_config = None
        self.dashboard_manager = None
    
    def initialize(self) -> bool:
        """애플리케이션 초기화"""
        try:
            self.yaml_config = load_yaml_datasense() # YAML 파일 로드
            if self.yaml_config is None:
                st.error("YAML 파일을 로드할 수 없습니다.")
                return False
            
            # set_page_config(self.yaml_config) # 페이지 설정
            
            self.dashboard_manager = DashboardManager(self.yaml_config) # 대시보드 매니저 초기화
            
            return True
            
        except Exception as e:
            st.error(f"애플리케이션 초기화 중 오류 발생: {str(e)}")
            return False
    
    def run(self):
        """애플리케이션 실행"""
        try:
            success = self.dashboard_manager.display_dashboard()
                
        except Exception as e:
            st.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")

def main():
    """메인 함수"""
    app = FilesInformationApp()
    
    if app.initialize():
        app.run()
    else:
        st.error("애플리케이션 초기화 실패")

if __name__ == "__main__":
    main()
