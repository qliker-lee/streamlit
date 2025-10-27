#
# streamlit를 이용한 QDQM Analyzer : Master Information
# 2024. 11. 9.  Qliker
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

from graphviz import Digraph
from dataclasses import dataclass
from typing import Dict, Any, Optional
import plotly.graph_objects as go

SOLUTION_NAME = "Data Sense System"
SOLUTION_KOR_NAME = "데이터 센스 시스템"
APP_NAME = "Data Quality Information"

# CURRENT_DIR = Path(__file__).resolve()
# PROJECT_ROOT = CURRENT_DIR.parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT))

# 현재 파일의 상위 디렉토리를 path에 추가
CURRENT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CURRENT_DIR_PATH)

app_name = re.sub(r'\s*', '', re.sub(r'^\d+_|\.py$', '', os.path.basename(__file__)))
QDQM_ver = '2.0'

# Now import utils after adding to path
from function.Files_FunctionV20 import load_yaml_datasense, set_page_config

from function.Display import (
    create_metric_card,
    display_kpi_metrics
)


#-----------------------------------------------------------------------------------------
# Master KPI 
def Display_Master_KPIs(loaded_data):
    """ Master Statistics KPIs """
    def calculate_master_type_counts(df):
        """MasterType별 파일 수 계산"""
        if 'MasterType' not in df.columns or 'FileName' not in df.columns:
            return {}
        try:
            master_type_counts = df.groupby('MasterType')['FileName'].nunique()
            expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
            
            result = {}
            for master_type in expected_types:
                count = master_type_counts.get(master_type, 0)
                result[master_type] = f"{count:,}"
            return result
        except Exception as e:
            st.error(f"MasterType 계산 중 오류 발생: {str(e)}")
            return {}

    df = loaded_data['filestats']
    if df is None or df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    st.markdown("### File Statistics")
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
        "Code Type #": f"{total_master_types:,}",
        "Work Date": f"{work_date}"
    }

    # 각 메트릭에 대한 색상 정의
    metric_colors = {
        "Code File #": "#1f77b4",
        "Total Record #": "#2ca02c", 
        "Total File Size": "#ff7f0e",
        "Code Type #": "#d62728",
        "Work Date": "#9467bd"
    }

    # 메트릭 표시
    cols = st.columns(len(summary))
    for col, (key, value) in zip(cols, summary.items()):
        color = metric_colors.get(key, "#0072B2")
        col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

    # MasterType별 파일 수
    st.markdown("### Statistics by Code Type")
    master_type_counts = calculate_master_type_counts(df)
    
    if master_type_counts:
        metric_colors = {
            "Master": "#1f77b4",      # 파란색
            "Operation": "#2ca02c",    # 초록색
            "Reference": "#ff7f0e",    # 주황색
            "Attribute": "#d62728",    # 빨간색
            "Common": "#9467bd",       # 보라색
            "Validation": "#8c564b"    # 갈색
        }

        cols = st.columns(len(master_type_counts))
        for col, (key, value) in zip(cols, master_type_counts.items()):
            color = metric_colors.get(key, "#0072B2")
            col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

    return True

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

# #---------------------------------
# # Master KPI Total (임시)
# def Display_Master_KPIs_Total(loaded_data):
#     """ Master Statistics KPIs """
#     def Master_KPIs_Display(loaded_data):
#         df = loaded_data['filestats']
#         if df is None or df.empty:
#             st.warning("데이터가 없습니다.")
#             return
        
#         st.markdown("### Code File Statistics")
#         # KPI 계산
#         total_files = len(df['FileName'].unique()) if 'FileName' in df.columns else 0
#         total_records = df['RecordCnt'].sum() if 'RecordCnt' in df.columns else 0
#         total_filesize = df['FileSize'].sum() if 'FileSize' in df.columns else 0
#         total_master_types = len(df['MasterType'].unique()) if 'MasterType' in df.columns else 0
#         work_date = df['WorkDate'].max() if 'WorkDate' in df.columns else ''

#         if total_records < 1000:
#             total_records_unit = '건'
#         else:
#             total_records = total_records / 10000
#             total_records_unit = '만건'

#         if total_filesize < 1000:
#             total_filesize = total_filesize 
#             total_filesize_unit = 'Bytes'
#         elif total_filesize < 1000000:
#             total_filesize = total_filesize / 1000
#             total_filesize_unit = 'KB'
#         elif total_filesize < 1000000000:
#             total_filesize = total_filesize / 1000000
#             total_filesize_unit = 'MB'
#         else:
#             total_filesize = total_filesize / 1000000000
#             total_filesize_unit = 'GB'        

#         summary = {
#             "Code File #": f"{total_files:,}",
#             "Total Record #": f"{total_records:,.0f} {total_records_unit}",
#             "Total File Size": f"{total_filesize:,.0f} {total_filesize_unit}",
#             "Code Type #": f"{total_master_types:,}",
#             "Work Date": f"{work_date}"
#         }

#         # 각 메트릭에 대한 색상 정의
#         metric_colors = {
#             "Code File #": "#1f77b4",
#             "Total Record #": "#2ca02c", 
#             "Total File Size": "#ff7f0e",
#             "Code Type #": "#d62728",
#             "Work Date": "#9467bd"
#         }

#         # 메트릭 표시
#         cols = st.columns(len(summary))
#         for col, (key, value) in zip(cols, summary.items()):
#             color = metric_colors.get(key, "#0072B2")
#             col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

#         # MasterType별 파일 수
#         st.markdown("### Statistics by Code Type")
#         master_type_counts = calculate_master_type_counts(df)
        
#         if master_type_counts:
#             # 각 메트릭에 대한 색상 정의
#             metric_colors = {
#                 "Master": "#1f77b4",      # 파란색
#                 "Operation": "#2ca02c",    # 초록색
#                 "Reference": "#ff7f0e",    # 주황색
#                 "Attribute": "#d62728",    # 빨간색
#                 "Common": "#9467bd",       # 보라색
#                 "Validation": "#8c564b"    # 갈색
#             }

#             # 메트릭 표시
#             cols = st.columns(len(master_type_counts))
#             for col, (key, value) in zip(cols, master_type_counts.items()):
#                 color = metric_colors.get(key, "#0072B2")
#                 col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

#         return True
    
#     def Code_Definition(loaded_data): 
#         st.markdown("#### Code Type Definition")
#         df = loaded_data['mastertypedefinition']
#         df = df.set_index('No')
#         df = df.rename(columns={'MasterType': 'Code Type'})
#         df = df.drop(columns=['대안명칭']) # '대안명칭', '예시']
#         st.table(data=df)

    
#     def Display_MasterStats_Detail(loaded_data):
#         st.markdown("#### Master Code File Information")
#         loaded_masterformat_df = loaded_data['masterformat']
#         loaded_mastercharacter_df = loaded_data['mastercharacter']

#         character_stats_df = loaded_mastercharacter_df.groupby(['FileName', 'ColumnName']).agg({
#             'UnicodeCnt': 'sum',
#             'BrokenKoreanCnt': 'sum',
#             'ChineseCnt': 'sum',
#             'SpecialCnt': 'sum'
#         }).reset_index()

#         df = pd.merge(loaded_masterformat_df, character_stats_df, on=['FileName', 'ColumnName'], how='left')
#         df = df[['MasterType', 'ColumnName', 'UnicodeCnt', 'BrokenKoreanCnt', 'ChineseCnt', 'SpecialCnt', 
#                  'FormatCnt', 'Format', 'MinString', 'MaxString', 'ModeString']]

#         # MasterType들
#         expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
#         tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(expected_types)
#         for master_type, tab in zip(expected_types, [tab1, tab2, tab3, tab4, tab5, tab6]):
#             master_type_df = df[df['MasterType'] == master_type]
#             if master_type_df is not None and not master_type_df.empty:
#                 with tab:
#                     master_type_df = master_type_df.reset_index(drop=True)
#                     st.dataframe(data=master_type_df, width=1400, hide_index=True)
#             else:
#                 st.warning(f"{master_type} 통계 정보 파일을 로드할 수 없습니다.")

#     def Display_FileCharacter_Detail(loaded_data):
#         def display_kpi_metrics(summary, metric_colors, title):
#             """KPI 메트릭 표시 공통 함수"""
#             st.markdown(f"#### {title}")
            
#             # 메트릭 표시
#             cols = st.columns(len(summary))
#             for col, (key, value) in zip(cols, summary.items()):
#                 color = metric_colors.get(key, "#0072B2")  # 기본 색상
#                 col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

#         def Display_Character_KPIs(df):
#             """Reference Code 이상 문자 정보 KPI"""
            
#             # KPI 계산
#             file_cnt = len(df['FileName'].unique())
#             unicode_cnt = len(df[df['UnicodeCnt'] > 0])    
#             broken_korean_cnt = len(df[df['BrokenKoreanCnt'] > 0])
#             chinese_cnt = len(df[df['ChineseCnt'] > 0])
#             special_cnt = len(df[df['SpecialCnt'] > 0])

#             summary = {
#                 "Total Files #": f"{file_cnt:,}", 
#                 "Unicode": f"{unicode_cnt:,}",  
#                 "Broken Korean": f"{broken_korean_cnt:,}",
#                 "Chinese": f"{chinese_cnt:,}",
#                 "Special": f"{special_cnt:,}",
#             }
            
#             # 각 메트릭에 대한 색상 정의
#             metric_colors = {
#                 "Total Files #": "#1f77b4",      # 파란색
#                 "Unicode": "#2ca02c",            # 초록색
#                 "Broken Korean": "#ff7f0e",      # 주황색
#                 "Chinese": "#d62728",            # 빨간색
#                 "Special": "#9467bd"             # 보라색
#             }

#             display_kpi_metrics(summary, metric_colors, "Code Character Information")

#         loaded_masterformat_df = loaded_data['fileformat']
#         mastertype_df = loaded_masterformat_df[['FileName', 'ColumnName', 'MasterType']]

#         char_df = loaded_data['mastercharacter']
#         if char_df is not None and not mastertype_df.empty:
#             df = pd.merge(char_df, mastertype_df, on=['FileName', 'ColumnName'], how='inner')
#             Display_Character_KPIs(char_df)

#             df = df[(df['UnicodeCnt'] > 0) | (df['BrokenKoreanCnt'] > 0) | (df['ChineseCnt'] > 0)]
                                    
#             #  컬럼 순서 변경
#             df = df[['MasterType', 'ColumnName', 'UnicodeCnt', 'UnicodeChars', 
#                         'UnicodeOrdValues', 'UnicodeOrdInt', 'BrokenKoreanCnt', 'BrokenKoreanChars', 
#                         'ChineseCnt', 'ChineseChars', 'SpecialCnt', 'FileName']]
            
#             summary_df = df.groupby('MasterType').agg({
#                 'UnicodeCnt': 'sum',
#                 'BrokenKoreanCnt': 'sum',
#                 'ChineseCnt': 'sum',
#                 'SpecialCnt': 'sum'
#             }).reset_index()
            
#             st.markdown(f"#### Code Character Information Detail")
#             expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
#             tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(expected_types)
#             for master_type, tab in zip(expected_types, [tab1, tab2, tab3, tab4, tab5, tab6]):
#                 master_type_df = df[df['MasterType'] == master_type]
#                 if master_type_df is not None and not master_type_df.empty:
#                     with tab:
#                         master_type_df = master_type_df.reset_index(drop=True)
#                         st.dataframe(data=master_type_df, width=1400, hide_index=True)

#         return True
    
#     def Code_Usage_KPI(loaded_data):
#         st.write("-"*100)
#         st.markdown("## Code Usage Information")
        
#         expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
#         df = loaded_data['fieldinfo']
#         if df is None or df.empty:
#             st.warning("fieldInfo 데이터가 없습니다.")
#             return
        
#         try:
#             #--------------------------------------------------------------------------
#             master_cnt = df[df['MasterType'] == 'Master'][['FileName']].drop_duplicates().shape[0]
#             operation_cnt = df[df['MasterType'] == 'Operation'][['FileName']].drop_duplicates().shape[0]
#             attribute_cnt = df[df['MasterType'] == 'Attribute'][['FileName']].drop_duplicates().shape[0]
#             common_cnt = df[df['MasterType'] == 'Common'][['FileName']].drop_duplicates().shape[0]
#             reference_cnt = df[df['MasterType'] == 'Reference'][['FileName']].drop_duplicates().shape[0]
#             validation_cnt = df[df['MasterType'] == 'Validation'][['FileName']].drop_duplicates().shape[0]

#             summary = {
#                 "Master": f"{master_cnt:,}",
#                 "Operation": f"{operation_cnt:,}",
#                 "Attribute": f"{attribute_cnt:,}",
#                 "Common": f"{common_cnt:,}",
#                 "Reference": f"{reference_cnt:,}",
#                 "Validation": f"{validation_cnt:,}",
#             }

#             metric_colors = {
#                 "Master": "#1f77b4",
#                 "Operation": "#2ca02c",
#                 "Attribute": "#ff7f0e",
#                 "Common": "#d62728",
#                 "Reference": "#9467bd",
#                 "Validation": "#330000"
#             }
#             display_kpi_metrics(summary, metric_colors, "Code Type Usage by File")

#             #--------------------------------------------------------------------------
#             master_cnt = df[df['MasterType'] == 'Master'][['FileName', 'ColumnName']].drop_duplicates().shape[0]
#             operation_cnt = df[df['MasterType'] == 'Operation'][['FileName', 'ColumnName']].drop_duplicates().shape[0]
#             attribute_cnt = df[df['MasterType'] == 'Attribute'][['FileName', 'ColumnName']].drop_duplicates().shape[0]
#             common_cnt = df[df['MasterType'] == 'Common'][['FileName', 'ColumnName']].drop_duplicates().shape[0]
#             reference_cnt = df[df['MasterType'] == 'Reference'][['FileName', 'ColumnName']].drop_duplicates().shape[0]
#             validation_cnt = df[df['MasterType'] == 'Validation'][['FileName', 'ColumnName']].drop_duplicates().shape[0]

#             summary = {
#                 "Master": f"{master_cnt:,}",
#                 "Operation": f"{operation_cnt:,}",
#                 "Attribute": f"{attribute_cnt:,}",
#                 "Common": f"{common_cnt:,}",
#                 "Reference": f"{reference_cnt:,}",
#                 "Validation": f"{validation_cnt:,}",
#             }

#             metric_colors = {
#                 "Master": "#1f77b4",
#                 "Operation": "#2ca02c",
#                 "Attribute": "#ff7f0e",
#                 "Common": "#d62728",
#                 "Reference": "#9467bd",
#                 "Validation": "#330000"
#             }
#             display_kpi_metrics(summary, metric_colors, "Code Type Usage by Column")

#             #--------------------------------------------------------------------------
#             st.markdown(f"#### Code별 사용 횟수 집계 ")
#             tmp_df = df.copy()
#             tmp_df = tmp_df[tmp_df['MasterType'].isin(expected_types)]

#             # 각 집계를 개별적으로 수행하여 컬럼명 중복 문제 해결
#             tmp_file_df = tmp_df.drop_duplicates(subset=['MasterType', 'MasterCode', 'FileName'])
#             summary_file_df = tmp_file_df.groupby(['MasterType', 'MasterCode']).agg({
#                 'FileName': 'count',
#             }).reset_index()
#             summary_file_df = summary_file_df.rename(columns={'FileName': 'File_Usage'})

#             tmp_column_df = tmp_df.drop_duplicates(subset=['MasterType', 'MasterCode', 'FileName', 'ColumnName'])
#             summary_column_df = tmp_column_df.groupby(['MasterType', 'MasterCode']).agg({
#                 'ColumnName': 'count',
#             }).reset_index()
#             summary_column_df = summary_column_df.rename(columns={'ColumnName': 'Column_Usage'})
            
#             # Matched(%) 통계를 개별적으로 계산
#             matched_stats = tmp_df.groupby(['MasterType', 'MasterCode'])['Matched(%)'].agg([
#                 ('Matched(%)_Min', 'min'),
#                 ('Matched(%)_Max', 'max'),
#                 ('Matched(%)_Mean', 'mean'),
#             ]).reset_index()
            
#             # 두 데이터프레임을 병합
#             summary_df = summary_file_df.merge(summary_column_df, on=['MasterType', 'MasterCode'], how='left')
#             summary_df = summary_df.merge(matched_stats, on=['MasterType', 'MasterCode'], how='left')
#             st.dataframe(data=summary_df, width=1400, height=600, hide_index=True)  
            
#         except Exception as e:
#             st.error(f"Code Usage KPI 표시 중 오류 발생: {str(e)}")
#             return None  
#         return summary_df

#     def Code_Verify(loaded_data):
#         st.markdown("## All Code Verify Information")
        
#         expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
#         try:

#             #--------------------------------------------------------------------------
#             master_format_df = loaded_data['masterformat']
#             master_char_df = loaded_data['mastercharacter']
#             tmp_master_char_df = master_char_df.groupby(['FileName', 'ColumnName']).agg({
#                 'UnicodeCnt': 'sum',
#                 'BrokenKoreanCnt': 'sum',
#                 'ChineseCnt': 'sum',
#                 'SpecialCnt': 'sum',
#             }).reset_index()
#             master_format_df = master_format_df.merge(tmp_master_char_df, on=['FileName', 'ColumnName'], how='left')
#             master_format_df = master_format_df.rename(columns={'ColumnName': 'MasterCode'})

#             for index, row in master_format_df.iterrows():
#                 master_format_df.loc[index, '한글'] = '있음' if row['HasKor'] > 0 else ''
#                 master_format_df.loc[index, '공백'] = '있음' if row['HasBlank'] > 0 else ''
#                 master_format_df.loc[index, '특수문자'] = '있음' if row['SpecialCnt'] > 0 else ''
#                 master_format_df.loc[index, '유니코드'] = '있음' if row['UnicodeCnt'] > 0 else ''
#                 master_format_df.loc[index, '미완성한글'] = '있음' if row['BrokenKoreanCnt'] > 0 else ''
#                 master_format_df.loc[index, '한자'] = '있음' if row['ChineseCnt'] > 0 else ''
#                 master_format_df.loc[index, '특수문자'] = '있음' if row['SpecialCnt'] > 0 else ''
#                 if ((row['Format(%)'] > 95 and row['Format(%)'] < 100) or \
#                     (row['UnicodeCnt'] > 0 or row['BrokenKoreanCnt'] > 0 or row['ChineseCnt'] > 0)):
#                     master_format_df.loc[index, '점검대상'] = '점검'
#                 else:
#                     master_format_df.loc[index, '점검대상'] = ''


#             display_df = master_format_df[['MasterType', 'MasterCode', 'ValueCnt', 'Format', 'FormatCnt', 
#             'Format(%)', 'Format2nd', '점검대상', '한글', '공백', '특수문자', '유니코드', '미완성한글', '한자']]

#             display_df = display_df.sort_values('MasterCode', ascending=True)

#             display_df = display_df.rename(
#                 columns={'ValueCnt': 'Value #', 
#                          'Format': 'ValueType', 
#                          'FormatCnt': 'ValueType #', 
#                          'Format(%)': 'ValueType(%)', 
#                          'Format2nd': 'ValueType2nd'})

#             # MasterType들
#             expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
#             tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(expected_types)
#             for master_type, tab in zip(expected_types, [tab1, tab2, tab3, tab4, tab5, tab6]):
#                 df = display_df[display_df['MasterType'] == master_type]
#                 if df is not None and not df.empty:
#                     with tab:
#                         df = df.reset_index(drop=True)
#                         st.dataframe(data=df, width=1400, height=600, hide_index=True)

#         except Exception as e:
#             st.error(f"Code Verify 표시 중 오류 발생: {str(e)}")
#             return False
#         return True

#     def Usage_Code_Verify(loaded_data, usage_df):
#         st.markdown("## Code Usage Verify Information")
        
#         expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
#         df = loaded_data['fieldinfo']
#         if df is None or df.empty:
#             st.warning("fieldInfo 데이터가 없습니다.")
#             return False
#         if usage_df is None or usage_df.empty:
#             st.warning("Code Usage 데이터가 없습니다.")
#             return False
#         try:
#             #--------------------------------------------------------------------------
#             master_format_df = loaded_data['masterformat']
#             master_char_df = loaded_data['mastercharacter']
#             tmp_master_char_df = master_char_df.groupby(['FileName', 'ColumnName']).agg({
#                 'UnicodeCnt': 'sum',
#                 'BrokenKoreanCnt': 'sum',
#                 'ChineseCnt': 'sum',
#                 'SpecialCnt': 'sum',
#             }).reset_index()
#             master_format_df = master_format_df.merge(tmp_master_char_df, on=['FileName', 'ColumnName'], how='left')
#             master_format_df = master_format_df.rename(columns={'ColumnName': 'MasterCode'})

#             master_format_df = master_format_df.merge(usage_df, on=['MasterType', 'MasterCode'], how='inner')

#             for index, row in master_format_df.iterrows():
#                 master_format_df.loc[index, '한글'] = '있음' if row['HasKor'] > 0 else ''
#                 master_format_df.loc[index, '공백'] = '있음' if row['HasBlank'] > 0 else ''
#                 master_format_df.loc[index, '특수문자'] = '있음' if row['SpecialCnt'] > 0 else ''
#                 master_format_df.loc[index, '유니코드'] = '있음' if row['UnicodeCnt'] > 0 else ''
#                 master_format_df.loc[index, '미완성한글'] = '있음' if row['BrokenKoreanCnt'] > 0 else ''
#                 master_format_df.loc[index, '한자'] = '있음' if row['ChineseCnt'] > 0 else ''
#                 master_format_df.loc[index, '특수문자'] = '있음' if row['SpecialCnt'] > 0 else ''
#                 if ((row['Format(%)'] > 95 and row['Format(%)'] < 100) or \
#                     (row['UnicodeCnt'] > 0 or row['BrokenKoreanCnt'] > 0 or row['ChineseCnt'] > 0)):
#                     master_format_df.loc[index, '점검대상'] = '점검'
#                 else:
#                     master_format_df.loc[index, '점검대상'] = ''


#             display_df = master_format_df[['MasterType', 'MasterCode', 'ValueCnt', 'Format', 'FormatCnt', 
#             'Format(%)', 'Column_Usage', '점검대상', '한글', '공백', '특수문자', '유니코드', '미완성한글', '한자', 
#              'Matched(%)_Min', 'Matched(%)_Max', 'Matched(%)_Mean']]

#             display_df = display_df.sort_values('MasterCode', ascending=True)

#             display_df = display_df.rename(
#                 columns={'ValueCnt': 'Value #', 
#                          'Format': 'ValueType', 
#                          'FormatCnt': 'ValueType #', 
#                          'Format(%)': 'ValueType(%)', 
#                          'Format2nd': 'ValueType2nd'})

#             # MasterType들
#             expected_types = ['Master', 'Operation', 'Attribute', 'Common', 'Reference', 'Validation']
#             tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(expected_types)
#             for master_type, tab in zip(expected_types, [tab1, tab2, tab3, tab4, tab5, tab6]):
#                 df = display_df[display_df['MasterType'] == master_type]
#                 if df is not None and not df.empty:
#                     with tab:
#                         df = df.reset_index(drop=True)
#                         st.dataframe(data=df, width=1400, height=600, hide_index=True)

#         except Exception as e:
#             st.error(f"Code Verify 표시 중 오류 발생: {str(e)}")
#             return False
#         return True
#     # 마스터 정보 표시
#     try:
#         result = Master_KPIs_Display(loaded_data)
#         result = Display_MasterFormat_Detail(loaded_data)
#     except Exception as e:
#         st.error(f"Master KPIs 표시 중 오류 발생: {str(e)}")
#         return False

#     return True


# def create_metric_card(value, label, color="#0072B2"):
#     """메트릭 카드 HTML 생성"""
#     return f"""
#         <div style="text-align: center; padding: 1.2rem; background-color: #FFFFFF; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
#             <div style="color: {color}; font-size: 2em; font-weight: bold;">{value}</div>
#             <div style="color: #404040; font-size: 1em; margin-top: 0.5rem;">{label}</div>
#         </div>
#     """


@dataclass
class FileConfig:
    """파일 설정 정보"""
    fileformat: str
    filestats: str
    fileformatmapping: str

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
            fileformat=f"{self.root_path}/{files['fileformat']}",
            filestats=f"{self.root_path}/{files['filestats']}",
            fileformatmapping=f"{self.root_path}/{files['fileformatmapping']}",
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
            'fileformat': self.files_config.fileformat,
            'filestats': self.files_config.filestats,
            'fileformatmapping': self.files_config.fileformatmapping,
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
            
            loaded_data = self.file_loader.load_all_files() # 모든 파일 로드
           
            # 마스터 통계 표시
            Display_Master_KPIs(loaded_data)

            # 마스터 포맷 상세 표시
            Display_MasterFormat_Detail(loaded_data)

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
            
            set_page_config(self.yaml_config) # 페이지 설정
            
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
