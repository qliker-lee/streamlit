# -*- coding: utf-8 -*-
# DS_13_Code Relationship Analyzer.py
# 코드 관계 분석 프로그램은 파일 형식 매핑 결과와 룰 매핑 결과를 기반으로 코드 관계 분석을 수행합니다.
# 2025.12.24 Qliker

import pandas as pd
import numpy as np
import os
import sys
import logging

from pathlib import Path
import traceback
from typing import Dict, Any, Iterable, Optional, Sequence, List
from multiprocessing import Pool, cpu_count, Manager
from itertools import combinations

# ---------------------- 전역 기본값 ----------------------
DEBUG_MODE = True   # 디버그 모드 여부 (True: 디버그 모드, False: 운영 모드)

OUTPUT_FILE_NAME = 'CodeMapping'       # 코드 관계 분석 결과 파일 이름
OUTPUT_FILEFORMAT = 'FileFormatMapping' # 파일 형식 매핑 결과 파일 이름
OUTPUT_FILENUMERIC = 'FileNumericStats' # 숫자 형식 통계 결과 파일 이름

MATCH_RATE_THRESHOLD = 20 # 매핑 결과 중 MatchRate(%) 20% 이상인 레코드만 선택 (기본값: 20%)

# ---------------------- 함수 선언 ----------------------
def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 헤더를 정리하는 함수
    """
    out = df.copy()
    out.columns = [str(c).replace('\ufeff', '').strip() for c in out.columns]
    return out

# ---------------------------------------------------------
# 1. 워커 함수 (CPU 코어별로 독립 실행되는 비교 로직) internal mapping 에서 사용
# ---------------------------------------------------------
def compare_columns_worker(task_info):
    """
    각 코어에서 실행될 독립적인 비교 함수
    task_info: (a_meta, b_meta, a_set, b_set)
    """
    a, b, a_set, b_set = task_info
    
    if not a_set or not b_set:
        return None

    # 교집합 연산 (Set 연산은 파이썬에서 가장 빠름)
    intersection = a_set.intersection(b_set)
    compare_count = len(intersection)
    total_count = len(a_set)
    
    match_rate = round(compare_count / total_count * 100, 2) if total_count > 0 else 0.0

    # 임계치(예: 10%) 미만은 결과에서 제외하여 메모리 절약
    if match_rate < 10.0:
        return None

    return {
        "FilePath": a['FilePath'], "FileName": a['FileName'], "ColumnName": a['ColumnName'],
        "MasterType": "Internal",
        "MasterFilePath": b['FilePath'], "MasterFile": b['FileName'],
        "ReferenceMasterType": "Internal", "MasterColumn": b['ColumnName'],
        "CompareLength": a.get('CompareLength', 0),
        "CompareCount": compare_count, "SourceCount": total_count, "MatchRate(%)": match_rate
    }

#-----------------------------------------------------------------------------------------------------    
def Expand_Format(Source_df, Mode='Reference') -> pd.DataFrame:
    """    Source_df의 상위 3개 포맷(Format_1~3)을 행으로 펼쳐서 반환.    """
    try:

        # 1) 전처리
        s_df = Source_df.copy().rename(columns={
            'Format':        'Format_1',
            'Format2nd':     'Format_2',
            'Format3rd':     'Format_3',
            'FormatMin':     'FormatMin_1',
            'FormatMax':     'FormatMax_1',
            'FormatMedian':   'FormatMedian_1',
            'Format2ndMin':  'FormatMin_2',
            'Format2ndMax':  'FormatMax_2',
            'Format2ndMedian': 'FormatMedian_2',
            'Format3rdMin':  'FormatMin_3',
            'Format3rdMax':  'FormatMax_3',
            'Format3rdMedian': 'FormatMedian_3',
            'FormatValue':   'FormatValue_1',
            'Format2ndValue':'FormatValue_2',
            'Format3rdValue':'FormatValue_3',
            'Format(%)':     'Format(%)_1',
            'Format2nd(%)':  'Format(%)_2',
            'Format3rd(%)':  'Format(%)_3',
        })

        # 2) i=1..3 별로 분리 → 표준 컬럼명으로 통일 → 붙이기
        frames = []
        for i in (1, 2, 3):
            cols_i = [
                'FilePath', 'FileName', 'ColumnName', 'DetailDataType',  'MasterType', 'CompareLength', 'FormatCnt', 'UniqueCnt',
                f'Format_{i}', f'FormatMin_{i}', f'FormatMax_{i}', f'FormatMedian_{i}', 
                f'FormatValue_{i}', f'Format(%)_{i}'
            ]

            # 존재하는 컬럼만 선택
            available_cols = [c for c in cols_i if c in s_df.columns]
            if not available_cols:
                continue
                
            df_i = s_df[available_cols].copy()
            
            # 컬럼명 변경 (존재하는 컬럼만)
            rename_dict = {}
            if f'Format_{i}' in df_i.columns:
                rename_dict[f'Format_{i}'] = 'Format'
            if f'FormatMin_{i}' in df_i.columns:
                rename_dict[f'FormatMin_{i}'] = 'FormatMin'
            if f'FormatMax_{i}' in df_i.columns:
                rename_dict[f'FormatMax_{i}'] = 'FormatMax'
            if f'FormatMedian_{i}' in df_i.columns:
                rename_dict[f'FormatMedian_{i}'] = 'FormatMedian'
            if f'FormatValue_{i}' in df_i.columns:
                rename_dict[f'FormatValue_{i}'] = 'FormatValue'
            if f'Format(%)_{i}' in df_i.columns:
                rename_dict[f'Format(%)_{i}'] = 'Format(%)'
            
            if rename_dict:
                df_i = df_i.rename(columns=rename_dict)

            df_i['MatchNo'] = i

            # 빈/결측 포맷 제거 (Format 컬럼이 있는 경우만)
            if 'Format' in df_i.columns and not df_i.empty:
                format_series = df_i['Format']
                # Series인지 확인 (단일 컬럼 선택은 항상 Series 반환)
                if isinstance(format_series, pd.Series):
                    mask = format_series.notna() & (format_series.astype(str).str.strip() != '')
                    df_i = df_i[mask]
                # 이상 케이스: DataFrame이 반환된 경우는 스킵
                elif isinstance(format_series, pd.DataFrame):
                    continue

            # 숫자형 정리
            for col in ('FormatValue', 'Format(%)', 'CompareLength'):
                if col in df_i.columns:
                    df_i[col] = pd.to_numeric(df_i[col], errors='coerce')

            frames.append(df_i)

        if not frames:
            return pd.DataFrame(columns=[
                'FilePath', 'FileName', 'ColumnName', 'DetailDataType', 'MasterType', 'FormatCnt', 'UniqueCnt', 'MatchNo',
                'Format', 'FormatMin', 'FormatMax', 'FormatMedian', 'FormatValue', 'Format(%)', 'CompareLength'
            ])

        result_df = pd.concat(frames, ignore_index=True)

        # 정렬 & 중복 제거(선택)
        result_df = (result_df
                     .drop_duplicates()
                     .sort_values(['FilePath','FileName','ColumnName','MasterType','Format(%)'], ascending=[True, True, True, True, False])
                     .reset_index(drop=True))

        return result_df

    except Exception as e:
        print(f"전체 처리 중 오류 발생: {e}")
        raise


def Combine_Format(source_df, reference_df):
    """
    source_df와 reference_df를 조합하여 반환 (기술적 최적화 버전)
    """
    try:
        # 제외할 타입 리스트 (set으로 변환하여 검색 속도 향상)
        except_types = {
            'Time', 'Timestamp', 'Date', 'DateTime', 'DATECHAR', 'TIME', 'TIMESTAMP', 
            'DATE', 'DATETIME', 'YEAR', 'YEARMONTH', 'YYMMDD', 'LATITUDE', 'LONGITUDE', 
            'TEL', 'CELLPHONE', 'ADDRESS', 'Alpha_Flag', 'Num_Flag', 'YN_Flag', 
            'NUM_Flag', 'KOR_Flag', 'KOR_Name'
        }

        # 1. 필터링 최적화: 불필요한 copy()를 줄이고 필터링 후 필요한 컬럼만 선택
        if 'DetailDataType' in source_df.columns:
            s_df = source_df[~source_df['DetailDataType'].isin(except_types)]
        else:
            s_df = source_df

        if 'DetailDataType' in reference_df.columns:
            r_df = reference_df[~reference_df['DetailDataType'].isin(except_types)]
        else:
            r_df = reference_df

        # 2. Merge 전 필요한 컬럼만 추출 및 Rename (메모리 절약)
        rename_map = {
            'FilePath': 'MasterFilePath', 'FileName': 'MasterFile',
            'MasterType': 'ReferenceMasterType', 'ColumnName': 'MasterColumn',
            'FormatCnt': 'MasterFormatCnt', 'FormatMin': 'MasterMin',
            'FormatMax': 'MasterMax', 'FormatMedian': 'MasterMedian',
            'FormatValue': 'MasterValue', 'Format(%)': 'Master(%)',
            'UniqueCnt': 'MasterUniqueCnt'
        }
        
        # r_df에서 필요한 컬럼만 골라내며 바로 이름을 바꿉니다.
        r_cols = ['Format'] + list(rename_map.keys())
        r_df = r_df[r_cols].rename(columns=rename_map)

        # 3. Merge 및 중복 제거
        result_df = pd.merge(s_df, r_df, on='Format', how='left')
        result_df = result_df.dropna(subset=['MasterFile'])
        result_df = result_df.drop_duplicates(['FilePath', 'FileName', 'ColumnName', 'MasterFile', 'MasterColumn'])

        # 4. 숫자형 변환 및 결측치 처리 (Vectorized fillna)
        num_cols = [
            'FormatCnt', 'FormatValue', 'UniqueCnt', 'CompareLength',
            'MasterFormatCnt', 'MasterValue', 'MasterUniqueCnt'
        ]
        for c in num_cols:
            if c in result_df.columns:
                result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0)

        # 5. MasterCompareLength 계산 최적화
        # .str 연산은 무거우므로 결측치 먼저 처리 후 마지막 글자 추출
        mcol_series = result_df['MasterColumn'].astype(str)
        last_char = mcol_series.str[-1]
        result_df['MasterCompareLength'] = np.where(last_char.str.isdigit(), last_char, '0')

        # 6. 플래그 계산 (불필요한 Series 생성을 피하고 numpy 연산 활용)
        # result_df['Format']이 object일 수 있으므로 str.len() 연산 최적화
        fmt_len = result_df['Format'].astype(str).str.len()

        f0 = result_df['FormatMedian'].between(result_df['MasterMin'], result_df['MasterMax'])
        f1 = fmt_len > 1
        f2 = result_df['FormatCnt'] < result_df['MasterValue']
        f3 = ~( (result_df['FormatCnt'] >= result_df['MasterFormatCnt'] * 1.5) & 
                (result_df['FormatCnt'] >= 5) & 
                (result_df['MasterCompareLength'] == '0') )
        f5 = result_df['UniqueCnt'] >= 10
        f6 = result_df['FormatValue'] >= 10
        f8 = ~( (result_df['FilePath'] == result_df['MasterFilePath']) & 
                (result_df['FileName'] == result_df['MasterFile']) & 
                (result_df['ColumnName'] == result_df['MasterColumn']) )

        # 7. 최종 결과 할당 (bool을 int로 바로 변환)
        result_df['Match_Flag']  = f0.astype(int)
        result_df['Match_Flag1'] = f1.astype(int)
        result_df['Match_Flag2'] = f2.astype(int)
        result_df['Match_Flag3'] = f3.astype(int)
        result_df['Match_Flag4'] = 1
        result_df['Match_Flag5'] = f5.astype(int)
        result_df['Match_Flag6'] = f6.astype(int)
        result_df['Match_Flag7'] = 1
        result_df['Match_Flag8'] = f8.astype(int)

        # Final_Flag 연산 (논리 연산 & 가 산술 곱셈보다 빠름)
        # 모든 플래그가 1이어야 하므로 & 연산자를 사용합니다.
        result_df['Final_Flag'] = (
            result_df['Match_Flag'] & result_df['Match_Flag2'] & 
            result_df['Match_Flag3'] & result_df['Match_Flag5'] & 
            result_df['Match_Flag6'] & result_df['Match_Flag8']
        ).astype(int)

        return result_df

    except Exception as e:
        print(f"조합 처리 중 오류 발생: {e}")
        raise

def Combine_Format_old(source_df, reference_df):
    """    source_df와 reference_df를 조합하여 반환.    """
    try:
        s_df = source_df.copy()

        except_detail_data_types = ['Time', 'Timestamp', 'Date', 'DateTime', 'DATECHAR', 'TIME', 'TIMESTAMP', 
            'DATE', 'DATETIME', 'YEAR', 'YEARMONTH', 'YYMMDD', 'LATITUDE', 'LONGITUDE', 'TEL', 'CELLPHONE', 'ADDRESS',
            'Alpha_Flag', 'Num_Flag', 'YN_Flag', 'NUM_Flag', 'KOR_Flag', 'KOR_Name']

        # DetailDataType 컬럼이 있는 경우만 필터링
        if 'DetailDataType' in s_df.columns:
            s_df = s_df[~s_df['DetailDataType'].isin(except_detail_data_types)].copy()
        r_df = reference_df.copy()
        if 'DetailDataType' in r_df.columns:
            r_df = r_df[~r_df['DetailDataType'].isin(except_detail_data_types)].copy()

        r_df = r_df[['FilePath', 'FileName', 'MasterType', 'ColumnName', 'FormatCnt', 'Format', 'FormatMin', 'FormatMax', 
                     'FormatMedian', 'FormatValue', 'Format(%)', 'UniqueCnt']].copy().rename(columns={
            'FilePath': 'MasterFilePath',
            'FileName': 'MasterFile',
            'MasterType': 'ReferenceMasterType',
            'ColumnName': 'MasterColumn',
            'FormatCnt': 'MasterFormatCnt',
            'Format': 'Format',
            'FormatMin': 'MasterMin',
            'FormatMax': 'MasterMax',
            'FormatMedian': 'MasterMedian',
            'FormatValue': 'MasterValue', 
            'Format(%)': 'Master(%)',
            'UniqueCnt': 'MasterUniqueCnt',
            # 'CompareLength': 'MasterCompareLength',
        })

        result_df = pd.merge(s_df, r_df, on=['Format'], how='left')
        result_df = result_df[result_df['MasterFile'].notna()]
        result_df = result_df.drop_duplicates(['FilePath','FileName','ColumnName','MasterFile','MasterColumn'])

        # --- 숫자형 컬럼 일괄 변환 ---
        num_cols = [
            'FormatCnt','FormatValue','UniqueCnt','CompareLength',
            'MasterFormatCnt','MasterValue','MasterUniqueCnt'
        ]
        for c in num_cols:
            if c in result_df.columns:
                result_df[c] = pd.to_numeric(result_df[c], errors='coerce')

                # 결측 기본값 (비교 안전용)
        result_df['FormatCnt']        = result_df['FormatCnt'].fillna(0)
        result_df['FormatValue']      = result_df['FormatValue'].fillna(0)
        result_df['UniqueCnt']        = result_df['UniqueCnt'].fillna(0)
        result_df['MasterFormatCnt']  = result_df['MasterFormatCnt'].fillna(0)
        result_df['MasterValue']      = result_df['MasterValue'].fillna(0)
        result_df['MasterUniqueCnt']  = result_df['MasterUniqueCnt'].fillna(0)

        # --- MasterCompareLength 계산 (MasterColumn 끝자리가 숫자면 사용, 아니면 "0") ---
        mcol_str = result_df['MasterColumn'].astype(str)
        last_char = mcol_str.str[-1].fillna('')
        result_df['MasterCompareLength'] = np.where(last_char.str.isdigit(), last_char, '0')

        # --- 플래그 계산(벡터화) ---
        # 0) 포맷 중앙값이 마스터 범위 안에 있는가
        flag0 = result_df['FormatMedian'].ge(result_df['MasterMin']) & result_df['FormatMedian'].le(result_df['MasterMax'])

        # 1) 포맷 길이가 1 초과인가 (문자열 길이 기준)
        flag1 = result_df['Format'].astype(str).str.len().gt(1)

        # 2) 소스 포맷 카운트가 마스터 기준값보다 작은가 (작아야 1)
        flag2 = result_df['FormatCnt'].lt(result_df['MasterValue'])

        # 3) 과도한 포맷 카운트(=마스터 1.5배 이상 & 5 이상)인데 MasterCompareLength가 0이면 탈락
        flag3 = ~((result_df['FormatCnt'] >= result_df['MasterFormatCnt']*1.5) & (result_df['FormatCnt'] >= 5) & (result_df['MasterCompareLength'] == '0') )
 
        flag4 = pd.Series(True, index=result_df.index)  # 4) 항상 1 (유지)
 
        flag5 = result_df['UniqueCnt'].ge(10)  # 5) 유니크가 10 미만이면 탈락

        flag6 = result_df['FormatValue'].ge(10)  # 6) 포맷값이 10 미만이면 탈락

        # 7) 소스 유니크가 마스터보다 크면서 MasterCompareLength가 0이면 탈락
        flag7 = ~( (result_df['UniqueCnt'] > result_df['MasterUniqueCnt']) & (result_df['MasterCompareLength'] == '0') )

        # 8) FilePath = MasterFilePath 이고 FileName = MasterFile 이고 ColumnName = MasterColumn 이면 탈락
        flag8 = ~( (result_df['FilePath'] == result_df['MasterFilePath']) & (result_df['FileName'] == result_df['MasterFile']) & (result_df['ColumnName'] == result_df['MasterColumn']) )

        # 최종 플래그
        result_df['Match_Flag']  = flag0.astype(int)
        result_df['Match_Flag1'] = flag1.astype(int)
        result_df['Match_Flag2'] = flag2.astype(int)
        result_df['Match_Flag3'] = flag3.astype(int)
        result_df['Match_Flag4'] = flag4.astype(int)
        result_df['Match_Flag5'] = flag5.astype(int)
        result_df['Match_Flag6'] = flag6.astype(int)
        # result_df['Match_Flag7'] = flag7.astype(int)
        result_df['Match_Flag7'] = 1
        result_df['Match_Flag8'] = flag8.astype(int)
        result_df['Final_Flag'] = (
            result_df['Match_Flag']  *
            result_df['Match_Flag2'] *
            result_df['Match_Flag3'] *
            result_df['Match_Flag4'] *
            result_df['Match_Flag5'] *
            result_df['Match_Flag6'] *
            result_df['Match_Flag7'] *
            result_df['Match_Flag8']
        )

        return result_df

    except Exception as e:
        print(f"조합 처리 중 오류 발생: {e}")
        raise
#--------------[ 클래스 선언 ]--------------
# --- [1. 경로 및 설정 관리 클래스] ---
class DQConfig:
    ROOT_PATH = Path(__file__).resolve().parents[2]
    YAML_RELATIVE_PATH = 'DataSense/util/DS_Master.yaml'
    # CONTRACT_RELATIVE_PATH = 'DataSense/util/DQ_Contract.yaml'

    @staticmethod
    def get_path(rel_path):
        """EXE 빌드 환경과 일반 파이썬 환경 모두 대응"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, rel_path)
        return os.path.join(DQConfig.ROOT_PATH, rel_path)

# sys.path 추가 (내부 모듈 참조용)
if str(DQConfig.ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(DQConfig.ROOT_PATH))

try:
    from DataSense.util.io import Load_Yaml_File
    # from DataSense.util.dq_format import Expand_Format, Combine_Format
    from DataSense.util.dq_validate import (
        init_reference_globals,
        validate_date, validate_yearmonth, validate_latitude, validate_longitude,
        validate_YYMMDD, validate_year, validate_tel, validate_cellphone,
        validate_url, validate_email, validate_kor_name, validate_address,
        validate_country_code, validate_gender, validate_gender_en, validate_car_number,
        validate_time, validate_timestamp,
    )

except ImportError as e:
    print(f"필수 모듈 로드 실패: {e}")
    sys.exit(1)

class Initializing_Main_Class:
    def __init__(self, main_config):
        self.logger = self._setup_logger()
        self.config = main_config
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def process_files_mapping(self):
        try:
            # 1. 파일 로드 (유지보수 용이하게 경로 관리)
            output_dir = Path(self.config['ROOT_PATH']) / self.config['directories']['output']
            meta_dir = Path(self.config['ROOT_PATH']) / "DataSense" / "DS_Meta"
            
            f_format_path = output_dir / "FileFormat.csv"
            r_datatype_path = output_dir / "RuleDataType.csv"
            m_meta_path = meta_dir / "Master_Meta.xlsx"

            # 데이터 로드 시 에러 방지 (str 연산 오류 방지를 위해 모든 컬럼을 str로 읽거나 처리)
            df_ff = pd.read_csv(f_format_path, encoding='utf-8-sig', dtype=str).fillna('')
            df_rt = pd.read_csv(r_datatype_path, encoding='utf-8-sig', dtype=str).fillna('')
            df_mm = pd.read_excel(m_meta_path, dtype=str).fillna('')

            self.logger.info("모든 기본 파일 로드 완료")

            # 2.  'str' 연산 부분 수정 (Vectorized 연산 사용)  예: 특정 컬럼에만 str 연산 적용
            for df in [df_ff, df_rt, df_mm]:
                for col in df.columns:
                    # 데이터프레임 자체가 아닌, 개별 시리즈(컬럼)에 str.strip() 적용
                    df[col] = df[col].astype(str).str.strip()

            # 3. 코드 관계 분석 메인 로직 
            result_df = self.execute_relationship_analysis(df_ff, df_rt, df_mm)
            
            final_path = output_dir / "Code_Relationship_Result.csv"
            result_df.to_csv(final_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"분석 완료 및 저장: {final_path}")
            
            return True

        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def execute_relationship_analysis(self, df_ff, df_rt, df_mm) -> pd.DataFrame:
        """
        df_ff: 파일 형식 매핑 결과
        df_rt: 룰 매핑 결과
        df_mm: 마스터 매핑 결과
        """
        # 1) Reference
        reference_df = self.reference_mapping(df_ff)
        if DEBUG_MODE and reference_df is not None and not reference_df.empty:
            p = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILE_NAME + '_3rd_ref_mapping.csv')
            reference_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"reference mapping : {p} 저장")

        # 2) Rule
        rule_df = self.rule_mapping(df_ff, df_rt)
        if DEBUG_MODE and rule_df is not None and not rule_df.empty:
            p = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILE_NAME + '_4th_rule_mapping.csv')
            rule_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"rule mapping : {p} 저장")

        # 3) Numeric stats
        numeric_df = self.numeric_column_statistics(df_ff)
        if DEBUG_MODE and numeric_df is not None and not numeric_df.empty:
            p = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILENUMERIC + '.csv')
            numeric_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"numeric stats : {p} 저장")

        # 4) Internal
        internal_df = self.internal_mapping(df_ff) # "데이터가 생긴 모양(Pattern)이 같은 것들끼리만" 그룹핑하여 비교 대상을 확 줄여버립니다.
        if DEBUG_MODE and internal_df is not None and not internal_df.empty:
            p = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILE_NAME + '_7th_int_mapping.csv')
            internal_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"internal mapping : {p} 저장")

        # # 4) Internal New () N X N 조합을 만든 뒤 하나씩 검사하는 방법 (속도가 더 느림) 
        # internal_df_new = self.internal_mapping_new(df_ff)
        # if DEBUG_MODE and internal_df_new is not None and not internal_df_new.empty:
        #     p = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILE_NAME + '_7th_int_mapping_new.csv')
        #     internal_df_new.to_csv(p, index=False, encoding='utf-8-sig')
        #     self.logger.info(f"internal mapping_new : {p} 저장")

        # 5) concat + pivot + final
        concat_df = self.mapping_concat(reference_df, internal_df, rule_df)
        if DEBUG_MODE and concat_df is not None and not concat_df.empty:
            p = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILE_NAME + '_8th_concat.csv')
            concat_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"concat_df : {p} 저장")

        pivoted_df = self.mapping_pivot(internal_df) 
        if DEBUG_MODE and pivoted_df is not None and not pivoted_df.empty:
            p = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILE_NAME + '_9th_pivoted.csv')
            pivoted_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"pivoted_df : {p} 저장")

        final_df = self.final_mapping(df_ff, pivoted_df, reference_df, rule_df) # 새로운 방식 
        if DEBUG_MODE and final_df is not None and not final_df.empty:
            final_path = os.path.join(self.config['ROOT_PATH'], self.config['directories']['output'], OUTPUT_FILE_NAME + '_final.csv')
            final_df.to_csv(final_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"최종 : {final_path} 저장")
           
        return final_df

# ---------------------------------------------------------
# 2. 메인 클래스 내 확장 메서드 
# ---------------------------------------------------------
    def internal_mapping_new(self, fileformat_df: pd.DataFrame, sample_size: int = 10000):
        self.logger.info(f"🚀 초고속 Internal Mapping 시작 (샘플링: {sample_size}건)")
        
        # 1. 유니크 셋 사전 추출 (I/O 최소화)
        unique_sets = {}
        target_cols = fileformat_df.to_dict('records')
        
        self.logger.info(f"대상 컬럼 {len(target_cols)}개에 대한 데이터 로딩 및 샘플링 시작...")
        for col_meta in target_cols:
            fpath = col_meta['FilePath']
            cname = col_meta['ColumnName']
            
            try:
                # 필요한 컬럼만, 지정된 샘플만큼 읽기
                df_tmp = pd.read_csv(fpath, usecols=[cname], dtype=str, encoding='utf-8-sig', low_memory=False)
                if len(df_tmp) > sample_size:
                    series = df_tmp[cname].sample(n=sample_size, random_state=42)
                else:
                    series = df_tmp[cname]
                
                # 클렌징 후 Set 저장
                cleaned_set = set(series.dropna().str.strip().unique())
                unique_sets[(fpath, cname)] = cleaned_set
            except Exception:
                unique_sets[(fpath, cname)] = set()

        # 2. Pruning (가지치기) 기반 태스크 생성
        self.logger.info("메타데이터 기반 Pruning(가지치기) 수행 중...")
        tasks = []
        for a, b in combinations(target_cols, 2):
            # [조건 1] 같은 파일의 같은 컬럼은 제외
            if a['FilePath'] == b['FilePath'] and a['ColumnName'] == b['ColumnName']:
                continue
            
            # # [조건 2] 데이터 타입이 다르면 연산 가치 없음 (Pruning)
            # if a.get('DetailDataType') != b.get('DetailDataType'):
            #     continue
            
            # [조건 3] 값 범위(Min/Max)가 전혀 겹치지 않으면 스킵 (Pruning)
            a_min, a_max = str(a.get('FormatMin_1', '')), str(a.get('FormatMax_1', ''))
            b_min, b_max = str(b.get('FormatMin_1', '')), str(b.get('FormatMax_1', ''))
            
            if a_min and a_max and b_min and b_max:
                if a_max < b_min or a_min > b_max:
                    continue # 범위가 겹치지 않으므로 스킵

            # 비교 대상 리스트 추가
            set_a = unique_sets.get((a['FilePath'], a['ColumnName']), set())
            set_b = unique_sets.get((b['FilePath'], b['ColumnName']), set())
            
            if set_a and set_b:
                tasks.append((a, b, set_a, set_b))

        self.logger.info(f"최종 비교 대상 조합: {len(tasks)}개 (병렬 처리 시작)")

        # 3. 병렬 처리 실행
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(compare_columns_worker, tasks)

        # 4. 결과 정리
        final_results = [r for r in results if r is not None]
        self.logger.info(f"분석 완료! 유효 매핑 결과: {len(final_results)}건")
        
        return pd.DataFrame(final_results)

# ------------------ (2) Reference 값 비교 (new) ------------------
    def mapping_check_old(self, mapping_df: pd.DataFrame, sample: int = 10_000) -> pd.DataFrame:
        """Reference/Internal 매핑 비교 수행 + 필수 컬럼 보장"""
        
        def _clean_values(series: pd.Series, length_limit=0) -> pd.Series:
            s = (series.dropna().astype(str).str.strip()
                 .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})).dropna()
            if length_limit and length_limit > 0:
                s = s.str[:int(length_limit)]
            return s.drop_duplicates()

        def _to_int(x, default=0):
            try:
                v = pd.to_numeric(x, errors="coerce")
                return int(v) if pd.notna(v) else default
            except Exception:
                return default

        # ✅ 1. 캐시 저장소 초기화 (루프 밖)
        master_val_cache = {}
        src_cache, master_cache = {}, {}
        rows: List[Dict[str, Any]] = []

        mapping_df = mapping_df.copy()

        for _, r in mapping_df.sort_values(by='FilePath').iterrows():
            # ✅ 2. 루프 내부에서 변수 정의
            fpath = str(r['FilePath']).strip()
            fname = str(r['FileName']).strip()
            col   = str(r['ColumnName']).strip()
            mtype = str(r['MasterType']).strip()
            mpath = str(r['MasterFilePath']).strip()
            mfile = str(r.get('MasterFile', "")).strip()
            rtype = str(r['ReferenceMasterType']).strip()
            mcol  = str(r['MasterColumn']).strip()

            comp_len_src = _to_int(r.get('CompareLength', 0), 0)
            comp_len_mst = _to_int(r.get('MasterCompareLength', 0), 0)

            # --- 파일 로드 로직 (생략 방지용 유지) ---
            if fpath not in src_cache:
                try:
                    src_cache[fpath] = _clean_headers(pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False, dtype=str))
                except Exception: continue
            if mpath not in master_cache:
                try:
                    master_cache[mpath] = _clean_headers(pd.read_csv(mpath, encoding='utf-8-sig', low_memory=False, dtype=str))
                except Exception: continue

            df = src_cache[fpath]
            md = master_cache[mpath]

            if (col not in df.columns) or (mcol not in md.columns):
                continue

            # ✅ 3. 마스터 값 추출 및 캐싱 (변수가 모두 정의된 루프 안에서 수행)
            # 마스터 컬럼과 적용할 길이 제한을 키로 사용
            m_key = (mpath, mcol, comp_len_mst or comp_len_src)
            if m_key not in master_val_cache:
                # 마스터 데이터셋(md)에서 해당 컬럼(mcol)을 가져와 클렌징 후 캐시에 저장
                master_val_cache[m_key] = _clean_values(md[mcol], m_key[2])
            
            m_vals = master_val_cache[m_key]

            # --- 소스 데이터 샘플링 및 비교 ---
            s_series = df[col]
            if len(s_series) > sample:
                s_series = s_series.sample(sample, random_state=42)

            s_vals = _clean_values(s_series, comp_len_src or comp_len_mst)

            # 비교 수행
            compare_count = s_vals[s_vals.isin(m_vals)].count()
            total_count   = s_vals.count()
            match_rate    = round(compare_count / total_count * 100, 2) if total_count > 0 else 0.0

            rows.append({
                "FilePath": fpath, "FileName": fname, "ColumnName": col, "MasterType": mtype,
                "MasterFilePath": mpath, "MasterFile": mfile, "ReferenceMasterType": rtype,
                "MasterColumn": mcol, "CompareLength": comp_len_src,
                "CompareCount": compare_count, "SourceCount": total_count, "MatchRate(%)": match_rate
            })

        out = pd.DataFrame(rows)
        return out
    
    def mapping_check(self, mapping_df: pd.DataFrame, sample: int = 10000) -> pd.DataFrame:
        """기존 패턴 매핑 방식을 유지하되, 데이터 정제 과정을 캐싱하여 속도 최적화"""
        
        # --- 내부 유틸리티: 값을 한 번만 정제해서 저장 ---
        cleaned_cache = {} # (fpath, col, limit) -> cleaned_series_set

        def get_cleaned_values(fpath, col, df_source, limit):
            key = (fpath, col, limit)
            if key not in cleaned_cache:
                # 1. 샘플링 및 정제 (최초 1회만 수행)
                s = df_source[col].dropna().astype(str).str.strip()
                s = s.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA}).dropna()
                
                if len(s) > sample:
                    s = s.sample(sample, random_state=42)
                
                if limit > 0:
                    s = s.str[:int(limit)]
                
                # 교집합 연산을 위해 set으로 변환하여 캐싱
                cleaned_cache[key] = set(s.unique())
            return cleaned_cache[key]

        # --------------------------------------------------
        mapping_df = mapping_df.copy()
        rows = []
        src_cache = {} # 파일 객체 캐시

        # FilePath 순으로 정렬하여 파일 로드 횟수 최소화
        for _, r in mapping_df.sort_values(by=['FilePath', 'MasterFilePath']).iterrows():
            fpath, col = str(r['FilePath']), str(r['ColumnName'])
            mpath, mcol = str(r['MasterFilePath']), str(r['MasterColumn'])
            
            # 파일 로드 (캐시 활용)
            for path in [fpath, mpath]:
                if path not in src_cache:
                    try:
                        src_cache[path] = _clean_headers(pd.read_csv(path, dtype=str, encoding='utf-8-sig', low_memory=False))
                    except: continue

            if fpath not in src_cache or mpath not in src_cache: continue
            
            df, md = src_cache[fpath], src_cache[mpath]
            if col not in df.columns or mcol not in md.columns: continue

            # 비교 길이 설정
            comp_len = int(r.get('CompareLength', 0) or r.get('MasterCompareLength', 0))

            # ✅ 핵심: 이미 정제된 데이터셋을 가져옴 (중복 연산 0)
            s_vals_set = get_cleaned_values(fpath, col, df, comp_len)
            m_vals_set = get_cleaned_values(mpath, mcol, md, comp_len)

            # ✅ 고속 Set 교집합 연산
            intersection = s_vals_set.intersection(m_vals_set)
            compare_count = len(intersection)
            total_count = len(s_vals_set)
            match_rate = round(compare_count / total_count * 100, 2) if total_count > 0 else 0.0

            if match_rate >= MATCH_RATE_THRESHOLD:
                rows.append({
                    "FilePath": fpath, "FileName": r['FileName'], "ColumnName": col, 
                    "MasterType": r['MasterType'], "MasterFilePath": mpath, "MasterFile": r.get('MasterFile',''),
                    "ReferenceMasterType": r['ReferenceMasterType'], "MasterColumn": mcol,
                    "CompareLength": comp_len, "CompareCount": compare_count, 
                    "SourceCount": total_count, "MatchRate(%)": match_rate
                })

        return pd.DataFrame(rows)
    # # ------------------ (2) 피벗(Left-compact) ------------------
    # def mapping_pivot_old(self, df_merged: pd.DataFrame, valid_threshold: float = 10.0,
    #                   top_k: int = 3, drop_old_pivot_cols: bool = True) -> pd.DataFrame:
    #     """Left-compact pivot: 상위 top_k 후보를 CodeFilePath/CodeFile/CodeType/CodeColumn/Matched로 전개"""
    #     if df_merged is None or df_merged.empty:
    #         cols = ["FilePath","FileName","ColumnName","MasterType"]
    #         for b in ["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"]:
    #             cols += [f"{b}_{i}" for i in range(1, top_k+1)]
    #         return pd.DataFrame(columns=cols)

    #     df = df_merged.copy()
    #     # normalize numeric columns
    #     for numc in ("Matched","Matched(%)"):
    #         if numc in df.columns:
    #             df[numc] = pd.to_numeric(df[numc], errors='coerce').fillna(0)

    #     # keep only candidate rows that exceed thresholds
    #     mask = (df["Matched"].fillna(0) > 0) & (df["Matched(%)"].fillna(-1) > valid_threshold)
    #     df = df.loc[mask].copy()
    #     if df.empty:
    #         cols = ["FilePath","FileName","ColumnName","MasterType"]
    #         for b in ["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"]:
    #             cols += [f"{b}_{i}" for i in range(1, top_k+1)]
    #         return pd.DataFrame(columns=cols)

    #     sort_keys = ["FilePath","FileName","ColumnName","MasterType","Matched(%)","Matched"]
    #     df = df.sort_values(sort_keys, ascending=[True,True,True,True,False,False], kind="mergesort").reset_index(drop=True)

    #     grp_keys = ["FilePath","FileName","ColumnName","MasterType"]
    #     df = df.assign(rank=df.groupby(grp_keys).cumcount() + 1)
    #     df = df.loc[df["rank"] <= top_k].copy()

    #     wide = (
    #         df.pivot_table(
    #             index=grp_keys,
    #             columns="rank",
    #             values=["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"],
    #             aggfunc="first"
    #         )
    #     )
    #     # Normalize column names to previous naming (CodeFile / CodeColumn)
    #     # pivot produced e.g. ('CodeFilePath', 1)
    #     wide.columns = [f"{col[0]}_{int(col[1])}" for col in wide.columns]
    #     wide = wide.reset_index().copy()

    #     # Left-compact each block of parallel columns
    #     def _left_compact_block(block: pd.DataFrame) -> pd.DataFrame:
    #         arr = block.to_numpy(object)
    #         for r in range(arr.shape[0]):
    #             vals = [x for x in arr[r].tolist() if not (pd.isna(x) or str(x).strip() == "")]
    #             vals += [""] * (arr.shape[1] - len(vals))
    #             arr[r, :] = vals
    #         return pd.DataFrame(arr, columns=block.columns, index=block.index)

    #     # perform left-compact for groups
    #     for base in ["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"]:
    #         cols = [c for c in wide.columns if c.startswith(base + "_")]
    #         if cols:
    #             block = _left_compact_block(wide[cols].copy())
    #             wide[cols] = block

    #     # fillna -> empty string for object columns
    #     obj_cols = wide.select_dtypes(include="object").columns.tolist()
    #     if obj_cols:
    #         wide[obj_cols] = wide[obj_cols].fillna("")

    #     return wide

    # 2025. 12. 24 Qliker - 피벗(Left-compact) 수정
    def mapping_pivot(self, df_merged: pd.DataFrame, valid_threshold: float = 10.0,
                         top_k: int = 3, drop_old_pivot_cols: bool = True) -> pd.DataFrame:
        """Top-K 후보를 가로로 전개하는 개선된 피벗 로직"""
        
        if df_merged is None or df_merged.empty:
            return self._make_empty_pivot_df(top_k)

        # 1. 컬럼명 정리
        rename_map = {
            'MasterFilePath':'CodeFilePath', 'MasterFile':'CodeFile',
            'ReferenceMasterType':'CodeType', 'MasterColumn':'CodeColumn',
            'CompareCount':'Matched', 'MatchRate(%)':'Matched(%)'
        }
        df = df_merged.rename(columns=rename_map).copy()

        # 2. 숫자형 변환 및 필터링
        for numc in ["Matched", "Matched(%)"]:
            if numc in df.columns:
                df[numc] = pd.to_numeric(df[numc], errors='coerce').fillna(0)

        mask = (df["Matched"] > 0) & (df["Matched(%)"] >= valid_threshold)
        df = df.loc[mask].copy()
        
        if df.empty:
            return self._make_empty_pivot_df(top_k)

        # 3. 정렬 및 랭킹 부여 (Top-K 추출)
        grp_keys = ["FilePath", "FileName", "ColumnName", "MasterType"]
        sort_keys = grp_keys + ["Matched(%)", "Matched"]
        
        df = df.sort_values(sort_keys, ascending=[True]*4 + [False]*2, kind="mergesort")
        df['rank'] = df.groupby(grp_keys).cumcount() + 1
        df = df.loc[df["rank"] <= top_k]

        # 4. Pivot Table 생성
        value_vars = ["CodeFilePath", "CodeFile", "CodeType", "CodeColumn", "Matched", "Matched(%)"]
        wide = df.pivot_table(
            index=grp_keys,
            columns="rank",
            values=value_vars,
            aggfunc="first"
        )

        # 5. 컬럼명 평탄화 (Multi-index -> Single-index) 예: ('CodeFile', 1) -> 'CodeFile_1'
        wide.columns = [f"{c[0]}_{int(c[1])}" for c in wide.columns]
        wide = wide.reset_index()

        # 6. 컬럼 순서 정렬 (CodeFile_1, Matched_1, CodeFile_2, Matched_2... 순서로 정렬하고 싶을 때)
        ordered_cols = grp_keys.copy()
        for i in range(1, top_k + 1):
            for base in value_vars:
                col_name = f"{base}_{i}"
                if col_name in wide.columns:
                    ordered_cols.append(col_name)
                else:
                    wide[col_name] = "" if "Matched" not in base else 0
                    ordered_cols.append(col_name)

        return wide[ordered_cols].fillna("")

    def _make_empty_pivot_df(self, top_k):
        """빈 결과 데이터프레임 생성 유틸리티"""
        cols = ["FilePath", "FileName", "ColumnName", "MasterType"]
        bases = ["CodeFilePath", "CodeFile", "CodeType", "CodeColumn", "Matched", "Matched(%)"]
        for i in range(1, top_k + 1):
            for b in bases:
                cols.append(f"{b}_{i}")
        return pd.DataFrame(columns=cols)

    # ------------------ (2) 피벗(Left-compact) ------------------
    def mapping_pivot_old(self, df_merged: pd.DataFrame, valid_threshold: float = 10.0,
                      top_k: int = 3, drop_old_pivot_cols: bool = True) -> pd.DataFrame:
        """Left-compact pivot: 상위 top_k 후보를 CodeFilePath/CodeFile/CodeType/CodeColumn/Matched로 전개"""
        df_merged = df_merged.rename(columns={
            'MasterFilePath':'CodeFilePath',
            'MasterFile':'CodeFile',
            'ReferenceMasterType':'CodeType',
            'MasterColumn':'CodeColumn',
            'CompareLength':'CompareLength',
            'MatchRate(%)':'Matched(%)',
            'CompareCount':'Matched',
            'SourceCount':'SourceCount',
            'MatchRate(%)':'Matched(%)'
        })

        # df_merged = df_merged[df_merged['Matched(%)'] > MATCH_RATE_THRESHOLD]

        if df_merged is None or df_merged.empty:
            cols = ["FilePath","FileName","ColumnName","MasterType"]
            for b in ["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"]:
                cols += [f"{b}_{i}" for i in range(1, top_k+1)]
            return pd.DataFrame(columns=cols)

        df = df_merged.copy()
        # normalize numeric columns
        for numc in ("Matched","Matched(%)"):
            if numc in df.columns:
                df[numc] = pd.to_numeric(df[numc], errors='coerce').fillna(0)

        # keep only candidate rows that exceed thresholds
        mask = (df["Matched"].fillna(0) > 0) & (df["Matched(%)"].fillna(-1) > valid_threshold)
        df = df.loc[mask].copy()
        if df.empty:
            cols = ["FilePath","FileName","ColumnName","MasterType"]
            for b in ["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"]:
                cols += [f"{b}_{i}" for i in range(1, top_k+1)]
            return pd.DataFrame(columns=cols)

        sort_keys = ["FilePath","FileName","ColumnName","MasterType","Matched(%)","Matched"]
        df = df.sort_values(sort_keys, ascending=[True,True,True,True,False,False], kind="mergesort").reset_index(drop=True)

        grp_keys = ["FilePath","FileName","ColumnName","MasterType"]
        df = df.assign(rank=df.groupby(grp_keys).cumcount() + 1)
        df = df.loc[df["rank"] <= top_k].copy()

        wide = (
            df.pivot_table(
                index=grp_keys,
                columns="rank",
                values=["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"],
                aggfunc="first"
            )
        )
        # Normalize column names to previous naming (CodeFile / CodeColumn)
        wide.columns = [f"{col[0]}_{int(col[1])}" for col in wide.columns]
        wide = wide.reset_index().copy()

        # Left-compact each block of parallel columns
        def _left_compact_block(block: pd.DataFrame) -> pd.DataFrame:
            arr = block.to_numpy(object)
            for r in range(arr.shape[0]):
                vals = [x for x in arr[r].tolist() if not (pd.isna(x) or str(x).strip() == "")]
                vals += [""] * (arr.shape[1] - len(vals))
                arr[r, :] = vals
            return pd.DataFrame(arr, columns=block.columns, index=block.index)

        # perform left-compact for groups
        for base in ["CodeFilePath","CodeFile","CodeType","CodeColumn","Matched","Matched(%)"]:
            cols = [c for c in wide.columns if c.startswith(base + "_")]
            if cols:
                block = _left_compact_block(wide[cols].copy())
                wide[cols] = block

        # fillna -> empty string for object columns
        obj_cols = wide.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            wide[obj_cols] = wide[obj_cols].fillna("")

        return wide
    # ------------------ (3) Rule 매핑 ------------------
    def rule_mapping(
        self,
        fileformat_df: pd.DataFrame,
        ruldatatype_df: pd.DataFrame,
        valid_types: Sequence[str] = (
            'URL','YEAR','EMAIL','CELLPHONE','TEL','LATITUDE','LONGITUDE',
            'DATECHAR','YEARMONTH','YYMMDD','ADDRESS','KOR_NAME', 'TIME', 'TIMESTAMP',
            'COUNTRY_ISO3','국가코드','시도','차량번호','GENDER','GENDER_EN'
        ),
        encodings_try: Iterable[str] = ("utf-8-sig","utf-8","cp949"),
        sampling_rows: Optional[int] = None,
        use_valuecnt_fallback: bool = True
    ) -> pd.DataFrame:
        """
        ruldatatype_df를 보고 파일의 컬럼에 대해 validate_* 함수를 적용해서 룰 기반 매핑 결과를 만들어 반환
        """
        out_cols = [
            'FilePath','FileName','ColumnName','MasterType','MasterColumn',
            'CompareCount','MatchRate(%)','MasterFile','ReferenceMasterType',
            'MasterFilePath','CompareLength','SourceCount'
        ]
        if ruldatatype_df is None or ruldatatype_df.empty:
            self.logger.info("[rule_mapping] ruldatatype_df is empty")
            return pd.DataFrame(columns=out_cols)

        required_cols = ["FilePath","FileName","ColumnName","MasterType","ValueCnt","Rule","MatchedScoreList"]
        miss = set(required_cols) - set(ruldatatype_df.columns)
        if miss:
            raise ValueError(f"ruldatatype_df 필수 컬럼 누락: {sorted(miss)}")

        rule_clean = ruldatatype_df["Rule"].fillna("").astype(str).str.strip()
        mask = (ruldatatype_df["MasterType"] != "Reference") & (rule_clean != "")
        df_rule = ruldatatype_df.loc[mask, required_cols].copy()
        if df_rule.empty:
            return pd.DataFrame(columns=out_cols)

        # Matched(%) 계산 (MatchedScoreList의 첫 값)
        df_rule["Matched(%)"] = (
            df_rule["MatchedScoreList"].astype(str)
            .str.split(";").str[0].str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            .astype(float).fillna(0.0) * 100.0
        ).round(2)

        # rule name standardization
        rule_key_syn = {
            "주소": "ADDRESS", "국가코드": "COUNTRY_ISO3", "이메일": "EMAIL",
            "휴대폰": "CELLPHONE", "전화": "TEL", "위도": "LATITUDE", "경도": "LONGITUDE",
            "연월": "YEARMONTH", "연월일": "DATECHAR", "성씨": "KOR_NAME",
            "성별구분": "GENDER", "성별구분_영문": "GENDER_EN",
        }
        def _std_rule_name(x: str) -> str:
            x = (x or "").strip()
            xu = x.upper()
            if xu in (t.upper() for t in valid_types):
                return xu
            return rule_key_syn.get(x, xu)

        df_rule["Rule"] = df_rule["Rule"].map(_std_rule_name)
        vtypes = {t.upper() for t in valid_types}
        rule_df = df_rule[df_rule["Rule"].isin(vtypes)].copy()
        if rule_df.empty:
            return pd.DataFrame(columns=out_cols)

        mapper = {
            'URL': validate_url, 'YEAR': validate_year, 'EMAIL': validate_email,
            'CELLPHONE': validate_cellphone, 'TEL': validate_tel,
            'LATITUDE': validate_latitude, 'LONGITUDE': validate_longitude,
            'DATECHAR': validate_date, 'YEARMONTH': validate_yearmonth,
            'YYMMDD': validate_YYMMDD, 'ADDRESS': validate_address,
            'KOR_NAME': validate_kor_name,
            'COUNTRY_ISO3': validate_country_code, '국가코드': validate_country_code,
            '시도': validate_address, '차량번호': validate_car_number,
            'GENDER': validate_gender, 'GENDER_EN': validate_gender_en,
            'TIME': validate_time, 'TIMESTAMP': validate_timestamp,
        }

        results: List[Dict[str, Any]] = []
        # 파일 경로별 처리 (group by FilePath)
        for fpath, grp in rule_df.sort_values('FilePath').groupby('FilePath'):
            src_path = str(fpath).strip()
            if not os.path.exists(src_path):
                self.logger.warning(f"[rule_mapping] 파일 경로 확인 불가: {src_path}")
                continue

            # 전체 파일 읽기 (헤더 정리)
            try:
                df_src = pd.read_csv(src_path, encoding='utf-8-sig', on_bad_lines="skip", dtype=str, low_memory=False)
                df_src = _clean_headers(df_src)
            except Exception as e:
                self.logger.warning(f"[rule_mapping] 파일 로드 실패: {src_path} -> {e}")
                continue

            # sampling
            if sampling_rows and sampling_rows > 0 and len(df_src) > sampling_rows:
                df_src = df_src.sample(n=sampling_rows, random_state=42)

            for _, r in grp.iterrows():
                col = str(r['ColumnName']).replace('\ufeff','').strip()
                if col not in df_src.columns:
                    continue

                series = df_src[col].dropna().astype(str)
                non_null = len(series)
                if non_null == 0:
                    continue

                key = str(r['Rule']).strip().upper()
                fn = mapper.get(key)
                if fn is None:
                    self.logger.debug(f"[rule_mapping] 미지원 Rule: {key}")
                    continue

                try:
                    # apply validator and count True
                    valid_count = int(series.apply(fn).sum())
                except Exception as e:
                    self.logger.warning(f"[rule_mapping] validate 실패: {src_path}::{col} ({key}) -> {e}")
                    continue

                if valid_count <= 0:
                    continue

                rate = round(valid_count / max(non_null, 1) * 100, 2)
                results.append({
                    'FilePath': src_path,
                    'FileName': os.path.basename(src_path),
                    'ColumnName': col,
                    'MasterType': str(r['MasterType']).strip(),
                    'MasterColumn': key,
                    'CompareCount': int(valid_count),
                    'MatchRate(%)': float(rate),
                    'MasterFilePath': 'Rule',
                    'MasterFile': 'Rule',
                    'ReferenceMasterType': 'Rule',
                    'CompareLength': '',
                    'SourceCount': int(non_null),
                })

        if not results:
            return pd.DataFrame(columns=out_cols)

        out = pd.DataFrame(results)[out_cols].copy()
        for c in ('CompareCount','SourceCount','MatchRate(%)'):
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out['MatchRate(%)'] = out['MatchRate(%)'].astype("float32").round(2)
        return out

    # ------------------ (4) 숫자 통계 ------------------
    def numeric_column_statistics(self, fileformat_df: pd.DataFrame, vSamplingRows: int = 10_000) -> Optional[pd.DataFrame]:
        """fileformat_df의 DetailDataType이 비어있는 항목들에 대해 숫자 통계 계산"""
        def calc_numeric(file_path: str, cols: List[str]) -> Optional[pd.DataFrame]:
            try:
                if not os.path.exists(file_path):
                    self.logger.debug(f"[numeric] 파일 존재하지 않음: {file_path}")
                    return None
                if file_path.lower().endswith('.csv'):
                    df = pd.read_csv(file_path, low_memory=False)
                elif file_path.lower().endswith('.pkl'):
                    df = pd.read_pickle(file_path)
                else:
                    df = pd.read_excel(file_path)
                if len(df) > vSamplingRows:
                    df = df.sample(n=vSamplingRows, random_state=42)

                rows=[]
                for c in cols:
                    if c not in df.columns:
                        continue
                    s = pd.to_numeric(df[c], errors='coerce').dropna()
                    if s.empty:
                        continue
                    desc = s.describe()
                    mean, std = desc['mean'], desc['std']
                    lcl, ucl = mean - 3*std, mean + 3*std
                    rows.append({
                        'FilePath': file_path, 'FileName': os.path.basename(file_path), 'ColumnName': c,
                        'dtype': str(s.dtype), 'Count': int(desc['count']), 'Mean': float(mean), 'Std': float(std),
                        'Min': float(desc['min']), '25%': float(desc['25%']), '50%': float(desc['50%']),
                        '75%': float(desc['75%']), 'Max': float(desc['max']),
                        'LCL': float(lcl), 'UCL': float(ucl),
                        'BelowLCL': int((s < lcl).sum()), 'AboveUCL': int((s > ucl).sum())
                    })
                return pd.DataFrame(rows) if rows else None
            except Exception as e:
                self.logger.warning(f"[numeric] 처리 오류: {file_path} -> {e}")
                return None

        self.logger.info("Numeric Column Statistics 시작")
        # select candidates: DetailDataType empty AND Format(%) < 90 AND LenCnt > 2
        len_cnt = pd.to_numeric(fileformat_df.get('LenCnt', pd.Series(dtype='float')), errors='coerce').fillna(0)
        fmt_pct = pd.to_numeric(fileformat_df.get('Format(%)', pd.Series(dtype='float')), errors='coerce').fillna(0)
        target = fileformat_df[
            (len_cnt > 2) & (fmt_pct < 90) & (
                fileformat_df.get('DetailDataType').isna() |
                (fileformat_df.get('DetailDataType').astype(str).str.len() == 0)
            )
        ].copy()

        if target.empty:
            self.logger.info("Numeric: 처리 대상 없음")
            return None

        blocks=[]
        for fpath, grp in target.groupby('FilePath'):
            cols = grp['ColumnName'].tolist()
            r = calc_numeric(fpath, cols)
            if r is not None and not r.empty:
                blocks.append(r)
        if not blocks:
            self.logger.info("Numeric 결과 없음")
            return None
        return pd.concat(blocks, ignore_index=True)

    # ------------------ (5) Reference / Internal / Concat ------------------
    def reference_mapping(self, fileformat_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Reference Code Mapping 시작")

        expand_df = Expand_Format(fileformat_df)
        expand_df['Format(%)'] = pd.to_numeric(expand_df.get('Format(%)', 0), errors='coerce').fillna(0)
        expand_df = expand_df.loc[expand_df['Format(%)'] > 10].copy()
        source_df = expand_df.loc[expand_df['MasterType'] != 'Reference'].copy()
        reference_df = expand_df.loc[expand_df['MasterType'] == 'Reference'].copy()
        combine_df = Combine_Format(source_df, reference_df)
        # Combine_Format must produce columns expected by mapping_check (MasterFilePath etc.)
        combine_df = combine_df[combine_df.get('Final_Flag', 0) == 1].copy()
        if combine_df.empty:
            return pd.DataFrame()
        mapping_df = self.mapping_check(combine_df)
        mapping_df = mapping_df[mapping_df['MatchRate(%)'] > MATCH_RATE_THRESHOLD]
        mapping_df = mapping_df.sort_values(by=['FilePath','ColumnName','MatchRate(%)'], ascending=[True,True,False])
        return mapping_df

    def internal_mapping(self, fileformat_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Internal Code Mapping 시작")

        expand_df = Expand_Format(fileformat_df)
        expand_df['Format(%)'] = pd.to_numeric(expand_df.get('Format(%)', 0), errors='coerce').fillna(0)
        expand_df = expand_df.loc[expand_df['Format(%)'] > 10].copy()
        source_df = expand_df.loc[expand_df['MasterType'] != 'Reference'].copy()
        combine_df = Combine_Format(source_df, source_df) # Match 된 레코드 중 조건을 충족하는 레코드만 선택
        combine_df = combine_df[combine_df.get('Final_Flag', 0) == 1].copy()
        if combine_df.empty:
            return pd.DataFrame()
        mapping_df = self.mapping_check(combine_df)
        mapping_df = mapping_df[mapping_df['MatchRate(%)'] > MATCH_RATE_THRESHOLD]
        mapping_df = mapping_df.sort_values(by=['FilePath','ColumnName','MatchRate(%)'], ascending=[True,True,False])
        return mapping_df

    def mapping_concat(self, reference_df: pd.DataFrame, internal_df: pd.DataFrame, rule_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("모든 매핑 파일을 통합합니다.")
        required_cols = [
            'FilePath','FileName','ColumnName','MasterType','MasterFilePath',
            'MasterFile','ReferenceMasterType','MasterColumn','CompareLength',
            'CompareCount','MatchRate(%)'
        ]
        # safe slicing: if any missing -> create empty frame with those columns
        def safe_slice(df):
            if df is None or df.empty:
                return pd.DataFrame(columns=required_cols)
            cols_present = [c for c in required_cols if c in df.columns]
            missing = [c for c in required_cols if c not in df.columns]
            out = df.copy()
            for m in missing:
                out[m] = "" if "Count" not in m and "Rate" not in m else 0
            return out[required_cols]

        rref = safe_slice(reference_df)
        rint = safe_slice(internal_df)
        rrul = safe_slice(rule_df)
        concat_df = pd.concat([rref, rint, rrul], ignore_index=True)
        # rename to pivot-friendly names
        concat_df = concat_df.rename(columns={
            'MasterFilePath':'CodeFilePath','MasterFile':'CodeFile',
            'ReferenceMasterType':'CodeType','MasterColumn':'CodeColumn',
            'CompareCount':'Matched','MatchRate(%)':'Matched(%)'
        })
        # keep meaningful candidates only (>=20% match)
        concat_df['Matched(%)'] = pd.to_numeric(concat_df.get('Matched(%)', 0), errors='coerce').fillna(0)
        concat_df = concat_df[concat_df['Matched(%)'] > MATCH_RATE_THRESHOLD]
        concat_df = concat_df.sort_values(by=['FilePath','FileName','ColumnName','MasterType','Matched(%)'],
                                          ascending=[True,True,True,True,False])
        return concat_df

    # 2025-11-26 새로운 방식으로 변경함 
    def final_mapping(self, fileformat_df, pivoted_df, reference_df, rule_df) -> pd.DataFrame:
        """fileformat_df와 ruldatatype(preset)과 pivoted_df를 합쳐 최종 산출"""
        #---------------------------------------------------------
        #  rule_df 읽어옴. 
         #---------------------------------------------------------
        self.logger.info("최종 매핑 파일을 생성합니다.")
        df_rule = rule_df.copy()
        if df_rule.empty:
            self.logger.debug("ruldatatype 비어있음 -> 룰 반영 스킵")
        rule_required_cols = ["FilePath","FileName","ColumnName","MasterType", "Rule","MatchedScoreList"]
        # safe: fill missing rule cols if necessary
        for c in rule_required_cols:
            if c not in df_rule.columns:
                df_rule[c] = ""

        df_rule = df_rule[rule_required_cols].copy()
        #---------------------------------------------------------
        #  reference_df 읽어옴. 
        #---------------------------------------------------------
        ref_cols = ["FilePath","FileName","ColumnName","MasterType","MasterFilePath","MasterFile","ReferenceMasterType","MasterColumn","CompareLength","CompareCount","SourceCount","MatchRate(%)"]
        ref_df = reference_df[ref_cols].copy()
        ref_df = ref_df.sort_values(by=['FilePath','FileName','ColumnName','MasterType','MatchRate(%)'], ascending=[True,True,True,True,False])
        ref_df = ref_df.groupby(['FilePath', 'ColumnName'], as_index=False).head(1)
        if ref_df.empty:
            self.logger.debug("reference_df 비어있음 -> 참조 반영 스킵")
        ref_df = ref_df.rename(columns={
            'MasterFilePath':'CodeFilePath_4',
            'MasterFile':'CodeFile_4',
            'ReferenceMasterType':'CodeType_4',
            'MasterColumn':'CodeColumn_4',
            'CompareCount':'Matched_4',
            'MatchRate(%)':'Matched(%)_4'
        })

        #---------------------------------------------------------
        #  pivoted_df 읽어옴. 
        #---------------------------------------------------------
        # pivoted_df may be empty -> create empty with expected columns
        pivot_cols = [
            'FilePath','FileName','ColumnName','MasterType',
            'CodeColumn_1','CodeFile_1','CodeFilePath_1','CodeType_1','Matched_1','Matched(%)_1',
            'CodeColumn_2','CodeFile_2','CodeFilePath_2','CodeType_2','Matched_2','Matched(%)_2'
        ]
        if pivoted_df is None or pivoted_df.empty:
            pivoted_df = pd.DataFrame(columns=pivot_cols)
        else:  # ensure all pivot cols exist
            for c in pivot_cols:
                if c not in pivoted_df.columns:
                    pivoted_df[c] = ""

        # merge
        df = pd.merge(fileformat_df, df_rule, on=['FilePath','FileName','ColumnName','MasterType'], how='left', suffixes=("","_rule"))
        df = pd.merge(df, pivoted_df, on=['FilePath','FileName','ColumnName','MasterType'], how='left', suffixes=("","_pivot"))
        df = pd.merge(df, ref_df, on=['FilePath','FileName','ColumnName','MasterType'], how='left', suffixes=("","_ref"))

        #---------------------------------------------------------
        #  Attribute 컬럼 생성
        #---------------------------------------------------------
        # Rule 컬럼에서 세미콜론 기준 첫 번째 값 추출하여 Attribute에 설정
        df['Attribute'] = ""
        if 'Rule' in df.columns:
            # Rule 컬럼을 문자열로 변환하고 NaN 처리
            df['Rule'] = df['Rule'].fillna("").astype(str).str.strip()
            # 세미콜론 기준으로 첫 번째 값 추출
            rule_first_value = df['Rule'].str.split(';').str[0].str.strip()
            # 값이 있으면 Attribute에 설정
            mask_rule = rule_first_value != ""
            df.loc[mask_rule, 'Attribute'] = rule_first_value[mask_rule]
        
        # Rule에서 값이 없는 경우 CodeColumn_4 값 사용
        if 'CodeColumn_4' in df.columns:
            df['CodeColumn_4'] = df['CodeColumn_4'].fillna("").astype(str).str.strip()
            mask_no_rule = (df['Attribute'] == "") & (df['CodeColumn_4'] != "")
            df.loc[mask_no_rule, 'Attribute'] = df.loc[mask_no_rule, 'CodeColumn_4']

        #---------------------------------------------------------
        #  PK -> FK mapping (if PK column present in fileformat_df) 추가
        #---------------------------------------------------------
        if 'PK' in fileformat_df.columns:
            pk_numeric = pd.to_numeric(fileformat_df['PK'], errors='coerce').fillna(0).astype(int)
            mask_pk = pk_numeric == 1
            tmp_df = fileformat_df.loc[mask_pk, ['FilePath','ColumnName']].copy()
            tmp_df = tmp_df.rename(columns={'FilePath':'CodeFilePath_1','ColumnName':'CodeColumn_1'})
            tmp_df['FK'] = 'FK'
            df = pd.merge(df, tmp_df, on=['CodeFilePath_1','CodeColumn_1'], how='left')

        return df

# 메인 실행부
if __name__ == "__main__":
    import time
    start_time = time.time()
    main_config = Load_Yaml_File(DQConfig.get_path(DQConfig.YAML_RELATIVE_PATH))
    analyzer = Initializing_Main_Class(main_config)
    analyzer.process_files_mapping()

    print("="*50)
    print(f"총 처리 시간: {time.time()-start_time:.2f}초")
    print("="*50)


# 다음은 성능향상을 위하여 적용한 기법들 입니다.  
# 📊 성능 최적화 요약 (43s → 17.9s)최적화 단계적용 기술효과
# 1단계: 필터링set 기반 검색 및 중복 copy() 
# 제거초기 데이터 로딩 및 메모리 점유율 감소
# 2단계: 병합(Merge)필요한 컬럼만 선택하여 조인
# 조인 연산 시 발생하는 오버헤드 최소화
# 3단계: 연산(Flag)between, & 논리 연산자 활용
# CPU 수준의 비트 연산으로 계산 속도 극대화
# 4단계: 문자열 처리
# np.where와 벡터화된 .str 접근문자열 루프 처리 비용 절감