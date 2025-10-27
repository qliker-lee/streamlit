### QDQM_01_FormatBuilder.py 참조
### Master_Code 하위의 모든 폴더 및 파일에 대한 Format 분석 결과를 저장합니다. 

import os
import pandas as pd
from datetime import datetime
from typing import List
from multiprocessing import Pool, cpu_count
import logging

from util.io import Load_Yaml_File
# from util.io import Directory_Recreate
from util.io import Backup_File


from util.dq_validate import (
    validate_date,
    validate_yearmonth,
    validate_latitude,
    validate_longitude,
    validate_YYMMDD,
    validate_tel,
    validate_cellphone,
)

# 상수 정의를 클래스로 분리
class Constants:
    FORMAT_MAX_VALUE = 100
    FORMAT_AVG_LENGTH = 50
    SAMPLING_ROWS = 10000
    
# 설정 관련 클래스 분리
class Config:
    def __init__(self, yaml_path):
        self.config = self.load_config(yaml_path)
        
    @staticmethod
    def load_config(yaml_path):
        return Load_Yaml_File(yaml_path)

# Output 파일 이름 설정
YAML_PATH = 'QDQM_Master_Code/util/QDQM_Master.yaml'
MASTERFORMAT = 'MasterFormat'
MASTERSTATS = 'MasterStats'
#-------------------------------------------------------------------------------------------
# Data Type 변환 함수 선언 
#-------------------------------------------------------------------------------------------
def Get_Oracle_Type(series): # Pandas 데이터 타입을 Oracle 데이터 타입으로 변환
    
    dtype = str(series.dtype)
    
    # 정수형
    if dtype.startswith('int'):
        non_null_values = series.dropna()
        if len(non_null_values) > 0:
            # 최대 자릿수 계산 (음수 고려하여 절대값 사용)
            max_digits = max(len(str(abs(x))) for x in non_null_values if not pd.isna(x))
            return f"NUMBER({max_digits})"
        return 'NUMBER'
    
    # 실수형
    if dtype.startswith('float'):
        non_null_values = series.dropna()
        if len(non_null_values) > 0:
            max_int_digits = max(len(str(int(abs(x)))) for x in non_null_values if not pd.isna(x))
            # 소수점 이하 자릿수 계산
            decimal_values = [str(abs(x)).split('.')[-1] for x in non_null_values if not pd.isna(x) and '.' in str(x) and str(abs(x)).split('.')[-1] != '0']
            if decimal_values:
                max_decimal_digits = max(len(x) for x in decimal_values)
                return f"NUMBER({max_int_digits + max_decimal_digits},{max_decimal_digits})"
            else:
                return f"NUMBER({max_int_digits})"
        return 'NUMBER'
    
    # 날짜/시간형
    if dtype == 'datetime64[ns]':
        return 'DATE'
    
    # 문자열
    if dtype == 'object':
        max_length = series.astype(str).str.len().max()
        if max_length <= 4000:
            return f"VARCHAR2({max_length})"
        return 'CLOB'
    return 'VARCHAR2(4000)'


def Find_Unique_Combination(df: pd.DataFrame) -> List[str]: # 필드를 조합하여 유니크한 키를 찾는 함수
    columns = df.columns.tolist()
    current_combo = []
    
    # 첫 번째 컬럼부터 순차적으로 조합 확인
    for column in columns:
        current_combo.append(column)
        # 현재까지의 컬럼 조합으로 그룹화하여 유니크한지 확인
        if df.groupby(current_combo).size().max() == 1:
            return current_combo
    
    return current_combo  # 모든 컬럼을 조합해도 유니크하지 않으면 전체 컬럼 반환

#-------------------------------------------------------------------------------------------
# Format Builder 함수 
#-------------------------------------------------------------------------------------------
def Get_String_Pattern(value, dtype=None):
    """문자열의 패턴을 반환합니다."""

    # 패턴 매핑 딕셔너리를 상수로 정의
    PATTERN_MAP = {
        'digit': 'n',
        'upper': 'A',
        'lower': 'a',
        'special': {'tab': '_', 'newline': '_', 'return': '_'},
        'keep': {'-', '=', '.', ' ', ':', '(', ')', '@', '/'}
    }
    
    str_value = str(value)
    if dtype == 'float64' and str_value.endswith('.0'):
        str_value = str_value[:-2]
    
    pattern = []
    for char in str_value:
        if char.isdigit():
            pattern.append(PATTERN_MAP['digit'])
        # elif is_korean(char):
        elif ord('가') <= ord(char) <= ord('힣'):
            pattern.append('K')
        # elif is_parenthesis(char):
        elif char in '(){}[]':
            pattern.append(char)
        elif char.isalpha():
            pattern.append(PATTERN_MAP['upper'] if char.isupper() else PATTERN_MAP['lower'])
        elif char in PATTERN_MAP['special']:
            pattern.append(PATTERN_MAP['special'][char])
        elif char in PATTERN_MAP['keep']:
            pattern.append(char)
        else:
            pattern.append('s')
    
    pattern = ''.join(pattern)
    return f"'{pattern}" if pattern.startswith('-') else pattern
    

def Calculate_Format_Statistics(values):  
    # NA 값 제거
    values = [str(v) for v in values if pd.notna(v)]  # 모든 값을 문자열로 변환
    if not values:
        return {'MinString': '', 'MaxString': '', 'ModeString': '', 'Median': ''}

    # 정렬된 값들
    sorted_values = sorted(values)
    
    # 중앙값 계산
    n = len(sorted_values)
    if n % 2 == 0:  # 짝수 개수일 경우
        median_val = sorted_values[n//2 - 1]  # 두 중앙값 중 작은 값 선택
    else:  # 홀수 개수일 경우
        median_val = sorted_values[n//2]

    min_val = str(min(values))  # 문자열로 변환
    max_val = str(max(values))  # 문자열로 변환

    # 최빈값
    from collections import Counter
    mode_val = str(Counter(values).most_common(1)[0][0]) if values else ''  # 문자열로 변환

    return {
        'MinString': min_val,
        'MaxString': max_val,
        'ModeString': mode_val,
        'Median': median_val
    }
  
def Analyze_Column_Format(series): # 필드 포맷을 분석하는 함수

    from collections import Counter
    
    # 초기 검증
    max_length = series.astype(str).str.len().max()
    if max_length > Constants.FORMAT_MAX_VALUE or max_length == 0:
        return {
            'pattern_type_cnt': 0,
            'most_common_pattern': '',
            'most_common_count': 0,
            'second_common_pattern': '',
            'second_common_count': 0,
            'third_common_pattern': '',
            'third_common_count': 0,
            'FormatMin': '',
            'FormatMax': '',
            'FormatMode': '',
            'FormatMedian': '',
            'Format2ndMin': '',
            'Format2ndMax': '',
            'Format2ndMode': '',
            'Format3rdMin': '',
            'Format3rdMax': '',
            'Format3rdMode': ''
        }
    
    # 패턴 분석
    series_clean = series.dropna()
    patterns = pd.Series([Get_String_Pattern(val, series.dtype) for val in series_clean], dtype='object')  # dtype 명시적 지정
    pattern_counts = Counter(patterns)
    
    if not pattern_counts:
        return {
            'pattern_type_cnt': 0,
            'most_common_pattern': '',
            'most_common_count': 0,
            'second_common_pattern': '',
            'second_common_count': 0,
            'third_common_pattern': '',
            'third_common_count': 0,
            'FormatMin': '',
            'FormatMax': '',
            'FormatMode': '',
            'FormatMedian': '',
            'Format2ndMin': '',
            'Format2ndMax': '',
            'Format2ndMode': '',
            'Format3rdMin': '',
            'Format3rdMax': '',
            'Format3rdMode': ''
        }
    
    # 상위 3개 패턴 추출
    top_patterns = pattern_counts.most_common(3)
    
    # 패턴별 값 추출
    pattern_values = {}
    for pattern, _ in top_patterns:
        mask = patterns == pattern
        if mask.any():
            pattern_values[pattern] = series_clean.iloc[mask.index[mask]].head(5).tolist()
    
    # 통계 계산
    stats = {
        'pattern_type_cnt': len(pattern_counts),
        'most_common_pattern': top_patterns[0][0],
        'most_common_count': top_patterns[0][1],
        'second_common_pattern': top_patterns[1][0] if len(top_patterns) > 1 else '',
        'second_common_count': top_patterns[1][1] if len(top_patterns) > 1 else 0,
        'third_common_pattern': top_patterns[2][0] if len(top_patterns) > 2 else '',
        'third_common_count': top_patterns[2][1] if len(top_patterns) > 2 else 0,
    }
    
    # 각 패턴별 통계 계산
    format_stats = Calculate_Format_Statistics(pattern_values.get(top_patterns[0][0], []))
    second_format_stats = Calculate_Format_Statistics(pattern_values.get(top_patterns[1][0], [])) if len(pattern_counts) > 1 else {'MinString': '', 'MaxString': '', 'ModeString': '', 'Median': ''}
    third_format_stats = Calculate_Format_Statistics(pattern_values.get(top_patterns[2][0], [])) if len(pattern_counts) > 2 else {'MinString': '', 'MaxString': '', 'ModeString': '', 'Median': ''}

    stats.update({
        'FormatMin': format_stats['MinString'],
        'FormatMax': format_stats['MaxString'],
        'FormatMode': format_stats['ModeString'],
        'FormatMedian': format_stats['Median'],
        'Format2ndMin': second_format_stats['MinString'],
        'Format2ndMax': second_format_stats['MaxString'],
        'Format2ndMode': second_format_stats['ModeString'],
        'Format3rdMin': third_format_stats['MinString'],
        'Format3rdMax': third_format_stats['MaxString'],
        'Format3rdMode': third_format_stats['ModeString']
    })
    
    return stats

def Determine_Detail_Type(pattern, pattern_type_cnt, format_stats, total_stats, 
                          max_length, unique_count, non_null_count):
    """패턴과 포맷 통계를 기반으로 detail_type을 결정합니다."""
    detail_type = ''
    
    # 최대 길이 체크
    max_length = int(max_length) if isinstance(max_length, str) else max_length
    if max_length > 4000:
        return 'CLOB'   
    if (max_length > Constants.FORMAT_MAX_VALUE or len(pattern) > Constants.FORMAT_AVG_LENGTH or format_stats['pattern_type_cnt'] > 20):
        return 'TEXT'

    # TimeStamp 체크
    if (pattern in ['nnnn-nn-nn nn:nn:nn', 'nnnn-nn-nn nn:nn:nn.nnnnnn'] 
        and pattern_type_cnt == 1):
        return 'TimeStamp'

    if (pattern in ['nnnnnnnnnnnnnn', 'nnnnnnnnnnnn'] and 
        validate_date(format_stats['FormatMode'])):
        return 'TimeChar'

    # 위도 체크
    if (pattern in ['nn.nnnn', 'nn.nnnnn', 'nn.nnnnnn', 'nn.nnnnnnn', 'nn.nnnnnnnn'] 
        and validate_latitude(format_stats['FormatMode'])):
        return 'LATITUDE'   
    # 경도 체크
    if (pattern in ['nnn.nnnn', 'nnn.nnnnn', 'nnn.nnnnnn', 'nnn.nnnnnnn', 'nnn.nnnnnnnn'] 
        and validate_longitude(format_stats['FormatMode'])):
        return 'LONGITUDE'       
    
    # TEL 체크
    if (pattern in ['nnn-nnn-nnnn', 'nn-nnnn-nnnn', 'nn-nnn-nnnn', 'nnn-nnnn', 'nnnn-nnnn', 'nnnnnnn', 'nnnnnnnn', 'nnnnnnnnnn'] 
        and validate_tel(format_stats['FormatMedian'])
        and validate_tel(format_stats['FormatMin'])
        and validate_tel(format_stats['FormatMax'])
        ):
        return 'TEL'   

    # CELLPHONE 체크
    if (pattern in ['nnn-nnnn-nnnn', 'nnnnnnnnnnn'] 
        and validate_cellphone(format_stats['FormatMedian'])):
        return 'CELLPHONE'   
    
    # YYMMDD 체크
    if (pattern in ['nnnnnn', 'nn-nn-nn', 'nn/nn/nn', 'nn.nn.nn'] 
        and validate_YYMMDD(format_stats['FormatMedian'], total_stats['MinString'], total_stats['MaxString'])):
        return 'YYMMDD'   
    
    # DateChar 체크
    if (pattern in ['nnnnnnnn', 'nnnn-nn-nn', 'nnnn/nn/nn', 'nnnnKnnKnnK', 'nnnn.nn.nn', 'nnnn. n. nn.'] 
        and validate_date(format_stats['FormatMedian'])):
        return 'DateChar'
    
    # YearMonth 체크
    if (pattern in ['nnnnnn', 'nnnn-nn', 'nnnn.nn', 'nnnnKnnK'] and 
        validate_yearmonth(format_stats['FormatMin']) and
        validate_yearmonth(format_stats['FormatMax']) and
        validate_yearmonth(format_stats['FormatMedian']) and
        float(format_stats['most_common_count']) / 
            (float(format_stats['most_common_count']) + float(format_stats['second_common_count']) + float(format_stats['third_common_count'])) > 0.95):
        return 'YearMonth'

    # Year 체크
    if pattern in ['nnnn'] and format_stats['FormatMedian']:
        year = int(float(format_stats['FormatMedian']))
        if (1900 < year < 2999 and 
            float(format_stats['most_common_count']) / 
            (float(format_stats['most_common_count']) + float(format_stats['second_common_count']) + float(format_stats['third_common_count'])) > 0.95):
            return 'YEAR'

    if '@' in pattern and pattern.count('.') >= 1 and pattern.count('.') <= 2:
        return 'EMail'    
    
    if '://' in pattern and pattern.count('.') >= 1:
        return 'URL'
    
    if len(pattern) == 0 : 
        return 'NULL'
    
    # Text 체크
    pattern_type_cnt = int(pattern_type_cnt) if isinstance(pattern_type_cnt, str) else pattern_type_cnt
    if pattern_type_cnt > 0:
        count = pattern_type_cnt
        # 주소 체크
        if len(pattern) > 10 and count > 10 and 'K' in pattern and pattern.count(' ') >= 3 and max_length > 20:
            return 'ADDRESS'
               
        if ((count > 15 and 'K' in pattern and pattern.count(' ') > 2) or  
            (len(pattern) > 10 and pattern.count(' ') > 2)  or 
            max_length > Constants.FORMAT_MAX_VALUE):
            return 'DESCRIPTION'       

        if (format_stats['most_common_pattern'] == 'A' and 
            total_stats['MinString'] == 'N' and 
            total_stats['MaxString'] == 'Y' and
            format_stats['pattern_type_cnt'] == 1):
            return 'YN FLAG'
        
        if (format_stats['most_common_pattern'] == 'n' and 
            total_stats['MinString'] == '0' and 
            total_stats['MaxString'] == '1' and
            format_stats['pattern_type_cnt'] == 1):
            return 'TrueFalse FLAG'
        
        if((format_stats['most_common_pattern'] == 'A' or 
            format_stats['most_common_pattern'] == 'a') and 
            format_stats['pattern_type_cnt'] == 1): 
            return 'ALPHA FLAG'
        
        if (format_stats['most_common_pattern'] == 'n' and 
            format_stats['pattern_type_cnt'] == 1): 
            return 'NUM FLAG'
        
        if (format_stats['most_common_pattern'] == 'K' and 
            format_stats['pattern_type_cnt'] == 1): 
            return 'KOR FLAG'
        
        if (pattern[:3] == 'KKK' or pattern[-3:] == 'KKK') and len(pattern) > 1 and count > 3:
            return 'KOR NAME'
        
        # SEQUENCE 체크 추가
        try:
            total_min = total_stats.get('MinString')
            total_max = total_stats.get('MaxString')
            
            # None이나 빈 문자열이 아닌지 명시적으로 체크
            if total_min is not None and total_max is not None and total_min != '' and total_max != '':
                total_min = float(total_min)
                total_max = float(total_max)
                
                if total_min.is_integer() and total_max.is_integer():
                    total_min = int(total_min)
                    total_max = int(total_max)
                    expected_count = total_max - total_min + 1
                    
                    if expected_count > 0 and expected_count == unique_count:
                        return 'SEQUENCE'
        except (ValueError, TypeError):
            pass

        if unique_count == 1:
            return 'SINGLE VALUE'                
        if non_null_count > 0 and non_null_count == unique_count:
            return 'UNIQUE'
    return detail_type

def Calculate_Statistics(series):
    """시리즈의 전체 통계를 계산합니다."""
    # NA 값 제거
    non_null_series = series.dropna()
    
    # 시리즈가 비어있는 경우
    if len(non_null_series) == 0:
        return {'MinString': '', 'MaxString': '', 'ModeString': '', 'Median': ''}

    try:
        # 숫자형으로 변환 시도
        numeric_series = pd.to_numeric(non_null_series, errors='coerce')
        numeric_series = numeric_series.dropna()  # 변환 실패한 NA 값 제거
        
        if len(numeric_series) > 0:
            # 숫자형 데이터가 있는 경우
            min_val = str(int(numeric_series.min()) if numeric_series.min().is_integer() else numeric_series.min())
            max_val = str(int(numeric_series.max()) if numeric_series.max().is_integer() else numeric_series.max())
            mode_val = str(int(numeric_series.mode().iloc[0]) if numeric_series.mode().iloc[0].is_integer() else numeric_series.mode().iloc[0]) if not numeric_series.mode().empty else ''
            median_val = str(int(numeric_series.median()) if numeric_series.median().is_integer() else numeric_series.median())
        else:
            # 숫자형 변환에 실패한 경우 문자열로 처리
            str_series = non_null_series.astype(str)
            min_val = min(str_series)
            max_val = max(str_series)
            mode_val = str_series.mode().iloc[0] if not str_series.mode().empty else ''
            sorted_series = sorted(str_series)
            median_val = sorted_series[len(sorted_series)//2]
    except:
        # 예외 발생 시 문자열로 처리
        str_series = non_null_series.astype(str)
        min_val = min(str_series)
        max_val = max(str_series)
        mode_val = str_series.mode().iloc[0] if not str_series.mode().empty else ''
        sorted_series = sorted(str_series)
        median_val = sorted_series[len(sorted_series)//2]

    return {
        'MinString': min_val,
        'MaxString': max_val,
        'ModeString': mode_val,
        'Median': median_val
    }

def Analyze_Column_Statistics(df, column):
    series = df[column]
    str_series = series.astype(str)
    # 각 값의 길이를 계산하고 서로 다른 길이의 개수를 세기
    length_counts = str_series.str.len().value_counts()
    len_cnt = len(length_counts)  # 서로 다른 길이의 개수
    min_length = str_series.str.len().min()
    max_length = str_series.str.len().max()
    avg_length = str_series.str.len().mean()
    mode_length = str_series.str.len().mode().iloc[0]

    # 기본 통계 계산
    non_null_mask = series.notna()
    non_null_count = non_null_mask.sum()
    null_count = len(series) - non_null_count
    unique_count = series.nunique()

    # 전체 통계 계산 (Min, Max, Mode 등)
    total_stats = Calculate_Statistics(series)
    
    # 새로 추가: Mode와 일치하는 행의 갯수를 계산 및 비율 산출
    total_mode_value = str(total_stats['ModeString'])
    total_mode_count = series.astype(str).eq(total_mode_value).sum()
    total_mode_ratio = (total_mode_count / non_null_count * 100) if non_null_count > 0 else 0

    # 전체 문자열이 조건에 맞는지 체크 (문자열 전체가 해당 패턴만으로 구성되었는지)
    has_only_alpha    = str_series.str.contains(r'^[a-zA-Z]+$', na=False).sum()
    has_only_num      = str_series.str.contains(r'^[0-9]+$', na=False).sum()
    has_only_kor      = str_series.str.contains(r'^[가-힣]+$', na=False).sum()
    has_only_alphanum = str_series.str.contains(r'^[a-zA-Z0-9]+$', na=False).sum()

    # 각 문자열의 첫번째 글자만 추출하여 조건별 체크
    first_char = str_series.str[0]
    first_char_is_kor     = first_char.str.contains(r'[가-힣]', na=False).sum()
    first_char_is_num     = first_char.str.contains(r'^[0-9]$', na=False).sum()
    first_char_is_alpha   = first_char.str.contains(r'^[a-zA-Z]$', na=False).sum()
    # 특수문자: 알파벳, 숫자, 한글, 공백을 제외한 문자
    first_char_is_special = first_char.str.contains(r'[^a-zA-Z0-9가-힣\s]', na=False).sum()

    # 공백이 포함된 행수, 특수문자 행수 등
    whitespace_count = str_series.str.contains(r"\s", na=False).sum()
    dash_count = str_series.str.contains(r"-", na=False).sum()
    dot_count = str_series.str.contains(r"\.", na=False).sum()
    at_count = str_series.str.contains(r"@", na=False).sum()
    eng_count = str_series.str.contains(r"[a-zA-Z]", na=False).sum()
    kor_count = str_series.str.contains(r"[가-힣]", na=False).sum()
    numeric_count = str_series.str.contains(r"[0-9]", na=False).sum()
    bracket_count = str_series.str.contains(r"[(){}\[\]]", na=False).sum()

    # 음수값 개수 계산 (시리즈가 숫자형 타입인 경우에만)
    negative_count = 0
    if pd.api.types.is_numeric_dtype(series):
        negative_count = (series < 0).sum()

    # 오라클 타입, 포맷 등의 통계
    oracle_type = Get_Oracle_Type(series)
    format_stats = Analyze_Column_Format(series)

    detail_type = Determine_Detail_Type(
        format_stats['most_common_pattern'], 
        format_stats['pattern_type_cnt'], 
        format_stats, 
        total_stats, 
        max_length, 
        unique_count, 
        non_null_count
    )

    # CLOB 타입일 경우 통계 빈값 처리
    if detail_type == 'CLOB':
        total_stats = {'MinString': '', 'MaxString': '', 'ModeString': ''}

    stats = {
        'DataType': str(series.dtype),
        'OracleType': oracle_type,
        'DetailDataType': detail_type,
        'LenCnt': int(len_cnt),
        'LenMin': int(min_length),
        'LenMax': int(max_length),
        'LenAvg': int(avg_length),
        'LenMode': int(mode_length),
        'SampleRows': str(len(series)),
        'ValueCnt': str(non_null_count),
        'NullCnt': str(null_count),
        'Null(%)': f"{(null_count / len(series) * 100):.2f}" if len(series) > 0 else "0",
        'UniqueCnt': str(unique_count),
        'Unique(%)': f"{(unique_count / len(series) * 100):.2f}" if len(series) > 0 else "0",
        'FormatCnt': str(format_stats['pattern_type_cnt']),
        'Format': format_stats['most_common_pattern'],
        'FormatLength': str(len(format_stats['most_common_pattern']) if format_stats['most_common_pattern'] else 0),
        'FormatValue': str(format_stats['most_common_count']),
        'Format(%)': f"{(float(format_stats['most_common_count']) / float(non_null_count) * 100):.2f}" if float(non_null_count) > 0 else "0",
        'Format2nd': format_stats['second_common_pattern'],
        'Format2ndValue': format_stats['second_common_count'],
        'Format2nd(%)': f"{(float(format_stats['second_common_count']) / float(non_null_count) * 100):.2f}" if float(non_null_count) > 0 else "0",
        'Format3rd': format_stats['third_common_pattern'],
        'Format3rdValue': format_stats['third_common_count'],
        'Format3rd(%)': f"{(float(format_stats['third_common_count']) / float(non_null_count) * 100):.2f}" if float(non_null_count) > 0 else "0",
        'MinString': str(total_stats['MinString']),
        'MaxString': str(total_stats['MaxString']),
        'ModeString': str(total_stats['ModeString']),
        'ModeCnt': total_mode_count,
        'Mode(%)': f"{total_mode_ratio:.2f}",
        'FormatMin': str(format_stats['FormatMin']),
        'FormatMax': str(format_stats['FormatMax']),
        'FormatMode': str(format_stats['FormatMode']),
        'FormatMedian': str(format_stats['FormatMedian']),
        'Format2ndMin': str(format_stats['Format2ndMin']),
        'Format2ndMax': str(format_stats['Format2ndMax']),
        'Format2ndMode': str(format_stats['Format2ndMode']),
        'Format3rdMin': str(format_stats['Format3rdMin']),
        'Format3rdMax': str(format_stats['Format3rdMax']),
        'Format3rdMode': str(format_stats['Format3rdMode']),
        'HasBlank': whitespace_count,
        'HasDash': dash_count,
        'HasDot': dot_count,
        'HasAt': at_count,
        'HasAlpha': eng_count,
        'HasKor': kor_count,
        'HasNum': numeric_count,
        'HasBracket': bracket_count,
        'HasMinus': negative_count,  # 음수값 개수
        'HasOnlyAlpha': has_only_alpha,
        'HasOnlyNum': has_only_num,
        'HasOnlyKor': has_only_kor,
        'HasOnlyAlphanum': has_only_alphanum,
        'FirstChrKor': first_char_is_kor,
        'FirstChrNum': first_char_is_num,
        'FirstChrAlpha': first_char_is_alpha,
        'FirstChrSpecial': first_char_is_special
    }

    return stats

def Read_Source_File(file_path, code_type, extension, sample_size=Constants.SAMPLING_ROWS): 
    try:
        # 파일 크기를 먼저 확인
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"경고: {file_path}가 비어있습니다.")
            return None, 0, None
            
        # 전체 데이터를 읽은 후 필요한 경우 샘플링
        if extension == '.csv':
            df = pd.read_csv(file_path, low_memory=False)  # low_memory=False 옵션 추가
        else:
            df = pd.read_pickle(file_path)
        row_count = len(df)
        Column_count = len(df.columns)
        
        # 샘플링된 데이터프레임 생성
        sampled_df = df
        if len(df) > sample_size:
            sampled_df = df.sample(n=sample_size, random_state=42)
        
        # 파일 통계 데이터프레임 생성
        file_stats = pd.DataFrame({
            'FileName': file_name,
            'MasterType': [code_type],
            'FileSize': [file_size],
            'RecordCnt': [row_count],
            'ColumnCnt': [Column_count],
            'SamplingRows': [len(sampled_df)],
            'Sampling(%)': [round(len(sampled_df) / row_count * 100, 2) if row_count > 0 else 0], 
            'WorkDate': [datetime.now().strftime('%Y-%m-%d')]
        })

        print(f"{[file_name]}   (Total Record # : {row_count:,})")
        return sampled_df, row_count, file_stats
        
    except Exception as e:
        print(f"파일 읽기 오류: {str(e)}")
        return None, 0, None

def Format_Process_File(file_info):
    """파일 처리 함수 최적화"""
    file, Source_folder, code_type, extension = file_info
    file_path = os.path.join(Source_folder, file)
    
    try:
        # 데이터 로드 최적화
        df, row_count, file_stats = Read_Source_File(file_path, code_type, extension) 

        if df is None:
            return [], None

        unique_columns = Find_Unique_Combination(df)

        # 컬럼별 병렬 처리를 위한 준비
        results = []
        for column in df.columns:
            stats = Analyze_Column_Statistics(df, column)
            Column_info = {
                'FileName': str(file),
                'ColumnName': str(column),
                'MasterType': code_type,
                'DataType': stats.get('DataType', ''),  
                'OracleType': stats.get('OracleType', ''),
                'PK': 1 if column in unique_columns else 0,
                'DetailDataType': stats.get('DetailDataType', ''),
                'LenCnt': stats.get('LenCnt', ''),
                'LenMin': stats.get('LenMin', 0),
                'LenMax': stats.get('LenMax', 0),
                'LenAvg': stats.get('LenAvg', 0),
                'LenMode': stats.get('LenMode', 0),
                'RecordCnt': row_count,
                'SampleRows': stats.get('SampleRows', 0),
                'ValueCnt': stats.get('ValueCnt', 0),
                'NullCnt': stats.get('NullCnt', 0),
                'Null(%)': stats.get('Null(%)', 0),
                'UniqueCnt': stats.get('UniqueCnt', 0),
                'Unique(%)': stats.get('Unique(%)', 0),
                'FormatCnt': stats.get('FormatCnt', 0),
                'Format': str(stats.get('Format', '')),
                'FormatLength': stats.get('FormatLength', 0),
                'FormatValue': stats.get('FormatValue', 0),
                'Format(%)': stats.get('Format(%)', 0),
                'Format2nd': str(stats.get('Format2nd', '')),
                'Format2ndValue': stats.get('Format2ndValue', 0),
                'Format2nd(%)': stats.get('Format2nd(%)', 0),
                'Format3rd': str(stats.get('Format3rd', '')),
                'Format3rdValue': stats.get('Format3rdValue', 0),
                'Format3rd(%)': stats.get('Format3rd(%)', 0),
                'MinString': str(stats.get('MinString', '')),
                'MaxString': str(stats.get('MaxString', '')),
                'ModeString': str(stats.get('ModeString', '')),
                'ModeCnt': stats.get('ModeCnt', 0),
                'Mode(%)': stats.get('Mode(%)', 0),
                'FormatMin': str(stats.get('FormatMin', '')),
                'FormatMax': str(stats.get('FormatMax', '')),
                'FormatMode': str(stats.get('FormatMode', '')),
                'FormatMedian': str(stats.get('FormatMedian', '')),
                'Format2ndMin': str(stats.get('Format2ndMin', '')),
                'Format2ndMax': str(stats.get('Format2ndMax', '')),
                'Format2ndMode': str(stats.get('Format2ndMode', '')),
                'Format3rdMin': str(stats.get('Format3rdMin', '')),
                'Format3rdMax': str(stats.get('Format3rdMax', '')),
                'Format3rdMode': str(stats.get('Format3rdMode', '')),
                'HasBlank': stats.get('HasBlank', 0), 
                'HasDash': stats.get('HasDash', 0), 
                'HasDot': stats.get('HasDot', 0), 
                'HasAt':stats.get('HasAt', 0), 
                'HasAlpha': stats.get('HasAlpha', 0), 
                'HasKor': stats.get('HasKor', 0), 
                'HasNum': stats.get('HasNum', 0), 
                'HasBracket': stats.get('HasBracket', 0), 
                'HasMinus': stats.get('HasNegativeCnt', 0),
                'HasOnlyAlpha': stats.get('HasOnlyAlpha', 0),
                'HasOnlyNum': stats.get('HasOnlyNum', 0),
                'HasOnlyKor': stats.get('HasOnlyKor', 0),
                'HasOnlyAlphanum': stats.get('HasOnlyAlphanum', 0),
                'FirstChrKor': stats.get('FirstChrKor', 0),
                'FirstChrNum': stats.get('FirstChrNum', 0),
                'FirstChrAlpha': stats.get('FirstChrAlpha', 0),
                'FirstChrSpecial': stats.get('FirstChrSpecial', 0)
            }
            results.append(Column_info)
            
        return results, file_stats
        
    except Exception as e:
        print(f"\n파일 처리 중 오류: {file}\n{str(e)}")
        return [], None
    
def Format_Build(InputDir, OutputDir, Code_Type, Extension):
    """
    파일 처리 함수
    Returns:
        tuple: (success: bool, columns_result: list, file_stats_result: list)
    """
    all_Columns = []
    all_file_stats = []
    processed_count = 0
    error_count = 0

    try:
        # 디렉토리 존재 여부 확인
        if not os.path.exists(InputDir):
            print(f"경고: 입력 디렉토리가 존재하지 않습니다: {InputDir}")
            return False, [], []

        # 파일 목록 가져오기
        if Extension == '.csv':
            files = [f for f in os.listdir(InputDir) if f.lower().endswith('.csv')]
        else:
            files = [f for f in os.listdir(InputDir) if f.lower().endswith('.pkl')]

        if not files:
            print(f"경고: {Extension} 파일을 찾을 수 없습니다: {InputDir}")
            return False, [], []

        total_files = len(files)
        print(f"Code Type: {Code_Type} 총 {total_files}개의 파일을 처리합니다.")
        file_info = [(f, InputDir, Code_Type, Extension) for f in files]
        
        # 멀티프로세싱 실행
        with Pool(processes=cpu_count()) as pool:
            for i, (result, file_stats) in enumerate(pool.imap_unordered(Format_Process_File, file_info), 1):
                if result:  # None이 아닌 경우에만 추가
                    all_Columns.extend(result)
                if file_stats is not None:
                    all_file_stats.append(file_stats)
                processed_count += len(result) if result else 0

        # 처리 결과 확인
        if processed_count == 0:
            print("경고: 처리된 데이터가 없습니다.")
            return False, [], []

        return True, all_Columns, all_file_stats

    except Exception as e:
        print(f"Format_Build 실행 중 오류 발생: {str(e)}")
        return False, [], []

