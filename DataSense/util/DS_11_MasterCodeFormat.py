# ======================================================================
# DataSense DQ Profiling (Refactored Final)
# ----------------------------------------------------------------------
import os
import re
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from collections import Counter
from random import Random
import fnmatch
import traceback
import sys
from pathlib import Path

# --- 외부 유틸/룰 엔진 -----------------------------------------------
# 현재 파일의 상위 2단계 폴더를 path에 추가
ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from DataSense.util.io import Load_Yaml_File, Backup_File

from DataSense.util.dq_function import (DataType_Analysis, create_standard_df, Determine_Detail_Type,
     Get_Oracle_Type, _ensure_severity_column, Find_Unique_Combination, 
     build_top_issue_reports, save_or_load_baseline, compute_proxy_drift,
     make_df_for_distribution, build_dist_snapshot_for_df, collect_value_samples, 
     dist_topk_categories, add_dq_scores, apply_score_importance,
     compute_snapshot_drift, 
     )
# ======================================================================
# Global Config
# ======================================================================
FORMAT_MAX_VALUE   = 1000   # Format 검사 최대 길이 한계
FORMAT_AVG_LENGTH  = 50     # 평균 길이 기준(문장형 텍스트 추정)
vSamplingRows      = 10000
YAML_PATH          = 'DataSense/util/DS_Master.yaml' # 'DataSense/util/DS_Diretory_Config.yaml' #
CODEY_NAME_HINT    = re.compile(r"(zip|postal|우편|code|코드|id|식별|번호)$", re.IGNORECASE)
_FLOAT_ZERO_RE = re.compile(r'^[+-]?\d+\.0+$')
# ===== Global Contract Holder =====
GLOBAL_CONTRACT = None
def set_global_contract(contract_dict):
    """자식 프로세스에서 접근할 계약 객체를 전역에 주입"""
    global GLOBAL_CONTRACT
    GLOBAL_CONTRACT = contract_dict

# ======================================================================
# IO
# ======================================================================
def read_csv_any(path: str) -> pd.DataFrame:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise FileNotFoundError(path)

def _to_pct(x) -> float:
    """'12.34' / '12.34%' / '' / None 등을 안전하게 float(%)로."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).strip().replace('%', '')
    if s == '' or s.lower() == 'nan':
        return 0.0
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except:
        return 0.0
def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')

def _proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total

def _strip_decimal_zero_if_numeric_str(x: object) -> str:
    """
    '123.0' '123.000' 같은 '숫자.0+' 형태만 → '123'으로.
    선행 0가 있는 코드('01000')나 실제 소수('1.25'), 버전('v1.0') 등은 건드리지 않음.
    """
    s = str(x)
    if _FLOAT_ZERO_RE.fullmatch(s):
        return s.split('.', 1)[0]
    return s 

def safe_int(x, default=0):
    try:
        if pd.isna(x) or x == '':
            return default
        return int(x)
    except Exception:
        return default

def safe_float(x, default=0.0):
    try:
        if pd.isna(x) or x == '':
            return default
        return float(x)
    except Exception:
        return default

# ======================================================================
# Format Builder / Pattern
# ======================================================================
def Get_String_Pattern(value, dtype=None) -> str:
    """문자열의 패턴을 반환합니다. (모든 입력 문자열 기반, 안전 정규화 포함)"""
    str_value = _strip_decimal_zero_if_numeric_str(value)[:20]

    pattern = []
    for ch in str_value:
        if ch.isdigit():                 pattern.append('n')
        elif '가' <= ch <= '힣':         pattern.append('K')
        elif ch in '(){}[]':             pattern.append(ch)
        elif ch.isalpha():               pattern.append('A' if ch.isupper() else 'a')
        elif ch in {'-', '=', '.', ' ', ':', '@', '/'}: pattern.append(ch)
        elif ch in {'\t','\n','\r'}:     pattern.append('_')
        else:                            pattern.append('s')
    pat = ''.join(pattern)
    return f"'{pat}" if pat.startswith('-') else pat

def _normalize_for_pattern(value, dtype_str: str) -> str:
    """패턴 통계용 값 정규화 (문자열에서도 '숫자.0+' → '숫자')"""
    return _strip_decimal_zero_if_numeric_str(value)

def Analyze_Column_Format(series):
    """
    컬럼 내 문자열 패턴 분석:
    - 최빈 패턴 Top3
    - 각 패턴의 Min/Max/Mode/Median 값
    """
    max_length = series.astype(str).str.len().max()
    columns = ['pattern_type_cnt','most_common_pattern','most_common_count',
               'second_common_pattern','second_common_count',
               'third_common_pattern','third_common_count',
               'FormatMin','FormatMax','FormatMode','FormatMedian',
               'Format2ndMin','Format2ndMax','Format2ndMode','Format2ndMedian',
               'Format3rdMin','Format3rdMax','Format3rdMode','Format3rdMedian']
    if max_length > FORMAT_MAX_VALUE or max_length == 0:
        return {k: (0 if 'count' in k or 'Cnt' in k else '') for k in columns}

    s_clean   = series.dropna()
    dtype_str = str(series.dtype)
    pats      = pd.Series([Get_String_Pattern(v, series.dtype) for v in s_clean], dtype='object')
    cnt       = Counter(pats.tolist())
    if not cnt:
        return {k: (0 if 'count' in k or 'Cnt' in k else '') for k in columns}

    top = cnt.most_common(3)

    def _stats_for_pat(pat):
        mask    = (pats == pat).values
        vals_rw = s_clean[mask]
        if vals_rw.empty:
            return {'MinString':'','MaxString':'','ModeString':'','Median':''}
        vals = vals_rw.map(lambda x: _normalize_for_pattern(x, dtype_str)).astype(str)
        sorted_vals = vals.sort_values(kind='mergesort')
        n = len(sorted_vals)
        median_val = sorted_vals.iloc[(n-1)//2]
        mode_series = sorted_vals.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else ''
        return {'MinString': sorted_vals.iloc[0],
                'MaxString': sorted_vals.iloc[-1],
                'ModeString': mode_val,
                'Median':   median_val}

    stats1 = _stats_for_pat(top[0][0])
    stats2 = _stats_for_pat(top[1][0]) if len(top) > 1 else {'MinString':'','MaxString':'','ModeString':'','Median':''}
    stats3 = _stats_for_pat(top[2][0]) if len(top) > 2 else {'MinString':'','MaxString':'','ModeString':'','Median':''}

    out = {
        'pattern_type_cnt':      len(cnt),
        'most_common_pattern':   top[0][0],
        'most_common_count':     top[0][1],
        'second_common_pattern': top[1][0] if len(top) > 1 else '',
        'second_common_count':   top[1][1] if len(top) > 1 else 0,
        'third_common_pattern':  top[2][0] if len(top) > 2 else '',
        'third_common_count':    top[2][1] if len(top) > 2 else 0,
        'FormatMin':             stats1['MinString'],
        'FormatMax':             stats1['MaxString'],
        'FormatMode':            stats1['ModeString'],
        'FormatMedian':          stats1['Median'],
        'Format2ndMin':          stats2['MinString'],
        'Format2ndMax':          stats2['MaxString'],
        'Format2ndMode':         stats2['ModeString'],
        'Format2ndMedian':       stats2['Median'],
        'Format3rdMin':          stats3['MinString'],
        'Format3rdMax':          stats3['MaxString'],
        'Format3rdMode':         stats3['ModeString'],
        'Format3rdMedian':       stats3['Median'],
    }
    try:
        if out['most_common_pattern']:
            pat_min = Get_String_Pattern(out['FormatMin'], series.dtype)
            out['Pattern_Mismatch'] = (pat_min != out['most_common_pattern'])
    except Exception:
        out['Pattern_Mismatch'] = True
    return out

# ======================================================================
# Column Statistics
# ======================================================================
def Calculate_Statistics(s: pd.Series):
    """ 시리즈의 통계를 계산하는 함수입니다. """
    s = s.dropna()
    if len(s) == 0:
        return {"min": None, "max": None, "mode": None, "median": None}
    if pd.api.types.is_datetime64_any_dtype(s):
        min_val, max_val = s.min(), s.max()
        mode_val = s.mode().iloc[0] if not s.mode().empty else None
        median_val = s.sort_values().iloc[len(s)//2]
    elif pd.api.types.is_numeric_dtype(s):
        min_val, max_val = s.min(), s.max()
        mode_val = s.mode().iloc[0] if not s.mode().empty else None
        median_val = s.median()
    else:
        s = s.astype(str)
        s = s[(s != '') & (s.str.lower() != 'nan')]
        if len(s) == 0:
            return {"min": None, "max": None, "mode": None, "median": None}
        min_val, max_val = s.min(), s.max()
        mode_val = s.mode().iloc[0] if not s.mode().empty else None
        arr = sorted(s); mid = len(arr)//2
        median_val = arr[mid] if len(arr)%2==1 else arr[mid-1]
    return {"min": min_val, "max": max_val, "mode": mode_val, "median": median_val}

def Analyze_Column_Statistics(df, column):
    """데이터 품질 점검 도구의 필드 통계 분석 함수 (안전 변환 적용 버전)"""
    series = df[column]
    non_null_series = series.dropna()
    str_nonnull     = non_null_series.astype(str)

    # -------------------------------
    # 처음 두 문자 & 마지막 두 문자 다빈도값 (추가)
    # -------------------------------
    first1 = str_nonnull.str[:1]
    last1  = str_nonnull.str[-1:]
    first2 = str_nonnull.str[:2]
    last2  = str_nonnull.str[-2:]
    first3 = str_nonnull.str[:3]
    last3  = str_nonnull.str[-3:]

    def top3_modes(s: pd.Series):
        """Series에서 Top3 모드 추출 (문자화, 없는 경우 빈 문자열)"""
        if s.empty:
            return ["", "", ""], ["0", "0", "0"]
        counts = s.value_counts()
        top_values = counts.index.astype(str).tolist()[:3]
        top_counts = counts.astype(str).tolist()[:3]
        # 부족할 경우 채우기
        while len(top_values) < 3:
            top_values.append("")
            top_counts.append("0")
        return top_values, top_counts

    def top10_modes(s: pd.Series):
        """Series에서 Top10 모드 추출 (JSON 직렬화)"""
        if s.empty:
            return json.dumps([])
        counts = s.value_counts().head(10)  # 상위 10개
        # result = [{"value": str(idx), "count": int(cnt)} for idx, cnt in counts.items()]
        result = [str(idx) for idx, cnt in counts.items()]
        # return result
        return json.dumps(result, ensure_ascii=False)

    first1_modes, first1_counts = top3_modes(first1)
    last1_modes, last1_counts   = top3_modes(last1)
    first2_modes, first2_counts = top3_modes(first2)
    last2_modes, last2_counts   = top3_modes(last2)
    first3_modes, first3_counts = top3_modes(first3)
    last3_modes, last3_counts   = top3_modes(last3)


    first1_top10 =  top10_modes(first1)
    last1_top10 =   top10_modes(last1)
    first2_top10 =  top10_modes(first2)
    last2_top10 =   top10_modes(last2)
    first3_top10 =  top10_modes(first3)
    last3_top10 =   top10_modes(last3)

    lengths         = str_nonnull.str.len()
    length_counts   = lengths.value_counts()
    non_null_count  = len(non_null_series)
    null_count      = len(series) - non_null_count
    unique_count    = non_null_series.nunique(dropna=True)
    len_cnt         = len(length_counts)
    
    # 안전한 변환 적용
    min_length      = safe_int(lengths.min())
    max_length      = safe_int(lengths.max())
    avg_length      = safe_float(lengths.mean())
    mode_length     = safe_int(lengths.mode().iloc[0]) if not lengths.mode().empty else 0
    
    total_stats      = Calculate_Statistics(series)
    total_mode_value = str(total_stats['mode'])
    total_mode_count = str_nonnull.eq(total_mode_value).sum()
    total_mode_ratio = (total_mode_count / non_null_count * 100) if non_null_count > 0 else 0

    has_only_alpha    = str_nonnull.str.fullmatch(r'[a-zA-Z]+').sum()
    has_only_num      = str_nonnull.str.fullmatch(r'[0-9]+').sum()
    has_only_kor      = str_nonnull.str.fullmatch(r'[가-힣]+').sum()
    has_only_alphanum = str_nonnull.str.fullmatch(r'[a-zA-Z0-9]+').sum()

    first_char = str_nonnull.str[0]
    first_char_is_kor     = first_char.str.contains(r'[가-힣]', na=False).sum()
    first_char_is_num     = first_char.str.contains(r'^[0-9]$', na=False).sum()
    first_char_is_alpha   = first_char.str.contains(r'^[a-zA-Z]$', na=False).sum()
    first_char_is_special = first_char.str.contains(r'[^a-zA-Z0-9가-힣\s]', na=False).sum()

    whitespace_count = str_nonnull.str.contains(r"\s", na=False).sum()
    dash_count       = str_nonnull.str.contains(r"-",  na=False).sum()
    dot_count        = str_nonnull.str.contains(r"\.", na=False).sum()
    at_count         = str_nonnull.str.contains(r"@",  na=False).sum()
    eng_count        = str_nonnull.str.contains(r"[a-zA-Z]", na=False).sum()
    kor_count        = str_nonnull.str.contains(r"[가-힣]",   na=False).sum()
    numeric_count    = str_nonnull.str.contains(r"[0-9]",     na=False).sum()
    bracket_count    = str_nonnull.str.contains(r"[(){}\[\]]",na=False).sum()

    num_ser =       _safe_to_numeric(series)
    negative_count = safe_int((num_ser < 0).sum())

    oracle_type =  Get_Oracle_Type(series, column_name=column)
    # oracle_type = ''

    format_stats = Analyze_Column_Format(series)
    if not isinstance(format_stats, dict):
        format_stats = {}


    # Top10 카테고리 분포(문자열 기준)
    labels, props = dist_topk_categories(series, k=10)
    top10 = json.dumps(labels, ensure_ascii=False)
    top10_pct = json.dumps(props.tolist())

    detail_type = Determine_Detail_Type(
        format_stats.get('most_common_pattern',''),
        format_stats.get('pattern_type_cnt',0),
        format_stats,
        total_stats,
        max_length,
        unique_count,
        non_null_count, 
        top10
    )

    if detail_type == 'CLOB':
        total_stats = {'min':'','max':'','mode':'','median':''}

    stats = {
        'DataType':       str(series.dtype),
        'OracleType':     oracle_type,
        'DetailDataType': detail_type,
        'LenCnt':         safe_int(len_cnt),
        'LenMin':         safe_int(min_length),
        'LenMax':         safe_int(max_length),
        'LenAvg':         safe_int(avg_length),
        'LenMode':        safe_int(mode_length),
        'SampleRows':     safe_int(len(series)),
        'ValueCnt':       safe_int(non_null_count),
        'NullCnt':        safe_int(null_count),
        'Null(%)':        f"{(null_count / len(series) * 100):.2f}" if len(series) > 0 else "0.00",
        'UniqueCnt':      safe_int(unique_count),
        'Unique(%)':      f"{(unique_count / len(series) * 100):.2f}" if len(series) > 0 else "0.00",
        'FormatCnt':      safe_int(format_stats.get('pattern_type_cnt',0)),
        'Format':         str(format_stats.get('most_common_pattern','')),
        'FormatLength':   safe_int(len(format_stats.get('most_common_pattern',''))),
        'FormatBlankCnt': safe_int(format_stats.get('most_common_pattern','').count(' ')),
        'FormatValue':    safe_int(format_stats.get('most_common_count',0)),
        'Format(%)':      f"{(safe_float(format_stats.get('most_common_count',0)) / non_null_count * 100):.2f}" if non_null_count > 0 else "0.00",
        'Format2nd':      str(format_stats.get('second_common_pattern','')),
        'Format2ndValue': safe_int(format_stats.get('second_common_count',0)),
        'Format2nd(%)':   f"{(safe_float(format_stats.get('second_common_count',0)) / non_null_count * 100):.2f}" if non_null_count > 0 else "0.00",
        'Format3rd':      str(format_stats.get('third_common_pattern','')),
        'Format3rdValue': safe_int(format_stats.get('third_common_count',0)),
        'Format3rd(%)':   f"{(safe_float(format_stats.get('third_common_count',0)) / non_null_count * 100):.2f}" if non_null_count > 0 else "0.00",
        'MinString':      str(total_stats.get('min','')),
        'MaxString':      str(total_stats.get('max','')),
        'ModeString':     str(total_stats.get('mode','')),
        'MedianString':   str(total_stats.get('median','')),
        'ModeCnt':        safe_int(total_mode_count),
        'Mode(%)':        f"{total_mode_ratio:.2f}",
        'FormatMin':      str(format_stats.get('FormatMin','')),
        'FormatMax':      str(format_stats.get('FormatMax','')),
        'FormatMode':     str(format_stats.get('FormatMode','')),
        'FormatMedian':   str(format_stats.get('FormatMedian','')),
        'Format2ndMin':   str(format_stats.get('Format2ndMin','')),
        'Format2ndMax':   str(format_stats.get('Format2ndMax','')),
        'Format2ndMode':  str(format_stats.get('Format2ndMode','')),
        'Format2ndMedian':str(format_stats.get('Format2ndMedian','')),
        'Format3rdMin':   str(format_stats.get('Format3rdMin','')),
        'Format3rdMax':   str(format_stats.get('Format3rdMax','')),
        'Format3rdMode':  str(format_stats.get('Format3rdMode','')),
        'Format3rdMedian':str(format_stats.get('Format3rdMedian','')),

        'Top10':        top10,
        'Top10(%)':     top10_pct,

        'First2M1':     first2_modes[0], 'First2Cnt1': first2_counts[0],
        'First2M2':     first2_modes[1], 'First2Cnt2': first2_counts[1],
        'First2M3':     first2_modes[2], 'First2Cnt3': first2_counts[2],
        'First1M1':     first1_modes[0], 'First1Cnt1': first1_counts[0],
        'First1M2':     first1_modes[1], 'First1Cnt2': first1_counts[1],
        'First1M3':     first1_modes[2], 'First1Cnt3': first1_counts[2],
        'Last2M1':      last2_modes[0],  'Last2Cnt1':  last2_counts[0],
        'Last2M2':      last2_modes[1],  'Last2Cnt2':  last2_counts[1],
        'Last2M3':      last2_modes[2],  'Last2Cnt3':  last2_counts[2],
        'Last1M1':      last1_modes[0],  'Last1Cnt1':  last1_counts[0],
        'Last1M2':      last1_modes[1],  'Last1Cnt2':  last1_counts[1],
        'Last1M3':      last1_modes[2],  'Last1Cnt3':  last1_counts[2],
        'First3M1':     first3_modes[0], 'First3Cnt1': first3_counts[0],
        'First3M2':     first3_modes[1], 'First3Cnt2': first3_counts[1],
        'First3M3':     first3_modes[2], 'First3Cnt3': first3_counts[2],
        'Last3M1':      last3_modes[0],  'Last3Cnt1':  last3_counts[0],
        'Last3M2':      last3_modes[1],  'Last3Cnt2':  last3_counts[1],
        'Last3M3':      last3_modes[2],  'Last3Cnt3':  last3_counts[2],

        'First1Top10':  first1_top10,
        'Last1Top10':   last1_top10,
        'First2Top10':  first2_top10,
        'Last2Top10':   last2_top10,
        'First3Top10':  first3_top10,
        'Last3Top10':   last3_top10,

        'HasBlank':       safe_int(whitespace_count),
        'HasDash':        safe_int(dash_count),
        'HasDot':         safe_int(dot_count),
        'HasAt':          safe_int(at_count),
        'HasAlpha':       safe_int(eng_count),
        'HasKor':         safe_int(kor_count),
        'HasNum':         safe_int(numeric_count),
        'HasBracket':     safe_int(bracket_count),
        'HasMinus':       safe_int(negative_count),
        'HasOnlyAlpha':   safe_int(has_only_alpha),
        'HasOnlyNum':     safe_int(has_only_num),
        'HasOnlyKor':     safe_int(has_only_kor),
        'HasOnlyAlphanum':safe_int(has_only_alphanum),
        'FirstChrKor':    safe_int(first_char_is_kor),
        'FirstChrNum':    safe_int(first_char_is_num),
        'FirstChrAlpha':  safe_int(first_char_is_alpha),
        'FirstChrSpecial':safe_int(first_char_is_special),
        'CompareLength':  0,
        'Format1st':      str(format_stats.get('Format','')),

    }
    return stats
# ======================================================================
# File IO (기본 force_str=True)
# ======================================================================
def Read_Source_File(file_path, code_type, extension, sample_size=vSamplingRows, force_str=True):
    try:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"경고: {file_path}가 비어있습니다.")
            return None, 0, None

        read_kwargs_csv  = dict(low_memory=False, on_bad_lines="skip",
                                na_values=['nan','NaN','NULL','','#N/A','N/A','NA'],
                                keep_default_na=True)
        read_kwargs_xlsx = dict(na_values=['nan','NaN','NULL','','#N/A','N/A','NA'],
                                keep_default_na=True)

        if extension == '.csv':
            df = pd.read_csv(file_path, dtype=str, **read_kwargs_csv) if force_str else pd.read_csv(file_path, **read_kwargs_csv)
        elif extension == '.xlsx':
            df = pd.read_excel(file_path, dtype=str, **read_kwargs_xlsx) if force_str else pd.read_excel(file_path, **read_kwargs_xlsx)
        elif extension == '.pkl':
            df = pd.read_pickle(file_path)
        elif extension == '.json':
            df = pd.read_json(file_path, dtype="string")
        else:
            print(f"지원하지 않는 파일 형식: {extension}")
            return None, 0, None

        row_count    = len(df)
        column_count = len(df.columns)
        sampled_df   = df if len(df) <= sample_size else df.sample(n=sample_size, random_state=42)

        file_stats = pd.DataFrame({
            'FilePath':     file_path,
            'FileName':     file_name,
            'MasterType':   str(code_type),
            'FileSize':     [file_size],
            'RecordCnt':    [row_count],
            'ColumnCnt':    [column_count],
            'SamplingRows': [len(sampled_df)],
            'Sampling(%)':  [round(len(sampled_df) / row_count * 100, 2) if row_count > 0 else 0],
            'WorkDate':     [datetime.now().strftime('%Y-%m-%d')]
        })
        return sampled_df, row_count, file_stats

    except Exception as e:
        print(f"파일 읽기 오류: {str(e)}")
        return None, 0, None

# ======================================================================
# Per-file processing
# ======================================================================
def Format_Process_File(file_info):
    file, Source_folder, code_type, extension = file_info
    file_path = os.path.join(Source_folder, file)
    file_path = file_path.replace("\\", "/")
    try:
        # 기본: 모두 문자열로 읽기(force_str=True) → 선행 0 보존
        df, row_count, file_stats = Read_Source_File(file_path, code_type, extension, force_str=True)
        if df is None:
            return [], None, [], []

        unique_columns = Find_Unique_Combination(df)

        results, rules = [], []
        for column in df.columns:
            stats = Analyze_Column_Statistics(df, column)
            Column_info = {
                'FilePath':     file_path,
                'FileName':     str(file),
                'ColumnName':   str(column),
                'MasterType':   str(code_type),
                'DataType':     stats.get('DataType',''),
                'OracleType':   stats.get('OracleType',''),
                'PK':           1 if column in unique_columns else 0,
                'DetailDataType': stats.get('DetailDataType',''),
                'LenCnt':       stats.get('LenCnt',''),
                'LenMin':       stats.get('LenMin',0),
                'LenMax':       stats.get('LenMax',0),
                'LenAvg':       stats.get('LenAvg',0),
                'LenMode':      stats.get('LenMode',0),
                'RecordCnt':    row_count,
                'SampleRows':   stats.get('SampleRows',0),
                'ValueCnt':     stats.get('ValueCnt',0),
                'NullCnt':      stats.get('NullCnt',0),
                'Null(%)':      stats.get('Null(%)',0),
                'UniqueCnt':    stats.get('UniqueCnt',0),
                'Unique(%)':    stats.get('Unique(%)',0),
                'FormatCnt':    stats.get('FormatCnt',0),
                'Format':       stats.get('Format',0),
                'FormatLength': stats.get('FormatLength',0),
                'FormatBlankCnt': stats.get('FormatBlankCnt',0),
                'FormatValue':  stats.get('FormatValue',0),
                'Format(%)':    stats.get('Format(%)',0),
                'Format2nd':    str(stats.get('Format2nd','')),
                'Format2ndValue': stats.get('Format2ndValue',0),
                'Format2nd(%)': stats.get('Format2nd(%)',0),
                'Format3rd':    str(stats.get('Format3rd','')),
                'Format3rdValue': stats.get('Format3rdValue',0),
                'Format3rd(%)': stats.get('Format3rd(%)',0),
                'MinString':    str(stats.get('MinString','')),
                'MaxString':    str(stats.get('MaxString','')),
                'ModeString':   str(stats.get('ModeString','')),
                'MedianString': str(stats.get('MedianString','')),
                'ModeCnt':      stats.get('ModeCnt',0),
                'Mode(%)':      stats.get('Mode(%)',0),
                'FormatMin':    str(stats.get('FormatMin','')),
                'FormatMax':    str(stats.get('FormatMax','')),
                'FormatMode':   str(stats.get('FormatMode','')),
                'FormatMedian': str(stats.get('FormatMedian','')),
                'Format2ndMin': str(stats.get('Format2ndMin','')),
                'Format2ndMax': str(stats.get('Format2ndMax','')),
                'Format2ndMode': str(stats.get('Format2ndMode','')),
                'Format2ndMedian': str(stats.get('Format2ndMedian','')),
                'Format3rdMin': str(stats.get('Format3rdMin','')),
                'Format3rdMax': str(stats.get('Format3rdMax','')),
                'Format3rdMode': str(stats.get('Format3rdMode','')),
                'Format3rdMedian': str(stats.get('Format3rdMedian','')),

                'Top10':        stats.get('Top10',''),
                'Top10(%)':     stats.get('Top10(%)',''),

                'First2M1':     str(stats.get('First2M1','')),
                'First2Cnt1':   safe_int(stats.get('First2Cnt1','')),
                'First2M2':     str(stats.get('First2M2','')),
                'First2Cnt2':   safe_int(stats.get('First2Cnt2','')),
                'First2M3':     str(stats.get('First2M3','')),
                'First2Cnt3':   safe_int(stats.get('First2Cnt3','')),
                'Last2M1':      str(stats.get('Last2M1','')),
                'Last2Cnt1':    safe_int(stats.get('Last2Cnt1','')),
                'Last2M2':      str(stats.get('Last2M2','')),
                'Last2Cnt2':    safe_int(stats.get('Last2Cnt2','')),
                'Last2M3':      str(stats.get('Last2M3','')),
                'Last2Cnt3':    safe_int(stats.get('Last2Cnt3','')),
                'First1M1':     str(stats.get('First1M1','')),
                'First1Cnt1':   safe_int(stats.get('First1Cnt1','')),
                'First1M2':     str(stats.get('First1M2','')),
                'First1Cnt2':   safe_int(stats.get('First1Cnt2','')),
                'First1M3':     str(stats.get('First1M3','')),
                'First1Cnt3':   safe_int(stats.get('First1Cnt3','')),
                'Last1M1':      str(stats.get('Last1M1','')),
                'Last1Cnt1':    safe_int(stats.get('Last1Cnt1','')),
                'Last1M2':      str(stats.get('Last1M2','')),
                'Last1Cnt2':    safe_int(stats.get('Last1Cnt2','')),
                'Last1M3':      str(stats.get('Last1M3','')),
                'Last1Cnt3':    safe_int(stats.get('Last1Cnt3','')),
                'First3M1':     str(stats.get('First3M1','')),
                'First3Cnt1':   safe_int(stats.get('First3Cnt1','')),
                'First3M2':     str(stats.get('First3M2','')),
                'First3Cnt2':   safe_int(stats.get('First3Cnt2','')),
                'First3M3':     str(stats.get('First3M3','')),
                'First3Cnt3':   safe_int(stats.get('First3Cnt3','')),
                'Last3M1':      str(stats.get('Last3M1','')),
                'Last3Cnt1':    safe_int(stats.get('Last3Cnt1','')),
                'Last3M2':      str(stats.get('Last3M2','')),
                'Last3Cnt2':    safe_int(stats.get('Last3Cnt2','')),
                'Last3M3':      str(stats.get('Last3M3','')),
                'Last3Cnt3':    safe_int(stats.get('Last3Cnt3','')),

                'First1Top10':  stats.get('First1Top10',''),
                'Last1Top10':   stats.get('Last1Top10',''),
                'First2Top10':  stats.get('First2Top10',''),
                'Last2Top10':   stats.get('Last2Top10',''),
                'First3Top10':  stats.get('First3Top10',''),
                'Last3Top10':   stats.get('Last3Top10',''),

                'HasBlank':     stats.get('HasBlank',0),
                'HasDash':      stats.get('HasDash',0),
                'HasDot':       stats.get('HasDot',0),
                'HasAt':        stats.get('HasAt',0),
                'HasAlpha':     stats.get('HasAlpha',0),
                'HasKor':       stats.get('HasKor',0),
                'HasNum':       stats.get('HasNum',0),
                'HasBracket':   stats.get('HasBracket',0),
                'HasMinus':     stats.get('HasMinus',0),
                'HasOnlyAlpha': stats.get('HasOnlyAlpha',0),
                'HasOnlyNum':   stats.get('HasOnlyNum',0),
                'HasOnlyKor':   stats.get('HasOnlyKor',0),
                'HasOnlyAlphanum': stats.get('HasOnlyAlphanum',0),
                'FirstChrKor':  stats.get('FirstChrKor',0),
                'FirstChrNum':  stats.get('FirstChrNum',0),
                'FirstChrAlpha': stats.get('FirstChrAlpha',0),
                'FirstChrSpecial': stats.get('FirstChrSpecial',0),
                'CompareLength': 0,
                'Format1st':    str(stats.get('Format','')),
            }
            results.append(Column_info)

        # 분포/샘플: 계약 기반 zfill 등 정규화 반영
        df_for_dist = make_df_for_distribution(df, file_name=str(file), contract_dict=GLOBAL_CONTRACT)
        dist_rows   = build_dist_snapshot_for_df(df_for_dist, file_name=str(file))
        samples     = collect_value_samples(df, file_name=str(file), contract_dict=GLOBAL_CONTRACT,
                                            per_col=200, random_state=42, normalize=True)

        return results, file_stats, dist_rows, samples

    except Exception as e:
        print(f"\n파일 처리 중 오류: {file}\n{str(e)}")
        return [], None, [], []

def Format_Build(InputDir, Code_Type, Extension, contract_dict=None):
    all_Columns, all_file_stats, all_dist, all_samples = [], [], [], []
    ext = Extension.lower()

    if ext in ('.csv', '.xlsx', '.pkl', '.json'):
        files = [f for f in os.listdir(InputDir) if f.lower().endswith(ext)]
    elif ext == 'all':
        files = [f for f in os.listdir(InputDir) if f.lower().endswith(('.csv','.xlsx','.pkl','.json'))]
    else:
        raise ValueError(f"지원하지 않는 확장자: {Extension}")

    print(f"Code Type: {Code_Type} 총 {len(files)}개의 파일을 처리합니다.")
    file_info = [(f, InputDir, Code_Type, Extension) for f in files]

    nproc = min(cpu_count(), max(1, len(files)))
    with Pool(processes=nproc, initializer=set_global_contract, initargs=(contract_dict,)) as pool:
        for result, file_stats, dist_rows, samples in pool.imap_unordered(Format_Process_File, file_info):
            all_Columns.extend(result)
            if file_stats is not None:
                all_file_stats.append(file_stats)
            all_dist.extend(dist_rows)
            all_samples.extend(samples)
    return True, all_Columns, all_file_stats, all_dist, all_samples

# ======================================================================
# Data Quality 분석
# ======================================================================
def Data_Quality_Analysis(config, source_dir_list):
    print("\n=== Data Quality 분석 시작 ===")

    base_path  = config['ROOT_PATH']
    output_dir = f"{base_path}/{config['directories']['output']}"
    fileformat = f"{config['files']['fileformat']}"
    filestats  = f"{config['files']['filestats']}"

    datatype_path = f"{config['ROOT_PATH']}/{config['files']['datatype']}.csv"

    datatype_df = read_csv_any(datatype_path)
    if datatype_df.empty:
        print("⚠️ DataType 파일이 없습니다.")
        return 1

    # 0) 백업
    _ = Backup_File(output_dir, fileformat, 'csv')
    _ = Backup_File(output_dir, filestats, 'csv')

    # 1) 규칙 로드(+ 전역 세팅)
    from .dq_rules import load_yaml_contract, validate_with_yaml_contract, build_error_samples, suggest_autofix
    from .dq_report_html import render_summary_html

    with open("DataSense/util/DQ_Contract.yaml", "r", encoding="utf-8") as f:
        DQ_CONTRACT_DICT = yaml.safe_load(f)
    contract = load_yaml_contract(DQ_CONTRACT_DICT)
    set_global_contract(DQ_CONTRACT_DICT)
    
    # 전역 변수로 설정
    import DataSense.util.dq_function as dq_func
    dq_func.DQ_CONTRACT_DICT = DQ_CONTRACT_DICT

    # 2) Data Quality 분석 
    all_Columns, all_file_stats, all_dists, all_samples = [], [], [], []
    for source_config in source_dir_list:
        print(f"\n{source_config['type']} 코드 Master 분석 Directory : {source_config['source']}")
        SourceDir = f"{base_path}/{source_config['source']}"
        success, columns_result, file_stats_result, dist_rows, samples = Format_Build(
            SourceDir, source_config['type'], source_config['extension'], contract_dict=DQ_CONTRACT_DICT
        )
        all_Columns.extend(columns_result)
        all_file_stats.extend(file_stats_result)
        all_dists.extend(dist_rows)
        all_samples.extend(samples)
        print("Format 분석이 성공적으로 완료되었습니다." if success else "Format 분석 중 오류가 발생했습니다.")

    # 3) 파일 통계 저장
    if all_file_stats:
        file_stats_df = pd.concat(all_file_stats, ignore_index=True)
        file_stats_df.insert(0, 'FileNo', range(1, len(file_stats_df)+1))
        file_stats_df.to_csv(f"{base_path}/{filestats}.csv", index=False, encoding='utf-8-sig')

    if not all_Columns:
        print("⚠️ 분석할 컬럼 결과가 없어 DQ/드리프트/규칙/리포트를 생략합니다.")
        return 0

    # 4) 결과 → DQ 스코어 산출(+ 중요도 가중 점수 추가)
    dq_df = pd.DataFrame(all_Columns)
    
    scored_df = add_dq_scores(dq_df)
    if DQ_CONTRACT_DICT is not None:
        scored_df = apply_score_importance(scored_df, DQ_CONTRACT_DICT)
     #-------------------------------------------------------------------------------------------------
    # Data type & Data Quality 분석 결과를 통합하여 저장함  PD_DataType
    #-------------------------------------------------------------------------------------------------  
    datatype_df = datatype_df[['FilePath', 'FileName', 'MasterType', 'ColumnName',  'DataType', 'OracleType']]

    scored_df.drop(columns=['DataType','OracleType'], inplace=True)

    format_df = pd.merge(datatype_df, scored_df,  on=['FilePath', 'FileName', 'ColumnName', 'MasterType'], how='left')

    format_df.insert(0, 'No', range(1, len(format_df)+1))
    format_df.to_csv(f"{base_path}/{fileformat}.csv", index=False, encoding='utf-8-sig')

    print(f"\nData Type & Data Quality 분석 결과가 저장되었습니다. \n ({base_path}/{fileformat}.csv)")
    # #-------------------------------------------------------------------------------------------------
    # 표준 카탈로그
    if not scored_df.empty:
        standard_df = create_standard_df(scored_df)
        standard_df.to_csv(f"{base_path}/{fileformat}_standard.csv", index=False, encoding='utf-8-sig')

    # 5) Top 이슈 리포트 저장
    reports = build_top_issue_reports(scored_df, top_n=20)
    reports["worst_columns"].to_csv(f"{base_path}/{fileformat}_top_issues.csv", index=False, encoding="utf-8-sig")
    reports["by_file"].to_csv(   f"{base_path}/{fileformat}_by_file.csv",     index=False, encoding="utf-8-sig")
    reports["issue_catalog"].to_csv(f"{base_path}/{fileformat}_issue_catalog.csv", index=False, encoding="utf-8-sig")

    # 6) 드리프트(요약 PSI: 패턴/Null/Unique)
    baseline_path = f"{base_path}/{fileformat}_baseline.csv"
    baseline_df   = save_or_load_baseline(scored_df, baseline_path, update=False)
    drift_df      = compute_proxy_drift(scored_df, baseline_df, key_cols=('FileName','ColumnName'))
    drift_df.to_csv(f"{base_path}/{fileformat}_drift.csv", index=False, encoding='utf-8-sig')

    # 7) 분포 스냅샷 + Full PSI
    dist_snapshot_df = pd.DataFrame(all_dists)
    dist_drift_df    = pd.DataFrame()
    if not dist_snapshot_df.empty:
        dist_snapshot_df.to_csv(f"{base_path}/{fileformat}_distribution.csv", index=False, encoding='utf-8-sig')

        dist_baseline_path = f"{base_path}/{fileformat}_dist_baseline.csv"
        if not os.path.exists(dist_baseline_path):
            dist_snapshot_df.to_csv(dist_baseline_path, index=False, encoding='utf-8-sig')
            dist_baseline_df = dist_snapshot_df.copy()
        else:
            dist_baseline_df = pd.read_csv(dist_baseline_path, encoding='utf-8-sig')

        dist_drift_df = compute_snapshot_drift(
            dist_snapshot_df, dist_baseline_df, key_cols=('FileName','ColumnName','DistType')
        )
        # baseline 라벨 불일치로 결과가 공집합이면 자동 리베이스
        if dist_drift_df.empty and not dist_snapshot_df.empty:
            print("[INFO] Dist baseline misaligned. Rebuilding baseline this run.")
            dist_snapshot_df.to_csv(dist_baseline_path, index=False, encoding='utf-8-sig')
            dist_baseline_df = dist_snapshot_df.copy()
            dist_drift_df = compute_snapshot_drift(
                dist_snapshot_df, dist_baseline_df, key_cols=('FileName','ColumnName','DistType')
            )

        dist_drift_df.to_csv(f"{base_path}/{fileformat}_dist_drift.csv", index=False, encoding='utf-8-sig')

    # 8) 규칙 검증(샘플 포함) + Severity 표준화 + Autofix
    sample_df = pd.DataFrame(all_samples) if all_samples else None
    try:
        rule_report_df, scored_df2 = validate_with_yaml_contract(scored_df, contract, sample_df=sample_df)
        if rule_report_df is not None and not rule_report_df.empty:
            rule_report_df.to_csv(f"{base_path}/{fileformat}_rule_report.csv", index=False, encoding='utf-8-sig')
        else:
            print("[WARN] rule_report_df is empty")
        if scored_df2 is not None and not scored_df2.empty:
            scored_df2.to_csv(f"{base_path}/{fileformat}_scored_df.csv", index=False, encoding='utf-8-sig')
        else:
            print("[WARN] scored_df2 is empty")
    except Exception as e:
        print(f"[WARN] validate_with_yaml_contract 실패: {e}")
        rule_report_df = pd.DataFrame()
        scored_df2     = scored_df.copy()  # 파이프라인 지속

    rule_report_df = _ensure_severity_column(rule_report_df)
    if rule_report_df is not None and not rule_report_df.empty:
        rule_report_df.to_csv(f"{base_path}/{fileformat}_rule_report2.csv", index=False, encoding='utf-8-sig')
        try:
            if sample_df is not None and not sample_df.empty:
                from .dq_rules import build_error_samples, suggest_autofix  # 재임포트 안전
                err_log_csv = f"{base_path}/{fileformat}_rule_errors_samples.csv"
                _ = build_error_samples(rule_report_df, sample_df, err_log_csv, max_samples_per_col=50)
        except Exception as e:
            print(f"[WARN] build_error_samples 중단: {e}")

        try:
            from .dq_rules import suggest_autofix
            autofix_df = suggest_autofix(rule_report_df, contract)
            autofix_df.to_csv(f"{base_path}/{fileformat}_autofix_suggestions.csv",
                              index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] suggest_autofix 중단: {e}")

    # 9) 규칙 반영 스코어 재계산
    try:
        scored_df2 = add_dq_scores(scored_df2)
        if DQ_CONTRACT_DICT is not None:
            scored_df2 = apply_score_importance(scored_df2, DQ_CONTRACT_DICT)
    except Exception as e:
        print(f"[WARN] add_dq_scores 재계산 중단: {e}")
        scored_df2 = scored_df.copy()

    # 10) 요약 HTML 리포트
    try:
        from .dq_report_html import render_summary_html
        reports2 = build_top_issue_reports(scored_df2, top_n=20)
        html_out = os.path.join(base_path, f"{fileformat}_summary.html")
        render_summary_html(
            out_path=html_out,
            title="Data Quality Summary",
            scored_df=scored_df2,
            reports=reports2,
            drift_df=drift_df,
            dist_drift_df=dist_drift_df,
            rule_report_df=_ensure_severity_column(rule_report_df)
        )
    except Exception as e:
        print(f"[WARN] render_summary_html 중단: {e}")

    # 11) 최종 CSV 저장
    try:
        scored_df2.to_csv(os.path.join(base_path, f"{fileformat}_2.csv"), index=False, encoding="utf-8-sig")
        _ensure_severity_column(rule_report_df).to_csv(
            os.path.join(base_path, f"{fileformat}_rule_report.csv"),
            index=False, encoding="utf-8-sig"
        )
    except Exception as e:
        print(f"[WARN] 결과 저장 중단: {e}")

    return 0

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    import time
    start_time = time.time()
    config = Load_Yaml_File(YAML_PATH)
    codelist_meta_file = f"{config['ROOT_PATH']}/{config['files']['codelist_meta']}"
    codelist_df = pd.read_excel(codelist_meta_file)
    codelist_df = codelist_df[codelist_df['execution_flag'] == 'Y']
    codelist_list = codelist_df.to_dict(orient='records')

    datatype_df = DataType_Analysis(config, codelist_list)
    if datatype_df is None or datatype_df.empty:
        print("⚠️ DataType 분석 결과가 없습니다.")
    else:
        file_path = f"{config['ROOT_PATH']}/{config['files']['datatype']}.csv"
        datatype_df.to_csv(file_path, index=False, encoding="utf-8-sig")

        print(f"Data Type 분석 결과가 저장되었습니다 \n ({file_path})")

    error_count = Data_Quality_Analysis(config, codelist_list)

    if error_count == 0:
        processing_time = time.time() - start_time
        print("\n--------------------------------------")
        print(f"총 처리 시간: {processing_time:.2f}초")
        print("--------------------------------------")
    else:
        print("분석 중 오류가 발생했습니다.")

