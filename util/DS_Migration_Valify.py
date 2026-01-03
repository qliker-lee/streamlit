import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
import os
import logging

MASTER_DIR = Path.cwd() / 'QDQM_Master_Code'
META_DIR = MASTER_DIR / 'QDQM_Meta'
META_FILENAME = 'Migration_Meta.xlsx'
OUTPUT_DIR = MASTER_DIR / 'QDQM_Output'
OUTPUT_FILENAME = 'Migration_Result.csv'
OUTPUT_DETAIL_DIR = OUTPUT_DIR / 'Migration_Detail'
MAX_NUMERIC_COLS = 10
MAX_COMPARE_COLS = 5 # 분석할 최대 비교 컬럼 수
REL_TOLERANCE = 0.00001

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = Path("C:/projects/myproject/QDQM/QDQM_Master_Code/test/mig")

def calculate_checksum(file_path):
    """파일의 체크섬을 계산하는 함수  
    Args:
        file_path (str): 체크섬을 계산할 파일 경로    
    Returns:
        str: MD5 체크섬 값 또는 None (파일이 존재하지 않는 경우)
    """
    if os.path.isabs(file_path):
        path = Path(file_path)
    else:
        path = BASE_PATH / file_path

    if not path.exists():
        logger.error(f"오류: {path} 파일이 존재하지 않습니다.")
        return None

    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def file_read(file_name):
    """파일 경로를 받아서 파일을 읽어오는 함수
    Args:
        file_name (str): 읽을 파일 경로  
    Returns:
        DataFrame: 읽어온 데이터프레임 또는 None (오류 발생 시)
    """
    # 파일 경로가 이미 절대 경로인 경우 그대로 사용
    if os.path.isabs(file_name):
        file_path = Path(file_name)
    else:
        file_path = BASE_PATH / file_name

    if not file_path.exists():
        logger.error(f"오류: {file_path} 파일이 존재하지 않습니다.")
        return None
    
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            logger.error(f"오류: {file_path} 파일 형식이 지원되지 않습니다.")
            return None
        return df
    except Exception as e:
        logger.error(f"파일 읽기 오류: {file_path}, 오류: {str(e)}")
        return None

def calculate_columns_checksum(df, columns):
    """지정된 컬럼들의 데이터로 체크섬을 계산하는 함수
    Args:
        df (DataFrame): 체크섬을 계산할 데이터프레임
        columns (list): 체크섬 계산에 사용할 컬럼 목록
    Returns:
        str: MD5 체크섬 값
    """
    try:
        # 존재하는 컬럼만 선택
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            return None
            
        # 지정된 컬럼만 선택하고 정렬
        df_subset = df[valid_columns].sort_values(by=valid_columns)
        # 데이터프레임을 문자열로 변환하여 체크섬 계산
        return hashlib.md5(df_subset.to_string().encode()).hexdigest()
    except Exception as e:
        logger.error(f"체크섬 계산 오류: {str(e)}")
        return None

def compare_numeric_stats(df1, df2, statistics_columns_list):
    """숫자형 컬럼의 통계값을 비교하는 함수
    Args:
        df1 (DataFrame): 첫 번째 데이터프레임
        df2 (DataFrame): 두 번째 데이터프레임  
        statistics_columns_list (list): 통계 컬럼 목록
    Returns:
        dict: 컬럼별 통계 비교 결과
    """
    try:
        numeric_cols1 = df1.select_dtypes(include=[np.number]).columns
        numeric_cols2 = df2.select_dtypes(include=[np.number]).columns
        
        # 공통 숫자형 컬럼만 선택
        common_numeric_cols = list(set(numeric_cols1) & set(numeric_cols2))
        common_numeric_cols = [col for col in common_numeric_cols if col in statistics_columns_list]

        stats_comparison = {}
        all_equal_sum = True  # 모든 컬럼의 합계 일치 여부 확인을 위한 변수
        all_equal_sum_exact = True
        
        for col in common_numeric_cols:
            stats1 = {
                'count': df1[col].count(),
                'sum': df1[col].sum(),
                'mean': df1[col].mean(),
                'std': df1[col].std(),
                'min': df1[col].min(),
                'max': df1[col].max()
            }
            stats2 = {
                'count': df2[col].count(),
                'sum': df2[col].sum(),
                'mean': df2[col].mean(),
                'std': df2[col].std(),
                'min': df2[col].min(),
                'max': df2[col].max()
            }
            
            # 각 컬럼별 합계 일치 여부
            is_equal_sum = abs(stats1['sum'] - stats2['sum']) < 1e-10
            is_exact_equal_sum = abs(stats1['sum'] - stats2['sum']) < REL_TOLERANCE
            is_equal_mean = abs(stats1['mean'] - stats2['mean']) < 1e-10
            
            stats_comparison[col] = {
                'stats1': stats1,
                'stats2': stats2,
                'is_equal': all(abs(stats1[k] - stats2[k]) < 1e-10 for k in stats1),
                'is_equal_sum': is_equal_sum,
                'is_exact_equal_sum': is_exact_equal_sum,
                'is_equal_mean': is_equal_mean
            }
            
            # 하나라도 합계가 일치하지 않으면 all_equal_sum은 False
            if not is_equal_sum:
                all_equal_sum = False
            if not is_exact_equal_sum:
                all_equal_sum_exact = False
        

        # 전체 결과에 모든 컬럼의 합계 일치 여부 추가
        stats_comparison['is_all_equal_sum'] = all_equal_sum
        stats_comparison['is_all_equal_sum_rel'] = all_equal_sum_exact
        
        return stats_comparison
    except Exception as e:
        logger.error(f"통계 비교 오류: {str(e)}")
        return {}

def prepare_common_columns(df_source, df_target):
    """두 데이터프레임의 공통 컬럼을 준비하고 데이터 타입을 맞추는 함수
    Args:
        df_source (DataFrame): 소스 데이터프레임
        df_target (DataFrame): 타겟 데이터프레임  
    Returns:
        list: 공통 컬럼 목록
    """
    # 필드명이 동일한 컬럼만 선택
    common_columns = [col for col in df_source.columns if col in df_target.columns]
    if not common_columns:
        return []
    
    # 데이터 타입 변환
    for col in common_columns:
        if df_source[col].dtype != df_target[col].dtype:
            try:
                df_source[col] = df_source[col].astype(str)
                df_target[col] = df_target[col].astype(str)
            except:
                continue
    
    return common_columns

def calculate_missing_records(df_source, df_target):
    """source에는 있지만 target에는 없는 레코드 수를 계산하는 함수
    Args:
        df_source (DataFrame): 소스 데이터프레임
        df_target (DataFrame): 타겟 데이터프레임 
    Returns:
        int: 누락된 레코드 수
    """
    try:
        common_columns = prepare_common_columns(df_source, df_target)
        if not common_columns:
            return 0
        
        # 공통 컬럼으로 merge하여 source에만 있는 레코드 수 계산
        merged = pd.merge(df_source[common_columns], df_target[common_columns], 
                         how='left', indicator=True)
        missing_count = len(merged[merged['_merge'] == 'left_only'])
        return missing_count
    except Exception as e:
        logger.error(f"누락 레코드 계산 오류: {str(e)}")
        return 0

def calculate_added_records(df_source, df_target):
    """target에는 있지만 source에는 없는 레코드 수를 계산하는 함수
    Args:
        df_source (DataFrame): 소스 데이터프레임
        df_target (DataFrame): 타겟 데이터프레임
    Returns:
        int: 추가된 레코드 수
    """
    try:
        common_columns = prepare_common_columns(df_source, df_target)
        if not common_columns:
            return 0
        
        # 공통 컬럼으로 merge하여 target에만 있는 레코드 수 계산
        merged = pd.merge(df_source[common_columns], df_target[common_columns], 
                         how='right', indicator=True)
        added_count = len(merged[merged['_merge'] == 'right_only'])
        return added_count
    except Exception as e:
        logger.error(f"추가 레코드 계산 오류: {str(e)}")
        return 0

def parse_column_list(column_str):
    """컬럼 문자열을 리스트로 파싱하는 함수
    Args:
        column_str (str): 쉼표로 구분된 컬럼 문자열
    Returns:
        list: 컬럼 리스트
    """
    if not column_str or not isinstance(column_str, str):
        return []
    
    # 공백과 쌍따옴표, 홑따옴표를 제거하고, 쉼표로 분리
    return [col.strip().replace('"', '').replace("'", '') for col in column_str.split(',') if col.strip()]

def calculate_group_statistics(df_source, df_target, compare_columns_list, statistics_columns_list, col_idx=None, col=None):
    """그룹별 통계를 계산하는 함수
    
    Args:
        df_source (DataFrame): 소스 데이터프레임
        df_target (DataFrame): 타겟 데이터프레임
        compare_columns_list (list): 그룹화할 컬럼 리스트
        statistics_columns_list (list): 통계를 계산할 컬럼 리스트
        col_idx (int, optional): 컬럼 인덱스
        col (str, optional): 통계를 계산할 특정 컬럼
        
    Returns:
        tuple: (상태, 체크섬, 통계 결과)
    """
    try:
        if col: # 단일 컬럼 통계인 경우
            # 컬럼이 숫자형인지 확인
            if col not in df_source.columns or col not in df_target.columns:
                logger.error(f"컬럼 {col}이 소스 또는 타겟 데이터프레임에 존재하지 않습니다.")
                return 'E', None, None
                
            # 숫자형으로 변환 시도
            try:
                if df_source[col].dtype == 'object':
                    df_source[col] = pd.to_numeric(df_source[col], errors='coerce')
                if df_target[col].dtype == 'object':
                    df_target[col] = pd.to_numeric(df_target[col], errors='coerce')
            except Exception as e:
                logger.warning(f"컬럼 {col}을 숫자형으로 변환 중 오류 발생: {str(e)}")
                
            sum_dict = {col: 'sum'}
            source_sum_grouped = df_source.groupby(compare_columns_list).agg(sum_dict).reset_index()
            target_sum_grouped = df_target.groupby(compare_columns_list).agg(sum_dict).reset_index()
            
            # 그룹별 합계에 대한 통계 계산
            agg_dict = {col: ['sum']}  # mean은 제외하고 sum만 사용
            
            # 데이터가 숫자형인 경우에만 mean 추가
            if pd.api.types.is_numeric_dtype(df_source[col]) and pd.api.types.is_numeric_dtype(df_target[col]):
                agg_dict[col].append('mean')
                
            source_grouped = source_sum_grouped.groupby(compare_columns_list).agg(agg_dict).reset_index()
            target_grouped = target_sum_grouped.groupby(compare_columns_list).agg(agg_dict).reset_index()
            
            # 그룹화된 데이터 정렬 후 체크섬 계산
            source_grouped_sorted = source_grouped.sort_values(by=compare_columns_list)
            target_grouped_sorted = target_grouped.sort_values(by=compare_columns_list)
            
            source_checksum = hashlib.md5(source_grouped_sorted.to_string().encode()).hexdigest()
            target_checksum = hashlib.md5(target_grouped_sorted.to_string().encode()).hexdigest()
            
            status = 'O' if source_checksum == target_checksum else 'X'
            
            # 통계 결과 생성
            stats = {'sum': source_grouped[(col, 'sum')].sum()}
            if 'mean' in agg_dict[col]:
                stats['mean'] = source_grouped[(col, 'mean')].mean()
            else:
                stats['mean'] = None
                
            return status, source_checksum, stats
        
        # 모든 통계 컬럼에 대한 종합 계산
        else:
            # 숫자형 컬럼만 필터링
            valid_stat_columns = []
            for col in statistics_columns_list:
                if col in df_source.columns and col in df_target.columns:
                    try:
                        if df_source[col].dtype == 'object':
                            df_source[col] = pd.to_numeric(df_source[col], errors='coerce')
                        if df_target[col].dtype == 'object':
                            df_target[col] = pd.to_numeric(df_target[col], errors='coerce')
                        valid_stat_columns.append(col)
                    except:
                        logger.warning(f"컬럼 {col}은 숫자형으로 변환할 수 없어 통계 계산에서 제외됩니다.")
            
            if not valid_stat_columns:
                logger.warning("유효한 통계 컬럼이 없습니다.")
                return 'N/A', None, None
                
            # 모든 통계 컬럼들의 그룹별 합계 계산
            sum_dict = {col: 'sum' for col in valid_stat_columns}
            source_sum_grouped = df_source.groupby(compare_columns_list).agg(sum_dict).reset_index()
            target_sum_grouped = df_target.groupby(compare_columns_list).agg(sum_dict).reset_index()
            
            # 그룹별 합계에 대한 통계 계산 (sum만 사용)
            agg_dict = {col: ['sum'] for col in valid_stat_columns}
            
            # 숫자형 컬럼에 대해서만 mean 추가
            for col in valid_stat_columns:
                if pd.api.types.is_numeric_dtype(df_source[col]) and pd.api.types.is_numeric_dtype(df_target[col]):
                    agg_dict[col].append('mean')
            
            source_grouped = source_sum_grouped.groupby(compare_columns_list).agg(agg_dict).reset_index()
            target_grouped = target_sum_grouped.groupby(compare_columns_list).agg(agg_dict).reset_index()
            
            # 그룹화된 데이터 정렬 후 체크섬 계산
            source_grouped_sorted = source_grouped.sort_values(by=compare_columns_list)
            target_grouped_sorted = target_grouped.sort_values(by=compare_columns_list)
            
            source_checksum = hashlib.md5(source_grouped_sorted.to_string().encode()).hexdigest()
            target_checksum = hashlib.md5(target_grouped_sorted.to_string().encode()).hexdigest()
            
            status = 'O' if source_checksum == target_checksum else 'X'
            return status, source_checksum, None
            
    except Exception as e:
        error_msg = f"그룹 통계 계산 중 오류 발생"
        if col:
            error_msg += f" - 컬럼: {col}"
        logger.error(f"{error_msg}: {str(e)}")
        return 'E', None, None

def added_deleted_dim(df_source, df_target, compare_columns_list, source_file, target_file, output_df_main):
    """원본과 비교하여 추가/삭제된 차원 값을 분석하고, output_df_main을 업데이트하며, 로그 리스트를 반환합니다."""
    log_entries = []

    if compare_columns_list:
        for idx, comp_col in enumerate(compare_columns_list):
            if idx >= MAX_COMPARE_COLS:
                logger.warning(f"최대 비교 컬럼 개수({MAX_COMPARE_COLS})를 초과하여 {comp_col} 컬럼은 차원 값 분석에서 제외됩니다.")
                break
            
            output_df_main.loc[0, f'Comp_Col_{idx+1}_Name'] = comp_col
            if comp_col in df_source.columns and comp_col in df_target.columns:
                try:
                    source_values = set(df_source[comp_col].dropna().unique())
                    target_values = set(df_target[comp_col].dropna().unique())
                    
                    added_dim_values = sorted(list(target_values - source_values))
                    deleted_dim_values = sorted(list(source_values - target_values))
                    
                    added_dim_cnt = len(added_dim_values)
                    deleted_dim_cnt = len(deleted_dim_values)

                    output_df_main.loc[0, f'Comp_Col_{idx+1}_Added_Dim_Cnt'] = added_dim_cnt
                    output_df_main.loc[0, f'Comp_Col_{idx+1}_Deleted_Dim_Cnt'] = deleted_dim_cnt

                    if added_dim_cnt == 0 and deleted_dim_cnt == 0:
                        output_df_main.loc[0, f'Comp_Col_{idx+1}_Added_Dim_Val'] = ''
                        output_df_main.loc[0, f'Comp_Col_{idx+1}_Deleted_Dim_Val'] = ''
                    else:
                        output_df_main.loc[0, f'Comp_Col_{idx+1}_Added_Dim_Val'] = str(added_dim_values)
                        output_df_main.loc[0, f'Comp_Col_{idx+1}_Deleted_Dim_Val'] = str(deleted_dim_values)

                        for val in added_dim_values:
                            log_entries.append({
                                'log_type': 'dimension_added',
                                'source_file': source_file,
                                'target_file': target_file,
                                'column': comp_col,
                                'value': val
                            })
                        
                        for val in deleted_dim_values:
                            log_entries.append({
                                'log_type': 'dimension_deleted',
                                'source_file': source_file,
                                'target_file': target_file,
                                'column': comp_col,
                                'value': val
                            })
                except Exception as e:
                    logger.error(f"비교 컬럼 '{comp_col}'의 차원 값 분석 중 오류 발생: {str(e)}")
                    # 에러 발생 시 빈 문자열로 설정
                    output_df_main.loc[0, f'Comp_Col_{idx+1}_Added_Dim_Val'] = ''
                    output_df_main.loc[0, f'Comp_Col_{idx+1}_Deleted_Dim_Val'] = ''
                    output_df_main.loc[0, f'Comp_Col_{idx+1}_Added_Dim_Cnt'] = 0
                    output_df_main.loc[0, f'Comp_Col_{idx+1}_Deleted_Dim_Cnt'] = 0

    return log_entries

def save_mismatched_agg_group_details(df_source, df_target, compare_columns_list, 
                                      stat_col_to_check, source_file, target_file, 
                                      agg_idx):
    """
    Agg_X_Status가 'X'일 때, 불일치하는 그룹의 상세 정보를 추출하여 로그 리스트로 반환합니다.

    Args:
        df_source (DataFrame): 소스 데이터프레임
        df_target (DataFrame): 타겟 데이터프레임
        compare_columns_list (list): 그룹화 기준 컬럼 리스트
        stat_col_to_check (str): 현재 검사 중인 통계 컬럼명
        source_file (str): 소스 파일명 (로그용)
        target_file (str): 타겟 파일명 (로그용)
        agg_idx (int): 현재 처리 중인 Agg 인덱스 (예: 1 for Agg_1)
        
    Returns:
        list: 불일치 상세 정보를 담은 딕셔너리의 리스트. 불일치가 없으면 빈 리스트.
    """
    log_list = []
    try:
        # 소스 데이터 그룹화 및 집계 (sum 기준)
        source_grouped_agg = df_source.groupby(
            compare_columns_list, dropna=False, as_index=False
        ).agg(source_val=(stat_col_to_check, 'sum'))
        
        # 타겟 데이터 그룹화 및 집계 (sum 기준)
        target_grouped_agg = df_target.groupby(
            compare_columns_list, dropna=False, as_index=False
        ).agg(target_val=(stat_col_to_check, 'sum'))

        # 그룹화된 결과 병합
        # suffixes를 추가하여 동일 컬럼명 충돌 방지 (이미 source_val, target_val로 agg하므로 필요 없을 수 있으나 안전장치)
        merged_agg_diff = pd.merge(source_grouped_agg, target_grouped_agg, 
                                   on=compare_columns_list, how='outer', suffixes=('_source_merge', '_target_merge'))

        # NaN 값을 0으로 대체하여 비교 (실제 데이터 특성에 따라 조정 필요)
        merged_agg_diff['source_val'] = merged_agg_diff['source_val'].fillna(0)
        merged_agg_diff['target_val'] = merged_agg_diff['target_val'].fillna(0)
        
        # 값이 다른 그룹 필터링
        source_col_is_numeric = pd.api.types.is_numeric_dtype(df_source[stat_col_to_check])
        target_col_is_numeric = pd.api.types.is_numeric_dtype(df_target[stat_col_to_check])

        if source_col_is_numeric and target_col_is_numeric:
            diff_condition = ~np.isclose(merged_agg_diff['source_val'], merged_agg_diff['target_val'])
        else: 
            diff_condition = merged_agg_diff['source_val'].astype(str) != merged_agg_diff['target_val'].astype(str)

        mismatched_agg_groups = merged_agg_diff[diff_condition].copy()

        if not mismatched_agg_groups.empty:
            # DataFrame을 딕셔너리 리스트로 변환
            raw_log_list = mismatched_agg_groups.to_dict('records')
            for entry in raw_log_list:
                log_entry = {
                    'log_type': 'aggregation_mismatch',
                    'source_file': source_file,
                    'target_file': target_file,
                    'statistic_column': stat_col_to_check,
                    'agg_index': agg_idx
                }

                log_entry['source_value_sum'] = entry.get('source_val') # .get으로 안전하게 접근
                log_entry['target_value_sum'] = entry.get('target_val') # .get으로 안전하게 접근
                # 그룹 바이 컬럼 및 해당 값 추가
                for idx, col_name in enumerate(compare_columns_list): # entry에서 그룹 컬럼 값을 가져옴
                    log_entry[f"group_{idx+1}_name"] = col_name
                    log_entry[f"group_{idx+1}_value"] = entry.get(col_name)

                log_list.append(log_entry)
            # logger.info(f"Agg_{agg_idx} ({stat_col_to_check}) 불일치 {len(log_list)}건에 대한 로그 데이터 생성.")
            
    except Exception as e:
        logger.error(f"Agg_{agg_idx} 불일치 상세 정보 추출 중 오류 발생 (컬럼: {stat_col_to_check}): {str(e)}")
    return log_list

def validate_migration(row):
    """마이그레이션 검증을 수행하는 함수
    
    Args:
        row (Series): 검증할 마이그레이션 메타데이터 행
        
    Returns:
        DataFrame: 검증 결과 데이터프레임
    """
    detailed_logs = [] # 모든 상세 로그를 수집할 리스트

    source_file = row['Source_File']
    target_file = row['Target_File']
    compare_columns = row['Compare_Columns']
    statistics_columns = row['Statistics_Columns']

    # logger.info(f"검증 시작: {source_file} -> {target_file}")
    output_data = {
        'Mig_ID': [row['Mig_ID']],
        'Source_File': [source_file],
        'Target_File': [target_file],
        'Org_Status': ['O'],
        'Agg_Status': ['O'],
        'Rel_Status': ['O'],
        'Rel_Precision': [f"{REL_TOLERANCE:.10f}"],  # 소수점 5자리 고정 포맷
        'Common_Status': ['O'],
        'Common_Sort_Status': ['O'],
        'Rec_Cnt': [0],
        'Rec_Ins' : [0],   
        'Col_Cnt': [0],
        'Added_Col_Cnt': [0],
        'Added_Col': [''],
        'Deleted_Col_Cnt': [0],
        'Deleted_Col': [''],
        'Dim_Change_Cnt': [0],
        'Dim_Change_Col': [''],
        'Measure_Change_Cnt': [0],
        'Measure_Change_Col': [''],
        'Sum_Equal': [True],
        'Sum_Rel_Equal': [True],
        'Common_Col_Cnt': [0],
        'Common_Cols': [''],
        'Compare_Cols': [compare_columns],
        'Statistics_Cols': [statistics_columns],
        'Missed_Rec': [0],
        'Added_Rec': [0],
    }

    output_df = pd.DataFrame(output_data)
    df_source = file_read(row['Source_File'])
    df_target = file_read(row['Target_File'])
    
    if df_source is None or df_target is None:
        logger.error(f"오류: {row['Source_File']} 또는 {row['Target_File']} 파일이 없습니다.")
        return output_df
    
    # 파일의 기본 정보 수집 
    output_df.loc[0, 'Rec_Cnt'] = len(df_source)
    output_df.loc[0, 'Rec_Ins'] = len(df_target) - len(df_source)
    output_df.loc[0, 'Col_Cnt'] = len(df_source.columns)

    # 1st 원본 파일 체크섬 비교 (물리적으로 완전일치할 경우 나머지 )
    output_df.loc[0, 'Org_Status'] = 'O' if calculate_checksum(source_file) == calculate_checksum(target_file) else 'X'

    if  output_df.loc[0, 'Org_Status'] == 'O':
        print(f"파일 {index+1}/{len(mig_meta_df)} {row['Source_File']} -> {row['Target_File']} 검증 중 (원본 파일 체크섬 비교 통과)")
        return output_df
    
        # 원본과 비교하여 추가된 컬럼 목록 추출
    added_columns = [col for col in df_target.columns if col not in df_source.columns]
    if added_columns:
        output_df.loc[0, 'Added_Col'] = str(added_columns)
        output_df.loc[0, 'Added_Col_Cnt'] = len(added_columns)
    else:
        output_df.loc[0, 'Added_Col'] = ''
        output_df.loc[0, 'Added_Col_Cnt'] = 0

    # 대상과 비교하여 삭제된 컬럼 목록 추출
    deleted_columns = [col for col in df_source.columns if col not in df_target.columns]
    if deleted_columns:
        output_df.loc[0, 'Deleted_Col'] = str(deleted_columns)
        output_df.loc[0, 'Deleted_Col_Cnt'] = len(deleted_columns)
    else:
        output_df.loc[0, 'Deleted_Col'] = ''
        output_df.loc[0, 'Deleted_Col_Cnt'] = 0

    # 비교 컬럼 및 통계 컬럼 파싱
    compare_columns_list = parse_column_list(compare_columns)
    statistics_columns_list = parse_column_list(statistics_columns)

    # 2nd 공통 컬럼으로 체크섬 비교
    common_columns = list(set(df_source.columns) & set(df_target.columns))
    common_columns_checksum_original = calculate_columns_checksum(df_source, common_columns)
    common_columns_checksum_migrated = calculate_columns_checksum(df_target, common_columns)

    output_df.loc[0, 'Common_Col_Cnt'] = len(common_columns)
    output_df.loc[0, 'Common_Cols'] = str(common_columns)
    output_df.loc[0, 'Common_Status'] = 'O' if common_columns_checksum_original == common_columns_checksum_migrated else 'X'

    # 공통 컬럼이 모두 일치할 경우 나머지 처리 중단 
    if output_df.loc[0, 'Common_Status'] == 'O':
        return output_df
    
    # 공통 컬럼을 소팅 후 체크섬 비교
    common_columns_sorted = sorted(common_columns)

    df_source_sorted = df_source[common_columns_sorted].sort_values(by=common_columns_sorted)
    df_target_sorted = df_target[common_columns_sorted].sort_values(by=common_columns_sorted)
    df_source_sorted = df_source_sorted.reset_index(drop=True)
    df_target_sorted = df_target_sorted.reset_index(drop=True)
    common_columns_checksum_original_sorted = calculate_columns_checksum(df_source_sorted, common_columns_sorted)
    common_columns_checksum_migrated_sorted = calculate_columns_checksum(df_target_sorted, common_columns_sorted)

    output_df.loc[0, 'Common_Sort_Status'] = 'O' if common_columns_checksum_original_sorted == common_columns_checksum_migrated_sorted else 'X'
    
    # 공통 컬럼을 소팅 후 체크섬 비교 일치할 경우 나머지 처리 중단 
    if output_df.loc[0, 'Common_Sort_Status'] == 'O':
        return output_df
    
    # 공통컬럼으로 비교하였을 때 값이 변경된 컬럼 목록 추출
    Dim_Change_Columns = []
    for col in common_columns:
        if df_source[col].dtype != df_target[col].dtype: # 컬럼 타입이 다른 경우
            Dim_Change_Columns.append(col)
        else: # 각 컬럼의 유니크한 값들을 정렬하여 비교     
            try:
                # 두 데이터프레임에서 해당 컬럼의 고유값 추출 및 정렬
                source_unique_values = sorted(df_source[col].dropna().unique())
                target_unique_values = sorted(df_target[col].dropna().unique())
                
                # 유니크한 값들로 체크섬 계산
                source_checksum = hashlib.md5(str(source_unique_values).encode()).hexdigest()
                target_checksum = hashlib.md5(str(target_unique_values).encode()).hexdigest()
                
                # 체크섬이 다르면 변경된 컬럼으로 추가
                if source_checksum != target_checksum:
                    Dim_Change_Columns.append(col)
            except Exception as e:
                logger.warning(f"컬럼 {col} 비교 중 오류 발생: {str(e)}")
                # 오류 발생 시 기존 방식으로 비교
                source_checksum = calculate_columns_checksum(df_source, [col])
                target_checksum = calculate_columns_checksum(df_target, [col])
                if source_checksum != target_checksum:
                    Dim_Change_Columns.append(col)
      
    if Dim_Change_Columns: # 차원값이 변경된 컬럼 정보 수집
        # statistics_columns_list에 있는 항목 제외
        filtered_Dim_Change_Columns = [col for col in Dim_Change_Columns if col not in statistics_columns_list]
        output_df.loc[0, 'Dim_Change_Col'] = str(filtered_Dim_Change_Columns)
        output_df.loc[0, 'Dim_Change_Cnt'] = len(filtered_Dim_Change_Columns)
    
    if output_df.loc[0, 'Dim_Change_Cnt'] == 0:
        output_df.loc[0, 'Dim_Change_Col'] = ''

    # compare_columns 기준으로 차원 값 변경 분석 및 로그 수집
    dim_change_logs = added_deleted_dim(df_source, df_target, compare_columns_list, source_file, target_file, output_df)
    if dim_change_logs:
        detailed_logs.extend(dim_change_logs)

    # 누락 및 추가 레코드 계산
    output_df.loc[0, 'Missed_Rec'] = calculate_missing_records(df_source, df_target)
    output_df.loc[0, 'Added_Rec'] = calculate_added_records(df_source, df_target)

    # statistics_columns_list에 있는 컬럼들을 숫자형으로 변환 시도
    for col in statistics_columns_list:
        if col in df_source.columns:
            try:
                df_source[col] = pd.to_numeric(df_source[col], errors='coerce')
            except Exception as e:
                logger.warning(f"소스 데이터프레임의 '{col}' 컬럼을 숫자형으로 변환 중 오류: {e}")
        if col in df_target.columns:
            try:
                df_target[col] = pd.to_numeric(df_target[col], errors='coerce')
            except Exception as e:
                logger.warning(f"타겟 데이터프레임의 '{col}' 컬럼을 숫자형으로 변환 중 오류: {e}")

    # 숫자형 컬럼 통계 비교
    stats_comparison = compare_numeric_stats(df_source, df_target, statistics_columns_list)
    # 각 숫자형 컬럼에 대한 통계 정보를 저장
    for idx, (col_name, stats) in enumerate(stats_comparison.items()):
        if col_name == 'is_all_equal_sum' or col_name == 'is_all_equal_sum_rel':  # is_all_equal_sum 키는 건너뛰기
            continue
        output_df.loc[0, f'Field_{idx+1}'] = col_name
        output_df.loc[0, f'Sum_{idx+1}_S'] = stats['stats1']['sum']
        output_df.loc[0, f'Sum_{idx+1}_T'] = stats['stats2']['sum']
        # output_df.loc[0, f'Mean_{idx+1}_S'] = stats['stats1']['mean']  
        # output_df.loc[0, f'Mean_{idx+1}_T'] = stats['stats2']['mean']
        output_df.loc[0, f'Sum_Status_{idx+1}'] = 'O' if stats['is_equal_sum'] else 'X'
        # output_df.loc[0, f'Mean_Status_{idx+1}'] = 'O' if stats['is_equal_mean'] else 'X'
    
    output_df.loc[0, 'Sum_Equal'] = stats_comparison.get('is_all_equal_sum', False)
    output_df.loc[0, 'Sum_Rel_Equal'] = stats_comparison.get('is_all_equal_sum_rel', False)
    
    
    # 그룹별 통계 검증
    failed_columns = []  # 실패한 컬럼을 저장할 리스트
    agg_columns = []  # 컬럼 순서를 위한 리스트
    
    for idx, col in enumerate(statistics_columns_list):
        if idx >= MAX_NUMERIC_COLS:  # 최대 처리할 Numeric 컬럼 수
            break
            
        # 기본 통계 비교
        output_df.loc[0, f'Agg_{idx+1}'] = col
        agg_columns.append(f'Agg_{idx+1}')
        agg_columns.append(f'Agg_{idx+1}_Status')
        
        # compare_columns과 statistics_columns이 모두 있는 경우 그룹별 통계 비교
        if compare_columns_list and statistics_columns_list:
            status, _, stats = calculate_group_statistics(
                df_source, df_target, compare_columns_list, statistics_columns_list, idx, col
            )
            
            output_df.loc[0, f'Agg_{idx+1}_Status'] = status
            
            # 실패한 컬럼 추적
            if status == 'X':
                failed_columns.append(col)
        else:
            output_df.loc[0, f'Agg_{idx+1}_Status'] = output_df.loc[0, 'Org_Status']  # 'N/A' 대신 Org_Status 값 사용
        
    # 실패한 컬럼 목록을 Measure_Change_Col 필드에 저장
    output_df.loc[0, 'Measure_Change_Cnt'] = len(failed_columns)
    output_df.loc[0, 'Measure_Change_Col'] = ', '.join(failed_columns) if failed_columns else ''
    
    # 'Agg_Status' 컬럼 추가 - 모든 개별 Agg 컬럼의 상태를 종합
    agg_statuses = []
    if statistics_columns_list: # statistics_columns_list가 비어있지 않은 경우에만 루프 실행
        for i in range(len(statistics_columns_list)):
            if i < MAX_NUMERIC_COLS:
                status_col_name = f'Agg_{i+1}_Status'
                if status_col_name in output_df.columns:
                    agg_statuses.append(output_df.loc[0, status_col_name])
            else:
                break
    
    if agg_statuses: # agg_statuses 리스트가 비어있지 않은 경우에만 실행
        output_df.loc[0, 'Agg_Status'] = 'O' if all(status == 'O' for status in agg_statuses) else 'X'
    elif not statistics_columns_list: # 통계 컬럼 자체가 없는 경우
        output_df.loc[0, 'Agg_Status'] = 'N/A' # 또는 'O' 또는 상황에 맞는 값
    else: # 통계 컬럼은 있으나 MAX_NUMERIC_COLS 제한 등으로 처리된 Agg_X_Status가 없는 경우
        output_df.loc[0, 'Agg_Status'] = 'N/A' # 또는 다른 적절한 기본값
    
    # Agg_n_Status가 'X'인 경우, 불일치 상세 그룹 정보 저장/수집
    if statistics_columns_list and compare_columns_list: 
        for idx, stat_col_to_check in enumerate(statistics_columns_list):
            if idx >= MAX_NUMERIC_COLS: 
                break
            
            agg_status_col_name = f'Agg_{idx+1}_Status'
            # 해당 Agg_X_Status 컬럼이 존재하고, 그 값이 'X'인 경우에만 상세 분석 실행
            if agg_status_col_name in output_df.columns and output_df.loc[0, agg_status_col_name] == 'X':
                # 통계 컬럼이 소스 및 타겟 데이터프레임에 모두 존재하는지 확인
                if stat_col_to_check in df_source.columns and stat_col_to_check in df_target.columns:
                    mismatch_details = save_mismatched_agg_group_details(
                        df_source,
                        df_target,
                        compare_columns_list,
                        stat_col_to_check,
                        source_file,
                        target_file,
                        idx + 1 # agg_idx는 1부터 시작
                    )
                    if mismatch_details:
                        detailed_logs.extend(mismatch_details)
                else:
                    logger.warning(f"통계 컬럼 '{stat_col_to_check}'이 소스 또는 타겟에 없어 Agg_{idx+1} 불일치 상세 정보를 추출할 수 없습니다 ({source_file} -> {target_file}).")
    # 비교 컬럼이 없는데 Agg 상태가 'X'인 경우 (통계 컬럼은 있을 수 있음) 경고 로그
    elif not compare_columns_list and statistics_columns_list:
        # Agg_X_Status 중 하나라도 'X'가 있는지 확인
        is_any_agg_fail = False
        for i in range(min(len(statistics_columns_list), MAX_NUMERIC_COLS)):
            status_col = f'Agg_{i+1}_Status'
            if status_col in output_df.columns and output_df.loc[0, status_col] == 'X':
                is_any_agg_fail = True
                break
        if is_any_agg_fail:
            logger.warning(f"비교 컬럼(Compare_Columns)이 지정되지 않아 Agg 불일치 상세 정보를 추출할 수 없습니다 ({source_file} -> {target_file}).")

    # 수집된 모든 상세 로그를 하나의 파일로 저장
    if detailed_logs:
        s_file_name_part = Path(source_file).stem
        t_file_name_part = Path(target_file).stem
        # 파일명에 source와 target 파일명을 모두 포함
        log_detail_filename = f"{s_file_name_part}_{t_file_name_part}_validation_details.csv"
        
        log_path = OUTPUT_DETAIL_DIR
        os.makedirs(log_path, exist_ok=True)
        
        log_detail_df = pd.DataFrame(detailed_logs)
        try:
            log_detail_df.to_csv(log_path / log_detail_filename, index=False, encoding='utf-8-sig')
            # logger.info(f"상세 검증 로그 저장: {log_path / log_detail_filename}")
        except Exception as e:
            logger.error(f"상세 검증 로그 파일 저장 오류 ({log_path / log_detail_filename}): {e}")
            
    return output_df

def define_mig_case(result_df):
    if result_df.loc[0, 'Org_Status'] == 'O':
        mig_case = 1
    # Common_Status=='O' & Col_Status=='O'  & Added_Col_Cnt==0 & Deleted_Col_Cnt==0 
    elif result_df.loc[0, 'Common_Status'] == 'O' and result_df.loc[0, 'Col_Status'] == 'O' and result_df.loc[0, 'Added_Col_Cnt'] == 0 and result_df.loc[0, 'Deleted_Col_Cnt'] == 0:
        mig_case = 2
    # Common_Status=='X' & Col_Status=='O'  & Common_Sort_Status == 'O'
    elif result_df.loc[0, 'Common_Status'] == 'X' and result_df.loc[0, 'Col_Status'] == 'O' and result_df.loc[0, 'Common_Sort_Status'] == 'O':
        mig_case = 3
    # Common_Status='O' & Col_Status=='X'  & Common_Sort_Status == 'O' & Added_Col_Cnt > 0 & Deleted_Col_Cnt==0 
    elif result_df.loc[0, 'Common_Status'] == 'O' and result_df.loc[0, 'Col_Status'] == 'X' and result_df.loc[0, 'Common_Sort_Status'] == 'O' and result_df.loc[0, 'Added_Col_Cnt'] > 0 and result_df.loc[0, 'Deleted_Col_Cnt'] == 0:
        mig_case = 4
    # Common_Status='O' & Col_Status=='X'  & Common_Sort_Status == 'O' & Added_Col_Cnt == 0 & Deleted_Col_Cnt>0    
    elif result_df.loc[0, 'Common_Status'] == 'O' and result_df.loc[0, 'Col_Status'] == 'X' and result_df.loc[0, 'Common_Sort_Status'] == 'O' and result_df.loc[0, 'Added_Col_Cnt'] == 0 and result_df.loc[0, 'Deleted_Col_Cnt'] > 0:
        mig_case = 5
    # Common_Status='O'  &  Common_Sort_Status == 'O' & Added_Col_Cnt > 0 & Deleted_Col_Cnt>0 
    elif result_df.loc[0, 'Common_Status'] == 'O' and result_df.loc[0, 'Common_Sort_Status'] == 'O' and result_df.loc[0, 'Added_Col_Cnt'] > 0 and result_df.loc[0, 'Deleted_Col_Cnt'] > 0:
        mig_case = 6
    # Mig_Status=='O' & Common_Status='X' & Rec_Ins > 0
    elif result_df.loc[0, 'Mig_Status'] == 'O' and result_df.loc[0, 'Common_Status'] == 'X' and result_df.loc[0, 'Rec_Ins'] > 0:
        mig_case = 7
    # Mig_Status=='O' & Common_Status='X' & Rec_Ins < 0
    elif result_df.loc[0, 'Mig_Status'] == 'O' and result_df.loc[0, 'Common_Status'] == 'X' and result_df.loc[0, 'Rec_Ins'] < 0:
        mig_case = 8
    elif result_df.loc[0, 'Rel_Status'] == 'O' and result_df.loc[0, 'Dim_Status'] == 'O' and result_df.loc[0, 'Rec_Status'] == 'O':
        mig_case = 9
    elif result_df.loc[0, 'Rel_Status'] == 'O' and result_df.loc[0, 'Dim_Status'] == 'O' and result_df.loc[0, 'Rec_Status'] == 'X':
        mig_case = 10


    # Mig_Status=='X' & Rec_Status=='X' & Col_Status =='O' & Sum_Status=='O' & Dim_Change_Cnt > 0 
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Status'] == 'X' and result_df.loc[0, 'Col_Status'] == 'O' and result_df.loc[0, 'Sum_Status'] == 'O' and result_df.loc[0, 'Dim_Change_Cnt'] > 0:
        mig_case = 11
    # Mig_Status=='X' & Rec_Status=='O' & Col_Status=='O' & Sum_Status=='X' & Dim_Change_Cnt==0 & Measure_Change_Cnt > 0
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Status'] == 'O' and result_df.loc[0, 'Col_Status'] == 'O' and result_df.loc[0, 'Sum_Status'] == 'X' and result_df.loc[0, 'Dim_Change_Cnt'] == 0 and result_df.loc[0, 'Measure_Change_Cnt'] > 0:
        mig_case = 12
    # Mig_Status=='X' & Rec_Status=='O' & Col_Status=='X' & Sum_Status=='X' & Dim_Change_Cnt>0 & Added_Col_Cnt>0 & Deleted_Col_Cnt==0
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Status'] == 'O' and result_df.loc[0, 'Col_Status'] == 'X' and result_df.loc[0, 'Sum_Status'] == 'X' and result_df.loc[0, 'Dim_Change_Cnt'] > 0 and result_df.loc[0, 'Added_Col_Cnt'] > 0 and result_df.loc[0, 'Deleted_Col_Cnt'] == 0:
        mig_case = 13
    # Mig_Status=='X' & Rec_Status=='O' & Col_Status=='X' & Sum_Status=='O' & Dim_Change_Cnt==0 & Added_Col_Cnt==0 & Deleted_Col_Cnt>0
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Status'] == 'O' and result_df.loc[0, 'Col_Status'] == 'X' and result_df.loc[0, 'Sum_Status'] == 'O' and result_df.loc[0, 'Dim_Change_Cnt'] == 0 and result_df.loc[0, 'Added_Col_Cnt'] == 0 and result_df.loc[0, 'Deleted_Col_Cnt'] > 0:
        mig_case = 14
    # Mig_Status=='X' & Rec_Status=='O' & Col_Status=='X' & Sum_Status=='X' & Dim_Change_Cnt==0 
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Status'] == 'O' and result_df.loc[0, 'Col_Status'] == 'X' and result_df.loc[0, 'Sum_Status'] == 'X' and result_df.loc[0, 'Dim_Change_Cnt'] == 0:
        mig_case = 15
    # Mig_Status=='X' & Rec_Ins > 0 & Sum_Status=='O'
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Ins'] > 0 and result_df.loc[0, 'Sum_Status'] == 'O':
        mig_case = 16
    # Mig_Status=='X' & Rec_Ins > 0 & Sum_Status=='X'
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Ins'] > 0 and result_df.loc[0, 'Sum_Status'] == 'X':
        mig_case = 17
    # Mig_Status=='X' & Rec_Ins < 0 & Sum_Status=='O'
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Ins'] < 0 and result_df.loc[0, 'Sum_Status'] == 'O':
        mig_case = 18
    # Mig_Status=='X' & Rec_Ins < 0 & Sum_Status=='X'
    elif result_df.loc[0, 'Mig_Status'] == 'X' and result_df.loc[0, 'Rec_Ins'] < 0 and result_df.loc[0, 'Sum_Status'] == 'X':
        mig_case = 19

    else:
        mig_case = 99
    return mig_case

def migration_status_analysis(result_df):
    status_df = pd.DataFrame()
    status_df.loc[0, 'Mig_ID'] = result_df.loc[0, 'Mig_ID']
    status_df.loc[0, 'Source_File'] = result_df.loc[0, 'Source_File']
    status_df.loc[0, 'Target_File'] = result_df.loc[0, 'Target_File']
    status_df.loc[0, 'Mig_Status'] = '[X]'

    if result_df is None:
        return None
    
    status_df.loc[0, 'Rec_Status'] = 'X' if result_df.loc[0, 'Rec_Ins']  else 'O'
    status_df.loc[0, 'Col_Status'] = 'X' if result_df.loc[0, 'Added_Col_Cnt'] or result_df.loc[0, 'Deleted_Col_Cnt'] else 'O'
    status_df.loc[0, 'Dim_Status'] = 'X' if result_df.loc[0, 'Dim_Change_Cnt'] else 'O'
    status_df.loc[0, 'Measure_Status'] = 'X' if result_df.loc[0, 'Measure_Change_Cnt'] else 'O'

    # 모든 검증 결과가 'O'인 경우 검증 통과
    if (result_df.loc[0, 'Org_Status'] == 'O' or 
        result_df.loc[0, 'Common_Status'] == 'O' or 
        result_df.loc[0, 'Common_Sort_Status'] == 'O' or 
        # result_df.loc[0, 'Agg_Status'] == 'O' or
        result_df.loc[0, 'Rel_Status'] == 'O'):
        status_df.loc[0, 'Mig_Status'] = 'O'
    else:
        status_df.loc[0, 'Mig_Status'] = 'X'

    # # 모든 검증 결과가 'O'인 경우 검증 통과
    # if (result_df.loc[0, 'Org_Status'] == 'O' or 
    #     result_df.loc[0, 'Common_Status'] == 'O' or 
    #     result_df.loc[0, 'Common_Sort_Status'] == 'O' or 
    #     result_df.loc[0, 'Rel_Status'] == 'O'):
    #     status_df.loc[0, 'Rel_Mig_Status'] = 'O'
    # else:
    #     status_df.loc[0, 'Rel_Mig_Status'] = 'X'

    status_df.loc[0, 'Sum_Status'] = 'O' if result_df.loc[0, 'Sum_Equal'] else 'X' 

    if result_df.loc[0, 'Rec_Ins'] > 0:
        status_df.loc[0, 'Relation'] = '1:n 관계'
    elif result_df.loc[0, 'Rec_Ins'] < 0:
        status_df.loc[0, 'Relation'] = 'n:1 관계'
    else:
        status_df.loc[0, 'Relation'] = '1:1 관계'   

    output_df = pd.merge(
        result_df,
        status_df,
        on=['Mig_ID', 'Source_File', 'Target_File'],
        how='left'
    )
    # Mig_Case 결정
    mig_case = define_mig_case(output_df)
    output_df.loc[0, 'Mig_Case'] = mig_case

    # 최종 컬럼 순서를 정의합니다.
    final_columns_order = []
    
    # 1. 기본 정보 컬럼 (output_data에 정의된 순서 및 주요 상태/결과 컬럼)
    initial_cols = [ 'Mig_ID', 
        'Source_File', 'Target_File', 'Mig_Status', 'Rel_Mig_Status', 'Mig_Case', 'Org_Status', 
        'Sum_Status', 'Agg_Status', 'Rel_Status', 'Common_Status', 'Common_Sort_Status',
        'Rec_Status', 'Col_Status', 'Dim_Status', 'Measure_Status', 'Relation',
    ]
    final_columns_order.extend(initial_cols)

    # initial_cols에 포함되지 않은 나머지 컬럼들을 output_df에서 현재 순서대로 가져와 final_columns_order에 추가합니다.
    current_df_columns = list(output_df.columns)
    for col in current_df_columns:
        if col not in final_columns_order: # initial_cols에 이미 포함된 컬럼은 다시 추가하지 않습니다.
            final_columns_order.append(col)
    
    # 최종적으로 final_columns_order에 있는 컬럼 중 output_df에 실제로 존재하는 컬럼만으로 순서를 확정합니다.
    final_columns_order = [col for col in final_columns_order if col in current_df_columns]
    
    output_df = output_df[final_columns_order]
    return output_df

# ============================
# ✅ 추가: 상대 오차 기반 그룹 통계 비교 함수
# ============================

import pandas as pd
import hashlib

def calculate_group_statistics_with_rel_diff(
        df_source, 
        df_target, 
        compare_columns_list, 
        statistics_columns_list, 
        col_idx=None, col=None, 
        rel_tol=REL_TOLERANCE):
    try:
        # compare_columns_list의 각 컬럼명 정제
        cleaned_compare_columns_list = []
        if compare_columns_list:
            for c_col in compare_columns_list:
                if isinstance(c_col, str):
                    cleaned_compare_columns_list.append(c_col.strip().replace('"', '').replace("'", ""))
                else:
                    # 문자열이 아닌 경우 원본 유지 또는 오류 처리 (여기서는 원본 유지)
                    cleaned_compare_columns_list.append(c_col)
        else:
            # compare_columns_list가 비어있거나 None인 경우 그대로 사용
            cleaned_compare_columns_list = compare_columns_list

        if col:
            # 컬럼 이름에서 앞뒤 공백 및 따옴표 제거
            col = col.strip().replace('"', '').replace("'", "")
            # 컬럼 존재 여부 확인
            if col not in df_source.columns or col not in df_target.columns:
                return 'N/A', None, None, None
                # logger.warning(f"컬럼 '{col}'이 소스 또는 타겟 데이터프레임에 존재하지 않아 상대 오차 그룹 통계 비교 건너뜀")
                # return 'N/A', None, None, None

            if df_source[col].dtype == 'object':
                df_source[col] = pd.to_numeric(df_source[col], errors='coerce')
            if df_target[col].dtype == 'object':
                df_target[col] = pd.to_numeric(df_target[col], errors='coerce')

            # 정제된 compare_columns_list 사용
            src_group = df_source.groupby(cleaned_compare_columns_list, as_index=False)[col].sum().rename(columns={col: 'sum_s'})
            tgt_group = df_target.groupby(cleaned_compare_columns_list, as_index=False)[col].sum().rename(columns={col: 'sum_t'})
            
            # groupby 결과가 비어 있을 경우를 대비하여 컬럼명 명시 (sum 결과 컬럼만 있도록)
            if 'sum_s' not in src_group.columns and not cleaned_compare_columns_list: # 전체 합계 케이스
                 src_group = pd.DataFrame({'sum_s': [df_source[col].sum()]})
            if 'sum_t' not in tgt_group.columns and not cleaned_compare_columns_list: # 전체 합계 케이스
                 tgt_group = pd.DataFrame({'sum_t': [df_target[col].sum()]})

            if not cleaned_compare_columns_list: # 그룹핑 컬럼이 없을 경우
                s_sum = src_group['sum_s'].iloc[0] if not src_group.empty else 0
                t_sum = tgt_group['sum_t'].iloc[0] if not tgt_group.empty else 0
                merged = pd.DataFrame({'sum_s': [s_sum], 'sum_t': [t_sum]})
            else:
                merged = pd.merge(src_group, tgt_group, on=cleaned_compare_columns_list, how='outer')

            merged = merged.fillna(0) # sum_s 또는 sum_t가 NaN (그룹이 없거나 모든 값이 NaN)인 경우 0으로 채움

            merged['abs_diff'] = (merged['sum_s'] - merged['sum_t']).abs()

            merged['rel_diff'] = merged.apply(
                lambda row: row['abs_diff'] / abs(row['sum_s']) if abs(row['sum_s']) > 1e-10 else (0 if row['abs_diff'] == 0 else float('inf')),
                axis=1
            )

            status = 'O' if all(merged['rel_diff'] <= REL_TOLERANCE) else 'X'
            
            # src_group이 비어있을 수 있으므로, to_string 전에 확인 또는 빈 DataFrame 처리
            checksum_source_df = src_group[cleaned_compare_columns_list + ['sum_s']] if cleaned_compare_columns_list else src_group[['sum_s']]
            checksum_source = hashlib.md5(checksum_source_df.sort_values(by=cleaned_compare_columns_list if cleaned_compare_columns_list else ['sum_s']).to_string().encode()).hexdigest()

            current_sum_s = src_group['sum_s'].sum() # src_group이 비어있으면 0

            rel_diff_max_val = 0.0
            # rel_diff_avg_val = 0.0

            if not merged.empty and 'rel_diff' in merged.columns and not merged['rel_diff'].empty:
                # inf 값을 제외하고 max, mean 계산 (필요에 따라)
                finite_rel_diffs = merged['rel_diff'][np.isfinite(merged['rel_diff'])]
                if not finite_rel_diffs.empty:
                    rel_diff_max_val = finite_rel_diffs.max()
                    # rel_diff_avg_val = finite_rel_diffs.mean()
                elif not merged['rel_diff'].empty: # 모든 값이 inf인 경우
                    rel_diff_max_val = float('inf')
                    # rel_diff_avg_val = float('inf')

            stats = {
                'sum': current_sum_s,
                'rel_diff_max': rel_diff_max_val,
                # 'rel_diff_avg': rel_diff_avg_val
            }

            return status, checksum_source, stats, merged
        else: # col이 None이거나 비어있는 경우
            return 'N/A', None, None, None

    except Exception as e:
        logger.error(f"❗ 그룹 통계 비교 오류 (컬럼: {col}) - {str(e)}")
        return 'E', None, None, None


# # ============================
# # ✅ 추가: validate_migration 확장 함수 정의
# # ============================
def update_validate_migration_with_rel_diff(validate_func, rel_diff_func, rel_tol=REL_TOLERANCE):
    def modified_validate_migration(row):
        result_df = validate_func(row)
        if result_df is None or result_df.empty:
            return result_df

        # Org_Status 또는 Common_Sort_Status가 'O'인 경우 추가 검증 없이 반환
        if (result_df.loc[0, 'Org_Status'] == 'O' or 
            result_df.loc[0, 'Common_Sort_Status'] == 'O'):
            # 필요한 컬럼들을 'N/A'로 초기화
            # for idx in range(MAX_NUMERIC_COLS):
            #     result_df.loc[0, f'Agg_{idx+1}_Rel_Status'] = 'O'
            #     result_df.loc[0, f'Rel_Field_{idx+1}'] = ''
            #     result_df.loc[0, f'Rel_Sum_{idx+1}'] = None
            #     result_df.loc[0, f'Rel_Diff_Max_{idx+1}'] = None
            #     result_df.loc[0, f'Rel_Diff_Avg_{idx+1}'] = None
            
            result_df.loc[0, 'Rel_Status'] = 'O'
            result_df.loc[0, 'Rel_Measure_Change_Cnt'] = 0
            result_df.loc[0, 'Rel_Measure_Change_Col'] = ''
            return result_df

        source_file = row['Source_File']
        target_file = row['Target_File']
        compare_columns_list = [col.strip() for col in str(row['Compare_Columns']).split(',') if col.strip()]
        statistics_columns_list = [col.strip() for col in str(row['Statistics_Columns']).split(',') if col.strip()]

        df_source = file_read(source_file)
        df_target = file_read(target_file)
        if df_source is None or df_target is None:
            return result_df

        new_columns_data = {} # 모든 새로운 컬럼 데이터를 저장할 딕셔너리
        rel_failed_columns = []

        for idx, col in enumerate(statistics_columns_list):
            if idx >= MAX_NUMERIC_COLS: # 최대 처리 컬럼 수 상수 사용
                break

            status_col_name = f'Agg_{idx+1}_Rel_Status'
            rel_field_col_name = f'Rel_Field_{idx+1}'
            rel_sum_col_name = f'Rel_Sum_{idx+1}'
            rel_diff_max_col_name = f'Rel_Diff_Max_{idx+1}'
            rel_diff_avg_col_name = f'Rel_Diff_Avg_{idx+1}'

            status, _, stats, mismatch_df = rel_diff_func(
                df_source, df_target,
                compare_columns_list,
                statistics_columns_list, 
                col_idx=idx, 
                col=col, 
                rel_tol=rel_tol
            )

            new_columns_data[rel_field_col_name] = col
            new_columns_data[status_col_name] = status
            
            if stats: 
                new_columns_data[rel_sum_col_name] = stats.get('sum')
                new_columns_data[rel_diff_max_col_name] = stats.get('rel_diff_max')
                new_columns_data[rel_diff_avg_col_name] = stats.get('rel_diff_avg')
            else: 
                new_columns_data[rel_sum_col_name] = None
                new_columns_data[rel_diff_max_col_name] = None
                new_columns_data[rel_diff_avg_col_name] = None

            if status == 'X':
                rel_failed_columns.append(col)

        new_columns_data['Rel_Measure_Change_Cnt'] = len(rel_failed_columns)
        new_columns_data['Rel_Measure_Change_Col'] = ', '.join(rel_failed_columns) if rel_failed_columns else ''

        # Rel_Status 결정 로직 수정
        final_Rel_Status = 'N/A' # 기본값
        processed_rel_statuses = []

        if statistics_columns_list:
            for i in range(min(len(statistics_columns_list), MAX_NUMERIC_COLS)):
                status_key = f'Agg_{i+1}_Rel_Status'
                if status_key in new_columns_data: # 해당 키로 상태가 new_columns_data에 저장되었는지 확인
                    processed_rel_statuses.append(new_columns_data[status_key])
        
        if not statistics_columns_list:
            final_Rel_Status = 'N/A'
        elif not processed_rel_statuses: # 통계 컬럼 리스트는 있으나, 실제로 처리된 상태가 없는 경우
            final_Rel_Status = 'N/A'
        else:
            if all(s == 'O' for s in processed_rel_statuses):
                final_Rel_Status = 'O'
            else: # 'O'가 아닌 상태('X', 'E', 'N/A' 등)가 하나라도 있으면 'X'로 처리
                final_Rel_Status = 'X'
        
        new_columns_data['Rel_Status'] = final_Rel_Status
        
        # result_df에 모든 새로운 컬럼들을 한번에 할당
        result_df = result_df.assign(**new_columns_data)
        
        return result_df

    return modified_validate_migration

if __name__ == "__main__":
    import time
    start_time = time.time()
    logger.info("마이그레이션 검증 시작")

    # 상대 오차 기반 확장 버전 함수 생성
    validate_migration_with_rel = update_validate_migration_with_rel_diff(
        validate_migration,
        calculate_group_statistics_with_rel_diff, # 파일 내에 이미 정의된 함수 사용
        rel_tol=REL_TOLERANCE
    )

    try:
        mig_meta_file = META_DIR / META_FILENAME
        if not mig_meta_file.exists():
            logger.error(f"메타 파일이 존재하지 않습니다: {mig_meta_file}")
            exit(1)
           
        mig_meta_df = pd.read_excel(mig_meta_file)
        mig_meta_df['Mig_ID'] = mig_meta_df.index + 1
        output_dfs = []  # DataFrame 리스트로 변경
        
        for index, row in mig_meta_df.iterrows():
            logger.info(f"파일 {index+1}/{len(mig_meta_df)} {row['Source_File']} -> {row['Target_File']} 검증 중")
            result_df = validate_migration_with_rel(row)
            if result_df is not None and not result_df.empty:
                analysis_df = migration_status_analysis(result_df)
                if not analysis_df.empty:
                    # NA 값을 적절한 기본값으로 채우기
                    analysis_df = analysis_df.fillna({
                        'Added_Col': '',
                        'Deleted_Col': '',
                        'Dim_Change_Col': '',
                        'Measure_Change_Col': '',
                        'Common_Cols': '',
                        'Compare_Cols': '',
                        'Statistics_Cols': '',
                        'Added_Col_Cnt': 0,
                        'Deleted_Col_Cnt': 0,
                        'Dim_Change_Cnt': 0,
                        'Measure_Change_Cnt': 0,
                        'Common_Col_Cnt': 0,
                        'Rec_Cnt': 0,
                        'Rec_Ins': 0,
                        'Missed_Rec': 0,
                        'Added_Rec': 0
                    })
                    output_dfs.append(analysis_df)

        # 모든 DataFrame을 한 번에 연결
        if output_dfs:
            # 모든 DataFrame의 컬럼 타입을 일치시키기
            common_cols = set.intersection(*[set(df.columns) for df in output_dfs])
            
            # 각 DataFrame의 컬럼 타입을 미리 통일
            for df in output_dfs:
                for col in common_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].astype('float64')
                    else:
                        df[col] = df[col].astype('str')
                        
                # 빈 값이나 NA 값을 포함한 컬럼 제거
                df.dropna(axis=1, how='all', inplace=True)

            # concat 전에 모든 DataFrame이 동일한 컬럼을 가지도록 보장
            all_columns = list(common_cols)
            for df in output_dfs:
                df = df[all_columns]

            output_df = pd.concat(output_dfs, ignore_index=True)
            output_file = OUTPUT_DIR / OUTPUT_FILENAME
            output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            end_time = time.time()
            print("--------------------------------")
            print(f"검증 결과 파일 : {output_file}")
            print(f"마이그레이션 검증 실행 시간: {end_time - start_time:.2f}초")
            print("--------------------------------")
        else:
            logger.warning("검증 결과가 없습니다.")
            
    except Exception as e:
        logger.error(f"마이그레이션 검증 중 오류 발생: {str(e)}")


