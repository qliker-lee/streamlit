# -*- coding: utf-8 -*-
"""
DS_15_PK_FK_Generator
PK, FK만 생성하는 별도 프로그램
2025.12.XX Qliker
- DS_11_MasterCodeFormat.py의 PK 생성 로직
- DS_13_Code Relationship Analyzer.py의 FK 생성 로직
을 통합하여 PK, FK만 생성합니다.
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# 경로 설정
# -------------------------------------------------------------------
ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

YAML_PATH = ROOT_PATH / "DataSense" / "util" / "DS_Master.yaml"

from DataSense.util.io import Load_Yaml_File, Backup_File, setup_logger, read_csv_any, _clean_headers
from DataSense.util.dq_function import Find_Unique_Combination

# ---------------------- 전역 기본값 ----------------------
DEBUG_MODE = True
OUTPUT_FILE_NAME = 'PK_FK_Result'

# # ---------------------- 로깅 설정 ----------------------
# def setup_logger() -> logging.Logger:
#     log_dir = Path('logs')
#     log_dir.mkdir(exist_ok=True)
#     log_file = log_dir / f"pk_fk_generator_{datetime.now():%Y%m%d_%H%M%S}.log"
#     level = logging.DEBUG if DEBUG_MODE else logging.INFO
#     logging.basicConfig(
#         level=level,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
#     )
#     return logging.getLogger(__name__)

# # ---------------------- 파일 읽기 ----------------------
# def read_csv_any(path: str) -> pd.DataFrame:
#     """다양한 인코딩으로 CSV 파일 읽기"""
#     path = os.path.expanduser(os.path.expandvars(str(path)))
#     for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
#         try:
#             return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
#         except Exception:
#             continue
#     raise FileNotFoundError(path)

# def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
#     """헤더 정리"""
#     out = df.copy()
#     out.columns = [str(c).replace('\ufeff', '').strip() for c in out.columns]
#     return out

# ---------------------- PK 생성 로직 ----------------------
def process_file_for_pk(file_info: tuple) -> List[Dict[str, Any]]:
    """
    파일을 읽어서 PK 정보를 추출합니다.
    DS_11_MasterCodeFormat.py의 Format_Process_File 로직 참조
    복합 PK도 지원합니다.
    """
    file_path, code_type = file_info
    results = []
    
    try:
        # 파일 읽기
        df = read_csv_any(file_path)
        df = _clean_headers(df)
        
        if df.empty:
            return results
        
        file_name = os.path.basename(file_path)
        
        # Find_Unique_Combination을 사용하여 PK 찾기 (복합 PK 지원)
        unique_columns = Find_Unique_Combination(df, max_try_cols=15)
        
        # PK 그룹 식별자 생성 (복합 PK의 경우 같은 그룹으로 식별)
        # 단일 PK인 경우: PKGroup = FilePath + "_PK1"
        # 복합 PK인 경우: PKGroup = FilePath + "_PK1" (모든 PK 컬럼이 같은 그룹)
        pk_group_id = f"{file_path}_PK1" if unique_columns else ""
        
        # 각 컬럼에 대해 PK 정보 생성
        for column in df.columns:
            is_pk = 1 if column in unique_columns else 0
            pk_group = pk_group_id if is_pk == 1 else ""
            results.append({
                'FilePath': file_path,
                'FileName': file_name,
                'ColumnName': str(column),
                'MasterType': str(code_type),
                'PK': is_pk,
                'PKGroup': pk_group,  # 복합 PK 그룹 식별자
                'FK': 0,  # FK는 나중에 설정
                'MCFK': ''  # MCFK는 나중에 설정
            })
        
    except Exception as e:
        print(f"파일 처리 중 오류: {file_path} -> {e}")
        return []
    
    return results

# ---------------------- FK 생성 로직 ----------------------
def generate_fk_from_pk(pk_df: pd.DataFrame) -> pd.DataFrame:
    """
    PK 정보를 기반으로 FK를 생성합니다.
    DS_13_Code Relationship Analyzer.py의 final_mapping 로직 참조
    
    로직:
    1. 단일 컬럼 PK: 다른 파일의 단일 PK 컬럼과 같은 이름을 가진 컬럼 → FK = 'FK', MCFK = 'S'
    2. 멀티 컬럼 PK: 다른 파일의 복합 PK의 모든 컬럼과 같은 이름을 가진 컬럼 조합 → FK = 'FK', MCFK = 'M'
    """
    if pk_df.empty:
        return pk_df
    
    result_df = pk_df.copy()
    
    # MCFK 컬럼이 없으면 추가
    if 'MCFK' not in result_df.columns:
        result_df['MCFK'] = ''
    
    # FK 참조 정보 컬럼 추가
    if 'FK_Ref_File' not in result_df.columns:
        result_df['FK_Ref_File'] = ''
    if 'FK_Ref_Column' not in result_df.columns:
        result_df['FK_Ref_Column'] = ''
    
    # PK가 1인 컬럼들을 찾아서 CodeFilePath_1, CodeColumn_1로 매핑
    pk_mask = (
        result_df['PK'].astype(str).str.upper().isin(['1', 'Y', 'TRUE']) |
        (pd.to_numeric(result_df['PK'], errors='coerce').fillna(0) == 1)
    )
    
    # PK 컬럼 정보 추출 (단일 PK용)
    pk_df_subset = result_df.loc[pk_mask, ['FilePath', 'ColumnName', 'PKGroup', 'FileName']].copy()
    pk_df_subset = pk_df_subset.rename(columns={
        'FilePath': 'CodeFilePath_1',
        'ColumnName': 'CodeColumn_1',
        'PKGroup': 'CodePKGroup',
        'FileName': 'CodeFileName_1'
    })
    
    if pk_df_subset.empty:
        # PK가 없으면 FK도 없음
        result_df['FK'] = 0
        result_df['MCFK'] = ''
        result_df['FK_Ref_File'] = ''
        result_df['FK_Ref_Column'] = ''
        return result_df
    
    # PK가 아닌 컬럼에 대해 다른 파일의 PK와 매칭 확인
    result_df['FK'] = 0  # 기본값은 0
    result_df['MCFK'] = ''  # 기본값은 빈 문자열
    result_df['FK_Ref_File'] = ''  # 기본값은 빈 문자열
    result_df['FK_Ref_Column'] = ''  # 기본값은 빈 문자열
    
    # PK가 아닌 컬럼만 필터링
    non_pk_mask = pd.to_numeric(result_df['PK'], errors='coerce').fillna(0) == 0
    
    if non_pk_mask.any() and not pk_df_subset.empty:
        # 각 non-PK 컬럼에 대해 다른 파일의 PK와 매칭 확인
        non_pk_df = result_df.loc[non_pk_mask].copy()
        non_pk_df = non_pk_df.reset_index()  # 인덱스를 컬럼으로 변환
        
        # 멀티 컬럼 FK로 처리된 컬럼을 추적하기 위한 집합
        multi_fk_processed = set()
        
        # 각 인덱스별로 여러 참조 관계를 저장하기 위한 딕셔너리
        # key: 인덱스, value: {'refs': list of (file, column) tuples, 'mcfk': set()}
        fk_ref_info = {}
        
        # === 1. 멀티 컬럼 FK 찾기 (먼저 처리) ===
        # 각 파일별로 복합 PK 그룹 정보 수집
        # PKGroup별로 그룹화하여 복합 PK 정보 추출
        pk_groups = result_df.loc[pk_mask].groupby(['FilePath', 'PKGroup'])['ColumnName'].apply(list).to_dict()
        
        # 각 파일의 복합 PK 그룹에 대해 다른 파일의 컬럼 조합이 매칭되는지 확인
        for (pk_file_path, pk_group), pk_columns in pk_groups.items():
            if len(pk_columns) <= 1:
                continue  # 단일 PK는 나중에 처리
            
            # 참조하는 PK 파일명 (파일명만 추출)
            pk_file_name = os.path.basename(pk_file_path)
            # 참조하는 PK 컬럼명 (콤마로 구분) - 전체 PK 컬럼명
            pk_column_names = ', '.join(pk_columns)
            
            # 다른 파일의 컬럼들 중에서 이 복합 PK의 일부 또는 전체 컬럼과 같은 이름을 가진 컬럼 찾기
            # 자기 참조는 제외 (같은 파일의 PK를 참조하는 것은 FK가 아님)
            for current_file_path in result_df['FilePath'].unique():
                if current_file_path == pk_file_path:
                    continue  # 같은 파일은 제외
                
                # 현재 파일의 모든 컬럼 (PK 여부와 무관하게 모두 포함)
                # 복합 PK의 일부 컬럼이 현재 파일의 PK일 수도 있으므로 모든 컬럼을 확인
                current_file_df = result_df[
                    (result_df['FilePath'] == current_file_path)
                ]
                current_file_columns = current_file_df['ColumnName'].tolist()
                
                # 복합 PK의 일부 컬럼만 있어도 FK로 설정 (모든 컬럼이 있을 필요 없음)
                # PK 컬럼 중 현재 파일에 있는 컬럼만 찾기 (PK 여부와 무관)
                matching_columns = [col for col in pk_columns if col in current_file_columns]
                
                if matching_columns:  # 하나 이상의 PK 컬럼이 매칭되면
                    # 매칭된 컬럼들을 FK로 설정
                    # 자기 참조는 제외 (같은 파일의 PK를 참조하는 것은 FK가 아님)
                    # 자신의 테이블에 실제로 존재하는 컬럼만 표시 (matching_columns 사용)
                    actual_referenced_columns = ', '.join(sorted(matching_columns))  # 자신의 테이블에 있는 컬럼만
                    
                    for col in matching_columns:
                        fk_mask = (
                            (result_df['FilePath'] == current_file_path) &
                            (result_df['ColumnName'] == col)
                        )
                        matched_indices = result_df.loc[fk_mask].index
                        for idx in matched_indices:
                            multi_fk_processed.add((current_file_path, col))
                            
                            # 여러 참조 관계를 저장하기 위해 딕셔너리에 추가
                            # 복합 PK의 경우, 자신의 테이블에 실제로 존재하는 컬럼만 표시
                            if idx not in fk_ref_info:
                                fk_ref_info[idx] = {'refs': [], 'mcfk': set()}
                            # 자신의 테이블에 있는 컬럼만 저장 (예: "CompanyCode" 또는 "CompanyCode, Dept")
                            ref_tuple = (pk_file_name, actual_referenced_columns)
                            if ref_tuple not in fk_ref_info[idx]['refs']:  # 중복 방지
                                fk_ref_info[idx]['refs'].append(ref_tuple)
                            fk_ref_info[idx]['mcfk'].add('M')  # 멀티 컬럼 FK
                            
                            result_df.loc[idx, 'FK'] = 'FK'
        
        # === 2. 단일 컬럼 FK 찾기 (멀티 FK로 처리되지 않은 컬럼만) ===
        # 멀티 FK로 처리되지 않은 컬럼만 필터링
        non_pk_df_filtered = non_pk_df[
            ~non_pk_df.apply(
                lambda row: (row['FilePath'], row['ColumnName']) in multi_fk_processed,
                axis=1
            )
        ]
        
        if not non_pk_df_filtered.empty:
            # 단일 PK만 필터링 (복합 PK는 제외)
            # PKGroup별로 그룹화하여 단일 PK만 선택
            pk_group_counts = result_df.loc[pk_mask].groupby(['FilePath', 'PKGroup']).size()
            single_pk_groups = pk_group_counts[pk_group_counts == 1].index
            
            if len(single_pk_groups) > 0:
                # 단일 PK 그룹에 속한 PK만 필터링
                single_pk_df = result_df.loc[pk_mask].copy()
                single_pk_df = single_pk_df[
                    single_pk_df.apply(
                        lambda row: (row['FilePath'], row['PKGroup']) in single_pk_groups,
                        axis=1
                    )
                ]
                
                if not single_pk_df.empty:
                    single_pk_subset = single_pk_df[['FilePath', 'ColumnName']].copy()
                    single_pk_subset = single_pk_subset.rename(columns={
                        'FilePath': 'CodeFilePath_1',
                        'ColumnName': 'CodeColumn_1'
                    })
                    
                    # 모든 컬럼과 단일 PK 매칭 (PK 여부와 무관하게 모두 확인)
                    # 자기 참조도 허용하므로 non_pk_df_filtered 대신 모든 컬럼 확인
                    all_columns_df = result_df.reset_index().rename(columns={'index': 'orig_index'})
                    all_columns_df = all_columns_df[
                        ~all_columns_df.apply(
                            lambda row: (row['FilePath'], row['ColumnName']) in multi_fk_processed,
                            axis=1
                        )
                    ]
                    
                    # 다른 파일의 단일 PK와 매칭 (같은 ColumnName을 가진 다른 파일의 PK)
                    # 자기 참조는 제외 (같은 파일의 PK를 참조하는 것은 FK가 아님)
                    merged_single = pd.merge(
                        all_columns_df[['orig_index', 'FilePath', 'ColumnName']],
                        single_pk_subset,
                        left_on='ColumnName',
                        right_on='CodeColumn_1',
                        how='inner'
                    )
                    
                    # 다른 파일의 PK인 경우만 필터링 (같은 파일의 PK는 제외)
                    merged_single = merged_single[merged_single['FilePath'] != merged_single['CodeFilePath_1']]
                    
                    if not merged_single.empty:
                        # 단일 컬럼 FK 설정
                        # merged_single에서 직접 참조 정보 추출
                        for _, row in merged_single.iterrows():
                            idx = row['orig_index']
                            ref_file = row['CodeFilePath_1']
                            ref_file_name = os.path.basename(ref_file)
                            ref_column = row['CodeColumn_1']
                            
                            # 여러 참조 관계를 저장하기 위해 딕셔너리에 추가
                            if idx not in fk_ref_info:
                                fk_ref_info[idx] = {'refs': [], 'mcfk': set()}
                            fk_ref_info[idx]['refs'].append((ref_file_name, ref_column))  # (파일명, 컬럼명) 튜플로 저장
                            fk_ref_info[idx]['mcfk'].add('S')  # 단일 컬럼 FK
                            
                            result_df.loc[idx, 'FK'] = 'FK'
        
        # 모든 참조 관계를 File:Column 형식으로 변환하여 저장
        for idx, ref_data in fk_ref_info.items():
            # refs 리스트를 파일별로 그룹화
            file_to_columns = {}
            for file_name, column_names in ref_data['refs']:
                if file_name not in file_to_columns:
                    file_to_columns[file_name] = []
                file_to_columns[file_name].append(column_names)
            
            # File:Column1, Column2|File2:Column1, Column2 형식으로 변환
            ref_parts = []
            for file_name in sorted(file_to_columns.keys()):
                # 같은 파일의 참조 관계들을 중복 제거 (각 참조 관계는 독립적으로 유지)
                # 복합 PK의 경우 "CompanyCode, Dept" 같은 전체 PK 컬럼명이 저장됨
                unique_refs = sorted(set(column_names.strip() for column_names in file_to_columns[file_name] if column_names and column_names.strip()))
                
                # 각 참조 관계를 독립적으로 표시
                for ref_columns in unique_refs:
                    ref_parts.append(f"{file_name}:{ref_columns}")
            
            # | 구분자로 연결
            ref_combined = '|'.join(ref_parts)
            result_df.loc[idx, 'FK_Ref_File'] = ref_combined
            result_df.loc[idx, 'FK_Ref_Column'] = ''  # 더 이상 사용하지 않으므로 빈 값
            
            # MCFK 값 설정: 'M'과 'S'가 모두 있으면 'M,S', 하나만 있으면 해당 값
            mcfk_values = sorted(ref_data['mcfk'])
            if len(mcfk_values) > 1:
                result_df.loc[idx, 'MCFK'] = ', '.join(mcfk_values)
            elif len(mcfk_values) == 1:
                result_df.loc[idx, 'MCFK'] = mcfk_values[0]
    
    return result_df

# ---------------------- 메인 클래스 ----------------------
class PK_FK_Generator:
    """PK, FK 생성 클래스"""
    
    def __init__(self, yaml_path: Optional[str] = None):
        import os.path as osp
        app_name = osp.basename(__file__)
        print(app_name)
        self.logger = setup_logger(app_name, DEBUG_MODE)
        self.config: Dict[str, Any] = {}
        self.yaml_path = Path(yaml_path or YAML_PATH)
        
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {self.yaml_path}")
        
        self.config = Load_Yaml_File(str(self.yaml_path))
        
        # ROOT_PATH가 YAML에 없으면 자동으로 설정
        if 'ROOT_PATH' not in self.config or not self.config.get('ROOT_PATH'):
            self.config['ROOT_PATH'] = str(ROOT_PATH)
            self.logger.info(f"ROOT_PATH 자동 설정: {self.config['ROOT_PATH']}")
    
    def get_source_files(self) -> List[tuple]:
        """소스 파일 목록 가져오기"""
        files_list = []
        
        # YAML 설정에서 source_dir_list 가져오기
        base_path = Path(self.config['ROOT_PATH'])
        
        # codelist_meta 파일에서 실행 대상 가져오기
        codelist_meta_file = base_path / self.config['files'].get('codelist_meta', 'DS_Meta/Master_Meta.xlsx')
        
        try:
            if codelist_meta_file.exists():
                codelist_df = pd.read_excel(codelist_meta_file)
                codelist_df = codelist_df[codelist_df.get('execution_flag', pd.Series(dtype=str)) == 'Y']
                codelist_list = codelist_df.to_dict(orient='records')
            else:
                # codelist_meta가 없으면 기본 설정 사용
                self.logger.warning(f"codelist_meta 파일이 없습니다: {codelist_meta_file}")
                # 기본 source_dir_list 사용
                directories = self.config.get('directories', {})
                source_dir = base_path / directories.get('input', 'DS_Input')
                if source_dir.exists():
                    codelist_list = [{
                        'type': 'Master',
                        'source': str(source_dir.relative_to(base_path)),
                        'extension': '.csv'
                    }]
                else:
                    return []
        except Exception as e:
            self.logger.error(f"codelist_meta 파일 읽기 실패: {e}")
            return []
        
        # 각 source_config에 대해 파일 목록 수집
        for source_config in codelist_list:
            source_type = source_config.get('type', 'Master')
            source_path = base_path / source_config.get('source', 'DS_Input')
            extension = source_config.get('extension', '.csv')
            
            if not source_path.exists():
                self.logger.warning(f"소스 디렉토리가 없습니다: {source_path}")
                continue
            
            # 확장자에 맞는 파일 찾기
            if extension.lower() == '.csv':
                files = list(source_path.glob('*.csv'))
            elif extension.lower() == '.xlsx':
                files = list(source_path.glob('*.xlsx'))
            elif extension.lower() == 'all':
                files = list(source_path.glob('*.csv')) + list(source_path.glob('*.xlsx'))
            else:
                files = list(source_path.glob(f'*{extension}'))
            
            for file_path in files:
                files_list.append((str(file_path), source_type))
        
        return files_list
    
    def generate_pk_fk(self) -> pd.DataFrame:
        """PK, FK 생성 메인 함수"""
        self.logger.info("PK, FK 생성 시작")
        
        # 소스 파일 목록 가져오기
        files_list = self.get_source_files()
        
        if not files_list:
            self.logger.warning("처리할 파일이 없습니다.")
            return pd.DataFrame()
        
        self.logger.info(f"총 {len(files_list)}개의 파일을 처리합니다.")
        
        # 멀티프로세싱으로 PK 생성
        nproc = min(cpu_count(), max(1, len(files_list)))
        all_results = []
        
        with Pool(processes=nproc) as pool:
            for result in pool.imap_unordered(process_file_for_pk, files_list):
                all_results.extend(result)
        
        if not all_results:
            self.logger.warning("PK 생성 결과가 없습니다.")
            return pd.DataFrame()
        
        # DataFrame 생성
        pk_df = pd.DataFrame(all_results)
        
        # FK 생성
        result_df = generate_fk_from_pk(pk_df)
        
        self.logger.info(f"PK, FK 생성 완료: {len(result_df)}개 레코드")
        
        return result_df
    
    def save_result(self, df: pd.DataFrame) -> bool:
        """결과 저장"""
        if df.empty:
            self.logger.warning("저장할 데이터가 없습니다.")
            return False
        
        base_path = Path(self.config['ROOT_PATH'])
        output_dir = base_path / self.config['directories'].get('output', 'DS_Output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 백업
        output_file = output_dir / f"{OUTPUT_FILE_NAME}.csv"
        _ = Backup_File(str(output_dir), OUTPUT_FILE_NAME, 'csv')
        
        # 저장
        try:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"결과 저장 완료: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            return False

# ---------------------- main ----------------------
def main():
    start = time.time()
    try:
        generator = PK_FK_Generator()
        result_df = generator.generate_pk_fk()
        
        if not result_df.empty:
            success = generator.save_result(result_df)
            if success:
                print("Success : PK, FK 생성 완료")
            else:
                print("Fail : 결과 저장 실패")
        else:
            print("Fail : PK, FK 생성 결과가 없습니다.")
        
        print("="*50)
        print(f"총 처리 시간: {time.time()-start:.2f}초")
        print("="*50)
    except Exception as e:
        print(f"PK, FK 생성 프로그램 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

