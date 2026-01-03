# -*- coding: utf-8 -*-
"""
DS_17_Create_Logical_ERD
CodeMapping_relationship.csv 파일을 읽어서 논리적 ERD를 생성하여 파일로 저장
2025.12.16 Qliker
"""

import pandas as pd
import os
import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import time
import io
import re
from typing import Dict, Any, Optional, List

# --- 1. 설정 및 상수 ---
# 현재 파일의 상위 2단계 폴더를 path에 추가
ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# YAML 파일 경로
YAML_PATH = ROOT_PATH / "DataSense" / "util" / "DS_Master.yaml"

from DataSense.util.io import Load_Yaml_File, Backup_File, setup_logger 

#----------------------------------------------------------------------
# 상수 선언 
#----------------------------------------------------------------------
MAX_PK_COMBINATION = 20  # PK 탐색 시 최대 컬럼 조합 수 (성능을 위해 제한)
OUTPUT_ERD_FILE = 'DS_ERD'
OUTPUT_LOGICAL_FILE = 'DS_Logical'

DEBUG_MODE = True  # 실제 운영 시 False로 변경
# ------------------------------------------------------------------------------------------------
# 개선된 PK 탐색 함수 (클래스 밖 정의 유지)
# ------------------------------------------------------------------------------------------------
def find_unique_combo_sequential(df: pd.DataFrame, max_try_cols: int = 6) -> list:
    """순차적으로 컬럼을 추가하며 유니크한 조합을 찾습니다."""
    if df.empty:
        return None  # [] 대신 None 반환 통일
    
    columns_to_check = df.columns.tolist()[:max_try_cols]
    current_combo = []
    
    for column in columns_to_check:
        current_combo.append(column)
        
        # 성능 최적화: 유니크 검사
        if df.duplicated(subset=current_combo, keep=False).sum() == 0:
            return current_combo
            
    return None # PK 조합을 찾지 못했음을 명확히 표시


class DataAnalyzer:
    """
    지정된 폴더에서 파일을 로드하고 PK/FK 관계를 추론하며 결과를 CSV로 저장하는 클래스
    """
    def __init__(self, yaml_path: Optional[str] = None):
        import os.path as osp
        app_name = osp.basename(__file__)
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
        
        base_path = Path(self.config['ROOT_PATH'])
        directories = self.config.get('directories', {})
        self.base_path = base_path / directories.get('input', 'DS_Input')
        self.output_path = base_path / directories.get('output', 'DS_Output')
        
        self.dataframes = {}
        self.table_pks = {}
        self.table_fks = defaultdict(list)
        self.fk_column_map = defaultdict(lambda: defaultdict(list))
        self.file_paths = {}


    def get_source_files(self) -> List[tuple]:
        """소스 파일 목록 가져오기 (DS_Meta/Master_Meta.xlsx에서 읽기)"""
        files_list = []
        
        # YAML 설정에서 base_path 가져오기
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
    
    def load_files(self):
        """소스 파일 목록을 가져와서 DataFrame으로 로드합니다."""
        self.logger.info("1st Step: 파일 로드 시작")

        try:
            files_list = self.get_source_files()
            if not files_list:
                raise FileNotFoundError(f"처리할 파일이 없습니다. codelist_meta 파일을 확인하세요.")
            
            if DEBUG_MODE:
                self.logger.info(f"총 {len(files_list)}개의 파일을 로드합니다.")
            
            for file_path, source_type in files_list:
                table_name = Path(file_path).name  # 확장자 포함
                
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path, dtype=str, keep_default_na=False, low_memory=False)
                    elif file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
                    else:
                        self.logger.warning(f"지원하지 않는 파일 형식: {file_path}")
                        continue
                    
                    self.dataframes[table_name] = df
                    self.file_paths[table_name] = str(file_path)
                except Exception as e:
                    self.logger.error(f"  - ERROR: {table_name} 로드 중 오류 발생: {e}")
                    continue
            if DEBUG_MODE:
                self.logger.info(f"파일 로드 완료: {len(self.dataframes)}개 파일")
            return True
        except Exception as e:
            self.logger.error(f"load_files 실패: {e}")
            return False

    def discover_pk(self):
        """Primary Key (PK)를 추론합니다."""
        self.logger.info("2nd Step: Primary Key (PK) 추론 시작")
        
        try:
            for table_name, df in self.dataframes.items():
                if df.empty:
                    self.table_pks[table_name] = None
                    continue
                
                # 1단계: 단일 필드가 유니크 값을 갖는지 확인
                single_pk = None
                for col in df.columns:
                    if df[col].duplicated().sum() == 0:  # 해당 컬럼이 유니크한지 확인 (중복이 없는지)
                        single_pk = [col]
                        break
                
                if single_pk:  # 단일 필드 PK 발견
                    self.table_pks[table_name] = single_pk
                else: # 2단계: 단일 필드 PK가 없으면 복합 PK 탐색
                    pk_combo = find_unique_combo_sequential(df, MAX_PK_COMBINATION) 

                    if pk_combo:
                        self.table_pks[table_name] = pk_combo
                    else:
                        self.table_pks[table_name] = None
            
            pk_count = sum(1 for pk in self.table_pks.values() if pk is not None)
            return True
        except Exception as e:
            self.logger.error(f"discover_pk 실패: {e}")
            return False

    # # find_unique_combo_sequential 함수를 클래스 메서드로 추가
    # def find_unique_combo_sequential(self, df: pd.DataFrame, max_try_cols: int = 6) -> list:
    #     # 클래스 밖의 전역 함수 find_unique_combo_sequential을 호출하거나,
    #     # 편의를 위해 클래스 메서드로 재정의 (여기서는 클래스 메서드 사용)
    #     return find_unique_combo_sequential(df, max_try_cols)


    def discover_fk(self):
        """Foreign Key (FK) 관계를 설정하고, fk_column_map을 채웁니다."""

        self.logger.info("3rd Step: Foreign Key (FK) 관계 설정 시작")
        
        try:
            fk_count = 0
            for parent_table, pk_columns in self.table_pks.items():
                if not pk_columns:
                    continue
                
                for child_table, child_df in self.dataframes.items():
                    if parent_table == child_table:
                        continue
                    
                    child_cols = set(child_df.columns)
                    pk_set = set(pk_columns)
                    
                    if pk_set.issubset(child_cols):
                        # 데이터 존재 여부 확인
                        has_data = all(len(child_df[col].loc[child_df[col] != ''].unique()) > 1 for col in pk_columns)

                        if has_data:
                            fk_info = {
                                'referencing_table': child_table, 
                                'referenced_table': parent_table,   
                                'fk_columns': list(pk_columns)      
                            }
                            
                            if fk_info not in self.table_fks[child_table]:
                                self.table_fks[child_table].append(fk_info)
                                fk_count += 1
                                
                                for col in pk_columns:
                                    self.fk_column_map[child_table][col].append(parent_table)
            return True
        except Exception as e:
            self.logger.error(f"discover_fk 실패: {e}")
            return False
        
    def create_physical_erd(self) -> pd.DataFrame:
        """추론된 PK 및 FK 결과를 물리적 ERD 파일로 저장합니다."""
        try:
            if not self.output_path.exists():
                self.output_path.mkdir(parents=True)
                self.logger.info(f"출력 폴더 생성: {self.output_path}")

            # PK 결과 저장
            pk_data = []
            for table, pk_tuple in self.table_pks.items():
                pk_str = ', '.join(pk_tuple) if pk_tuple else 'N/A'
                file_path = self.file_paths.get(table, '')
                file_path = file_path.replace('\\', '/')
                pk_data.append([table, file_path, pk_str])

            pk_df = pd.DataFrame(pk_data, columns=['FileName', 'FilePath', 'PK Columns'])
            if DEBUG_MODE:
                pk_file_path = self.output_path / f"{OUTPUT_ERD_FILE}_PK.csv"
                pk_df.to_csv(pk_file_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"PK 결과 저장: {pk_file_path}")

            # FK 결과 저장
            fk_data = []
            for child_table, fk_relations in self.table_fks.items():
                for rel in fk_relations:
                    fk_cols_str = ', '.join(rel['fk_columns'])
                    child_file_path = self.file_paths.get(rel['referencing_table'], '')
                    parent_file_path = self.file_paths.get(rel['referenced_table'], '')
                    fk_data.append([
                        rel['referencing_table'], 
                        child_file_path,
                        rel['referenced_table'],
                        parent_file_path,
                        fk_cols_str
                    ])

            fk_df = pd.DataFrame(fk_data, columns=['Child Table', 'Child FilePath', 'Parent Table', 'Parent FilePath', 'FK Columns'])
            if DEBUG_MODE:
                fk_file_path = self.output_path / f"{OUTPUT_ERD_FILE}_FK.csv"
                fk_df.to_csv(fk_file_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"FK 결과 저장: {fk_file_path}")

            integrated_data = []
            
            for table_name, df in self.dataframes.items():
            
                pk_cols_list = self.table_pks.get(table_name)
                pk_cols_set = set(pk_cols_list) if pk_cols_list else set()
                
                fk_info_table = self.fk_column_map.get(table_name, {})
                
                file_path = self.file_paths.get(table_name, '')
                file_path = file_path.replace('\\', '/')
                
                for column_name in df.columns:
                    
                    is_pk = 1 if column_name in pk_cols_set else ''
                    
                    fk_parent_files = fk_info_table.get(column_name)
                    
                    if fk_parent_files:
                        is_fk = 1 #'FK'
                        fk_files_str = ', '.join(sorted(set(fk_parent_files)))
                        # parent_table: 이 컬럼이 참조하는 테이블들 (FK인 경우)
                        parent_table_str = ', '.join(sorted(set(fk_parent_files)))
                        # child_table: 현재 테이블 (FK 컬럼을 가진 테이블)
                        child_table_str = table_name
                    else:
                        is_fk = ''
                        fk_files_str = ''
                        parent_table_str = ''
                        child_table_str = ''
                        
                    integrated_data.append({
                        'FilePath': file_path,
                        'FileName': table_name,
                        'Column': column_name,
                        'PK': is_pk,
                        'FK': is_fk,
                        'Related Files': fk_files_str,
                        'Related File #': len(fk_parent_files) if fk_parent_files else 0,
                    })

            integrated_df = pd.DataFrame(integrated_data)
            integrated_file_path = self.output_path / f"{OUTPUT_ERD_FILE}.csv"

            integrated_df.to_csv(integrated_file_path, index=False, encoding='utf-8-sig')

            return integrated_df
        except Exception as e:
            self.logger.error(f"create_physical_erd 함수 수행 실패: {e}")
            return None
    
    # def export_integrated_results_to_csv(self):
    #     """
    #     FileName | FilePath | Column | PK | FK | Related Files | Related File # 
    #     """
    #     integrated_data = []
        
    #     for table_name, df in self.dataframes.items():
            
    #         # PK 정보 로드 (Tuple/List 형태)
    #         pk_cols_list = self.table_pks.get(table_name)
    #         pk_cols_set = set(pk_cols_list) if pk_cols_list else set()
            
    #         # FK 정보 로드 (fk_column_map 사용)
    #         fk_info_table = self.fk_column_map.get(table_name, {})
            
    #         # FilePath 가져오기
    #         file_path = self.file_paths.get(table_name, '')
    #         file_path = file_path.replace('\\', '/')
            
    #         for column_name in df.columns:
                
    #             is_pk = 1 if column_name in pk_cols_set else ''
                
    #             # 해당 컬럼이 FK인지 확인 및 부모 파일 리스트 추출
    #             fk_parent_files = fk_info_table.get(column_name)
                
    #             if fk_parent_files:
    #                 is_fk = 1 #'FK'
    #                 fk_files_str = ', '.join(sorted(set(fk_parent_files)))
    #                 # parent_table: 이 컬럼이 참조하는 테이블들 (FK인 경우)
    #                 parent_table_str = ', '.join(sorted(set(fk_parent_files)))
    #                 # child_table: 현재 테이블 (FK 컬럼을 가진 테이블)
    #                 child_table_str = table_name
    #             else:
    #                 is_fk = ''
    #                 fk_files_str = ''
    #                 parent_table_str = ''
    #                 child_table_str = ''
                    
    #             integrated_data.append({
    #                 'FilePath': file_path,
    #                 'FileName': table_name,
    #                 'Column': column_name,
    #                 'PK': is_pk,
    #                 'FK': is_fk,
    #                 'Related Files': fk_files_str,
    #                 'Related File #': len(fk_parent_files) if fk_parent_files else 0,
    #             })

    #     integrated_df = pd.DataFrame(integrated_data)
    #     integrated_file_path = self.output_path / f"{OUTPUT_ERD_FILE}.csv"
        
    #     try:
    #         integrated_df.to_csv(integrated_file_path, index=False, encoding='utf-8-sig')
            
    #         return True
    #     except Exception as e:
    #         self.logger.error(f"최종 통합 CSV 저장 실패: {e}")
    #         return False
    
    # def summary_erd_results_to_csv(self):
    #     """
    #     FileName 기준으로 집계하여 요약 정보를 저장합니다.
    #     FileName | FilePath | Column_count | Parent table count | Child table count
    #     """
    #     try:
    #         # ERD 파일 로드
    #         erd_file_path = self.output_path / f"{OUTPUT_ERD_FILE}.csv"
    #         if not erd_file_path.exists():
    #             self.logger.error(f"ERD 파일을 찾을 수 없습니다: {erd_file_path}")
    #             return False
            
    #         integrated_df = pd.read_csv(erd_file_path, encoding='utf-8-sig')
            
    #         summary_data = []
            
    #         # FileName별로 그룹화
    #         for file_name in integrated_df['FileName'].unique():
    #             file_data = integrated_df[integrated_df['FileName'] == file_name]

    #             file_path = file_data['FilePath'].iloc[0] if len(file_data) > 0 else ''

    #             column_count = file_data['Column'].nunique()

    #             pk_column_count = file_data[file_data['PK'].astype(str) == '1']['Column'].nunique()

    #             fk_column_count = file_data[file_data['FK'].astype(str) == '1']['Column'].nunique()

    #             # PK & FK 동시에 만족한 컬럼 수
    #             pk_fk_column_count = file_data[(file_data['PK'].astype(str) == '1') & (file_data['FK'].astype(str) == '1')]['Column'].nunique()
                
    #             # PK 컬럼 목록 추출
    #             pk_columns = set(file_data[file_data['PK'].astype(str) == '1']['Column'].unique())
                
    #             # FK 컬럼 목록 추출
    #             fk_columns = set(file_data[file_data['FK'].astype(str) == '1']['Column'].unique())
                
    #             # Parent table count: FK인 컬럼들이 참조하는 고유한 parent_table 수
    #             parent_tables = set()
    #             fk_rows = file_data[file_data['FK'].astype(str) == '1']
    #             for parent_table_str in fk_rows['Related Files']:
    #                 if pd.notna(parent_table_str) and str(parent_table_str).strip():
    #                     # 쉼표로 구분된 parent_table들을 분리
    #                     parent_tables.update([t.strip() for t in str(parent_table_str).split(',') if t.strip()])
    #             parent_table_count = len(parent_tables)
                
    #             # Child table count: 이 파일을 참조하는 고유한 child_table 수
    #             child_tables = set()
    #             for child_table, fk_relations in self.table_fks.items():
    #                 for rel in fk_relations:
    #                     if rel['referenced_table'] == file_name:
    #                         # 이 파일을 참조하는 child_table 추가
    #                         child_tables.add(rel['referencing_table'])
    #             child_table_count = len(child_tables)
                
    #             summary_data.append({
    #                 'FilePath': file_path,
    #                 'FileName': file_name,
    #                 'Column #': column_count,
    #                 'Parent Table #': parent_table_count,
    #                 'Child Table #': child_table_count,
    #                 'PK Column #': pk_column_count,
    #                 'FK Column #': fk_column_count, 
    #                 'PK & FK Column #': pk_fk_column_count, 
    #                 'Parent Table List': ', '.join(sorted(parent_tables)) if parent_tables else '',
    #                 'Child Table List': ', '.join(sorted(child_tables)) if child_tables else '', 
    #                 'PK Column List': ', '.join(sorted(pk_columns)) if pk_columns else '',
    #                 'FK Column List': ', '.join(sorted(fk_columns)) if fk_columns else '',
    #             })
            
    #         summary_df = pd.DataFrame(summary_data)
    #         # 이미 문자열로 변환되었으므로 추가 변환 불필요
    #         # summary_df = summary_df.sort_values(by='FileName')
    #         summary_output_file = self.output_path / f"{OUTPUT_ERD_FILE}_Summary.csv"
            
    #         summary_df.to_csv(summary_output_file, index=False, encoding='utf-8-sig')
    #         if DEBUG_MODE:
    #             self.logger.info(f"집계 요약 결과 저장 완료: {summary_output_file}")
    #         return True
    #     except Exception as e:
    #         self.logger.error(f"집계 요약 CSV 저장 실패: {e}")
    #         return False
    #----------------------------------------------------------------------
    # 논리적 ERD 생성 함수
    #----------------------------------------------------------------------
    def parse_relationship(self, relationship_str: str) -> List[Dict[str, str]]:
        """
        '파일1.컬럼1 -> 파일2.컬럼2 -> ...' 형태의 문자열에서 모든 1:1 관계 목록을 추출합니다.
        """
        if not isinstance(relationship_str, str) or '->' not in relationship_str:
            return []
        
        
        segments = relationship_str.split(' -> ') # '파일.컬럼' 형태로 분할
        
        relationships = []
        # 1:1 관계 리스트 생성 (Parent -> Child)
        # segments[i]가 부모, segments[i+1]이 자식 관계가 됩니다.
        for i in range(len(segments) - 1):
            parent_segment = segments[i].strip()
            child_segment = segments[i+1].strip()
            
            try:
                parent_file, parent_col = parent_segment.rsplit('.', 1)  # 파일명과 컬럼명 분리 (마지막 '.
                child_file, child_col = child_segment.rsplit('.', 1)
                
                # Level_Relationship: [상위 파일1 -> 하위 파일2]
                # ERD 관계: [하위 파일2 (Child) -> 상위 파일1 (Parent)]
                relationships.append({
                    'Child_Table': child_file,
                    'Child_Column': child_col,
                    'Parent_Table': parent_file,
                    'Parent_Column': parent_col
                })
                
            except ValueError:
                continue
                
        return relationships

    def create_logical_erd(self, input_file_name: str = "CodeMapping_relationship.csv"):
        """
        CodeMapping_relationship.csv 파일을 읽어서 논리적 ERD를 생성하여 파일로 저장합니다.
        """
        self.logger.info("4th Step: 논리적 ERD 생성 시작")
        
        try:
            input_file_path = self.output_path / input_file_name
            if not input_file_path.exists():
                base_path = Path(self.config['ROOT_PATH'])
                input_file_path = base_path / "DS_Output" / input_file_name
                if not input_file_path.exists():
                    self.logger.error(f"입력 파일을 찾을 수 없습니다: {input_file_name}")
                    return False

            df_raw = pd.read_csv(input_file_path, encoding='utf-8-sig')
            
            # 3.1. 모든 테이블 및 컬럼 정보 추출 (노드 정보)
            tables_data = {}  # {FileName: {ColumnName: PK_Status}}
            file_path_map = {}  # {FileName: FilePath} - 각 파일명에 대한 FilePath 저장
            
            for _, row in df_raw.iterrows():
                file_path = str(row.get('FilePath', '')).strip()
                file_path = file_path.replace('\\', '/')
                file_name = row['FileName']
                col_name = row['ColumnName']
                pk_status = 'PK' if str(row.get('PK', '')).strip() == '1' else ''
                
                if file_name not in tables_data:
                    tables_data[file_name] = defaultdict(lambda: {'PK': '', 'FK': '', 'Parent_Table': '', 'Parent_Column': '', 'child_table': '', 'child_column': ''})
                    if file_path:
                        file_path_map[file_name] = file_path
                
                tables_data[file_name][col_name]['PK'] = pk_status
            
            # 3.2. 관계 정보 추출 및 FK 업데이트 (엣지 정보)
            all_relationships = []
            for _, row in df_raw.iterrows():
                rel_str = row.get('Level_Relationship')
                
                if pd.isna(rel_str):
                    continue
                
                for rel in self.parse_relationship(str(rel_str)):
                    all_relationships.append(rel)
                    
                    child_table = rel['Child_Table']
                    child_col = rel['Child_Column']
                    parent_table = rel['Parent_Table']
                    parent_col = rel['Parent_Column']
                    
                    # Child Table의 해당 컬럼을 FK로 표시
                    if child_table in tables_data and child_col in tables_data[child_table]:
                        # FK로 표시
                        tables_data[child_table][child_col]['FK'] = 'FK'
                        
                        # 부모 테이블 목록 추가 (중복 제거)
                        current_parents = tables_data[child_table][child_col]['Parent_Table']
                        if current_parents:
                            parent_list = [p.strip() for p in current_parents.split(',')]
                            if parent_table not in parent_list:
                                tables_data[child_table][child_col]['Parent_Table'] += f", {parent_table}"
                        else:
                            tables_data[child_table][child_col]['Parent_Table'] = parent_table
                        
                        # 부모 컬럼 목록 추가 (중복 제거)
                        current_parents_col = tables_data[child_table][child_col]['Parent_Column']
                        if current_parents_col:
                            parent_col_list = [p.strip() for p in current_parents_col.split(',')]
                            if parent_col not in parent_col_list:
                                tables_data[child_table][child_col]['Parent_Column'] += f", {parent_col}"
                        else:
                            tables_data[child_table][child_col]['Parent_Column'] = parent_col
            
            # 3.3. 최종 통합 DataFrame 생성
            erd_data_list = []
            
            for file_name, columns in tables_data.items():
                file_path = file_path_map.get(file_name, '')
                for col_name, info in columns.items():
                    if info['Parent_Table']:
                        unique_parent_tables = set([t.strip() for t in info['Parent_Table'].split(',') if t.strip()])
                        parent_table_count = len(unique_parent_tables)
                        parent_table_str = ', '.join(sorted(unique_parent_tables))
                    else:
                        parent_table_count = 0
                        parent_table_str = ''
                    
                    if info['Parent_Column']:
                        unique_parent_columns = set([c.strip() for c in info['Parent_Column'].split(',') if c.strip()])
                        parent_column_str = ', '.join(sorted(unique_parent_columns))
                    else:
                        parent_column_str = ''
                    
                    erd_data_list.append({
                        'FilePath': file_path,
                        'FileName': file_name,
                        'Column': col_name,
                        'L_PK': 'PK' if info['PK'] == 'PK' else '',
                        'L_FK': 'FK' if info['FK'] == 'FK' else '',
                        'L_Related Table #': parent_table_count,
                        'L_Related Tables': parent_table_str,
                        'L_Related Columns': parent_column_str, 
                    })
            
            # 최종 ERD 관계 목록 (Parent -> Child 관계에 집중)
            unique_relationships = {}
            for rel in all_relationships:
                key = (rel['Child_Table'], rel['Parent_Table'])
                # 복합키/다중 관계 처리를 위해 컬럼 정보를 쉼표로 구분하여 추가
                if key not in unique_relationships:
                    unique_relationships[key] = {
                        'Child_Table': rel['Child_Table'],
                        'Parent_Table': rel['Parent_Table'],
                        'FK_Columns': set(),
                        'PK_Columns': set()
                    }
                
                unique_relationships[key]['FK_Columns'].add(rel['Child_Column'])
                unique_relationships[key]['PK_Columns'].add(rel['Parent_Column'])
            
            # 4. 최종 CSV 파일 저장
            df_erd_attributes = pd.DataFrame(erd_data_list)
            
            df_erd_relationships = pd.DataFrame([
                {
                    'Child_Table': rel['Child_Table'],
                    'Parent_Table': rel['Parent_Table'],
                    'FK_Columns': ', '.join(sorted(rel['FK_Columns'])),
                    'Parent_PK_Columns': ', '.join(sorted(rel['PK_Columns']))
                }
                for rel in unique_relationships.values()
            ])
            
            # 4.3. 파일 저장
            erd_attributes_file = self.output_path / f"{OUTPUT_LOGICAL_FILE}.csv"
            erd_relationships_file = self.output_path / f"{OUTPUT_LOGICAL_FILE}_List.csv"
            
            # 모든 컬럼 속성 정보를 저장
            df_erd_attributes.to_csv(erd_attributes_file, index=False, encoding='utf-8-sig')
            if DEBUG_MODE:
                self.logger.info(f"논리적 ERD 속성 정보 저장 완료: {erd_attributes_file}")
            
            # 관계 목록을 별도의 파일로 저장하여 ERD 생성 시 활용할 수 있도록 합니다.
            df_erd_relationships.to_csv(erd_relationships_file, index=False, encoding='utf-8-sig')
            if DEBUG_MODE:
                self.logger.info(f"논리적 ERD 관계 목록 저장 완료: {erd_relationships_file}")
            
            return df_erd_attributes, df_erd_relationships
        except Exception as e:
            self.logger.error(f"create_logical_erd 실패: {e}")
            return None, None

    def merge_erd_and_logical_erd(self):
        """
        ERD 및 논리적 ERD를 병합하여 하나의 파일로 저장합니다.
        """
        self.logger.info("5th Step: ERD 및 논리적 ERD 병합 시작")
        try:
            # ERD 파일 로드
            erd_file_path = self.output_path / f"{OUTPUT_ERD_FILE}.csv"
            erd_df = pd.read_csv(erd_file_path, encoding='utf-8-sig')
            
            # 논리적 ERD 파일 로드
            logical_erd_file_path = self.output_path / f"{OUTPUT_LOGICAL_FILE}.csv"
            logical_erd_df = pd.read_csv(logical_erd_file_path, encoding='utf-8-sig')
            
            # ERD 및 논리적 ERD 병합
            merged_df = pd.merge(erd_df, logical_erd_df, on=['FilePath', 'FileName', 'Column'], how='left')
            
            # 병합된 결과 저장
            merged_file_path = self.output_path / f"{OUTPUT_ERD_FILE}_Physical_Logical.csv"
            merged_df.to_csv(merged_file_path, index=False, encoding='utf-8-sig')

            self.logger.info(f"ERD 및 논리적 ERD 병합 결과 저장 완료: {merged_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"merge_erd_and_logical_erd 함수 수행 실패: {e}")
            return False

    #----------------------------------------------------------------------
    # 집계 결과 생성 함수
    #----------------------------------------------------------------------
    def erd_merge_summary(self, physical_erd_df: pd.DataFrame, logical_erd_df: pd.DataFrame) -> pd.DataFrame:
        """
        물리적 ERD 및 논리적 ERD를 병합하여 하나의 파일로 저장합니다.
        &
        FilePath와 FileName 기준으로 데이터를 집계하여 요약 데이터프레임을 생성합니다.

        집계 규칙:
        - Column #: count(Column)
        - PK Column #: count(PK = 1)
        - FK Column #: count(FK = 1)
        - Related File List: Related Files의 유니크한 값 목록 (문자열)
        - Related File #: Max(Related File #)
        - L_PK Column #: count(L_PK = 'PK')
        - L_PK Columns: L_PK Columns의 유니크한 값 목록 (문자열)
        - L_FK Column #: count(L_FK = 'FK')
        - L_FK Columns: L_FK Columns의 유니크한 값 목록 (문자열)
        - L_Related File List: L_Related Tables의 유니크한 값 목록 (문자열)
        - L_Related File #: Max(L_Related Table #)
        - L_Related Column List: L_Related Columns의 유니크한 값 목록 (문자열)
        - L_Related Column #: Max(L_Related Column #)
        """

        self.logger.info("5th Step: ERD 및 논리적 ERD 병합 시작")
        try:
            # # ERD 파일 로드
            # erd_file_path = self.output_path / f"{OUTPUT_ERD_FILE}.csv"
            # erd_df = pd.read_csv(erd_file_path, encoding='utf-8-sig')
            
            # # 논리적 ERD 파일 로드
            # logical_erd_file_path = self.output_path / f"{OUTPUT_LOGICAL_FILE}.csv"
            # logical_erd_df = pd.read_csv(logical_erd_file_path, encoding='utf-8-sig')

            # ERD 및 논리적 ERD 병합
            merged_df = pd.merge(physical_erd_df, logical_erd_df, on=['FilePath', 'FileName', 'Column'], how='left')
            
            # 병합된 결과 저장
            merged_file_path = self.output_path / f"{OUTPUT_ERD_FILE}_Physical_Logical.csv"
            merged_df.to_csv(merged_file_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            self.logger.error(f"erd_merge_summary 함수 수행 실패: {e}")
            return False

        #----------------------------------------------------------------------
        # 집계 결과 생성 함수
        #----------------------------------------------------------------------
        # 'Related Files'와 'L_Related Tables'에서 파일명(테이블명)만 추출하여 유니크한 집합을 구하는 함수
        def aggregate_unique_files(series: pd.Series) -> str:
            unique_files = set()
            for item in series.dropna().astype(str):
                files = item.split(',')
                for file in files:
                    file = file.strip()  # 앞뒤 공백만 제거
                    if file: # 빈 문자열 방지
                        unique_files.add(file)
            
            return ', '.join(sorted(list(unique_files)))

        # erd_file_path = self.output_path / f"{OUTPUT_ERD_FILE}_Physical_Logical.csv"
        # if not erd_file_path.exists():
        #     self.logger.error(f"ERD 파일을 찾을 수 없습니다: {erd_file_path}")
        #     return False
        
        # df = pd.read_csv(erd_file_path, encoding='utf-8-sig')

        df = merged_df.copy()

        aggregation_dict = {
            'Column': 'count', # Column #
            
            'PK': lambda x: (x == 1).sum(), # PK Column #
            'FK': lambda x: (x == 1).sum(), # FK Column #
            
            # Related Files 목록 집계
            'Related Files': aggregate_unique_files, # Related File List
            
            # L_PK/L_FK 개수 (조건부 합계 사용)
            'L_PK': lambda x: (x == 'PK').sum(), # L_PK Column #
            'L_FK': lambda x: (x == 'FK').sum(), # L_FK Column #

            # L_Related Tables 목록 집계
            'L_Related Tables': aggregate_unique_files, # L_Related File List
            'L_Related Columns': aggregate_unique_files, # L_Related Column List
        }

        # 2. 'FilePath', 'FileName' 기준으로 그룹화 및 집계 수행
        summary_df = df.groupby(['FilePath', 'FileName'], dropna=False).agg(aggregation_dict).reset_index()

        # 3. 컬럼 이름 변경
        summary_df.rename(columns={
            'Column': 'Column #',
            'PK': 'PK Column #',
            'FK': 'FK Column #',
            'Related Files': 'Related File List',
            'L_PK': 'L_PK Column #',
            'L_FK': 'L_FK Column #',
            'L_Related Tables': 'L_Related File List',
            'L_Related Columns': 'L_Related Column List',
        }, inplace=True)
        
        # 3.5. L_PK Column List와 L_FK Column List 생성 (조건부 집계)
        def get_pk_columns(group_df):
            """L_PK가 'PK'인 행들의 Column 값을 수집"""
            pk_cols = group_df[group_df['PK'] == 1]['Column'].dropna().unique()
            return ', '.join(sorted([str(col) for col in pk_cols if col]))
        
        def get_fk_columns(group_df):
            """L_FK가 'FK'인 행들의 Column 값을 수집"""
            fk_cols = group_df[group_df['FK'] == 1]['Column'].dropna().unique()
            return ', '.join(sorted([str(col) for col in fk_cols if col]))

        def get_L_PK_columns(group_df):
            """L_PK가 'PK'인 행들의 Column 값을 수집"""
            pk_cols = group_df[group_df['L_PK'] == 'PK']['Column'].dropna().unique()
            return ', '.join(sorted([str(col) for col in pk_cols if col]))
        
        def get_L_FK_columns(group_df):
            """L_FK가 'FK'인 행들의 Column 값을 수집"""
            fk_cols = group_df[group_df['L_FK'] == 'FK']['Column'].dropna().unique()
            return ', '.join(sorted([str(col) for col in fk_cols if col]))
        
        # 각 그룹별로 L_PK Column List와 L_FK Column List 생성
        pk_fk_lists = df.groupby(['FilePath', 'FileName'], dropna=False).apply(
            lambda g: pd.Series({
                'PK Column List': get_pk_columns(g),
                'FK Column List': get_fk_columns(g),
                'L_PK Column List': get_L_PK_columns(g),
                'L_FK Column List': get_L_FK_columns(g)
            })
        ).reset_index()
        
        # summary_df와 병합
        summary_df = summary_df.merge(pk_fk_lists, on=['FilePath', 'FileName'], how='left')
        
        # 3.6. L_Related File #를 L_Related File List에서 재계산 (정확성 보장)
        def count_from_list(file_list_str):
            if pd.isna(file_list_str) or file_list_str == '':
                return 0
            files = [f.strip() for f in str(file_list_str).split(',') if f.strip()]
            return len(set(files))  # 중복 제거
        
        summary_df['L_Related File #'] = summary_df['L_Related File List'].apply(count_from_list)
        summary_df['Related File #'] = summary_df['Related File List'].apply(count_from_list)
        summary_df['L_Related Column #'] = summary_df['L_Related Column List'].apply(count_from_list)
        
        # 4. 결과 컬럼 순서 조정
        final_columns = [
            'FilePath', 'FileName', 'Column #', 'PK Column #', 'PK Column List', 
            'FK Column #', 'FK Column List', 'Related File List', 'Related File #', 
            'L_PK Column #', 'L_PK Column List', 'L_FK Column #', 'L_FK Column List', 
            'L_Related File List', 'L_Related File #',
            'L_Related Column List', 'L_Related Column #',
        ]
        
        available_columns = [col for col in final_columns if col in summary_df.columns]
        summary_df = summary_df[available_columns]
        
        summary_output_file = self.output_path / f"{OUTPUT_ERD_FILE}_Summary.csv"
        summary_df.to_csv(summary_output_file, index=False, encoding='utf-8-sig')
        if DEBUG_MODE:
            self.logger.info(f"집계 요약 결과 저장 완료: {summary_output_file}")

        return True

            
# --- 4. 메인 실행 함수 ---
def main():
    start = time.time()
    try:
        analyzer = DataAnalyzer()
        
        #---------------------------------------
        # 물리적 ERD 생성
        #---------------------------------------
        # 1. 파일 로드
        result = analyzer.load_files()
        if not result:
            print("FATAL ERROR: 파일 로드 실패")
            sys.exit(1)
        
        # 2. PK 추론 (개선된 순차 방식 적용됨)
        result = analyzer.discover_pk()
        if not result:
            print("FATAL ERROR: PK 추론 실패")
            sys.exit(1)
        
        # 3. FK 추론
        result = analyzer.discover_fk()
        if not result:
            print("FATAL ERROR: FK 추론 실패")
            sys.exit(1)
        
        # 4. 결과 CSV 파일 저장 (PK/FK 개별 파일 + 최종 통합 파일)
        physical_erd_df = analyzer.create_physical_erd()
        if physical_erd_df is None:
            print("FATAL ERROR: 물리적 ERD 생성 실패")
            sys.exit(1)
        
        #---------------------------------------
        # 논리적 ERD 생성
        #---------------------------------------
        # 5. 논리적 ERD 생성 (CodeMapping_relationship.csv 기반)
        logical_erd_df, logical_erd_list_df = analyzer.create_logical_erd()
        if logical_erd_df is None or logical_erd_list_df is None:
            print("FATAL ERROR: 논리적 ERD 생성 실패")
            sys.exit(1)
        
        result = analyzer.erd_merge_summary(physical_erd_df, logical_erd_df)
        if not result:
            print("FATAL ERROR: 집계 결과 생성 실패")
            sys.exit(1)
        
        print("="*50)
        print("Success : ERD 및 Logical ERD 생성 완료")
        print("="*50)
        print(f"총 처리 시간: {time.time()-start:.2f}초")
        print("="*50)
        
    # except FileNotFoundError as e:
    #     print(f"\nFATAL ERROR: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()