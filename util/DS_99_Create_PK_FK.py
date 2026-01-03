# -*- coding: utf-8 -*-
"""
DS_16_Create_PK_FK
codelist_meta.xlsx 파일을 읽어서 소스 파일을 로드하고, 각 파일에 대한 PK, FK를 추론하여 파일로 저장
2025.12.09 Qliker
"""
import pandas as pd
import os
import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import time
import io
from typing import Dict, Any, Optional, List

# --- 1. 설정 및 상수 ---
# 현재 파일의 상위 2단계 폴더를 path에 추가
ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# YAML 파일 경로
YAML_PATH = ROOT_PATH / "DataSense" / "util" / "DS_Master.yaml"

from DataSense.util.io import Load_Yaml_File, Backup_File, setup_logger 

MAX_PK_COMBINATION = 20  # PK 탐색 시 최대 컬럼 조합 수 (성능을 위해 제한)
OUTPUT_FILE_NAME = 'DS_Key_Result'

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
        self.logger = setup_logger(app_name, True)
        self.config: Dict[str, Any] = {}
        self.yaml_path = Path(yaml_path or YAML_PATH)
        
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {self.yaml_path}")
        
        self.config = Load_Yaml_File(str(self.yaml_path))
        
        # ROOT_PATH가 YAML에 없으면 자동으로 설정
        if 'ROOT_PATH' not in self.config or not self.config.get('ROOT_PATH'):
            self.config['ROOT_PATH'] = str(ROOT_PATH)
            self.logger.info(f"ROOT_PATH 자동 설정: {self.config['ROOT_PATH']}")
        
        # base_path와 output_path 설정
        base_path = Path(self.config['ROOT_PATH'])
        directories = self.config.get('directories', {})
        self.base_path = base_path / directories.get('input', 'DS_Input')
        self.output_path = base_path / directories.get('output', 'DS_Output')
        
        self.dataframes = {}
        self.table_pks = {}
        self.table_fks = defaultdict(list)
        # FK 정보를 컬럼별/테이블별로 쉽게 조회하기 위한 딕셔너리 추가
        self.fk_column_map = defaultdict(lambda: defaultdict(list))
        # 파일 경로 저장 (FilePath 컬럼용)
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
        self.logger.info("파일 로드 시작")
        
        files_list = self.get_source_files()
        if not files_list:
            raise FileNotFoundError(f"처리할 파일이 없습니다. codelist_meta 파일을 확인하세요.")
        
        self.logger.info(f"총 {len(files_list)}개의 파일을 로드합니다.")
        
        for file_path, source_type in files_list:
            table_name = Path(file_path).stem
            try:
                # CSV 파일 읽기
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, dtype=str, keep_default_na=False, low_memory=False)
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
                else:
                    self.logger.warning(f"지원하지 않는 파일 형식: {file_path}")
                    continue
                
                self.dataframes[table_name] = df
                # 파일 경로 저장
                self.file_paths[table_name] = str(file_path)
                # self.logger.debug(f"  - {table_name} 로드 완료")
            except Exception as e:
                self.logger.error(f"  - ERROR: {table_name} 로드 중 오류 발생: {e}")
        
        self.logger.info(f"파일 로드 완료: {len(self.dataframes)}개 파일")

    def discover_pk(self):
        """Primary Key (PK)를 추론합니다."""
        self.logger.info("2nd Step: Primary Key (PK) 추론 시작")
        
        for table_name, df in self.dataframes.items():
            if df.empty:
                self.table_pks[table_name] = None
                continue
            
            # 1단계: 단일 필드가 유니크 값을 갖는지 확인
            single_pk = None
            for col in df.columns:
                # 해당 컬럼이 유니크한지 확인 (중복이 없는지)
                if df[col].duplicated().sum() == 0:
                    single_pk = [col]
                    break
            
            if single_pk:
                # 단일 필드 PK 발견
                self.table_pks[table_name] = single_pk
                # self.logger.debug(f"  - {table_name}: PK = {', '.join(single_pk)}")
            else:
                # 2단계: 단일 필드 PK가 없으면 복합 PK 탐색
                pk_combo = self.find_unique_combo_sequential(df, MAX_PK_COMBINATION) 
                
                if pk_combo:
                    self.table_pks[table_name] = pk_combo
                    # self.logger.debug(f"  - {table_name}: PK = {', '.join(pk_combo)}")
                else:
                    print(f"  - {table_name}: PK 없음") # DEBUG
                    self.table_pks[table_name] = None
                    # self.logger.debug(f"  - {table_name}: PK 없음")
        
        pk_count = sum(1 for pk in self.table_pks.values() if pk is not None)
        self.logger.info(f"PK 추론 완료: {pk_count}개 테이블에서 PK 발견")

    # find_unique_combo_sequential 함수를 클래스 메서드로 추가
    def find_unique_combo_sequential(self, df: pd.DataFrame, max_try_cols: int = 6) -> list:
        # 클래스 밖의 전역 함수 find_unique_combo_sequential을 호출하거나,
        # 편의를 위해 클래스 메서드로 재정의 (여기서는 클래스 메서드 사용)
        return find_unique_combo_sequential(df, max_try_cols)


    def establish_fk(self):
        """Foreign Key (FK) 관계를 설정하고, fk_column_map을 채웁니다."""
        self.logger.info("Foreign Key (FK) 관계 설정 시작")
        
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
                            # self.logger.debug(f"  - FK 발견: {child_table} -> {parent_table} ({', '.join(pk_columns)})")
                            
                            # FK 컬럼 맵 업데이트 (통합 CSV 생성에 사용)
                            for col in pk_columns:
                                self.fk_column_map[child_table][col].append(parent_table)
        
        self.logger.info(f"FK 관계 설정 완료: {fk_count}개 관계 발견")
        
    def export_results_to_csv(self):
        """추론된 PK 및 FK 결과를 각각 CSV 파일로 저장합니다."""
        
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
            self.logger.info(f"출력 폴더 생성: {self.output_path}")

        # PK 결과 저장
        pk_data = []
        for table, pk_tuple in self.table_pks.items():
            pk_str = ', '.join(pk_tuple) if pk_tuple else 'N/A'
            file_path = self.file_paths.get(table, '')
            pk_data.append([table, file_path, pk_str])
        pk_df = pd.DataFrame(pk_data, columns=['테이블명', 'FilePath', 'PK 컬럼'])
        pk_output_file = self.output_path / f"{OUTPUT_FILE_NAME}_PK.csv"
        pk_df.to_csv(pk_output_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"PK 결과 저장 완료: {pk_output_file}")

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
        fk_output_file = self.output_path / f"{OUTPUT_FILE_NAME}_FK.csv"
        fk_df.to_csv(fk_output_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"FK 결과 저장 완료: {fk_output_file}")
        
        # 통합 결과 저장 호출
        self.export_integrated_results_to_csv()
        
    
    def export_integrated_results_to_csv(self):
        """
        FileName | FilePath | Column | PK | FK | FK Files | File Count | parent_table | child_table
        """
        integrated_data = []
        
        for table_name, df in self.dataframes.items():
            
            # PK 정보 로드 (Tuple/List 형태)
            pk_cols_list = self.table_pks.get(table_name)
            pk_cols_set = set(pk_cols_list) if pk_cols_list else set()
            
            # FK 정보 로드 (fk_column_map 사용)
            fk_info_table = self.fk_column_map.get(table_name, {})
            
            # FilePath 가져오기
            file_path = self.file_paths.get(table_name, '')
            
            for column_name in df.columns:
                
                is_pk = 1 if column_name in pk_cols_set else ''
                
                # 해당 컬럼이 FK인지 확인 및 부모 파일 리스트 추출
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
                    'FileName': table_name,
                    'FilePath': file_path,
                    'Column': column_name,
                    'PK': is_pk,
                    'FK': is_fk,
                    'FK Files': fk_files_str,
                    'File Count': len(fk_parent_files) if fk_parent_files else 0,
                    'parent_table': parent_table_str,
                    'child_table': child_table_str
                })
        
        integrated_df = pd.DataFrame(integrated_data)
        integrated_output_file = self.output_path / f"{OUTPUT_FILE_NAME}_PK_FK.csv"
        
        try:
            integrated_df.to_csv(integrated_output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"최종 통합 결과 저장 완료: {integrated_output_file}")
            
            # 집계 결과 저장 호출
            self.export_summary_results_to_csv(integrated_df)
        except Exception as e:
            self.logger.error(f"최종 통합 CSV 저장 실패: {e}")
    
    def export_summary_results_to_csv(self, integrated_df: pd.DataFrame):
        """
        FileName 기준으로 집계하여 요약 정보를 저장합니다.
        FileName | FilePath | Column_count | Parent table count | Child table count
        """
        summary_data = []
        
        # FileName별로 그룹화
        for file_name in integrated_df['FileName'].unique():
            file_data = integrated_df[integrated_df['FileName'] == file_name]
            
            # FilePath (첫 번째 행에서 가져오기)
            file_path = file_data['FilePath'].iloc[0] if not file_data['FilePath'].empty else ''
            
            # Column_count: 고유한 컬럼 수
            column_count = file_data['Column'].nunique()
            
            # Parent table count: FK인 컬럼들이 참조하는 고유한 parent_table 수
            parent_tables = set()
            for parent_table_str in file_data[file_data['FK'] == 1]['parent_table']:
                if pd.notna(parent_table_str) and parent_table_str.strip():
                    # 쉼표로 구분된 parent_table들을 분리
                    parent_tables.update([t.strip() for t in str(parent_table_str).split(',') if t.strip()])
            parent_table_count = len(parent_tables)
            
            # Child table count: 이 파일을 참조하는 고유한 child_table 수
            # self.table_fks를 사용하여 더 효율적으로 계산
            child_tables = set()
            for child_table, fk_relations in self.table_fks.items():
                for rel in fk_relations:
                    if rel['referenced_table'] == file_name:
                        # 이 파일을 참조하는 child_table 추가
                        child_tables.add(rel['referencing_table'])
            child_table_count = len(child_tables)
            
            summary_data.append({
                'FileName': file_name,
                'FilePath': file_path,
                'Column #': column_count,
                'Parent Table #': parent_table_count,
                'Child Table #': child_table_count
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by='FileName')
        summary_output_file = self.output_path / f"{OUTPUT_FILE_NAME}_Summary.csv"
        
        try:
            summary_df.to_csv(summary_output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"집계 요약 결과 저장 완료: {summary_output_file}")
        except Exception as e:
            self.logger.error(f"집계 요약 CSV 저장 실패: {e}")


# --- 4. 메인 실행 함수 ---
def main():
    start = time.time()
    try:
        analyzer = DataAnalyzer()
        
        # 1. 파일 로드
        analyzer.load_files()
        
        # 2. PK 추론 (개선된 순차 방식 적용됨)
        analyzer.discover_pk()
        
        # 3. FK 관계 설정
        analyzer.establish_fk()
        
        # 4. 결과 CSV 파일 저장 (PK/FK 개별 파일 + 최종 통합 파일)
        analyzer.export_results_to_csv()
        
        print("="*50)
        print("Success : PK, FK 생성 완료")
        print("="*50)
        print(f"총 처리 시간: {time.time()-start:.2f}초")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()