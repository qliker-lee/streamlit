import pandas as pd
import numpy as np
import hashlib
import time
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MigrationValidator:
    """데이터 마이그레이션 정합성 검증 클래스"""
    
    def __init__(self):
        # 경로 설정
        self.BASE_PATH = Path("C:/projects/myproject/QDQM/QDQM_Master_Code/test/mig")
        self.MASTER_DIR = Path.cwd() / 'QDQM_Master_Code'
        self.META_DIR = self.MASTER_DIR / 'QDQM_Meta'
        self.OUTPUT_DIR = self.MASTER_DIR / 'QDQM_Output'
        self.OUTPUT_DETAIL_DIR = self.OUTPUT_DIR / 'Migration_Detail'
        
        # 파일 설정
        self.META_FILE = self.META_DIR / 'Migration_Meta.xlsx'
        self.RESULT_FILE = self.OUTPUT_DIR / 'Migration_Result.csv'
        
        # 검증 파라미터
        self.MAX_COMPARE_COLS = 5
        self.REL_TOLERANCE = 0.00001
        
        # 디렉토리 생성
        self.OUTPUT_DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    def calculate_checksum(self, file_path: str) -> Optional[str]:
        """파일의 MD5 체크섬 계산"""
        path = Path(file_path) if os.path.isabs(file_path) else self.BASE_PATH / file_path
        
        if not path.exists():
            logger.warning(f"파일 없음: {path}")
            return None
            
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def compare_numeric_data(self, src_df: pd.DataFrame, tgt_df: pd.DataFrame, cols: List[str]) -> Dict:
        """수치 데이터 정합성 비교 (합계 기반)"""
        results = {}
        for col in cols[:self.MAX_COMPARE_COLS]:
            if col in src_df.columns and col in tgt_df.columns:
                src_sum = pd.to_numeric(src_df[col], errors='coerce').sum()
                tgt_sum = pd.to_numeric(tgt_df[col], errors='coerce').sum()
                
                is_match = np.isclose(src_sum, tgt_sum, rtol=self.REL_TOLERANCE)
                results[col] = {
                    'src_sum': src_sum,
                    'tgt_sum': tgt_sum,
                    'status': 'Success' if is_match else 'Fail'
                }
        return results

    def process_migration_case(self, row: pd.Series) -> pd.DataFrame:
        """개별 마이그레이션 케이스 검증 실행"""
        mig_id = row['Mig_ID']
        src_file = row['Source_File']
        tgt_file = row['Target_File']
        comp_cols = str(row.get('Compare_Columns', '')).split(',') if pd.notna(row.get('Compare_Columns')) else []

        logger.info(f"[{mig_id}] 검증 시작: {src_file} -> {tgt_file}")
        
        # 1. 체크섬 계산
        src_md5 = self.calculate_checksum(src_file)
        tgt_md5 = self.calculate_checksum(tgt_file)
        
        # 2. 데이터 로드 및 건수 비교
        try:
            src_path = self.BASE_PATH / src_file
            tgt_path = self.BASE_PATH / tgt_file
            
            # 파일 포맷에 따른 로딩 (예시로 csv 기준, 필요시 확장)
            src_df = pd.read_csv(src_path) if src_path.suffix == '.csv' else pd.read_excel(src_path)
            tgt_df = pd.read_csv(tgt_path) if tgt_path.suffix == '.csv' else pd.read_excel(tgt_path)
            
            src_cnt, tgt_cnt = len(src_df), len(tgt_df)
            cnt_status = "Success" if src_cnt == tgt_cnt else "Fail"
            
            # 3. 수치 데이터 비교
            num_results = self.compare_numeric_data(src_df, tgt_df, comp_cols)
            
            # 결과 레코드 생성
            res = {
                'Mig_ID': mig_id,
                'Source_MD5': src_md5,
                'Target_MD5': tgt_md5,
                'Source_Count': src_cnt,
                'Target_Count': tgt_cnt,
                'Count_Status': cnt_status,
                'MD5_Status': "Success" if src_md5 == tgt_md5 else "Fail",
                'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 수치 결과 동적 추가
            for i, (col, data) in enumerate(num_results.items(), 1):
                res[f'Col{i}_Name'] = col
                res[f'Col{i}_Src_Sum'] = data['src_sum']
                res[f'Col{i}_Tgt_Sum'] = data['tgt_sum']
                res[f'Col{i}_Status'] = data['status']
                
            return pd.DataFrame([res])

        except Exception as e:
            logger.error(f"[{mig_id}] 처리 중 오류: {e}")
            return pd.DataFrame([{'Mig_ID': mig_id, 'Status': 'Error', 'Error_Msg': str(e)}])

    def run(self):
        """전체 검증 프로세스 실행"""
        start_time = time.time()
        
        if not self.META_FILE.exists():
            logger.error("메타 파일이 존재하지 않습니다.")
            return

        # 메타 정보 로드
        meta_df = pd.read_excel(self.META_FILE)
        output_dfs = []

        for _, row in meta_df.iterrows():
            result = self.process_migration_case(row)
            output_dfs.append(result)

        # 결과 병합 및 저장
        if output_dfs:
            final_df = pd.concat(output_dfs, ignore_index=True)
            # 모든 컬럼 문자열화 (CSV 저장 시 타입 충돌 방지)
            final_df = final_df.astype(str).replace('nan', '')
            
            final_df.to_csv(self.RESULT_FILE, index=False, encoding='utf-8-sig')
            
            duration = time.time() - start_time
            logger.info(f"검증 완료. 소요시간: {duration:.2f}초")
            logger.info(f"결과 저장: {self.RESULT_FILE}")

if __name__ == "__main__":
    validator = MigrationValidator()
    validator.run()