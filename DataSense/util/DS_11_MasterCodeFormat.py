# -*- coding: utf-8 -*-
"""
DataSense DQ Profiling System - Refactored for Maintenance & EXE
- 모든 기존 비즈니스 로직 보존
- 클래스 기반 모듈화로 가독성 향상
- 실행 파일(EXE) 경로 대응 로직 포함
"""

import os
import re
import sys
import yaml
import time
import json  # <--- 이 줄이 반드시 있어야 합니다.
import traceback
import platform
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

# --- [1. 경로 및 설정 관리 클래스] ---
class DQConfig:
    ROOT_PATH = Path(__file__).resolve().parents[2]
    YAML_RELATIVE_PATH = 'DataSense/util/DS_Master.yaml'
    CONTRACT_RELATIVE_PATH = 'DataSense/util/DQ_Contract.yaml'

    @staticmethod
    def get_path(rel_path):
        """EXE 빌드 환경과 일반 파이썬 환경 모두 대응"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, rel_path)
        return os.path.join(DQConfig.ROOT_PATH, rel_path)

# sys.path 추가 (내부 모듈 참조용)
if str(DQConfig.ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(DQConfig.ROOT_PATH))

# 외부 유틸 임포트 (기존 함수들)
try:
    from DataSense.util.io import Load_Yaml_File, Backup_File
    from DataSense.util.dq_function import (
        DataType_Analysis, create_standard_df, Determine_Detail_Type,
        Get_Oracle_Type, _ensure_severity_column, build_top_issue_reports, 
        save_or_load_baseline, compute_proxy_drift, make_df_for_distribution, 
        build_dist_snapshot_for_df, collect_value_samples, dist_topk_categories, 
        add_dq_scores, apply_score_importance, compute_snapshot_drift
    )
except ImportError as e:
    print(f"필수 모듈 로드 실패: {e}")
    sys.exit(1)

# --- [2. 유틸리티 클래스: 반복 로직 처리] ---
class DQUtils:
    _FLOAT_ZERO_RE = re.compile(r'^[+-]?\d+\.0+$')

    @staticmethod
    def strip_decimal_zero(x):
        s = str(x)
        return s.split('.', 1)[0] if DQUtils._FLOAT_ZERO_RE.fullmatch(s) else s

    @staticmethod
    def get_pattern(value):
        """문자열 패턴 추출 (n:숫자, K:한글, A/a:영문 등)"""
        s = DQUtils.strip_decimal_zero(value)[:20]
        p = []
        for ch in s:
            if ch.isdigit(): p.append('n')
            elif '가' <= ch <= '힣': p.append('K')
            elif ch.isalpha(): p.append('A' if ch.isupper() else 'a')
            elif ch in '(){}[]-=. :@/': p.append(ch)
            else: p.append('s')
        res = "".join(p)
        return f"'{res}" if res.startswith('-') else res

# --- [3. 분석 엔진 클래스: 컬럼별 통계 계산] ---
class ColumnProfiler:
    def __init__(self, df, column):
        self.col = column
        self.series = df[column]
        self.non_null = self.series.dropna()
        self.str_vals = self.non_null.astype(str)
        self.count = len(df)

    def profile(self):
        if self.count == 0: return {}

        val_cnt = len(self.non_null)
        null_cnt = self.count - val_cnt
        unique_cnt = self.non_null.nunique()
        
        res = {
            'ColumnName': self.col,
            'DataType': str(self.series.dtype),
            'OracleType': Get_Oracle_Type(self.series, self.col),
            'PK': 1 if unique_cnt == self.count and self.count > 0 else 0,
            'RecordCnt': self.count,
            'ValueCnt': val_cnt,
            'Null(%)': round(null_cnt / self.count * 100, 2) if self.count > 0 else 0,
            'UniqueCnt': unique_cnt,
            'Unique(%)': round(unique_cnt / val_cnt * 100, 2) if val_cnt > 0 else 0,
        }

        if val_cnt > 0:
            lens = self.str_vals.str.len()
            top_counts = self.str_vals.value_counts().head(10)
            top10_list = top_counts.index.tolist()
            top10_json = json.dumps(top10_list, ensure_ascii=False)
            
            # 패턴 분석
            patterns = self.str_vals.map(DQUtils.get_pattern)
            pattern_stats_list = Counter(patterns).most_common()
            top_p = pattern_stats_list[0][0] if pattern_stats_list else ""
            format_cnt = len(pattern_stats_list)

            # --- [추가 컬럼 로직 시작] ---
            # 1. CompareLength: 길이의 고정 여부 (Min과 Max가 같으면 0, 다르면 1)
            res['CompareLength'] = 1 if lens.min() != lens.max() else 0

            # --- [핵심] 1, 2, 3순위 포맷별 상세 통계 생성 루프 ---
            for i in range(1, 4):
                suffix = "" if i == 1 else "_2" if i == 2 else "_3"
                # 다른 프로그램 규격에 맞춘 컬럼 접미사 (_1, _2, _3)
                col_suffix = f"_{i}" 
                
                if len(pattern_stats_list) >= i:
                    p_val, p_cnt = pattern_stats_list[i-1]
                    
                    # 해당 포맷 명칭 및 기본 통계
                    res[f'Format{suffix}'] = p_val
                    res[f'Format{suffix}Value'] = p_cnt
                    res[f'Format{suffix}(%)'] = round(p_cnt / val_cnt * 100, 2)

                    # 해당 포맷에 해당하는 실제 값들 필터링
                    fmt_vals = self.str_vals[patterns == p_val]
                    if not fmt_vals.empty:
                        res[f'FormatMin{col_suffix}'] = fmt_vals.min()[:50]
                        res[f'FormatMax{col_suffix}'] = fmt_vals.max()[:50]
                        # 중간값 추출
                        sorted_vals = fmt_vals.sort_values()
                        res[f'FormatMedian{col_suffix}'] = sorted_vals.iloc[len(sorted_vals) // 2]
                    else:
                        res[f'FormatMin{col_suffix}'] = ""
                        res[f'FormatMax{col_suffix}'] = ""
                        res[f'FormatMedian{col_suffix}'] = ""
                else:
                    # 데이터가 없는 경우 빈값 처리
                    res[f'Format{suffix}'] = ""
                    res[f'Format{suffix}Value'] = 0
                    res[f'Format{suffix}(%)'] = 0.0
                    res[f'FormatMin{col_suffix}'] = ""
                    res[f'FormatMax{col_suffix}'] = ""
                    res[f'FormatMedian{col_suffix}'] = ""

            # --- [핵심] Determine_Detail_Type용 인자 생성 ---
            # 1. format_stats: 함수 내부에서 사용하는 키들 매핑
            f_stats = {
                'FormatMedian': self.str_vals.sort_values().iloc[val_cnt // 2], # 중간값 문자열
                'FormatMode': self.str_vals.mode()[0] if not self.str_vals.mode().empty else "",
                'most_common_pattern': top_p,
                'pattern_type_cnt': format_cnt
            }
            # 2. total_stats: 플래그 및 시퀀스 판별용
            t_stats = {
                'min': self.str_vals.min(),
                'max': self.str_vals.max(),
                'mode': f_stats['FormatMode']
            }

            try:
                # 함수 호출 (8개 인자 순서 준수)
                dt_result = Determine_Detail_Type(
                    top_p,            # 1. pattern
                    format_cnt,       # 2. pattern_type_cnt
                    f_stats,          # 3. format_stats
                    t_stats,          # 4. total_stats
                    int(lens.max()),  # 5. max_length
                    int(unique_cnt),  # 6. unique_count
                    int(val_cnt),     # 7. non_null_count
                    top10_json        # 8. top10 (is_tel 함수가 json.loads를 수행함)
                )
                res['DetailDataType'] = dt_result if dt_result else ""
            except Exception as e:
                res['DetailDataType'] = "Error"
                print(f"{self.col} 타입 판별 중 오류: {e}")

            # --- 포맷 통계 (1st, 2nd, 3rd) ---
            res['FormatCnt'] = format_cnt
            for i in range(1, 4):
                suffix = "" if i == 1 else "2nd" if i == 2 else "3rd"
                if len(pattern_stats_list) >= i:
                    p_val, p_cnt = pattern_stats_list[i-1]
                    res[f'Format{suffix}'] = p_val
                    res[f'Format{suffix}Value'] = p_cnt
                    res[f'Format{suffix}(%)'] = round(p_cnt / val_cnt * 100, 2)
                else:
                    res[f'Format{suffix}'] = ""; res[f'Format{suffix}Value'] = 0; res[f'Format{suffix}(%)'] = 0

            # 나머지 통계 (Min/Max/Top10/Slicing)
            res.update({
                'LenMin': int(lens.min()), 'LenMax': int(lens.max()), 'LenAvg': round(lens.mean(), 1),
                'MinString': self.str_vals.min()[:50], 'MaxString': self.str_vals.max()[:50],
                'ModeString': f_stats['FormatMode'], 'ModeCnt': int(top_counts.iloc[0]) if not top_counts.empty else 0,
                'Top10': top10_json, 'Top10(%)': round(top_counts.sum() / val_cnt * 100, 2)
            })
            
            for n in [1, 2, 3]:
                res.update(self._get_edge_stats('First', n))
                res.update(self._get_edge_stats('Last', n))
        
        return res

    def _get_edge_stats(self, side, n):
        stats = {}
        func = (lambda x: x[:n]) if side == 'First' else (lambda x: x[-n:])
        top = self.str_vals.apply(func).value_counts().head(3)
        for i in range(3):
            val = str(top.index[i]) if i < len(top) else ""
            cnt = int(top.iloc[i]) if i < len(top) else 0
            stats[f'{side}{n}M{i+1}'] = val
            stats[f'{side}{n}Cnt{i+1}'] = cnt
        return stats
        
# --- [4. 메인 파이프라인 엔진] ---
class DQEngine:
    def __init__(self, config_dict):
        self.config = config_dict
        # output_path를 안전하게 설정
        root = config_dict.get('ROOT_PATH', '')
        out_dir = config_dict.get('directories', {}).get('output', 'DS_Output')
        self.output_path = Path(root) / out_dir
        self.sampling_rows = 10000
        self.file_stats_list = [] 

    def process_file(self, file_meta):
        """
        수정 사항: 분석 결과(column_results)와 파일 통계(file_stats)를 튜플로 반환합니다.
        """
        source_dir = file_meta.get('source') or file_meta.get('file_path') or file_meta.get('FilePath')
        code_type = file_meta.get('type') or file_meta.get('code_type') or file_meta.get('Master_Type')
        extension = file_meta.get('ext', '.csv').lower()
        if extension and not extension.startswith('.'):
            extension = '.' + extension

        if not source_dir:
            return [], [] # 빈 결과 반환

        target_files = []
        try:
            if os.path.isdir(source_dir):
                for f in os.listdir(source_dir):
                    if f.lower().endswith(extension):
                        target_files.append(os.path.join(source_dir, f))
            elif os.path.isfile(source_dir):
                target_files.append(source_dir)
        except Exception:
            return [], []

        all_column_results = []
        local_file_stats = [] # 이 프로세스 내에서 수집할 통계

        for f_path in target_files:
            try:
                file_size = os.path.getsize(f_path)
                if f_path.lower().endswith('.csv'):
                    df = pd.read_csv(f_path, dtype=str, low_memory=False, encoding='utf-8-sig')
                else:
                    df = pd.read_excel(f_path, dtype=str)

                record_cnt = len(df)
                column_cnt = len(df.columns)
                sampling_rows = min(record_cnt, self.sampling_rows)
                sample_df = df if record_cnt <= self.sampling_rows else df.sample(n=self.sampling_rows)
                
                # 파일 통계 수집
                local_file_stats.append({
                    'FilePath': f_path,
                    'FileName': os.path.basename(f_path),
                    'MasterType': code_type,
                    'FileSize': file_size,
                    'RecordCnt': record_cnt,
                    'ColumnCnt': column_cnt,
                    'SamplingRows': sampling_rows,
                    'Sampling(%)': round((sampling_rows / record_cnt * 100), 2) if record_cnt > 0 else 0,
                    'WorkDate': datetime.now().strftime('%Y-%m-%d')
                })

                for col in sample_df.columns:
                    profiler = ColumnProfiler(sample_df, col)
                    col_res = profiler.profile()
                    col_res.update({
                        'MasterType': code_type,
                        'FileName': os.path.basename(f_path),
                        'FilePath': f_path,
                        'TotalRecords': record_cnt
                    })
                    all_column_results.append(col_res)
                
            except Exception as e:
                print(f"에러 발생 ({os.path.basename(f_path)}): {e}")
                continue
        
        # 중요: 컬럼 분석 결과와 파일 통계를 모두 리턴함
        return all_column_results, local_file_stats

    def run(self, codelist):
        print(f"분석 시작 (CPU Core: {cpu_count()})")
        
        with Pool(cpu_count()) as pool:
            # results는 [(col_res1, stat_res1), (col_res2, stat_res2), ...] 형태가 됨
            combined_results = pool.map(self.process_file, codelist)
        
        # 1. 분석 결과 통합 (Flatten)
        flat_column_results = []
        all_file_stats = []
        
        for col_res, stat_res in combined_results:
            flat_column_results.extend(col_res)
            all_file_stats.extend(stat_res)
        
        if not flat_column_results:
            print("Error: 분석된 데이터가 없습니다.")
            return

        # 2. DQ 결과 저장
        if flat_column_results:
            final_df = pd.DataFrame(flat_column_results)
            final_df = add_dq_scores(final_df)
            final_df.insert(0, 'No', range(1, len(final_df) + 1))
            # 최종 final_df 의 컬럼순서가 No, FilePath, FileName, MasterType, ColumnName, DataType, OracleType, DetailDataType, PK, ValueCnt 이후는 순서
            column_order = ['No', 'FilePath', 'FileName', 'MasterType', 'ColumnName', 'DataType', 'OracleType', 'DetailDataType', 'PK', 'ValueCnt']
            
            # 지정된 컬럼들을 앞에 배치하고, 나머지 컬럼들은 원래 순서대로 유지
            existing_columns = final_df.columns.tolist()
            ordered_columns = []
            
            # 1. column_order에 있는 컬럼들을 순서대로 추가 (존재하는 경우만)
            for col in column_order:
                if col in existing_columns:
                    ordered_columns.append(col)
            
            # 2. 나머지 컬럼들을 원래 순서대로 추가
            for col in existing_columns:
                if col not in ordered_columns:
                    ordered_columns.append(col)
            
            # 컬럼 순서 적용
            final_df = final_df[ordered_columns]
            
            save_path = self.output_path / "FileFormat.csv"
            final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"DQ 결과 저장 완료: {save_path}")

        # 3. FileStats 저장
        if all_file_stats:
            stats_df = pd.DataFrame(all_file_stats)
            stats_df.insert(0, 'FileNo', range(1, len(stats_df) + 1))
            
            # 경로 설정 보강
            out_path = self.config.get('output_path') or str(self.output_path)
            stats_final_path = os.path.join(out_path, "FileStats.csv")
            
            stats_df.to_csv(stats_final_path, index=False, encoding="utf-8-sig")
            print(f"파일 단위 통계 저장 완료: {stats_final_path}")


# --- [5. 실행부] ---
if __name__ == "__main__":
    start_time = time.time()
    
    # 설정 로드
    main_config = Load_Yaml_File(DQConfig.get_path(DQConfig.YAML_RELATIVE_PATH))
    
    # 실행 대상 리스트 확보 (Excel에서 Y인 것만)
    meta_path = os.path.join(main_config['ROOT_PATH'], main_config['files']['codelist_meta'])
    codelist_df = pd.read_excel(meta_path)
    codelist_list = codelist_df[codelist_df['execution_flag'] == 'Y'].to_dict(orient='records')

    # 엔진 구동
    engine = DQEngine(main_config)
    engine.run(codelist_list)

    print(f"총 소요 시간: {time.time() - start_time:.2f}초")

