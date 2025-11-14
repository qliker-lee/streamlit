# -*- coding: utf-8 -*-
"""
DS_13_CodeMapping 
- 모든 핵심 메서드(Reference/Internal/Rule mapping, pivot, final)를 포함합니다.
"""

from __future__ import annotations
import os
import re
import sys
import time
import logging
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Iterable, Optional, Sequence, List, Set

import numpy as np
import pandas as pd

# ✅ QDQM 루트 디렉토리 추적 (DataSense/util/ 까지 들어왔을 때)
ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# ✅ YAML 파일 절대 경로 설정
YAML_PATH = ROOT_PATH / "DataSense" / "util" / "DS_Master.yaml"

from DataSense.util.io import Load_Yaml_File, Backup_File   

# 외부 유틸 (기존 프로젝트의 util 패키지)
from DataSense.util.dq_format import Expand_Format, Combine_Format
from DataSense.util.dq_validate import (
    init_reference_globals,
    validate_date, validate_yearmonth, validate_latitude, validate_longitude,
    validate_YYMMDD, validate_year, validate_tel, validate_cellphone,
    validate_url, validate_email, validate_kor_name, validate_address,
    validate_country_code, validate_gender, validate_gender_en, validate_car_number,
    validate_time, validate_timestamp,
)

# ---------------------- 전역 기본값 ----------------------
DEBUG_MODE = True
OUTPUT_FILE_NAME = 'CodeMapping'
OUTPUT_FILEFORMAT = 'FileFormatMapping'
OUTPUT_FILENUMERIC = 'FileNumericStats'

# ---------------------- Dataclasses ----------------------
@dataclass
class DirectoriesConfig:
    root_path: Path
    input_dir: Path
    output_dir: Path
    meta_dir: Path
    master_csv_dir: Path
    reference_source_dir: Path

@dataclass
class FilesConfig:
    fileformat: Path
    master_meta: Path
    ruldatatype: Path

# ---------------------- Helpers ----------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\u3000", " ")
    return " ".join(s.split())

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace('\ufeff', '').strip() for c in out.columns]
    return out

# ---------------------- Main Class ----------------------
class Initializing_Main_Class:
    def __init__(self, yaml_path: Optional[str] = None):
        self.logger = None
        self.config: Dict[str, Any] = {}
        self.files_config: Optional[FilesConfig] = None
        self.directories_config: Optional[DirectoriesConfig] = None
        self.loaded_data: Dict[str, pd.DataFrame] = {}

        self._setup_logger()
        yaml_path = Path(yaml_path or self._get_default_yaml_path())
        if not yaml_path.exists():
            raise FileNotFoundError(f"Yaml 파일을 찾을 수 없습니다: {yaml_path}")
        self.logger.info(f"Yaml File: {yaml_path}")

        self.config = self._load_yaml_config(yaml_path)
        
        # ROOT_PATH가 YAML에 없으면 자동으로 설정
        if 'ROOT_PATH' not in self.config or not self.config.get('ROOT_PATH'):
            self.config['ROOT_PATH'] = str(ROOT_PATH)
            self.logger.info(f"ROOT_PATH 자동 설정: {self.config['ROOT_PATH']}")
        
        self.directories_config = self._setup_directories_config()

        # 전역 참조 세트 초기화 (시도/성씨/ISO3/연월일 등)
        init_reference_globals(self.directories_config.root_path, strict_columns=True, verbose=DEBUG_MODE)

        self.files_config = self._setup_files_config()
        self.loaded_data = self._load_files()

    # ------------ 초기화/로딩 ------------
    @staticmethod
    def _get_default_yaml_path() -> Path:
        return Path(__file__).parent / YAML_PATH

    def _setup_logger(self) -> None:
        log_dir = Path('logs'); log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"codemapping_{datetime.now():%Y%m%d_%H%M%S}.log"
        level = logging.DEBUG if DEBUG_MODE else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _load_yaml_config(self, yaml_path: Path) -> Dict[str, Any]:
        import yaml
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(yaml_path, 'r', encoding='euc-kr') as f:
                return yaml.safe_load(f)

    def _setup_directories_config(self) -> DirectoriesConfig:
        root = Path(self.config['ROOT_PATH'])
        directories = self.config.get('directories', {})
        dc = DirectoriesConfig(
            root_path=root,
            input_dir=root / directories.get('input', 'DS_Input').strip('/'),
            output_dir=root / directories.get('output', 'DS_Output').strip('/'),
            master_csv_dir=root / directories.get('master_csv', '@Master/csv').strip('/'),
            meta_dir=root / directories.get('meta', 'DS_Meta').strip('/'),
            reference_source_dir=root / directories.get('reference_source_dir', '5_Reference_Code/Source').strip('/'),
        )
        dc.output_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for name in ('input_dir', 'master_csv_dir', 'meta_dir'):
            p = getattr(dc, name)
            if not p.exists():
                missing.append(f"{name}={p}")
        if missing:
            raise RuntimeError("필수 디렉토리 없음: " + ", ".join(missing))
        return dc

    def _setup_files_config(self) -> FilesConfig:
        root = Path(self.config['ROOT_PATH'])
        files = self.config.get('files', {})
        # allow user to specify with or without extension
        return FilesConfig(
            fileformat = Path(f"{root}/{files.get('fileformat', 'DS_Output/CodeFormat')}"),
            ruldatatype= Path(f"{root}/{files.get('ruldatatype','DS_Output/RuleDataType')}"),
            master_meta= Path(f"{root}/{files.get('master_meta','DS_Meta/Master_Meta')}"),
        )

    def _resolve_file(self, path: Path) -> Optional[Path]:
        """주어진 Path (파일 또는 파일-기본명) -> 실제 존재하는 파일 Path 반환"""
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_file():
            return path_obj
        # try with common extensions
        for ext in ('.csv', '.xlsx', '.pkl'):
            cand = path_obj.with_suffix(ext)
            if cand.exists():
                return cand
        # if directory provided, try to find one file
        if path_obj.exists() and path_obj.is_dir():
            for ext in ('.csv', '.xlsx'):
                found = next(path_obj.glob(f"*{ext}"), None)
                if found:
                    return found
        return None

    def _load_files(self) -> Dict[str, pd.DataFrame]:
        """fileformat, ruldatatype, master_meta 파일을 로드하여 self.loaded_data 반환"""
        files_to_load = {
            'fileformat': self.files_config.fileformat,
            'ruldatatype': self.files_config.ruldatatype,
            'master_meta': self.files_config.master_meta,
        }
        loaded: Dict[str, pd.DataFrame] = {}
        for name, raw in files_to_load.items():
            resolved = self._resolve_file(Path(raw))
            if not resolved:
                raise RuntimeError(f"{name} 파일이 존재하지 않습니다: {raw}")
            try:
                if resolved.suffix.lower() == '.csv':
                    df = pd.read_csv(resolved, dtype=str, low_memory=False)
                elif resolved.suffix.lower() == '.xlsx':
                    df = pd.read_excel(resolved, dtype=str)
                elif resolved.suffix.lower() == '.pkl':
                    df = pd.read_pickle(resolved)
                else:
                    raise RuntimeError(f"{name} 파일 형식 미지원: {resolved.suffix}")
                loaded[name] = _clean_headers(df)
                self.logger.info(f"{name} 로드 완료: {resolved}")
            except Exception as e:
                self.logger.error(f"{name} 파일 로드 실패: {resolved} -> {e}")
                raise
        return loaded

    # ------------------ (1) Reference 값 비교 ------------------
    def mapping_check(self, mapping_df: pd.DataFrame, sample: int = 10_000) -> pd.DataFrame:
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

        mapping_df = mapping_df.copy()
        rows: List[Dict[str, Any]] = []
        src_cache, master_cache = {}, {}

        for _, r in mapping_df.sort_values(by='FilePath').iterrows():
            fpath = str(r['FilePath']).strip()
            fname = str(r['FileName']).strip()
            col   = str(r['ColumnName']).strip()
            mtype = str(r['MasterType']).strip()
            mpath = str(r['MasterFilePath']).strip()
            mfile = str(r.get('MasterFile', "")).strip()   # 안전 처리
            rtype = str(r['ReferenceMasterType']).strip()
            mcol  = str(r['MasterColumn']).strip()

            comp_len_src = _to_int(r.get('CompareLength', 0), 0)
            comp_len_mst = _to_int(r.get('MasterCompareLength', 0), 0)

            if fpath not in src_cache:
                try:
                    src_cache[fpath] = _clean_headers(
                        pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False, dtype=str)
                    )
                except Exception as e:
                    print(f"[❌ 파일 읽기 오류] {fpath} → {e}")
                    continue
            if mpath not in master_cache:
                try:
                    master_cache[mpath] = _clean_headers(
                        pd.read_csv(mpath, encoding='utf-8-sig', low_memory=False, dtype=str)
                    )
                except Exception as e:
                    print(f"[❌ 마스터 읽기 오류] {mpath} → {e}")
                    continue

            df = src_cache[fpath]
            md = master_cache[mpath]
            if (col not in df.columns) or (mcol not in md.columns):
                print(f"[경고] 컬럼 없음: {col} / {mcol}")
                continue

            s_series = df[col]
            if len(s_series) > sample:
                s_series = s_series.sample(sample, random_state=42)

            s_vals = _clean_values(s_series, comp_len_src or comp_len_mst)
            m_vals = _clean_values(md[mcol], comp_len_mst or comp_len_src)

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

        # 🔹 필수 컬럼 보장
        required_cols = [
            "FilePath","FileName","ColumnName","MasterType",
            "MasterFilePath","MasterFile","ReferenceMasterType","MasterColumn",
            "CompareLength","CompareCount","SourceCount","MatchRate(%)"
        ]
        for c in required_cols:
            if c not in out.columns:
                if "Count" in c or "Rate" in c:
                    out[c] = 0
                else:
                    out[c] = ""

        if out.empty:
            return pd.DataFrame(columns=required_cols)

        return out.drop_duplicates().reset_index(drop=True)[required_cols]


    # ------------------ (2) 피벗(Left-compact) ------------------
    def mapping_pivot(self, df_merged: pd.DataFrame, valid_threshold: float = 10.0,
                      top_k: int = 3, drop_old_pivot_cols: bool = True) -> pd.DataFrame:
        """Left-compact pivot: 상위 top_k 후보를 CodeFilePath/CodeFile/CodeType/CodeColumn/Matched로 전개"""
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
        # pivot produced e.g. ('CodeFilePath', 1)
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

        # # map original pivot keys to desired names
        # rename_map = {
        #     'MasterFile': 'CodeFile',  # we used MasterFile as pivot value, rename to CodeFile
        #     'MasterColumn': 'CodeColumn',
        # }
        # for orig, new in rename_map.items():
        #     cols = [c for c in wide.columns if c.startswith(orig + "_")]
        #     if cols:
        #         wide.rename(columns={c: c.replace(orig + "_", new + "_") for c in cols}, inplace=True)

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
        output_dir = self.directories_config.output_dir

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
        mapping_df = mapping_df[mapping_df['MatchRate(%)'] > 1]
        mapping_df = mapping_df.sort_values(by=['FilePath','ColumnName','MatchRate(%)'], ascending=[True,True,False])
        return mapping_df

    def internal_mapping(self, fileformat_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Internal Code Mapping 시작")
        output_dir = self.directories_config.output_dir

        expand_df = Expand_Format(fileformat_df)
        expand_df['Format(%)'] = pd.to_numeric(expand_df.get('Format(%)', 0), errors='coerce').fillna(0)
        expand_df = expand_df.loc[expand_df['Format(%)'] > 10].copy()
        source_df = expand_df.loc[expand_df['MasterType'] != 'Reference'].copy()
        combine_df = Combine_Format(source_df, source_df)
        combine_df = combine_df[combine_df.get('Final_Flag', 0) == 1].copy()
        if combine_df.empty:
            return pd.DataFrame()
        mapping_df = self.mapping_check(combine_df)
        mapping_df = mapping_df[mapping_df['MatchRate(%)'] > 1]
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
        concat_df = concat_df[concat_df['Matched(%)'] > 20]
        concat_df = concat_df.sort_values(by=['FilePath','FileName','ColumnName','MasterType','Matched(%)'],
                                          ascending=[True,True,True,True,False])
        return concat_df

    def final_mapping(self, fileformat_df: pd.DataFrame, pivoted_df: pd.DataFrame) -> pd.DataFrame:
        """fileformat_df와 ruldatatype(preset)과 pivoted_df를 합쳐 최종 산출"""
        self.logger.info("최종 매핑 파일을 생성합니다.")
        df_rule = self.loaded_data.get('ruldatatype', pd.DataFrame()).copy()
        if df_rule.empty:
            self.logger.debug("ruldatatype 비어있음 -> 룰 반영 스킵")
        rule_required_cols = ["FilePath","FileName","ColumnName","MasterType", "Rule","MatchedScoreList"]
        # safe: fill missing rule cols if necessary
        for c in rule_required_cols:
            if c not in df_rule.columns:
                df_rule[c] = ""

        df_rule = df_rule[rule_required_cols].copy()
        # pivoted_df may be empty -> create empty with expected columns
        pivot_cols = [
            'FilePath','FileName','ColumnName','MasterType',
            'CodeColumn_1','CodeFile_1','CodeFilePath_1','CodeType_1','Matched_1','Matched(%)_1',
            'CodeColumn_2','CodeFile_2','CodeFilePath_2','CodeType_2','Matched_2','Matched(%)_2'
        ]
        if pivoted_df is None or pivoted_df.empty:
            pivoted_df = pd.DataFrame(columns=pivot_cols)
        else:
            # ensure all pivot cols exist
            for c in pivot_cols:
                if c not in pivoted_df.columns:
                    pivoted_df[c] = ""

        # merge
        df = pd.merge(fileformat_df, df_rule, on=['FilePath','FileName','ColumnName','MasterType'], how='left', suffixes=("","_rule"))
        df = pd.merge(df, pivoted_df, on=['FilePath','FileName','ColumnName','MasterType'], how='left', suffixes=("","_pivot"))

        # 룰 매핑 반영: CodeColumn_1 비어있고 Rule이 있으면 반영
        df['Rule'] = df.get('Rule', "").fillna("").astype(str).str.strip()
        df['CodeColumn_1'] = df.get('CodeColumn_1', "").fillna("").astype(str)
        mask = (df['CodeColumn_1'].str.strip() == "") & (df['Rule'] != "")
        if mask.any():
            df.loc[mask, 'CodeColumn_1'] = df.loc[mask, 'Rule']
            df.loc[mask, 'CodeType_1'] = 'Rule'
            df.loc[mask, 'CodeFile_1'] = 'Rule'
            df.loc[mask, 'CodeFilePath_1'] = 'Rule'
            # ValueCnt might not exist -> safe
            if 'ValueCnt' in df.columns:
                df.loc[mask, 'Matched_1'] = pd.to_numeric(df.loc[mask, 'ValueCnt'], errors='coerce').fillna(0).astype(int)
            else:
                df.loc[mask, 'Matched_1'] = 0
            df.loc[mask, 'Matched(%)_1'] = 100
            df.loc[mask, 'CodeCheck'] = 'Y'

        # PK -> FK mapping (if PK column present in fileformat_df)
        if 'PK' in fileformat_df.columns:
            pk_numeric = pd.to_numeric(fileformat_df['PK'], errors='coerce').fillna(0).astype(int)
            mask_pk = pk_numeric == 1
            tmp_df = fileformat_df.loc[mask_pk, ['FilePath','ColumnName']].copy()
            tmp_df = tmp_df.rename(columns={'FilePath':'CodeFilePath_1','ColumnName':'CodeColumn_1'})
            tmp_df['FK'] = 'FK'
            df = pd.merge(df, tmp_df, on=['CodeFilePath_1','CodeColumn_1'], how='left')

        return df

    # ------------------ (6) 파이프라인 ------------------
    def process_files_mapping(self) -> bool:
        output_dir = self.directories_config.output_dir
        fileformat_df = self.loaded_data.get('fileformat', pd.DataFrame()).copy()
        ruldatatype_df = self.loaded_data.get('ruldatatype', pd.DataFrame()).copy()

        # 1) Reference
        reference_df = self.reference_mapping(fileformat_df)
        if DEBUG_MODE and reference_df is not None and not reference_df.empty:
            p = os.path.join(output_dir, OUTPUT_FILE_NAME + '_3rd_ref_mapping.csv')
            reference_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"reference mapping : {p} 저장")

        # 2) Rule
        rule_df = self.rule_mapping(fileformat_df, ruldatatype_df)
        if DEBUG_MODE and rule_df is not None and not rule_df.empty:
            p = os.path.join(output_dir, OUTPUT_FILE_NAME + '_4th_rule_mapping.csv')
            rule_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"rule mapping : {p} 저장")

        # 3) Numeric stats
        numeric_df = self.numeric_column_statistics(fileformat_df)
        if DEBUG_MODE and numeric_df is not None and not numeric_df.empty:
            p = os.path.join(output_dir, OUTPUT_FILENUMERIC + '.csv')
            numeric_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"numeric stats : {p} 저장")

        # 4) Internal
        internal_df = self.internal_mapping(fileformat_df)
        if DEBUG_MODE and internal_df is not None and not internal_df.empty:
            p = os.path.join(output_dir, OUTPUT_FILE_NAME + '_7th_int_mapping.csv')
            internal_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"internal mapping : {p} 저장")

        # 5) concat + pivot + final
        concat_df = self.mapping_concat(reference_df, internal_df, rule_df)
        if DEBUG_MODE and concat_df is not None and not concat_df.empty:
            p = os.path.join(output_dir, OUTPUT_FILE_NAME + '_8th_concat.csv')
            concat_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"concat_df : {p} 저장")

        pivoted_df = self.mapping_pivot(concat_df)
        if DEBUG_MODE and pivoted_df is not None and not pivoted_df.empty:
            p = os.path.join(output_dir, OUTPUT_FILE_NAME + '_9th_pivoted.csv')
            pivoted_df.to_csv(p, index=False, encoding='utf-8-sig')
            self.logger.info(f"pivoted_df : {p} 저장")

        final_df = self.final_mapping(fileformat_df, pivoted_df)
        final_path = os.path.join(output_dir, OUTPUT_FILE_NAME + '.csv')
        try:
            final_df.to_csv(final_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"최종 : {final_path} 저장")
            return True
        except Exception as e:
            self.logger.error(f"최종 파일 저장 실패: {e}")
            return False

# ---------------------- main ----------------------
def main():
    start = time.time()
    try:
        processor = Initializing_Main_Class()
        ok = processor.process_files_mapping()
        print("Success : Reference/Internal/Rule Mapping 완료" if ok else "Fail : 처리 실패")
        print("="*50)
        print(f"총 처리 시간: {time.time()-start:.2f}초")
        print("="*50)
    except Exception as e:
        print(f"Master Mapping 프로그램 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
