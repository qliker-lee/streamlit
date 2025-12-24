# -*- coding: utf-8 -*-
"""
DS_12_MasterRuleDataType.py
- Code File 에 대한 Rule & Data Type 프로파일링을 수행합니다.
"""

import argparse, ast, json, os, re
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ✅ QDQM 루트 디렉토리 추적 (DataSense/util/ 까지 들어왔을 때)
ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# ✅ YAML 파일 절대 경로 설정
YAML_PATH = ROOT_PATH / "DataSense" / "util" / "DS_Master.yaml"

from DataSense.util.io import Load_Yaml_File, Backup_File    
# ---------------- 유효성 함수(없으면 더미 대체) ----------------
try:
    from DataSense.util.dq_validate import (
        validate_date, validate_yearmonth, validate_latitude, validate_longitude,
        validate_YYMMDD, validate_tel, validate_cellphone, validate_address, validate_gender, validate_gender_en
    )
except Exception:
    def _false(*a, **k): return False
    validate_date = validate_yearmonth = validate_latitude = validate_longitude = _false
    validate_YYMMDD = validate_tel = validate_cellphone = validate_address = _false
    validate_gender = validate_gender_en = _false

# ---------------- IO ----------------
def read_csv_any(path: str) -> pd.DataFrame:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise FileNotFoundError(path)

def parse_jsonish_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip().strip('"').strip("'")
                for v in x if str(v).strip().lower() not in {"", "nan", "null", "none"}]
    s = str(x).strip()
    if not s: return []
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)):
            return [str(v).strip().strip('"').strip("'")
                    for v in obj if str(v).strip().lower() not in {"", "nan", "null", "none"}]
    except Exception:
        pass
    s = re.sub(r'^[\[\(\{]\s*|\s*[\]\)\}]$', '', s)
    tokens = re.split(r'[,\|;\/\n\r\t]+', s)
    return [t.strip().strip('"').strip("'")
            for t in tokens if t and t.strip().lower() not in {"", "nan", "null", "none"}]

def to_float(x, default=0.0):
    try:
        v = pd.to_numeric(x, errors="coerce")
        return float(v) if pd.notna(v) else default
    except Exception:
        return default

# ======================================================================
# Oracle Type Inference (name/digits heuristics)
# ======================================================================
CODEY_NAME_HINT = re.compile(r"(code|id|no|key)$", re.IGNORECASE)

def _safe_to_numeric(series):
    try:
        return pd.to_numeric(series, errors="coerce").dropna()
    except Exception:
        return pd.Series([], dtype=float)

def Get_Oracle_Type(series, column_name: str | None = None):
    s_all = series.dropna().astype(str).str.strip()
    if s_all.empty:
        return "NULL"

    name_is_codey = bool(column_name and CODEY_NAME_HINT.search(str(column_name)))

    date_like = s_all.str.fullmatch(
        r"\d{4}[-/.]?\d{2}[-/.]?\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?)?"
    ).mean()
    if date_like >= 0.98:
        return "DATE"

    num_like = s_all.str.fullmatch(r"[+-]?\d+(?:\.\d+)?").mean()
    has_leading_zero = s_all.str.match(r"^0\d+$").any()
    fixed_length = s_all.str.len().nunique(dropna=True) == 1

    if num_like >= 0.98 and not name_is_codey and not has_leading_zero and not fixed_length:
        nums = _safe_to_numeric(s_all)
        if nums.empty:
            maxlen = int(s_all.str.len().max())
            return f"VARCHAR2({min(maxlen, 4000)})"
        if (nums % 1 == 0).all():
            max_digits = nums.abs().astype("Int64").astype(str).str.len().max()
            return f"NUMBER({int(max_digits)})"
        else:
            parts = nums.abs().astype(str).str.split(".")
            int_digits = parts.str[0].str.len().astype(int).max()
            frac_digits = parts.str[1].str.len().fillna(0).astype(int).max()
            return f"NUMBER({int(int_digits + frac_digits)},{int(frac_digits)})"

    maxlen = int(s_all.str.len().max())
    return f"VARCHAR2({maxlen})" if maxlen <= 4000 else "CLOB"

# ======================================================================
# 파일별 컬럼 분석
# ======================================================================
def create_datatype_df(filepath, code_type, extension):
    """단일 파일의 컬럼별 Pandas/Oracle 타입 분석"""
    try:
        if extension in ("csv", ".csv"):
            df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
        elif extension in ("xlsx", ".xlsx"):
            df = pd.read_excel(filepath)
        elif extension in ("pkl", ".pkl"):
            df = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    except Exception as e:
        print(f"파일 로드 실패: {filepath} → {e}")
        return []

    summary = []
    for col in df.columns:
        pandas_dtype = str(df[col].dtype)
        try:
            oracle_type = Get_Oracle_Type(df[col], col)
        except Exception as e:
            oracle_type = f"ERROR({e})"

        summary.append({
            "FilePath": str(filepath).replace('\\', '/'),
            "FileName": os.path.basename(filepath),
            "MasterType": code_type,
            "ColumnName": col,
            "DataType": pandas_dtype,
            "OracleType": oracle_type,
        })

    return summary

# ======================================================================
# 전체 분석
# ======================================================================
def DataType_Analysis(config, source_dir_list):
    """모든 코드 파일에 대한 DataType 분석"""
    print("\n=== DataType 분석 시작 ===")

    base_path = Path(str(config["ROOT_PATH"]).rstrip("/\\"))
    output_subdir = str(config["directories"]["output"]).lstrip("/\\")
    output_dir = base_path / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    datatype_file = config["files"]["datatype"]
    # _ = Backup_File(str(output_dir), datatype_file, "csv")

    datatype_df = []

    for source_config in source_dir_list:
        source_subpath = str(source_config["source"]).lstrip("/\\")
        source_path = base_path / source_subpath

        file_pattern = f"*.{source_config['extension'].lstrip('.')}"
        files = list(source_path.glob(file_pattern))

        if not files:
            print(f"No files found in {source_path}")
            continue

        print(f"\n {source_config['type']} 코드 분석 중... (총 {len(files)}개 파일)")
        for file in files:
            try:
                datatype = create_datatype_df(file, source_config["type"], file.suffix)
                datatype_df.extend(datatype)
            except Exception as e:
                print(f"파일 처리 오류: {file.name} → {e}")

    if not datatype_df:
        print("처리된 데이터가 없습니다.")
        return 1

    result_df = pd.DataFrame(datatype_df)
    result_path = f"{base_path}/{datatype_file}.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print(f"결과 저장 완료: {result_path}")
    return 0
# ---------------- ValidData ----------------
def _read_valid_file(path: str, colname: str, strict_column: bool = True) -> list[str]:
    if not path: return []
    path = os.path.expanduser(os.path.expandvars(str(path).strip().strip('"').strip("'")))
    df = None
    for enc in ("utf-8-sig","cp949","utf-8"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        print(f"[Warning] 유효값 파일을 읽지 못했습니다: {path}")
        return []
    if strict_column and colname not in df.columns:
        print(f"[Warning] '{os.path.basename(path)}'에 '{colname}' 컬럼이 없습니다. 건너뜀.")
        return []
    series = df[colname] if colname in df.columns else df.select_dtypes(include="object").iloc[:,0]
    vals = (series.dropna().astype(str).map(lambda s: s.strip().strip('"').strip("'")))
    return [v for v in vals if v and v.lower() not in {"nan","null","none"}]

def build_valid_map(df_v: pd.DataFrame, *, base_dir: str | None = None,
                    case_sensitive: bool = True, strict_file_column: bool = True) -> dict[str, set]:
    def norm(v: str) -> str:
        v = (v or "").strip()
        return v if case_sensitive else v.lower()
    valid_map: dict[str, set] = {}
    for _, r in df_v.iterrows():
        name = str(r.get("valid_name","")).strip()
        if not name: continue
        list_or_file = str(r.get("ListOrFile","List")).strip()
        raw = r.get("valid_list","")
        values: list[str] = []
        if list_or_file.lower() == "list":
            values = parse_jsonish_list(raw)
        elif list_or_file.lower() == "file":
            path = str(raw or "").strip()
            if base_dir and path and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            values = _read_valid_file(path, colname=name, strict_column=strict_file_column)
        else:
            print(f"[Warning] 알 수 없는 ListOrFile='{list_or_file}' (valid_name={name})")
        values = [norm(v) for v in values if v]
        valid_map.setdefault(name, set()).update(values)
    return valid_map

# ---------------- RuleType helpers ----------------
FORMAT_MAX_VALUE = 4000
FORMAT_AVG_LENGTH = 100

def is_timestamp(pattern, pattern_cnt):
    return (pattern in ['nnnn-nn-nn nn:nn:nn','nnnn-nn-nn nn:nn:nn.nnnnnn','nnnn-nn-nn nn:nn:nn.']
            and int(pattern_cnt) == 1)

def is_time(pattern, pattern_cnt):
    return (pattern in ['nn:nn.n','nn:nn:nn','nn:nn:nn.nnnnnn','nn:nn:nn.']
            and int(pattern_cnt) == 1)

def is_datechar(pattern, median):
    return (pattern in ['nnnnnnnn','nnnn-nn-nn','nnnn/nn/nn','nnnnKnnKnnK','nnnn.nn.nn','nnnn. n. nn.']
            and validate_date(str(median)))

def is_yearmonth(pattern, median):
    return (pattern in ['nnnnnn','nnnn-nn','nnnn/nn','nnnn.nn','nnnnKnnK']
            and validate_yearmonth(str(median)))

def is_yymmdd(pattern, median):
    return (pattern in ['nnnnnn','nn-nn-nn','nn/nn/nn','nn.nn.nn','nn.nn.nn.']
            and validate_YYMMDD(str(median)))

def is_year(pattern, median, mode_string):
    if pattern == 'nnnn' and median:
        try:
            mode_val = float(mode_string)
            return 1990 < mode_val < 2999
        except Exception:
            return False
    return False

def is_latitude(pattern, median):
    return (pattern in ['nn.nnnn','nn.nnnnn','nn.nnnnnn','nn.nnnnnnn','nn.nnnnnnnn']
            and validate_latitude(median))

def is_longitude(pattern, median):
    return (pattern in ['nnn.nnnn','nnn.nnnnn','nnn.nnnnnn','nnn.nnnnnnn','nnn.nnnnnnnn']
            and validate_longitude(median))

def is_tel(pattern, top10, top_n: int = 10) -> bool: 
    def _ensure_list(x):
        if isinstance(x, (list, tuple, set)):
            return [str(v) for v in x]
        return parse_jsonish_list(x)  # 문자열이면 JSON/구분자 파싱

    tel_patterns = [
        'nnn-nnn-nnnn','nn-nnnn-nnnn','nn-nnn-nnnn','nnn-nnnn',
        'nnnn-nnnn','nnnnnnn','nnnnnnnn','nnnnnnnnnn'
    ]
    if pattern not in tel_patterns:
        return False

    values = _ensure_list(top10)
    top_values = [v for v in values if v != "__OTHER__"][:top_n]
    if not top_values:
        return False

    # 상위 값 중 전화번호로 판정되는 비율이 80% 이상이면 TEL
    valid_tel_count = sum(1 for v in top_values if validate_tel(v))
    return (valid_tel_count / len(top_values)) >= 0.9


def is_cellphone(pattern, median): return (pattern in ['nnn-nnnn-nnnn','nnnnnnnnnnn'] and validate_cellphone(median))
def is_car_number(pattern, pattern_cnt): return pattern in ['KKnnKnnnn','nnKnnnn','nnnKnnnn']
def is_company(pattern, pattern_cnt):    return (pattern in ['(K)KKKK','(K)KKKKK','(K)KKKKKK'] and int(pattern_cnt) > 5)
def is_email(pattern): return ('@' in pattern and 1 <= pattern.count('.') <= 2)
def is_url(pattern):   return ('://' in pattern and pattern.count('.') >= 1)
def is_address(pattern, median): return (len(pattern) >= 8 and pattern.count('K') >= 6 and pattern.count(' ') >= 2 and validate_address(median))

def is_flag(pattern, pattern_cnt, median, min_string, max_string):
    if (pattern == 'A' and min_string == 'N' and max_string == 'Y' and int(pattern_cnt) == 1): return 'YN_Flag'
    if (pattern == 'n' and min_string == '0' and max_string == '1' and int(pattern_cnt) == 1): return 'True_False_Flag'
    if (pattern in ['A','a'] and int(pattern_cnt) == 1): return 'Alpha_Flag'
    if (pattern == 'n' and int(pattern_cnt) == 1):       return 'Num_Flag'
    if (pattern == 'K' and int(pattern_cnt) == 1):       return 'Kor_Flag'
    if ((pattern == 'KKK') and int(pattern_cnt) < 6):    return 'KOR_NAME'
    return None

def is_text(pattern, max_length, pattern_cnt):
    return (int(max_length) > FORMAT_MAX_VALUE or len(pattern) > FORMAT_AVG_LENGTH or int(pattern_cnt) > 20)

def is_sequence(min_string, max_string, unique_count):
    try:
        if min_string not in (None,'') and max_string not in (None,''):
            a = float(min_string); b = float(max_string)
            if a.is_integer() and b.is_integer():
                ai = int(a); bi = int(b)
                expected = bi - ai + 1
                return expected > 0 and expected == int(unique_count)
    except Exception:
        pass
    return False

def safe_int(x, default=0):
    try:
        if pd.isna(x) or x == '': return default
        return int(float(x))
    except Exception:
        return default

def Determine_Rule_Type(r: pd.Series) -> str:
    try:
        pattern     = str(r.get('Format','') or '')
        pattern_cnt = safe_int(r.get('FormatCnt',0), 0)
        median      = str(r.get('FormatMedian','') or r.get('Median','') or r.get('MedianString','') or '')
        mode_string = str(r.get('FormatMode','') or r.get('ModeString','') or '')
        top10       = str(r.get('Top10','') or '')
        min_string  = str(r.get('FormatMin','') or r.get('MinString','') or '')
        max_string  = str(r.get('FormatMax','') or r.get('MaxString','') or '')
        max_length  = safe_int(r.get('LenMax',0), 0)
        unique_cnt  = safe_int(r.get('UniqueCnt',0), 0)
        value_cnt   = safe_int(r.get('ValueCnt',0), 0)
        has_alpha   = safe_int(r.get('HasAlpha',0), 0)
        has_num     = safe_int(r.get('HasNum',0), 0)
        has_kor     = safe_int(r.get('HasKor',0), 0)
        has_special = safe_int(r.get('HasSpecial',0), 0)
        unique_percent = float(r.get('Unique(%)',0))

        if max_length > FORMAT_MAX_VALUE: return 'CLOB'
        if len(pattern) == 0:            return 'NULL'

        if is_timestamp(pattern, pattern_cnt):      return 'TIMESTAMP'
        if is_time(pattern, pattern_cnt):           return 'TIME'
        if is_yymmdd(pattern, median):              return 'YYMMDD'
        if is_datechar(pattern, median):            return 'DATECHAR'
        if is_yearmonth(pattern, median):           return 'YEARMONTH'
        if is_year(pattern, median, mode_string):   return 'YEAR'

        if is_latitude(pattern, median):            return 'LATITUDE'
        if is_longitude(pattern, median):           return 'LONGITUDE'

        if is_tel(pattern, top10, 10) and unique_percent != 100 and has_alpha == 0 and has_kor == 0: return 'TEL'
        if is_cellphone(pattern, median) and has_alpha == 0 and has_kor == 0: return 'CELLPHONE'
        if is_car_number(pattern, pattern_cnt):     return 'CAR_NUMBER'
        if is_company(pattern, pattern_cnt):        return 'COMPANY'

        if is_email(pattern): return 'EMAIL'
        if is_url(pattern):   return 'URL'

        if pattern_cnt > 0 and pattern[:1] == 'K':
            if is_address(pattern, median):         return 'ADDRESS'
            if is_text(pattern, max_length, pattern_cnt): return 'Text'

        if pattern_cnt > 0:
            flag_type = is_flag(pattern, pattern_cnt, median, min_string, max_string)
            if flag_type: return flag_type
            if is_sequence(min_string, max_string, unique_cnt): return 'SEQUENCE'
            if unique_cnt == 1: return 'SINGLE VALUE'
        return ''
    except Exception as e:
        print(f"Error in Determine_Rule_Type: {e}")
        return ''

# ---------------- Env ----------------
def build_env(row: pd.Series, valid_map: dict, *, rule_type_auto: str = "") -> dict:
    env = {c: row[c] for c in row.index}

    # 숫자형 캐스팅
    for c in ("FormatCnt","FormatLength","HasBlank","LenCnt","LenMin","LenMax",
              "ValueCnt","SampleRows","UniqueCnt"):
        env[c] = to_float(row.get(c), 0.0)

    # Top10 계열은 list로
    for c in row.index:
        if str(c).endswith("Top10"):
            env[c] = parse_jsonish_list(row.get(c))

    # 주요 문자열들
    env["Format"]        = str(row.get("Format",""))
    env["MedianString"]  = str(row.get("FormatMedian","") or row.get("Median","") or row.get("MedianString","") or "")
    env["ModeString"]    = str(row.get("FormatMode","") or row.get("ModeString","") or "")
    env["MinString"]     = str(row.get("FormatMin","") or row.get("MinString","") or "")
    env["MaxString"]     = str(row.get("FormatMax","") or row.get("MaxString","") or "")
    env["Top10"]         = row.get("Top10")
    env["RuleTypeAuto"]  = rule_type_auto  # 규칙에서 바로 사용 가능

    # 유효값 집합
    for name, s in valid_map.items():
        env[name] = s
    return env

# ---------------- Safe eval ----------------
ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.Not, ast.Invert,
    ast.Compare, ast.Name, ast.Load, ast.Constant, ast.Num,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.USub, ast.UAdd, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Call, ast.Attribute,
)

# 기존 함수 대체
def _membership_ratio(left, right) -> float:
    # 컨테이너 대상: set/list/tuple
    if isinstance(right, (set, list, tuple)):
        rset = set(str(x) for x in right)
        if isinstance(left, (list, tuple, set)):
            L = [str(x) for x in left]
            return (sum(1 for x in L if x in rset) / len(L)) if L else 0.0
        return 1.0 if str(left) in rset else 0.0

    # 문자열 대상: 'left'가 'right'의 부분문자열이어야 함
    if isinstance(right, str):
        if isinstance(left, str):
            return 1.0 if left in right else 0.0
        if isinstance(left, (list, tuple, set)):
            L = [str(x) for x in left]
            return (sum(1 for x in L if x in right) / len(L)) if L else 0.0

    return 0.0


class SafeEval(ast.NodeVisitor):
    def __init__(self, env: dict, threshold: float, valid_map: dict, func_registry: dict):
        self.env = env
        self.threshold = threshold
        self.valid_map = valid_map
        self.funcs = func_registry
        self._ratios: list[float] = []

    def visit(self, node):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node): return self.visit(node.body)
    def visit_Constant(self, node):   return node.value
    def visit_Name(self, node):
        if node.id in self.env: return self.env[node.id]
        raise NameError(f"Unknown name: {node.id}")

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, (ast.Not, ast.Invert)): return not bool(v)
        if isinstance(node.op, ast.USub): return -float(v)
        if isinstance(node.op, ast.UAdd): return +float(v)
        raise ValueError("Bad unary")

    def visit_BinOp(self, node):
        L = self.visit(node.left); R = self.visit(node.right)
        if   isinstance(node.op, ast.BitAnd):  return bool(L) and bool(R)
        elif isinstance(node.op, ast.BitOr):   return bool(L) or bool(R)
        elif isinstance(node.op, ast.Add):     return L + R
        elif isinstance(node.op, ast.Sub):     return L - R
        elif isinstance(node.op, ast.Mult):    return L * R
        elif isinstance(node.op, ast.Div):     return L / R
        elif isinstance(node.op, ast.FloorDiv):return L // R
        elif isinstance(node.op, ast.Mod):     return L % R
        raise ValueError("Bad binop")

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not bool(self.visit(v)): return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if bool(self.visit(v)): return True
            return False
        raise ValueError("Bad boolop")

    def visit_Call(self, node):
        # --- len()/size() 허용 ---
        if isinstance(node.func, ast.Name) and node.func.id in {"len", "size"}:
            if len(node.args) != 1:
                raise ValueError("len(x)는 인자 1개만 허용합니다.")
            x = self.visit(node.args[0])
            if isinstance(x, (list, tuple, set, str)):
                return len(x)
            return 0

        # --- safe helper whitelist: 룰에서 직접 호출 허용 ---
        safe_fns = {
            "is_tel": is_tel,
            "is_datechar": is_datechar,
            "is_yearmonth": is_yearmonth,
            "is_yymmdd": is_yymmdd,
            "is_timestamp": is_timestamp,
            "is_time": is_time,
            "is_latitude": is_latitude,
            "is_longitude": is_longitude,
            "is_email": is_email,
            "is_url": is_url,
        }
        if isinstance(node.func, ast.Name) and node.func.id in safe_fns:
            fn = safe_fns[node.func.id]
            args = [self.visit(a) for a in node.args]
            try:
                return bool(fn(*args))
            except TypeError as e:
                # 인자 개수/타입 오류를 친절히 표시
                raise ValueError(f"{node.func.id} 호출 인자가 올바르지 않습니다: {e}")

        # --- 문자열 메서드 count:  "aaa".count("a")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "count":
            obj = self.visit(node.func.value)
            args = [self.visit(a) for a in node.args]
            if not isinstance(obj, str) or len(args) != 1 or not isinstance(args[0], str):
                raise ValueError("count()는 문자열.count('서브스트링')만 허용합니다.")
            return obj.count(args[0])

        # --- 함수형 count(x, sub)
        if isinstance(node.func, ast.Name) and node.func.id == "count":
            if len(node.args) != 2:
                raise ValueError("count(x, sub)는 2개의 인자만 허용합니다.")
            x   = self.visit(node.args[0])
            sub = self.visit(node.args[1])
            if not isinstance(sub, str):
                raise ValueError("count(x, sub)에서 sub는 문자열이어야 합니다.")
            if isinstance(x, str):
                return x.count(sub)
            if isinstance(x, (list, tuple, set)):
                return sum(1 for v in x if str(v) == sub)
            return 0

        raise ValueError("허용되지 않은 호출입니다.")


    def visit_Compare(self, node):
        left_val = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right_val = self.visit(comp)
            if isinstance(op, (ast.In, ast.NotIn)):
                if isinstance(right_val, str) and right_val in self.valid_map:
                    target = self.valid_map[right_val]
                else:
                    target = right_val
                ratio = _membership_ratio(left_val, target)
                self._ratios.append(ratio)
                ok = (ratio >= self.threshold)
                ok = ok if isinstance(op, ast.In) else (not ok)
            else:
                L, R = left_val, right_val
                try:
                    Lf = float(L); Rf = float(R)
                    if   isinstance(op, ast.Eq):    ok = (Lf == Rf)
                    elif isinstance(op, ast.NotEq): ok = (Lf != Rf)
                    elif isinstance(op, ast.Gt):    ok = (Lf >  Rf)
                    elif isinstance(op, ast.GtE):   ok = (Lf >= Rf)
                    elif isinstance(op, ast.Lt):    ok = (Lf <  Rf)
                    elif isinstance(op, ast.LtE):   ok = (Lf <= Rf)
                    else: raise ValueError("Bad compare")
                except Exception:
                    if   isinstance(op, ast.Eq):    ok = (str(L) == str(R))
                    elif isinstance(op, ast.NotEq): ok = (str(L) != str(R))
                    elif isinstance(op, ast.Gt):    ok = (str(L) >  str(R))
                    elif isinstance(op, ast.GtE):   ok = (str(L) >= str(R))
                    elif isinstance(op, ast.Lt):    ok = (str(L) <  str(R))
                    elif isinstance(op, ast.LtE):   ok = (str(L) <= str(R))
                    else: raise ValueError("Bad compare")
            if not ok: return False
            left_val = right_val
        return True

def eval_expr(expr: str, env: dict, valid_map: dict, threshold: float, func_registry: dict):
    txt = (expr or "").strip()
    if txt == "" or txt.lower() in {"nan","na","none"}:
        return True, []
    tree = ast.parse(txt, mode="eval")
    ev = SafeEval(env, threshold=threshold, valid_map=valid_map, func_registry=func_registry)
    ok = bool(ev.visit(tree))
    return ok, ev._ratios

# ---------------- RuleType 산출 ----------------
def apply_rules_and_type(dt: pd.DataFrame, ff: pd.DataFrame, rd: pd.DataFrame, vd: pd.DataFrame, threshold: float) -> pd.DataFrame:
    valid_map = build_valid_map(vd)

    # 함수 화이트리스트 레지스트리
    FUNC = {
        "is_datechar":  lambda fmt, med: is_datechar(str(fmt), str(med)),
        "is_yearmonth": lambda fmt, med: is_yearmonth(str(fmt), str(med)),
        "is_yymmdd":    lambda fmt, med: is_yymmdd(str(fmt), str(med)),
        "is_year":      lambda fmt, med, mode="": is_year(str(fmt), str(med), str(mode)),
        "is_latitude":  lambda fmt, med: is_latitude(str(fmt), str(med)),
        "is_longitude": lambda fmt, med: is_longitude(str(fmt), str(med)),
        "is_tel": lambda *args: is_tel(
            str(args[0]) if len(args) > 0 else "",
            args[1]      if len(args) > 1 else "",
            int(args[2]) if len(args) > 2 else 10
        ),
        "is_cellphone": lambda fmt, med: is_cellphone(str(fmt), str(med)),
        "is_car_number":lambda fmt, cnt=0: is_car_number(str(fmt), to_float(cnt,0)),
        "is_company":   lambda fmt, cnt=0: is_company(str(fmt), to_float(cnt,0)),
        "is_email":     lambda fmt: is_email(str(fmt)),
        "is_url":       lambda fmt: is_url(str(fmt)),
        "is_address":   lambda fmt, med: is_address(str(fmt), str(med)),
        # 대안: TypeIs("DATECHAR") 는 SafeEval 안에서 처리
    }

    out = ff.copy()
    matched_rules_col, matched_scores_list_col = [], []
    match_score_avg_col, max_score_col, rule_type_col = [], [], []

    for _, row in out.iterrows():
        # RuleType 먼저 산출하고 env에 넣어 규칙에서 사용 가능
        rule_type = Determine_Rule_Type(row)
        rule_type_col.append(rule_type)

        env = build_env(row, valid_map, rule_type_auto=rule_type)

        pairs = []
        for _, r in rd.iterrows():
            rn    = str(r.get("RuleName","") or "")
            rule1 = str(r.get("rule1","") or "")
            rule2 = str(r.get("rule2","") or "")

            # rule1
            try:
                ok1, _ = eval_expr(rule1, env, valid_map, threshold, FUNC)
            except Exception:
                ok1 = False

            # rule2 (비어있으면 통과)
            if rule2.strip() == "" or rule2.strip().lower() in {"nan","na","none"}:
                ok2, score = True, 1.0
            else:
                try:
                    ok2, ratios = eval_expr(rule2, env, valid_map, threshold, FUNC)
                    score = (sum(ratios)/len(ratios)) if ratios else 1.0
                except Exception:
                    ok2, score = False, 0.0

            if ok1 and ok2:
                pairs.append((rn, float(score)))

        if pairs:
            pairs.sort(key=lambda x: x[1], reverse=True)  # 점수 내림차순
            names  = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            matched_rules_col.append("; ".join(names))
            matched_scores_list_col.append("; ".join(f"{s:.4f}" for s in scores))
            match_score_avg_col.append(round(float(np.mean(scores)), 4))
            max_score_col.append(round(float(np.max(scores)), 4))
        else:
            matched_rules_col.append("")
            matched_scores_list_col.append("")
            match_score_avg_col.append("")
            max_score_col.append(0.0)

    out = out.assign(
        Rule             = matched_rules_col if any(matched_rules_col) else rule_type_col,
        RuleType         = rule_type_col,
        MatchedRule      = matched_rules_col,
        MatchedScoreList = matched_scores_list_col,
        MatchScoreAvg    = match_score_avg_col,
        MatchScoreMax    = max_score_col,
    ).sort_values(by=["MatchScoreMax","MatchScoreAvg"], ascending=[False, False]).reset_index(drop=True)


    final_columns = ["FilePath","FileName","ColumnName","MasterType","Rule","RuleType","MatchedRule","MatchedScoreList","MatchScoreAvg","MatchScoreMax", "ValueCnt"]
    out = out[final_columns].copy()
    out = pd.merge(dt, out, on=["FilePath","FileName","ColumnName", "MasterType"], how="left")
    return out

def MasterRuleDataType(config):
    """모든 코드 파일에 대한 Rule & DataType 분석"""
    print("\n=== Rule & DataType 분석 시작 ===")

    base_path = Path(str(config["ROOT_PATH"]).rstrip("/\\"))
    output_subdir = str(config["directories"]["output"]).lstrip("/\\")
    output_dir = base_path / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    ruldatatype_file = config["files"]["ruldatatype"]

    error_count = 0

    # --- 안전한 경로 결합 ---
    datatype_f = (base_path / config['files']['datatype']).with_suffix('.csv')
    fileformat_f = (base_path / config['files']['fileformat']).with_suffix('.csv')
    rule_definition_f = base_path / config['files']['rule_definition']
    rule_definition_validdata_f = base_path / config['files']['rule_definition_validdata']


    dt = read_csv_any(datatype_f)
    ff = read_csv_any(fileformat_f)
    rd = read_csv_any(rule_definition_f)
    if "Use_Flag" in rd.columns:
        rd = rd[rd["Use_Flag"] == "Y"].copy()
    vd = read_csv_any(rule_definition_validdata_f)

    res = apply_rules_and_type(dt, ff, rd, vd, threshold=0.30)

    out_path = (base_path / ruldatatype_file).with_suffix('.csv')
    # out_path = str(base_path) + ruldatatype_file + '.csv'
    res.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"결과 저장 완료: {out_path}")

    return error_count
# ---------------- main ----------------
def main():
    import time
    start_time = time.time()

    config = Load_Yaml_File(YAML_PATH)

    # DataType 분석
    codelist_meta_file = f"{config['ROOT_PATH']}/{config['files']['codelist_meta']}"
    codelist_df = pd.read_excel(codelist_meta_file)
    codelist_df = codelist_df[codelist_df["execution_flag"] == "Y"]
    codelist_list = codelist_df.to_dict(orient="records")

    error_count = DataType_Analysis(config, codelist_list)

    # MasterRuleDataType 분석
    error_count = MasterRuleDataType(config)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print("\n--------------------------------------")
    print(f"총 처리 시간: {processing_time:.2f}초")
    print("--------------------------------------")

if __name__ == "__main__":
    main()
