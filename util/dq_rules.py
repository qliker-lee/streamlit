# dq_rules.py
import re
import fnmatch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# ---------- 유틸 ----------
def _to_pct(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).strip().replace('%','')
    if s == '' or s.lower() == 'nan':
        return 0.0
    s = re.sub(r"[^0-9.\-]", "", s)
    try: return float(s)
    except: return 0.0

def _num_or_none(x):
    try:
        v = float(x)
        return v
    except:
        return None

def _mask_value(v: Any, keep=2) -> str:
    """PII 마스킹: 앞뒤 keep글자만 남기고 나머지는 * 처리"""
    s = str(v)
    if len(s) <= keep * 2:
        return "*" * len(s)
    return s[:keep] + ("*" * (len(s) - keep*2)) + s[-keep:]

# ---------- 계약(YAML dict) 로드/정규화 ----------
def load_yaml_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    - contract는 이미 dict로 들어온다고 가정(호출부에서 yaml.safe_load 완료)
    - 키 유효성/기본값 보강
    """
    out = {"data_contract": {"defaults": {}, "columns": []}}
    dc = contract.get("data_contract", {})
    defaults = dc.get("defaults", {})
    cols = dc.get("columns", [])

    out["data_contract"]["defaults"] = {
        "severity": defaults.get("severity", "medium"),
        "null": defaults.get("null", {}),          # {"max_pct": 5}
        "length": defaults.get("length", {}),      # {"min":0, "max": 200}
        "pattern": defaults.get("pattern", {}),    # {"regex": r"...", "fixed_length": 5, "allow_leading_zero": True}
        "numeric": defaults.get("numeric", {}),    # {"min":0, "max":100}
        "domain": defaults.get("domain", {}),      # {"values":[...]}
    }

    norm_cols = []
    for item in cols:
        norm_cols.append({
            "file": item.get("file", "*"),
            "column": item.get("column", "*"),
            "severity": item.get("severity", out["data_contract"]["defaults"]["severity"]),
            "rules": item.get("rules", {})
        })
    out["data_contract"]["columns"] = norm_cols
    return out

# ---------- 매칭(파일/컬럼) ----------
def _iter_targets(scored_df: pd.DataFrame, file_pat: str, col_pat: str):
    """scored_df에서 FileName/ColumnName을 glob 패턴으로 매칭"""
    for _, r in scored_df[["FileName", "ColumnName"]].drop_duplicates().iterrows():
        if fnmatch.fnmatch(str(r["FileName"]), file_pat) and fnmatch.fnmatch(str(r["ColumnName"]), col_pat):
            yield r["FileName"], r["ColumnName"]

# ---------- 규칙 검증 ----------
def _eval_null_rule(row: pd.Series, rule: Dict[str, Any]) -> Tuple[bool, float, str]:
    max_pct = _to_pct(rule.get("max_pct", None))
    if max_pct <= 0:
        return False, 0.0, ""
    cur = _to_pct(row.get("Null(%)", 0))
    if cur > max_pct:
        # 초과분을 실패율로 가정(안정적 대안: 단순 100%도 가능)
        over = max(0.0, cur - max_pct)
        return True, min(100.0, over), f"NULL {cur:.2f}% > {max_pct:.2f}%"
    return False, 0.0, ""

def _eval_length_rule(row: pd.Series, rule: Dict[str, Any]) -> Tuple[bool, float, str]:
    min_len = _num_or_none(rule.get("min", None))
    max_len = _num_or_none(rule.get("max", None))
    cur_min = _num_or_none(row.get("LenMin", None))
    cur_max = _num_or_none(row.get("LenMax", None))
    msg = []
    fail = False
    if min_len is not None and cur_min is not None and cur_min < min_len:
        fail = True; msg.append(f"LenMin {cur_min} < {min_len}")
    if max_len is not None and cur_max is not None and cur_max > max_len:
        fail = True; msg.append(f"LenMax {cur_max} > {max_len}")
    # 길이 위반은 비율 추정이 어려우므로 보수적으로 100 처리(필요 시 샘플 기반으로 보강)
    return (fail, 100.0 if fail else 0.0, "; ".join(msg))

def _eval_numeric_rule(row: pd.Series, rule: Dict[str, Any]) -> Tuple[bool, float, str]:
    min_v = _num_or_none(rule.get("min", None))
    max_v = _num_or_none(rule.get("max", None))
    cur_min = _num_or_none(row.get("MinString", None))
    cur_max = _num_or_none(row.get("MaxString", None))
    if cur_min is None or cur_max is None:
        return False, 0.0, ""  # 요약값으로 판단 불가 → 패스(추후 sample_df로 세분화 가능)
    msg = []
    fail = False
    if min_v is not None and cur_min < min_v:
        fail = True; msg.append(f"Min {cur_min} < {min_v}")
    if max_v is not None and cur_max > max_v:
        fail = True; msg.append(f"Max {cur_max} > {max_v}")
    return (fail, 100.0 if fail else 0.0, "; ".join(msg))

def _eval_pattern_rule(row: pd.Series, rule: Dict[str, Any], sample_df: pd.DataFrame | None,
                       file_name: str, col_name: str) -> Tuple[bool, float, str]:
    """
    sample_df가 있으면 실제 regex 매칭률로 계산, 없으면
    - fixed_length만 있으면 LenMin/LenMax로 대략 판단
    - 둘 다 없으면 '평가불가'로 통과 처리
    """
    regex = rule.get("regex", None)
    fixed_len = rule.get("fixed_length", None)
    allow_leading_zero = bool(rule.get("allow_leading_zero", False))

    # 샘플 기반 평가
    if sample_df is not None and not sample_df.empty and regex:
        sub = sample_df[(sample_df["FileName"] == file_name) & (sample_df["ColumnName"] == col_name)]
        if not sub.empty:
            pat = re.compile(regex)
            vals = sub["Value"].astype(str)
            total = len(vals)
            ok = vals.apply(lambda s: bool(pat.fullmatch(s))).sum()
            fail_rate = 100.0 * (1.0 - ok / total)
            return (fail_rate > 0.0, float(fail_rate), f"Regex({regex}) fail {fail_rate:.2f}% on samples({total})")

    # 샘플 없고 fixed_length만 제공된 경우: 길이로 근사
    if fixed_len is not None:
        lmin = _num_or_none(row.get("LenMin", None))
        lmax = _num_or_none(row.get("LenMax", None))
        if lmin is not None and lmax is not None and (lmin != fixed_len or lmax != fixed_len):
            return True, 100.0, f"Fixed length {fixed_len} violated: LenMin={lmin}, LenMax={lmax}"
        # allow_leading_zero는 여기서 별도 위반 여부 추정 어려움 → 샘플 있을 때에만 판단
        return False, 0.0, ""

    # 둘 다 없으면 평가 보류
    return False, 0.0, ""

def _eval_domain_rule(row: pd.Series, rule: Dict[str, Any], sample_df: pd.DataFrame | None,
                      file_name: str, col_name: str) -> Tuple[bool, float, str]:
    """
    domain: {"values": [...]}
    - sample_df가 있으면 샘플 값으로 허용 도메인 위반율 계산
    - 없으면 평가 보류(통과 처리)
    """
    vals = rule.get("values", None)
    if not vals:
        return False, 0.0, ""
    allowed = set(map(str, vals))
    if sample_df is None or sample_df.empty:
        return False, 0.0, ""  # 샘플 없으면 판단 보류
    sub = sample_df[(sample_df["FileName"] == file_name) & (sample_df["ColumnName"] == col_name)]
    if sub.empty:
        return False, 0.0, ""
    s = sub["Value"].astype(str)
    total = len(s)
    ok = s.isin(allowed).sum()
    fail_rate = 100.0 * (1.0 - ok / total)
    return (fail_rate > 0.0, float(fail_rate), f"Out-of-domain {fail_rate:.2f}% on samples({total})")

def validate_with_yaml_contract(scored_df: pd.DataFrame,
                                contract: Dict[str, Any],
                                sample_df: pd.DataFrame | None = None
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    반환: (rule_report_df, scored_df2)
    - rule_report_df: 컬럼별 규칙 위반 내역
    - scored_df2: scored_df에 'RuleFail(%)' 컬럼 추가/업데이트
    """
    ruleset = contract.get("data_contract", {})
    defaults = ruleset.get("defaults", {})
    col_rules = ruleset.get("columns", [])

    rows = []
    # 컬럼별 누적 실패율(최대값 채택)
    agg_fail = {}

    for item in col_rules:
        fpat = item.get("file", "*")
        cpat = item.get("column", "*")
        severity = item.get("severity", defaults.get("severity", "medium"))
        spec = item.get("rules", {})

        for fname, cname in _iter_targets(scored_df, fpat, cpat):
            row = scored_df[(scored_df["FileName"] == fname) & (scored_df["ColumnName"] == cname)]
            if row.empty:
                continue
            r = row.iloc[0]
            # 규칙들 평가
            checks = []

            # NULL
            if "null" in spec or "null" in defaults:
                rule = {**defaults.get("null", {}), **spec.get("null", {})}
                fail, rate, msg = _eval_null_rule(r, rule)
                checks.append(("NULL", fail, rate, msg))

            # LENGTH
            if "length" in spec or "length" in defaults:
                rule = {**defaults.get("length", {}), **spec.get("length", {})}
                if rule:  # 비어있으면 무시
                    fail, rate, msg = _eval_length_rule(r, rule)
                    checks.append(("LENGTH", fail, rate, msg))

            # NUMERIC RANGE
            if "numeric" in spec or "numeric" in defaults:
                rule = {**defaults.get("numeric", {}), **spec.get("numeric", {})}
                if rule:
                    fail, rate, msg = _eval_numeric_rule(r, rule)
                    checks.append(("NUMERIC", fail, rate, msg))

            # PATTERN
            if "pattern" in spec or "pattern" in defaults:
                rule = {**defaults.get("pattern", {}), **spec.get("pattern", {})}
                if rule:
                    fail, rate, msg = _eval_pattern_rule(r, rule, sample_df, fname, cname)
                    checks.append(("PATTERN", fail, rate, msg))

            # DOMAIN
            if "domain" in spec or "domain" in defaults:
                rule = {**defaults.get("domain", {}), **spec.get("domain", {})}
                if rule:
                    fail, rate, msg = _eval_domain_rule(r, rule, sample_df, fname, cname)
                    checks.append(("DOMAIN", fail, rate, msg))

            # 보고서 행 생성
            for rule_name, fail, rate, msg in checks:
                if fail:
                    rows.append({
                        "FileName": fname,
                        "ColumnName": cname,
                        "RuleName": rule_name,
                        "Severity": severity,
                        "FailRate(%)": round(rate, 2),
                        "Message": msg
                    })
                    # 컬럼별 RuleFail(%)는 최대값으로 집계
                    key = (fname, cname)
                    agg_fail[key] = max(agg_fail.get(key, 0.0), rate)

    rule_report_df = pd.DataFrame(rows)
    # scored_df에 RuleFail(%) 합치기
    scored_df2 = scored_df.copy()
    if agg_fail:
        map_df = pd.DataFrame(
            [(k[0], k[1], v) for k, v in agg_fail.items()],
            columns=["FileName", "ColumnName", "RuleFail(%)"]
        )
        scored_df2 = scored_df2.merge(map_df, on=["FileName","ColumnName"], how="left")
        scored_df2["RuleFail(%)"] = scored_df2["RuleFail(%)"].fillna(0.0)
    else:
        if "RuleFail(%)" not in scored_df2.columns:
            scored_df2["RuleFail(%)"] = 0.0

    return rule_report_df, scored_df2

# ---------- 오류 샘플(마스킹) ----------
def build_error_samples(rule_report_df: pd.DataFrame,
                        sample_df: pd.DataFrame,
                        out_csv: str,
                        max_samples_per_col: int = 50) -> pd.DataFrame:
    """
    입력 sample_df 스키마: [FileName, ColumnName, Value]
    규칙 상세 위반 판별은 최소화하고, 컬럼별로 샘플만 선별/마스킹해서 저장
    """
    if sample_df is None or sample_df.empty or rule_report_df is None or rule_report_df.empty:
        return pd.DataFrame(columns=["FileName","ColumnName","MaskedValue"])

    targets = rule_report_df[["FileName","ColumnName"]].drop_duplicates()
    out_rows = []
    for _, t in targets.iterrows():
        sub = sample_df[(sample_df["FileName"] == t["FileName"]) & (sample_df["ColumnName"] == t["ColumnName"])]
        if sub.empty: 
            continue
        # 랜덤/상위 샘플
        take = min(len(sub), max_samples_per_col)
        pick = sub.sample(n=take, random_state=42)
        for v in pick["Value"].tolist():
            out_rows.append({
                "FileName": t["FileName"],
                "ColumnName": t["ColumnName"],
                "MaskedValue": _mask_value(v)
            })

    err_df = pd.DataFrame(out_rows)
    if not err_df.empty:
        err_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return err_df

# ---------- Autofix 제안 ----------
def suggest_autofix(rule_report_df: pd.DataFrame,
                    contract: Dict[str, Any]) -> pd.DataFrame:
    """
    간단한 제안:
    - PATTERN.fixed_length + allow_leading_zero → zfill(N)
    - NULL → 기본값 치환
    - LENGTH.max → 우측 컷/트리밍
    """
    rows = []
    if rule_report_df is None or rule_report_df.empty:
        return pd.DataFrame(columns=["FileName","ColumnName","Suggestion"])

    # 컬럼별 규칙 수준 제안
    cdict = {}
    for item in contract.get("data_contract", {}).get("columns", []):
        cdict[(item.get("file","*"), item.get("column","*"))] = item.get("rules", {})

    # 규칙명 기반 간단 매핑
    for _, r in rule_report_df.iterrows():
        fname, cname, rname = r["FileName"], r["ColumnName"], r["RuleName"]
        suggestions = []
        # 패턴 길이 제약
        if rname == "PATTERN":
            # 해당 컬럼에 매칭되는 룰 찾기
            for (fp, cp), rules in cdict.items():
                if fnmatch.fnmatch(fname, fp) and fnmatch.fnmatch(cname, cp):
                    pat = rules.get("pattern", {})
                    flen = pat.get("fixed_length", None)
                    if flen:
                        if pat.get("allow_leading_zero", False):
                            suggestions.append(f"Pad with leading zeros to length {flen} (e.g., value.zfill({flen}))")
                        else:
                            suggestions.append(f"Enforce fixed length {flen}: truncate/left-pad as per business rule")
        # NULL
        if rname == "NULL":
            suggestions.append("Fill NULLs with business default (e.g., 'Unknown', 0, or recent date)")
        # LENGTH
        if rname == "LENGTH":
            suggestions.append("Trim whitespaces; if too long, cut to max allowed; if too short, left-pad or default")
        # NUMERIC
        if rname == "NUMERIC":
            suggestions.append("Clip to allowed numeric range or mark as outlier")

        if suggestions:
            rows.append({
                "FileName": fname,
                "ColumnName": cname,
                "Suggestion": " | ".join(suggestions)
            })

    return pd.DataFrame(rows).drop_duplicates()
