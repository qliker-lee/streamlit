# function/erd_from_mapping.py
# -*- coding: utf-8 -*-
"""
Code Relationship Diagram 생성 (CodeMapping 결과 기반, 단일 DataFrame 입력)
- 필수 컬럼: FileName, ColumnName, FK('FK'), CodeColumn_1, CodeType_1
- 선택 컬럼: PK(1/True/'Y'), CodeFile_1(참조 대상 테이블)
- 노드 색/필터는 CodeFile_1(타겟 테이블)의 CodeType_1을 기준으로 동작
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

# Graphviz import 시도
GRAPHVIZ_AVAILABLE = False
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, Exception) as e:
    GRAPHVIZ_AVAILABLE = False
    # Digraph를 더미 객체로 대체
    class Digraph:
        def __init__(self, *args, **kwargs):
            pass
        def attr(self, *args, **kwargs):
            pass
        def node(self, *args, **kwargs):
            pass
        def edge(self, *args, **kwargs):
            pass
        def render(self, *args, **kwargs):
            raise FileNotFoundError("Graphviz가 설치되지 않았습니다.")

# ------------------------------
# 유틸
# ------------------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def _to_set(x: pd.Series) -> Set[str]:
    return set(str(v).strip() for v in x.dropna().astype(str) if str(v).strip() != "")

def _norm_codetype(x: str) -> str:
    s = str(x or "").strip().upper()
    alias = {
        "MASTER": "MASTER",
        "OP": "OPERATION", "OPER": "OPERATION",
        "REF": "REFERENCE",
        "ATTR": "ATTRIBUTE",
        "VALID": "VALIDATE", "VAL": "VALIDATE",
        "RULES": "RULE",
        "COMMONS": "COMMON",
    }
    return alias.get(s, s) if s else "OTHER"

# 테이블 대표 타입 선택 우선순위(타겟/소스 모두 공통 사용)
_NODE_TYPE_PRIORITY = ["MASTER", "OPERATION", "REFERENCE", "VALIDATE", "RULE", "ATTRIBUTE", "COMMON", "OTHER"]

# ------------------------------
# PK 맵
# ------------------------------
def _build_pk_map(df_pk_source: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], bool]]:
    req = {"FileName", "ColumnName", "PK"}
    if not req.issubset(df_pk_source.columns):
        return {}, {}
    df = _norm_cols(df_pk_source)
    flag = (
        df["PK"].astype(str).str.upper().isin(["1", "Y", "TRUE"])
        | (pd.to_numeric(df["PK"], errors="coerce").fillna(0) == 1)
    )
    df_pk = df.loc[flag, ["FileName", "ColumnName"]].dropna().copy()

    pk_map: Dict[str, Set[str]] = {}
    pk_lookup: Dict[Tuple[str, str], bool] = {}
    for tname, g in df_pk.groupby("FileName"):
        cols = _to_set(g["ColumnName"])
        if cols:
            pk_map[tname] = cols
            for c in cols:
                pk_lookup[(tname, c)] = True
    return pk_map, pk_lookup

# ------------------------------
# 그래프 데이터 (색/필터 = CodeFile_1 기준)
# ------------------------------
def _build_graph_data(
    df_cm_raw: pd.DataFrame,
    pk_map: Dict[str, Set[str]],
    pk_lookup: Dict[Tuple[str, str], bool],
) -> Tuple[
    Dict[str, List[str]],
    List[Tuple[str, str, str, str]],
    Dict[str, str],            # node_types_final
    Dict[str, str],            # node_types_target (CodeFile_1기준)
]:
    """
    반환:
      - columns_by_table: {table -> [columns...]}
      - edges: [(src_table, tgt_table, pk_col_name, edge_code_type)]
      - node_types_final: {table -> 대표 타입}  (타겟 타입 우선, 없으면 소스 타입, 없으면 OTHER)
      - node_types_target: {table -> 대표 타입} (오직 CodeFile_1 기준; 필터에 사용)
    """
    need = {"FileName", "ColumnName", "FK", "CodeColumn_1", "CodeType_1"}
    missing = need - set(df_cm_raw.columns)
    if missing:
        raise ValueError(f"[CRD] 필요한 컬럼 누락: {sorted(missing)}")

    df = _norm_cols(df_cm_raw).copy()
    df["CodeType_1"] = df["CodeType_1"].map(_norm_codetype)

    # 테이블-컬럼(소스 기준 우선 수집)
    columns_by_table: Dict[str, Set[str]] = {}
    for tname, g in df.groupby("FileName"):
        columns_by_table.setdefault(tname, set()).update(_to_set(g["ColumnName"]))
    # PK 추가
    for t, pkcols in pk_map.items():
        columns_by_table.setdefault(t, set()).update(pkcols)

    # 타입 집계: (A) 타겟 기준(= CodeFile_1), (B) 소스 기준(= FileName)
    node_types_target: Dict[str, str] = {}
    if "CodeFile_1" in df.columns:
        for tname, g in df.dropna(subset=["CodeFile_1"]).groupby("CodeFile_1"):
            types = { _norm_codetype(x) for x in _to_set(g["CodeType_1"]) }
            # 우선순위에 따라 대표 타입 선택
            chosen = next((t for t in _NODE_TYPE_PRIORITY if t in types), "OTHER")
            node_types_target[str(tname).strip()] = chosen

    node_types_source: Dict[str, str] = {}
    for tname, g in df.groupby("FileName"):
        types = { _norm_codetype(x) for x in _to_set(g["CodeType_1"]) }
        chosen = next((t for t in _NODE_TYPE_PRIORITY if t in types), "OTHER")
        node_types_source[str(tname).strip()] = chosen

    # PK 컬럼 소유 테이블 역추적/폴백
    col_to_pk_tables: Dict[str, Set[str]] = {}
    for t, cols in pk_map.items():
        for c in cols:
            col_to_pk_tables.setdefault(c, set()).add(t)
    col_to_tables: Dict[str, Set[str]] = {}
    for tname, g in df.groupby("FileName"):
        for c in _to_set(g["ColumnName"]):
            col_to_tables.setdefault(c, set()).add(tname)

    # FK → 엣지/타겟 컬럼 보강
    use_cols = ["FileName", "ColumnName", "CodeColumn_1", "CodeType_1"] + (["CodeFile_1"] if "CodeFile_1" in df.columns else [])
    fk_mask = df["FK"].astype(str).str.upper().eq("FK")
    df_fk = df.loc[fk_mask, use_cols].dropna(subset=["FileName", "ColumnName", "CodeColumn_1", "CodeType_1"]).copy()

    edges: Set[Tuple[str, str, str, str]] = set()
    for _, r in df_fk.iterrows():
        src_t = str(r["FileName"]).strip()
        src_c = str(r["ColumnName"]).strip()
        tgt_pk_col = str(r["CodeColumn_1"]).strip()
        edge_type = _norm_codetype(r["CodeType_1"])

        tgt_candidates: Set[str] = set()
        if "CodeFile_1" in r.index and pd.notna(r["CodeFile_1"]):
            tf = str(r["CodeFile_1"]).strip()
            if tf:
                tgt_candidates.add(tf)
        if not tgt_candidates:
            tgt_candidates = col_to_pk_tables.get(tgt_pk_col, set())
        if not tgt_candidates:
            tgt_candidates = col_to_tables.get(tgt_pk_col, set())

        if not tgt_candidates:
            columns_by_table.setdefault(src_t, set()).add(src_c)
            continue

        for tgt_t in tgt_candidates:
            tgt_t = str(tgt_t).strip()
            edges.add((src_t, tgt_t, tgt_pk_col, edge_type, src_c))
            columns_by_table.setdefault(tgt_t, set()).add(tgt_pk_col)  # 타겟에도 PK 컬럼 보강

    # 노드 대표 타입: 타겟 타입 우선, 없으면 소스 타입, 없으면 OTHER
    all_nodes = set(columns_by_table.keys()) | {e[0] for e in edges} | {e[1] for e in edges}
    node_types_final: Dict[str, str] = {}
    for t in all_nodes:
        node_types_final[t] = node_types_target.get(t) or node_types_source.get(t) or "OTHER"

    # 리스트화/정렬
    columns_by_table_list: Dict[str, List[str]] = {
        t: sorted(cols, key=lambda x: (0 if pk_lookup.get((t, x), False) else 1, x.upper()))
        for t, cols in columns_by_table.items()
    }
    edge_list = sorted(list(edges))
    return columns_by_table_list, edge_list, node_types_final, node_types_target

# ------------------------------
# Graphviz 렌더링
# ------------------------------
def _render_erd(
    columns_by_table: Dict[str, List[str]],
    pk_lookup: Dict[Tuple[str, str], bool],
    edges: List[Tuple[str, str, str, str, str]],
    node_types: Dict[str, str],
    out_dir: Optional[Path] = None
) -> Path:
    out_dir = out_dir or (Path.cwd() / "QDQM_Master_Code" / "QDQM_Output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "CodeMapping_ERD.png"

        # metric_colors = {
        #     "Master":   "#ff7f0e",     # 주황색
        #     "Operation": "#2ca02c",    # 초록색
        #     "Reference": "#84994f",    # 녹색
        #     "Attribute": "#d62728",    # 빨간색
        #     "Common":     "#9467bd",   # 보라색
        #     "Validation": "#a7aae1"    # 블루
        # }

    node_color = {
        "MASTER" :   "#9ecad6", # 하늘색
        "OPERATION": "#2ca02c", # 빨강
        "REFERENCE": "#f4991a", # 앰버
        "INTERNAL":  "#ff7f0e", # 주황
        "RULE":      "#9467bd", # 보라
        "VALIDATE":  "#ffde63", # 노랑
        "ATTRIBUTE": "#d62728",
        "COMMON":    "#9467bd",
        "OTHER":     "#D3DA99",
    }

    edge_color = {
        "MASTER" :   "#9ecad6",
        "OPERATION": "#2ca02c",
        "REFERENCE": "#f4991a",
        "INTERNAL":  "#ff7f0e",
        "RULE":      "#9467bd",
        "VALIDATE":  "#ffde63",
        "ATTRIBUTE": "#d62728",
        "COMMON":    "#9467bd",
        "OTHER":     "#D3DA99",
    }

    dot = Digraph(comment="CodeMapping Code Relationship Diagram", encoding="utf-8")
    dot.attr(rankdir="LR", nodesep="0.5", ranksep="3", concentrate="true",
             overlap="scale", splines="curved", fontsize="12")
    dot.attr(size="10,10", ratio="auto")
    dot.attr("node", fontname="NanumGothic")
    dot.attr("edge", fontname="NanumGothic")

    # 노드
    for tname in sorted(columns_by_table.keys()):
        cols = columns_by_table[tname]
        head = node_color.get(_norm_codetype(node_types.get(tname, "OTHER")), "#c7c7c7")

        label = ['<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">']
        label.append(f'<TR><TD PORT="title" BGCOLOR="{head}"><B>{tname}</B></TD></TR>')

        # PK 먼저
        for col in cols:
            if pk_lookup.get((tname, col), False):
                safe = (col or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                label.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="#E6E6FA" PORT="{safe}"><B>🔑 {safe}</B></TD></TR>')
        # 나머지
        for col in cols:
            if pk_lookup.get((tname, col), False):
                continue
            safe = (col or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            label.append(f'<TR><TD ALIGN="LEFT" PORT="{safe}">{safe}</TD></TR>')

        label.append("</TABLE>>")
        dot.node(tname, "\n".join(label), shape="none")

    # 엣지
    dot.attr('edge', minlen='1')
    # for src_t, tgt_t, pk_col, code_type in edges:
    #     safe = (pk_col or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    #     color = edge_color.get(_norm_codetype(code_type), "#7f7f7f")
    #     # color = edge_color.get(_norm_codetype(code_type), "#99ff68")
    #     dot.edge(src_t, f"{tgt_t}:{safe}",
    #              fontsize="10", dir="both", arrowhead="none", arrowtail="crow",
    #              constraint="true", labeldistance="1.1", labelfloat="true",
    #              penwidth="1.2", color=color)

    for e in edges:
        if len(e) == 4:
            src_t, tgt_t, pk_col, code_type = e
        else:
            src_t, tgt_t, pk_col, code_type, _src_col = e  # src_col은 시각화에 사용 안함
        safe = (pk_col or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        color = edge_color.get(_norm_codetype(code_type), "#7f7f7f")
        dot.edge(src_t, f"{tgt_t}:{safe}",
                fontsize="10", dir="both", arrowhead="none", arrowtail="crow",
                constraint="true", labeldistance="1.1", labelfloat="true",
                penwidth="1.2", color=color)

    dot.attr(dpi="300")
    try:
        dot.render(str(png_path.with_suffix("")), format="png", cleanup=True)
    except FileNotFoundError as e:
        # Graphviz 바이너리가 없는 경우를 명시적으로 처리
        error_msg = str(e)
        if "PosixPath('dot')" in error_msg or "Graphviz executables" in error_msg or "'dot'" in error_msg:
            raise FileNotFoundError(
                f"Graphviz가 설치되지 않았습니다. {error_msg}\n"
                f"Streamlit Cloud에서는 Code Relationship Diagram을 생성할 수 없습니다."
            )
        raise
    return png_path

# ------------------------------
# 캐시: FK 표
# ------------------------------
@st.cache_data(show_spinner=False)
def _to_fk_df_cached(edges_tuple: Tuple[Tuple, ...]) -> pd.DataFrame:
    rows = []
    for e in edges_tuple:
        if len(e) == 4:
            src_t, tgt_t, pk_col, code_type = e
            src_col = ""
        else:
            src_t, tgt_t, pk_col, code_type, src_col = e
        rows.append({
            "From(FileName)": src_t,
            "FK (ColumnName)": src_col,          # 👈 추가
            "To (CodeFile)": tgt_t,
            "PK Column(CodeColumn)": pk_col,
            "CodeType": code_type,
        })
    return pd.DataFrame(rows)

# ------------------------------
# 공개 API
# ------------------------------
# Display_ERD 시그니처에 view_mode 추가
def Display_ERD(
    erd_df: pd.DataFrame,
    *,
    out_dir: Optional[Path] = None,
    img_width: int = 680,
    view_mode: str = "All",   # 👈 추가
) -> None:
    # Graphviz 사용 가능 여부 확인
    if not GRAPHVIZ_AVAILABLE:
        st.error("⚠️ Graphviz가 설치되지 않았습니다.")
        st.info("Streamlit Cloud에서는 Code Relationship Diagram을 생성할 수 없습니다.")
        st.info("로컬 환경에서는 다음 명령으로 설치할 수 있습니다:")
        st.code("pip install graphviz\n# Windows: winget install graphviz\n# Mac: brew install graphviz\n# Linux: apt-get install graphviz")
        return
    
    df_cm = _norm_cols(erd_df).copy()
    if "CodeType_1" not in df_cm.columns:
        st.error("erd_df 에 'CodeType_1' 컬럼이 필요합니다.")
        return
    df_cm["CodeType_1"] = df_cm["CodeType_1"].map(_norm_codetype)

    # PK 동작 동일
    df_pk = df_cm[["FileName","ColumnName","PK"]].copy() \
        if {"FileName","ColumnName","PK"}.issubset(df_cm.columns) \
        else pd.DataFrame(columns=["FileName","ColumnName","PK"])

    pk_map, pk_lookup = _build_pk_map(df_pk)
    try:
        columns_by_table, edges, node_types_final, node_types_target = _build_graph_data(
            df_cm, pk_map, pk_lookup
        )
    except ValueError as e:
        st.error(str(e)); return

    # 👇 외부에서 받은 view_mode 로 필터링
    view_mode_norm = _norm_codetype(view_mode)
    if view_mode_norm not in {"ALL", "OPERATION", "REFERENCE"}:
        view_mode_norm = "ALL"

    if view_mode_norm != "ALL":
        target_nodes = {t for t, ttype in node_types_target.items() if ttype == view_mode_norm}
        if not target_nodes:
            st.info(f"{view_mode} 타입의 타겟(CodeFile_1) 노드가 없습니다.")
            return
        keep_nodes = set(target_nodes)
        for s, t, *_ in edges:
            if t in target_nodes:
                keep_nodes.add(s)
        columns_by_table = {t: cols for t, cols in columns_by_table.items() if t in keep_nodes}
        node_types_final = {t: node_types_final.get(t, "OTHER") for t in keep_nodes}
        edges = [e for e in edges if (e[0] in keep_nodes and e[1] in keep_nodes)]

    # KPI/요약/이미지 렌더(기존 그대로)
    st.markdown("### Code Relationship Diagram (선택한 CodeMapping 기반)")
    st.write(f"- 대상 노드 수: **{len(columns_by_table):,}**, 엣지 수: **{len(edges):,}**")
    st.caption("노드 색은 CodeFile_1(타겟)의 CodeType_1 기준. 상단 필터는 CodeFile_1 타입으로 표시를 제어합니다.")

    fk_box = st.container()
    fk_box.write("##### PK vs FK 관계 요약")
    if edges:
        fk_df = _to_fk_df_cached(tuple(edges))
        if len(fk_df) <= 150:
            fk_box.table(fk_df)
        else:
            fk_box.dataframe(fk_df, hide_index=True, height=min(320, 28 * len(fk_df) + 42), use_container_width=False)
    else:
        fk_box.info("표시할 FK 엣지가 없습니다.")

    with st.spinner("Code Relationship Diagram 생성 중..."):
        try:
            png_path = _render_erd(columns_by_table, pk_lookup, edges, node_types_final, out_dir=out_dir)
            image = Image.open(png_path)
            st.image(image, caption="CodeMapping 기반 Code Relationship Diagram", width=img_width)
            st.caption(f"파일: {png_path}")
        except FileNotFoundError as e:
            if "PosixPath('dot')" in str(e) or "Graphviz executables" in str(e):
                st.error("Graphviz가 설치되지 않았습니다.")
                st.info("Streamlit Cloud에서는 Code Relationship Diagram을 생성할 수 없습니다. 로컬 환경에서 실행해주세요.")
            else:
                st.error(f"파일 오류: {e}")
        except Exception as e:
            # st.error(f"Code Relationship Diagram 생성 중 오류 발생: {e}")
            st.info("Cloud 환경에서는 Diagram을 생성할 수 없습니다. Local 환경에서 실행해주세요. 샘플 이미지를 표시합니다.")
            image = Image.open("DataSense/DS_Output/CRD_Sample.png") # 파일명 대소문자 구분합니다. 
            st.image(image, caption="샘플 이미지입니다.", width=480)
            return False