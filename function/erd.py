# function/erd.py
# -*- coding: utf-8 -*-
import os
import re
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from graphviz import Digraph


# ------------------------------
# 내부 유틸
# ------------------------------
def _read_csv_any(path: str):
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
        except Exception as e:
            last_err = e
    st.warning(f"[Skip] 파일 로드 실패: {path} -> {last_err}")
    return None


def _load_tables_from_selected(loaded_data: Dict[str, pd.DataFrame],
                               selected_files: List[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
    """
    selected_files: [(FileName, MasterType), ...]
    loaded_data['codemapping']에서 FilePath를 가져와 해당 CSV를 로드
    """
    tables: Dict[str, pd.DataFrame] = {}
    cm = loaded_data.get("codemapping", pd.DataFrame())
    if cm is None or cm.empty or not selected_files:
        return tables

    sel = pd.DataFrame(selected_files, columns=["FileName", "MasterType"])
    link = (cm.merge(sel, on=["FileName", "MasterType"], how="inner")
              [["FileName", "FilePath"]]
              .dropna()
              .drop_duplicates())

    for _, r in link.iterrows():
        fpath = str(r["FilePath"]).strip()
        if not fpath or not os.path.exists(fpath):
            continue
        df = _read_csv_any(fpath)
        if df is None:
            continue
        # 테이블 노드 이름: 파일명(확장자 제거)
        tables[Path(fpath).stem] = df

    return tables


# ------------------------------
# PK/FK 후보 탐색 (23_Make_ERD.py 기반)
# ------------------------------
def _find_composite_primary_keys(df: pd.DataFrame, max_comb_len: int = 5):
    """
    매우 단순화된 PK 후보 탐색:
    - 상단부터 n개 컬럼(최대 max_comb_len)을 유니크 여부로 검사
    - 첫 번째로 유니크가 되는 조합을 PK 후보로 간주
    """
    cols = df.columns.tolist()
    for i in range(len(cols)):
        cur = cols[: i + 1]
        if len(cur) > max_comb_len:
            break
        sub = df[cur].dropna()
        if len(sub) > 0 and sub.duplicated().sum() == 0:
            return [tuple(cur)]
    return []


def _find_foreign_keys(tables: Dict[str, pd.DataFrame]):
    """
    동일 컬럼명 기반, 간단한 휴리스틱으로 FK 후보 탐색
    (원 코드의 간소화/일반화 버전)
    """
    fk_candidates = set()
    names = list(tables.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t1, t2 = names[i], names[j]
            df1, df2 = tables[t1], tables[t2]

            for c1 in df1.columns:
                for c2 in df2.columns:
                    if c1.strip().upper() == c2.strip().upper():
                        # 코드성/ID/NO가 포함된 테이블명을 1측으로 간주하는 휴리스틱
                        if re.search(r'(CODE|_ID|_NO)$', t2.upper()):
                            fk_candidates.add((t1, t2, c2))  # t1(N) -> t2(1)
                        elif re.search(r'(CODE|_ID|_NO)$', t1.upper()):
                            fk_candidates.add((t2, t1, c1))  # t2(N) -> t1(1)
                        else:
                            # 더 짧은 이름을 1측으로
                            if len(t1) <= len(t2):
                                fk_candidates.add((t2, t1, c1))
                            else:
                                fk_candidates.add((t1, t2, c2))
    return list(fk_candidates)


def _create_pk_fk(tables: Dict[str, pd.DataFrame]):
    result_rows = []
    pk_results: Dict[str, List[Tuple[str, ...]]] = {}

    # PK
    for name, df in tables.items():
        pks = _find_composite_primary_keys(df)
        if pks:
            pk_results[name] = pks[:3]
            for pk_set in pk_results[name]:
                result_rows.append({
                    "파일명": name,
                    "구분": "PK",
                    "컬럼수": len(pk_set),
                    "PK조합": ",".join(pk_set),
                })

    # FK
    fk_candidates = _find_foreign_keys(tables)
    for from_table, to_table, col in fk_candidates:
        result_rows.append({
            "파일명": from_table,
            "구분": "FK",
            "컬럼수": 1,
            "PK조합": col,
        })

    result_df = pd.DataFrame(result_rows).drop_duplicates()
    return result_df, pk_results, fk_candidates


# ------------------------------
# ERD 생성 (Graphviz)
# ------------------------------
def _create_erd_png(tables: Dict[str, pd.DataFrame],
                    pk_candidates: Dict[str, List[Tuple[str, ...]]],
                    fk_candidates: List[Tuple[str, str, str]],
                    out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "PK_FK_ERD.png"

    dot = Digraph(comment="Database ERD", encoding="utf-8")
    dot.attr(rankdir="LR", nodesep="0.5", ranksep="3", concentrate="true",
             overlap="scale", splines="curved", fontsize="12")
    dot.attr(size="10,10", ratio="auto")
    dot.attr("node", fontname="NanumGothic")
    dot.attr("edge", fontname="NanumGothic")

    # 테이블 노드
    for tname in sorted(tables.keys()):
        pk_cols = list(pk_candidates.get(tname, [[]])[0]) if pk_candidates.get(tname) else []
        label = ['<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">']
        label.append(f'<TR><TD PORT="title" BGCOLOR="#FFD700"><B>{tname}</B></TD></TR>')

        for c in pk_cols:
            safe = (c or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            label.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="#E6E6FA" PORT="{safe}"><B>🔑 {safe}</B></TD></TR>')

        for c in tables[tname].columns:
            if c in pk_cols:
                continue
            safe = (c or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            label.append(f'<TR><TD ALIGN="LEFT" PORT="{safe}">{safe}</TD></TR>')

        label.append("</TABLE>>")
        dot.node(tname, "\n".join(label), shape="none")

    # FK 엣지
    dot.attr('edge', minlen='1')
    for from_table, to_table, column in fk_candidates:
        safe = (column or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        dot.edge(f"{from_table}:{safe}", f"{to_table}:{safe}",
                 fontsize="10", dir="both", arrowhead="none", arrowtail="crow",
                 constraint="true", labelangle="0", labeldistance="1.1",
                 labelfloat="true", penwidth="0.8")

    dot.attr(dpi="300")
    dot.render(str(png_path.with_suffix("")), format="png", cleanup=True)
    return png_path


def _display_erd_kpis(tables: Dict[str, pd.DataFrame]):
    st.markdown("### ERD KPI")
    info = [{
        "Table Name": name,
        "Column Count": len(df.columns),
        "Column List": list(df.columns)
    } for name, df in tables.items()]
    table_list = pd.DataFrame(info)

    # 간단 KPI
    cols = st.columns(2)
    cols[0].metric("Table Cnt #", f"{len(table_list):,}")
    cols[1].metric("Column Cnt #", f"{table_list['Column Count'].sum():,}")

    st.write("##### 테이블 정보")
    st.dataframe(table_list, hide_index=True)

# ------------------------------
# 공개 API
# ------------------------------
def Display_ERD(loaded_data: Dict[str, pd.DataFrame],
                selected_files: List[Tuple[str, str]],
                out_dir: Path | None = None) -> None:
    """
    2_Code Mapping 화면에서 선택된 파일들로 ERD 생성/표시
    """
    st.write(selected_files)

    df = loaded_data['codemapping'].copy()

        # ✅ FileName + MasterType 기준 필터링
    selected_df = df.merge(
        pd.DataFrame(selected_files, columns=['FileName','MasterType']),
        on=['FileName','MasterType'],
        how='inner'
    )

    selected_df = selected_df[['FileName', 'ColumnName', 'PK', 'FK', 'CodeColumn_1', 'CodeType_1', 'CodeFile_1']]
    st.dataframe(selected_df, hide_index=True, width=1000, height=500)
    return
# ------------------------------
# 공개 API
# ------------------------------
def Display_ERD_Old(loaded_data: Dict[str, pd.DataFrame],
                selected_files: List[Tuple[str, str]],
                out_dir: Path | None = None) -> None:
    """
    2_Code Mapping 화면에서 선택된 파일들로 ERD 생성/표시
    """
    if not selected_files:
        st.info("선택된 파일이 없습니다.")
        return

    tables = _load_tables_from_selected(loaded_data, selected_files)
    if not tables:
        st.warning("선택된 파일에서 로드할 테이블을 찾지 못했습니다.")
        return

    _display_erd_kpis(tables)

    if st.button("ERD 생성", key="btn_erd_generate"):
        with st.spinner("ERD 생성 중..."):
            result_df, pk_results, fk_candidates = _create_pk_fk(tables)
            # 결과 표
            st.write("##### 테이블 PK & FK 정보")
            st.dataframe(
                result_df,
                column_config={
                    "파일명": st.column_config.TextColumn(width=200),
                    "구분": st.column_config.TextColumn(width=50),
                    "컬럼수": st.column_config.NumberColumn(width=50),
                    "PK조합": st.column_config.TextColumn(width=700),
                },
                hide_index=True, width=1000, height=500
            )

            # ERD 이미지
            out_dir = out_dir or (Path.cwd() / "QDQM_Master_Code" / "QDQM_Output")
            png_path = _create_erd_png(tables, pk_results, fk_candidates, out_dir)
            try:
                image = Image.open(png_path)
                st.image(image, caption="생성된 ERD", width=None)
            except Exception as e:
                st.error(f"ERD 이미지 로드 중 오류: {e}")
