# 01_Directory_Inspector.py
# -*- coding: utf-8 -*-
"""
DS_Diretory_Config.yaml 을 읽어
- directories.* 경로
- source_directories.* 경로
의 파일/폴더 수 및 용량을 집계/표시하는 Streamlit 앱
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

import yaml
import pandas as pd
import streamlit as st

# =========================
# 유틸: YAML 로딩/경로 탐색
# =========================
def _safe_load_yaml(p: Path) -> Dict[str, Any]:
    encs = ("utf-8", "utf-8-sig", "euc-kr", "cp949")
    last_err = None
    for enc in encs:
        try:
            with open(p, "r", encoding=enc) as f:
                obj = yaml.safe_load(f) or {}
            return obj
        except Exception as e:
            last_err = e
    raise RuntimeError(f"YAML 로드 실패: {p} :: {last_err}")

def _auto_find_yaml() -> Optional[Path]:
    """
    DS_Diretory_Config.yaml 자동 탐색
    - CWD
    - CWD/util
    - 프로젝트 상위 추정 경로들
    - DataSense/util
    """
    candidates = [
        Path.cwd() / "DS_Diretory_Config.yaml",
        Path.cwd() / "util" / "DS_Diretory_Config.yaml",
        Path.cwd().parent / "DS_Diretory_Config.yaml",
        Path.cwd().parent / "util" / "DS_Diretory_Config.yaml",
        Path.cwd() / "DataSense" / "util" / "DS_Diretory_Config.yaml",
        Path.cwd().parent / "DataSense" / "util" / "DS_Diretory_Config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _expand_path(base: Path, p: str) -> Path:
    """환경변수/홈(~)/상대경로를 포함할 수 있는 문자열을 절대경로로 정규화"""
    s = os.path.expandvars(str(p or "").strip())
    s = os.path.expanduser(s)
    q = Path(s)
    if not q.is_absolute():
        q = (base / q)
    return q.resolve()

def _fmt_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PB"

# =========================
# 파일/폴더 집계
# =========================
def _iter_children(root: Path, recursive: bool, include_hidden: bool):
    """
    지정 디렉토리에서 파일/폴더를 순회하는 제너레이터.
    recursive=False 면 1레벨만, True 면 재귀
    """
    if not root.exists() or not root.is_dir():
        return
    if not recursive:
        for entry in os.scandir(root):
            if not include_hidden and Path(entry.path).name.startswith("."):
                continue
            yield entry
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            # 숨김 처리
            if not include_hidden:
                dirnames[:]  = [d for d in dirnames if not d.startswith(".")]
                filenames[:] = [f for f in filenames if not f.startswith(".")]
            # 파일
            for fn in filenames:
                full = Path(dirpath) / fn
                yield full

@st.cache_data(show_spinner=False)
def scan_dir_stats(
    dir_path_str: str, recursive: bool, include_hidden: bool
) -> Tuple[int, int, int]:
    """
    디렉토리 집계 결과 캐시:
      - file_count
      - dir_count (recursive=True인 경우에만 의미 있음)
      - total_size(bytes)
    """
    p = Path(dir_path_str)
    file_count = 0
    dir_count = 0
    total_size = 0

    if not p.exists() or not p.is_dir():
        return (0, 0, 0)

    if not recursive:
        # 1레벨만
        for e in _iter_children(p, recursive=False, include_hidden=include_hidden):
            try:
                if isinstance(e, os.DirEntry):
                    if e.is_file():
                        file_count += 1
                        try:
                            total_size += e.stat().st_size
                        except Exception:
                            pass
                    elif e.is_dir():
                        dir_count += 1
                else:
                    # pathlib Path 로 들어온 경우
                    if Path(e).is_file():
                        file_count += 1
                        try:
                            total_size += Path(e).stat().st_size
                        except Exception:
                            pass
                    elif Path(e).is_dir():
                        dir_count += 1
            except Exception:
                pass
    else:
        # 재귀: 파일/폴더 모두 카운트
        for dirpath, dirnames, filenames in os.walk(p):
            if not include_hidden:
                dirnames[:]  = [d for d in dirnames if not d.startswith(".")]
                filenames[:] = [f for f in filenames if not f.startswith(".")]
            dir_count += len(dirnames)
            file_count += len(filenames)
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    total_size += fp.stat().st_size
                except Exception:
                    pass

    return (file_count, dir_count, total_size)

def build_section_table(
    section_name: str,
    mapping: Dict[str, Any],
    root_path: Path,
    recursive: bool,
    include_hidden: bool,
) -> pd.DataFrame:
    """
    section_name: 'directories' 또는 'source_directories'
    mapping: {key: path_str}
    """
    rows = []
    for k, rel in (mapping or {}).items():
        try:
            abs_path = _expand_path(root_path, str(rel))
            exists = abs_path.exists()
            is_dir = abs_path.is_dir()
            if exists and is_dir:
                files, dirs, total = scan_dir_stats(str(abs_path), recursive, include_hidden)
            else:
                files, dirs, total = (0, 0, 0)
            rows.append({
                "section": section_name,
                "key": str(k),
                "path": str(rel),
                "resolved_path": str(abs_path),
                "exists": "Y" if exists else "N",
                "is_dir": "Y" if is_dir else "N",
                "file_count": files,
                "dir_count": dirs,
                "total_size": total,
                "total_size(human)": _fmt_bytes(total),
            })
        except Exception as e:
            rows.append({
                "section": section_name,
                "key": str(k),
                "path": str(rel),
                "resolved_path": "",
                "exists": "N",
                "is_dir": "N",
                "file_count": 0,
                "dir_count": 0,
                "total_size": 0,
                "total_size(human)": "-",
                "error": str(e),
            })
    return pd.DataFrame(rows)

# =========================
# Streamlit 앱
# =========================
def main():
    st.set_page_config(page_title="Directory Inspector", layout="wide", page_icon="🗂️")
    st.title("Directory Inspector") # st.title("Directory Inspector (DS_Diretory_Config.yaml)")

    st.markdown(
        "- **DS_Diretory_Config.yaml** 을 읽어 `directories`/`source_directories` 경로의 파일/폴더 수를 집계합니다.\n"
        "- 경로는 `ROOT_PATH`를 기준으로 상대경로를 절대경로로 변환합니다."
    )

    # ----- 입력 UI -----
    with st.expander("① YAML 선택", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            yaml_path_txt = st.text_input(
                "YAML 경로 (미입력 시 자동 탐색)",
                value=str(_auto_find_yaml() or ""),
                placeholder="예) C:/projects/DataSense/util/DS_Diretory_Config.yaml"
            )
        with col2:
            uploaded = st.file_uploader("또는 업로드", type=["yaml", "yml"])

        yaml_data: Dict[str, Any] = {}
        yaml_path_used: Optional[Path] = None

        if uploaded is not None:
            try:
                yaml_data = yaml.safe_load(uploaded.read().decode("utf-8")) or {}
                yaml_path_used = None
                st.success("업로드 YAML 로드 성공")
            except Exception as e:
                st.error(f"업로드 파싱 실패: {e}")
        else:
            if yaml_path_txt.strip():
                p = Path(yaml_path_txt.strip())
                if p.exists():
                    try:
                        yaml_data = _safe_load_yaml(p)
                        yaml_path_used = p
                        st.success(f"파일 로드 성공: {p}")
                    except Exception as e:
                        st.error(f"YAML 로드 실패: {e}")
                else:
                    st.warning("지정한 경로가 존재하지 않습니다. 자동 탐색 후보를 확인해 보세요.")
            else:
                st.info("경로 미입력 → 자동 탐색을 시도했습니다. 필요 시 직접 입력/업로드 하세요.")

    # ----- 옵션 -----
    with st.expander("② 옵션", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            recursive = st.checkbox("하위 폴더 포함(재귀)", value=True)
        with c2:
            include_hidden = st.checkbox("숨김 포함(.*)", value=False)
        with c3:
            root_override = st.text_input(
                "ROOT_PATH 재정의(선택, 비우면 YAML의 ROOT_PATH 사용)",
                value=""
            )

    # ----- 처리 -----
    if not yaml_data:
        st.stop()

    root_path = Path(root_override.strip() or yaml_data.get("ROOT_PATH") or (yaml_path_used.parent if yaml_path_used else Path.cwd()))
    st.info(f"ROOT_PATH: `{root_path}`")

    directories = yaml_data.get("directories", {})
    source_directories = yaml_data.get("source_directories", {})

    t0 = time.time()
    df_dirs = build_section_table("directories", directories, root_path, recursive, include_hidden)
    df_src = build_section_table("source_directories", source_directories, root_path, recursive, include_hidden)
    df_all = pd.concat([df_dirs, df_src], ignore_index=True)
    dt_sec = time.time() - t0

    # ----- 출력 -----
    st.markdown("### 결과 요약")
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        st.metric("총 항목수", f"{len(df_all):,}")
    with colB:
        st.metric("총 파일수", f"{int(df_all['file_count'].sum()):,}")
    with colC:
        st.metric("총 폴더수", f"{int(df_all['dir_count'].sum()):,}")
    with colD:
        st.metric("총 용량", _fmt_bytes(int(df_all["total_size"].sum())))
    st.caption(f"스캔 시간: {dt_sec:.2f}s   (재귀={recursive}, 숨김포함={include_hidden})")

    st.markdown("### 상세 표")
    st.dataframe(
        df_all[[
            "section","key","path","resolved_path","exists","is_dir",
            "file_count","dir_count","total_size(human)"
        ]].sort_values(["section","key"]),
        use_container_width=True,
        hide_index=True,
    )

    # 섹션별 필터 & 미존재 경로 하이라이트
    with st.expander("필터 & 점검", expanded=False):
        section_pick = st.multiselect("섹션", options=["directories","source_directories"], default=["directories","source_directories"])
        only_missing = st.checkbox("존재하지 않는 경로만 보기", value=False)
        df_view = df_all[df_all["section"].isin(section_pick)].copy()
        if only_missing:
            df_view = df_view[df_view["exists"] == "N"]
        if df_view.empty:
            st.info("표시할 항목이 없습니다.")
        else:
            st.dataframe(
                df_view.sort_values(["section","key"]),
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()
