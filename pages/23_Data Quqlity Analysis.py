# -------------------------------------------------
# 1. Path / Warning setup (Streamlit import 전)
# -------------------------------------------------
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util.streamlit_warnings import setup_streamlit_warnings
setup_streamlit_warnings()

# -------------------------------------------------
# 2. Standard / Third-party imports
# -------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

@st.cache_data
def load_and_refine_data():
    base_path = PROJECT_ROOT / 'DS_Output'
    
    # [A] 필요한 컬럼만 선택하여 로드 (메모리 절약 및 속도 향상)
    df_f = pd.read_csv(base_path / 'FileFormat.csv').fillna(0)
    df_r = pd.read_csv(base_path / 'CodeMapping_erd.csv', dtype=str).fillna('')
    
    # [B] 요구사항: 분석 불필요 도메인 사전 필터링
    exclude_types = ['Common', 'Reference', 'Validation']
    df_f = df_f[~df_f['MasterType'].isin(exclude_types)].copy()
    df_r = df_r[~df_r['MasterType'].isin(exclude_types)].copy()
    
    # [C] 실시간 DQ Scoring (자체 품질 지표 산출)
    df_f['DQ_Score'] = (
        (100 - pd.to_numeric(df_f['Null(%)'], errors='coerce').fillna(0)) * 0.4 +
        (pd.to_numeric(df_f['Format(%)'], errors='coerce').fillna(0)) * 0.3 +
        (100 - (df_f['HasBrokenKor'].astype(float).clip(0, 1) * 100)) * 0.3
    ).clip(0, 100)
    
    return df_f, df_r

def run_dashboard():
    # --- [UI/UX 설정] ---
    # 한글 가독성 강화 및 음영 제거, 와이드 레이아웃 설정
    st.set_page_config(layout="wide", page_title="DataSense Lineage Hub")
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
        * { font-family: 'Nanum Gothic', 'Malgun Gothic', sans-serif !important; text-shadow: none !important; }
        .main .block-container { max-width: 98%; padding-top: 1.5rem; }
        .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 10px; }
        </style>
        """, unsafe_allow_html=True)

    df_f, df_r = load_and_refine_data()

    st.title("🏛️ DataSense Smart Lineage & DQ Analyzer")
    st.caption("Pure Downstream 2단계 분석 모드 (순환 참조 제거 완료)")

    # 전역 필터
    all_mtypes = sorted(df_f['MasterType'].unique())
    selected_mtypes = st.multiselect("📂 분석 도메인(MasterType) 선택", options=all_mtypes, default=all_mtypes)
    
    df_f_sub = df_f[df_f['MasterType'].isin(selected_mtypes)]
    df_r_sub = df_r[df_r['MasterType'].isin(selected_mtypes)]

    # 탭 구성 (st.session_state 관리를 위해 탭 순서 고정)
    tabs = st.tabs(["💎 품질 대시보드", "🕸️ 순수 계보 분석", "🔍 컬럼 프로파일링", "🛡️ 결함 진단"])

    # --- Tab 1: 종합 품질 대시보드 ---
    with tabs[0]:
        st.subheader("📊 도메인 통합 품질 현황")
        m1, m2, m3 = st.columns(3)
        m1.metric("분석 대상 파일", f"{df_f_sub['FileName'].nunique()}개")
        m2.metric("평균 DQ 점수", f"{df_f_sub['DQ_Score'].mean():.1f}점")
        m3.metric("평균 적재율", f"{(100 - df_f_sub['Null(%)'].mean()):.1f}%")

        st.write("#### 📂 파일별 규모 및 품질 트리맵")
        fig_tree = px.treemap(df_f_sub, path=['MasterType', 'FileName'], values='RecordCnt', 
                              color='DQ_Score', color_continuous_scale='RdYlGn', height=600)
        st.plotly_chart(fig_tree, width='stretch')

    # --- Tab 2: 순수 계보 분석 (핵심 기능) ---
    with tabs[1]:
        st.subheader("🕸️ 파일 중심 하위 계보 추적 (Loop-Free)")
        
        # 파일 선택 시 탭 유지 안정성을 위해 고유 Key 사용
        start_node = st.selectbox("🎯 기준 파일(Source) 선택", options=sorted(df_r_sub['FileName'].unique()), key="sb_lineage")
        
        # [순환 참조 방지 알고리즘]
        links = []
        visited = {start_node}  # 방문 노드 기록 (자기 참조 및 상호 참조 차단용)

        # Level 1 추적
        l1_raw = df_r_sub[df_r_sub['FileName'] == start_node]
        l1_targets = []
        if not l1_raw.empty:
            l1_agg = l1_raw[l1_raw['Level1_File'] != ''].groupby(['FileName', 'Level1_File']).size().reset_index(name='v')
            for _, r in l1_agg.iterrows():
                target = r['Level1_File']
                if target not in visited:
                    links.append({'s': r['FileName'], 't': target, 'v': r['v'], 'c': "rgba(100, 181, 246, 0.6)", 'lvl': '1단계'})
                    l1_targets.append(target)
                    visited.add(target)

        # Level 2 추적
        if l1_targets:
            l2_raw = df_r_sub[df_r_sub['FileName'].isin(l1_targets)]
            if not l2_raw.empty:
                l2_agg = l2_raw[l2_raw['Level1_File'] != ''].groupby(['FileName', 'Level1_File']).size().reset_index(name='v')
                for _, r in l2_agg.iterrows():
                    target = r['Level1_File']
                    # 상호 참조($A \leftrightarrow B$) 차단: 이미 visited에 있으면 제외
                    if target not in visited:
                        links.append({'s': r['FileName'], 't': target, 'v': r['v'], 'c': "rgba(255, 193, 7, 0.6)", 'lvl': '2단계'})
                        visited.add(target)

        if not links:
            st.warning("순환 참조를 제외한 하위 계보가 존재하지 않습니다.")
        else:
            ldf = pd.DataFrame(links).drop_duplicates()
            all_nodes = sorted(list(set(ldf['s']) | set(ldf['t'])))
            node_idx = {name: i for i, name in enumerate(all_nodes)}

            # 대형 Sankey Chart (높이 850px)
            fig_sk = go.Figure(data=[go.Sankey(
                node = dict(pad=40, thickness=25, label=all_nodes, color="#CFD8DC", line=dict(color="#B0BEC5", width=1.2)),
                link = dict(source=ldf['s'].map(node_idx), target=ldf['t'].map(node_idx), value=ldf['v'], color=ldf['c'])
            )])
            fig_sk.update_layout(height=850, font_size=14, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_sk, width='stretch')

            # 범례 안내
            c1, c2 = st.columns(2)
            c1.markdown("<h4 style='color:#64B5F6;'>🔵 1단계 (Direct Downstream)</h4>", unsafe_allow_html=True)
            c2.markdown("<h4 style='color:#FFC107;'>🟡 2단계 (Extended Downstream)</h4>", unsafe_allow_html=True)

            st.divider()
            st.write(f"#### 📄 '{start_node}' 기반 계보 상세 매칭 리포트")
            st.dataframe(ldf[['lvl', 's', 't', 'v']].rename(columns={'lvl':'단계', 's':'출발파일', 't':'도착파일', 'v':'컬럼수'}), width='stretch')

    # --- Tab 3: 컬럼 프로파일링 ---
    with tabs[2]:
        st.subheader("🔍 파일 단위 상세 데이터 진단")
        f_name = st.selectbox("진단 대상 파일 선택", options=sorted(df_f_sub['FileName'].unique()), key="sb_prof")
        st.dataframe(df_f_sub[df_f_sub['FileName'] == f_name][['ColumnName', 'DataType', 'DQ_Score', 'Null(%)', 'Unique(%)', 'Format(%)', 'HasBrokenKor']], width='stretch')

    # --- Tab 4: 결함 진단 ---
    with tabs[3]:
        st.subheader("🛡️ 기술적 데이터 무결성 리포트")
        err_sum = df_f_sub.groupby('FileName')[['HasBrokenKor', 'HasUnicode2']].sum().reset_index()
        fig_err = px.bar(err_sum, x='FileName', y=['HasBrokenKor', 'HasUnicode2'], barmode='group', height=500)
        st.plotly_chart(fig_err, width='stretch')

if __name__ == "__main__":
    run_dashboard()