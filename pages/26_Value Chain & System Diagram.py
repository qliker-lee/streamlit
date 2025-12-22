# -*- coding: utf-8 -*-
"""
Value Chain & System Diagram Generator
DS_ValueChain.csv, DS_System.csv, DS_ValueChain_System_File.csv 파일을 이용하여 
Industry별 Value Chain Diagram과 System Architecture Diagram을 생성합니다.
Qliker 2025.
""" 
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import logging
import platform
import shutil
from graphviz import Digraph
from PIL import Image

# Streamlit 경고 억제
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# 경로 설정
OUTPUT_DIR = Path("DataSense/DS_Output")
IMAGE_DIR = OUTPUT_DIR / "images"
VALUECHAIN_CSV_PATH = OUTPUT_DIR / "DS_ValueChain.csv"
SYSTEM_CSV_PATH = OUTPUT_DIR / "DS_System.csv"
MAPPING_CSV_PATH = OUTPUT_DIR / "DS_ValueChain_System_File.csv"

# 디렉토리가 없으면 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 공통 유틸리티 함수
# ============================================================================

def is_cloud_env() -> bool:
    """Cloud 환경인지 확인합니다 (Graphviz dot 실행 파일 존재 여부로 판단)."""
    try:
        return shutil.which("dot") is None
    except Exception:
        return True

def show_sample_image(image_filename, caption):
    """Sample 이미지를 표시합니다."""
    try:
        sample_path = IMAGE_DIR / image_filename
        if sample_path.exists():
            image = Image.open(sample_path)
            st.image(image, caption=caption, use_container_width=True)
            st.info("**Cloud 환경에서는 Graphviz 실행이 제한됩니다. 실제 Diagram 대신 예제 이미지를 표시합니다.**")
        else:
            st.warning(f"⚠️ Sample 이미지를 찾을 수 없습니다: {image_filename}")
    except Exception as e:
        st.error(f"❌ Sample 이미지 로드 중 오류가 발생했습니다: {e}")

def load_data(path):
    """CSV 파일을 로드합니다."""
    if path.exists():
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            st.error(f"❌ 파일 로드 중 오류가 발생했습니다: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_valuechain_data():
    """Value Chain 데이터를 로드합니다."""
    df = load_data(VALUECHAIN_CSV_PATH)
    if df.empty:
        st.error(f"❌ '{VALUECHAIN_CSV_PATH.name}' 파일이 존재하지 않거나 비어있습니다.")
        st.info("먼저 'Value Chain & System Management' 페이지에서 Value Chain을 정의해주세요.")
    return df

def get_all_industries(df):
    """모든 Industry 목록을 반환합니다."""
    if df is None or df.empty:
        return []
    if "Industry" not in df.columns:
        return []
    return sorted(df["Industry"].unique().tolist())

# ============================================================================
# Value Chain Diagram 생성 함수
# ============================================================================

def create_valuechain_diagram(df, industry):
    """
    Value Chain Diagram 생성 (Primary 상단, Support 하단, 수직 정렬 완벽 최적화)
    """
    industry_df = df[df["Industry"] == industry].copy()
    if industry_df.empty:
        return None
    
    industry_df = industry_df.sort_values("Activity_Seq")
    
    # 1. 그래프 기본 설정
    graph = Digraph(name=f"ValueChain_{industry}", format='png', engine='dot')
    
    # rankdir='LR': 왼쪽에서 오른쪽으로 흐름
    # nodesep: 상하 노드 간격 (좁게 설정하여 두 그룹을 붙임)
    # ranksep: 좌우 노드 간격 (활동 간의 거리)
    graph.attr(rankdir='LR', size='20,20', nodesep='0.5', ranksep='0.6')
    
    # 폰트 설정
    font_name = 'Malgun Gothic' if platform.system() == 'Windows' else 'NanumGothic'

    # 모든 노드의 규격을 통일 (정렬의 핵심)
    graph.attr('node', shape='box', style='rounded,filled', fontname=font_name,
               fontsize='16', width='2.8', height='0.9', fixedsize='true')
    graph.attr('edge', fontname=font_name)
    
    primary_activities = industry_df[industry_df["Activity_Type"] == "Primary"].reset_index(drop=True)
    support_activities = industry_df[industry_df["Activity_Type"] == "Support"].reset_index(drop=True)
    
    max_cols = max(len(primary_activities), len(support_activities))

    # 2. 노드 생성 및 그룹화
    primary_node_ids = []
    support_node_ids = []

    for i in range(max_cols):
        # --- Primary 노드 생성 ---
        p_id = f"pri_{i}"
        if i < len(primary_activities):
            row = primary_activities.iloc[i]
            p_label = f"{row['Activity']}\\n({row['Activity_Kor']})"
            graph.node(p_id, label=p_label, fillcolor='#E3F2FD', color='#1E88E5')
        else:
            graph.node(p_id, label='', style='invis')
        primary_node_ids.append(p_id)

        # --- Support 노드 생성 ---
        s_id = f"sup_{i}"
        if i < len(support_activities):
            row = support_activities.iloc[i]
            s_label = f"{row['Activity']}\\n({row['Activity_Kor']})"
            graph.node(s_id, label=s_label, fillcolor='#FFF9C4', color='#FBC02D')
        else:
            graph.node(s_id, label='', style='invis')
        support_node_ids.append(s_id)

        # --- 수직 정렬 강제 (동일 선상 배치) ---
        # rank='same'과 투명 엣지를 사용하여 Primary가 무조건 위에, Support가 아래에 오도록 고정
        with graph.subgraph() as c:
            c.attr(rank='same')
            c.node(p_id)
            c.node(s_id)
        
        # Primary에서 Support로 투명선을 그어 상하 관계 확정
        graph.edge(p_id, s_id, style='invis')

    # 3. 수평 흐름 연결 (화살표)
    for i in range(max_cols - 1):
        # Primary 흐름 (실선 화살표)
        if i < len(primary_activities) - 1:
            graph.edge(primary_node_ids[i], primary_node_ids[i+1], 
                       style='bold', color='#1E88E5', arrowhead='vee')
        
        # Support 흐름 (점선 화살표)
        if i < len(support_activities) - 1:
            graph.edge(support_node_ids[i], support_node_ids[i+1], 
                       style='dashed', color='#FBC02D', arrowhead='vee')

    return graph

# ============================================================================
# System Architecture Diagram 생성 함수
# ============================================================================

def create_system_architecture_diagram(industry, mode="Summary"):
    """System Architecture Diagram을 생성합니다."""
    df_vc = load_data(VALUECHAIN_CSV_PATH)
    df_sys = load_data(SYSTEM_CSV_PATH)
    df_mapping = load_data(MAPPING_CSV_PATH)

    if df_vc.empty:
        return None

    industry_vc = df_vc[df_vc["Industry"] == industry].sort_values("Activity_Seq").copy()
    if industry_vc.empty:
        return None

    industry_mapping = df_mapping[df_mapping["Industry"] == industry]
    dot = Digraph(name=f"SysArch_{mode}_{industry}", format='png', engine='dot')
    
    # 상세 모드일 때는 노드 내부 텍스트가 길어지므로 간격을 조정합니다.
    rank_sep = '1.5' if mode == "Summary" else '2.5'
    dot.attr(rankdir='TB', size='40,30', nodesep='0.5', ranksep=rank_sep)
    
    font_name = 'Malgun Gothic' if platform.system() == 'Windows' else 'NanumGothic'

    # --- 1. 상단 레이어: 가로 정렬 Activity (수정 금지 로직) ---
    activity_node_ids = []
    with dot.subgraph() as s:
        s.attr(rank='same')
        for _, row in industry_vc.iterrows():
            act_id = f"act_{row['Activity']}"
            display_name = row['Activity_Kor'] if pd.notna(row.get('Activity_Kor')) else row['Activity']
            label = f"{display_name}\\n({row['Activity']})"
            
            if row.get('Activity_Type') == 'Primary':
                f_color, b_color = '#E3F2FD', '#1E88E5'
            else:
                f_color, b_color = '#FFF9C4', '#FBC02D'
            
            s.node(act_id, label=label, shape='box', style='filled,rounded', 
                   fillcolor=f_color, color=b_color, fontname=font_name, 
                   width='2.2', height='0.8', penwidth='2')
            activity_node_ids.append(act_id)

    for i in range(len(activity_node_ids) - 1):
        dot.edge(activity_node_ids[i], activity_node_ids[i+1], style='invis')

    # --- 2. 하단 레이어: IT Systems & Files ---
    mapped_systems = industry_mapping["System"].dropna().unique()
    system_node_ids = {}
    
    if len(mapped_systems) > 0:
        with dot.subgraph() as ss:
            ss.attr(rank='same')
            for sys_id in mapped_systems:
                if not sys_id: continue
                sys_info = df_sys[(df_sys["Industry"] == industry) & (df_sys["System"] == sys_id)]
                sys_kor = sys_info.iloc[0]["System_Kor"] if not sys_info.empty else sys_id
                
                node_id = f"sys_{sys_id}"
                
                if mode == "Summary":
                    # 요약 모드: 기존 컴포넌트 형태
                    ss.node(node_id, label=f"{sys_kor}\\n({sys_id})", shape='component', 
                           style='filled', fillcolor='#E8F5E9', color='#43A047', 
                           fontname=font_name, width='2.0', height='0.7')
                else:
                    # 상세 모드: 시스템명 아래에 파일 목록을 줄바꿈하여 포함
                    files = industry_mapping[industry_mapping["System"] == sys_id]["FileName"].dropna().unique()
                    file_list_str = "\\n".join([f"• {f}" for f in files]) if len(files) > 0 else "(No Files)"
                    
                    # HTML-like label을 사용하지 않고 단순 문자열과 구분선을 조합한 박스 형태
                    detail_label = f"{{ {sys_kor} ({sys_id}) | {file_list_str} }}"
                    
                    ss.node(node_id, label=detail_label, shape='record', # record 쉐이프 사용
                           style='filled', fillcolor='#F1F8E9', color='#2E7D32', 
                           fontname=font_name, penwidth='1.5')
                
                system_node_ids[sys_id] = node_id

    # --- 3. 연결선 (Activity -> System) ---
    unique_links = industry_mapping[["Activity", "System"]].drop_duplicates()
    for _, row in unique_links.iterrows():
        if f"act_{row['Activity']}" in activity_node_ids and row['System'] in system_node_ids:
            dot.edge(f"act_{row['Activity']}", system_node_ids[row['System']], 
                     color='#9E9E9E', arrowhead='vee', penwidth='1.5')

    return dot

# ============================================================================
# Value Chain & File Diagram 생성 함수
# ============================================================================

def create_valuechain_file_diagram(industry):
    """Value Chain Activity와 FileName을 직접 연결하는 Diagram을 생성합니다.
    System Architecture Detail과 비슷하게 Activity별로 FileName을 박스로 묶어서 표시합니다."""
    df_vc = load_data(VALUECHAIN_CSV_PATH)
    df_mapping = load_data(MAPPING_CSV_PATH)

    if df_vc.empty:
        return None

    industry_vc = df_vc[df_vc["Industry"] == industry].sort_values("Activity_Seq").copy()
    if industry_vc.empty:
        return None

    industry_mapping = df_mapping[df_mapping["Industry"] == industry]
    if industry_mapping.empty:
        return None
    
    dot = Digraph(name=f"ValueChain_File_{industry}", format='png', engine='dot')
    # 파일 목록이 포함되므로 간격을 넓게 설정
    dot.attr(rankdir='TB', size='40,30', nodesep='0.5', ranksep='2.5')
    
    font_name = 'Malgun Gothic' if platform.system() == 'Windows' else 'NanumGothic'

    # --- 1. 상단 레이어: 가로 정렬 Activity ---
    activity_node_ids = []
    with dot.subgraph() as s:
        s.attr(rank='same')
        for _, row in industry_vc.iterrows():
            act_id = f"act_{row['Activity']}"
            display_name = row['Activity_Kor'] if pd.notna(row.get('Activity_Kor')) else row['Activity']
            label = f"{display_name}\\n({row['Activity']})"
            
            if row.get('Activity_Type') == 'Primary':
                f_color, b_color = '#E3F2FD', '#1E88E5'
            else:
                f_color, b_color = '#FFF9C4', '#FBC02D'
            
            s.node(act_id, label=label, shape='box', style='filled,rounded', 
                   fillcolor=f_color, color=b_color, fontname=font_name, 
                   width='2.2', height='0.8', penwidth='2')
            activity_node_ids.append(act_id)

    for i in range(len(activity_node_ids) - 1):
        dot.edge(activity_node_ids[i], activity_node_ids[i+1], style='invis')

    # --- 2. 하단 레이어: Activity별 FileName 박스 ---
    file_box_node_ids = {}
    
    with dot.subgraph() as fs:
        fs.attr(rank='same')
        for _, row in industry_vc.iterrows():
            activity = row['Activity']
            act_id = f"act_{activity}"
            
            # 해당 Activity에 연결된 FileName들 가져오기
            files = industry_mapping[industry_mapping["Activity"] == activity]["FileName"].dropna().unique()
            
            if len(files) > 0:
                # 파일 목록을 줄바꿈으로 표시
                file_list_str = "\\n".join([f"• {f}" for f in files])
                
                # Activity 정보를 포함한 박스 레이블 생성
                display_name = row['Activity_Kor'] if pd.notna(row.get('Activity_Kor')) else activity
                detail_label = f"{{ {display_name} ({activity}) | {file_list_str} }}"
                
                # 각 Activity별 File 박스 노드 생성
                file_box_id = f"filebox_{activity}"
                fs.node(file_box_id, label=detail_label, shape='record',
                       style='filled', fillcolor='#F3E5F5', color='#9C27B0',
                       fontname=font_name, penwidth='1.5')
                
                file_box_node_ids[activity] = file_box_id

    # --- 3. 연결선 (Activity -> FileName 박스) ---
    for activity in file_box_node_ids.keys():
        act_id = f"act_{activity}"
        if act_id in activity_node_ids and activity in file_box_node_ids:
            dot.edge(act_id, file_box_node_ids[activity],
                     color='#9E9E9E', arrowhead='vee', penwidth='1.5')

    return dot

# ============================================================================
# 메인 UI 함수
# ============================================================================

def value_chain_tab(df, selected_industry):
    """Value Chain Diagram 탭"""
    st.markdown(f"### 📊 Value Chain Diagram: **{selected_industry}**")
    
    # 선택된 Industry의 데이터 확인
    industry_df = df[df["Industry"] == selected_industry].copy()
    if industry_df.empty:
        st.warning(f"⚠️ '{selected_industry}' Industry에 대한 데이터가 없습니다.")
        return
    
    # 데이터 미리보기
    with st.expander("📋 Value Chain Data Preview", expanded=False):
        display_df = industry_df[["Activity_Seq", "Activity_Type", "Activity", "Activity_Kor", "Activity_Description"]].copy()
        display_df.columns = ["Seq", "Type", "Activity (EN)", "Activity (KR)", "Description"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Cloud 환경 체크
    if is_cloud_env():
        show_sample_image("sample_ValueChain.png", f"Value Chain Diagram: {selected_industry} (Sample)")
        return
    
    # Diagram 생성
    try:
        graph = create_valuechain_diagram(df, selected_industry)
        
        if graph is None:
            st.warning("⚠️ Diagram을 생성할 수 없습니다.")
            return
        
        # Diagram 렌더링 및 저장
        file_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        png_filename = f"ValueChain_{selected_industry}_{file_time}.png"
        png_filepath = IMAGE_DIR / png_filename
        
        try:
            graph.attr(dpi='300')
            graph.render(str(png_filepath.with_suffix('')), format='png', cleanup=True)
            actual_png_filepath = IMAGE_DIR / f"{png_filepath.stem}.png"
            
            if actual_png_filepath.exists():
                # 이미지 표시
                image = Image.open(actual_png_filepath)
                st.image(image, caption=f"Value Chain Diagram: {selected_industry}", 
                       use_container_width=True)
                
                # 다운로드 버튼
                with open(actual_png_filepath, 'rb') as f:
                    png_data = f.read()
                if png_data:
                    st.download_button(
                        label="📥 Download PNG",
                        data=png_data,
                        file_name=actual_png_filepath.name,
                        mime="image/png",
                        key="vc_download"
                    )
            else:
                st.error("❌ PNG 파일 생성에 실패했습니다.")
                
        except Exception as e:
            st.error(f"❌ Diagram 생성 중 오류가 발생했습니다: {e}")
            st.info("Graphviz가 설치되어 있고 PATH에 추가되었는지 확인해주세요.")
            
            # SVG로 대체 시도
            try:
                svg_data = graph.pipe(format='svg').decode('utf-8')
                components.html(svg_data, height=800, scrolling=True)
                st.success("✅ Diagram이 SVG 형식으로 표시되었습니다.")
            except Exception as svg_e:
                st.error(f"SVG 생성도 실패했습니다: {svg_e}")
    
    except Exception as e:
        st.error(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        st.code(traceback.format_exc())

def system_architecture_tab(selected_industry, mode="Summary"):
    """System Architecture Diagram 탭"""
    if mode == "Summary":
        st.markdown(f"### 🏗️ System Architecture (Summary): **{selected_industry}**")
        st.markdown("##### Activity와 System 간의 연결 관계를 요약하여 표시합니다.")
    else:
        st.markdown(f"### 🔍 System Architecture (Detail): **{selected_industry}**")
        st.markdown("##### System 내부의 파일 목록을 포함한 상세 구성도를 표시합니다.")
    
    # 데이터 로드
    df_vc = load_data(VALUECHAIN_CSV_PATH)
    df_sys = load_data(SYSTEM_CSV_PATH)
    df_mapping = load_data(MAPPING_CSV_PATH)
    
    # 선택된 Industry의 데이터 필터링
    industry_vc = df_vc[df_vc["Industry"] == selected_industry].copy() if not df_vc.empty else pd.DataFrame()
    industry_sys = df_sys[df_sys["Industry"] == selected_industry].copy() if not df_sys.empty else pd.DataFrame()
    industry_mapping = df_mapping[df_mapping["Industry"] == selected_industry].copy() if not df_mapping.empty else pd.DataFrame()
    
    # 데이터 미리보기
    with st.expander("📋 System Architecture Data Preview", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Value Chain Activities**")
            if not industry_vc.empty:
                vc_display = industry_vc[["Activity_Seq", "Activity_Type", "Activity", "Activity_Kor"]].copy()
                vc_display.columns = ["Seq", "Type", "Activity (EN)", "Activity (KR)"]
                st.dataframe(vc_display, use_container_width=True, hide_index=True)
            else:
                st.info("No Value Chain data")
        
        with col2:
            st.markdown("**Systems**")
            if not industry_sys.empty:
                sys_display = industry_sys[["System_Seq", "System", "System_Kor"]].copy()
                sys_display.columns = ["Seq", "System (EN)", "System (KR)"]
                st.dataframe(sys_display, use_container_width=True, hide_index=True)
            else:
                st.info("No System data")
        
        with col3:
            st.markdown("**Activity-System-File Mapping**")
            if not industry_mapping.empty:
                mapping_display = industry_mapping[["Activity", "System", "FileName"]].copy()
                mapping_display.columns = ["Activity", "System", "File Name"]
                st.dataframe(mapping_display, use_container_width=True, hide_index=True)
            else:
                st.info("No Mapping data")
    
    # Cloud 환경 체크
    if is_cloud_env():
        if mode == "Summary":
            show_sample_image("sample_SysArch_Summary.png", f"System Architecture (Summary): {selected_industry} (Sample)")
        else:
            show_sample_image("sample_SysArch_Detailed.png", f"System Architecture (Detail): {selected_industry} (Sample)")
        return
    
    try:
        graph = create_system_architecture_diagram(selected_industry, mode=mode)
        if graph:
            file_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            path = IMAGE_DIR / f"SysArch_{mode}_{selected_industry}_{file_time}"
            graph.attr(dpi='300')
            graph.render(str(path), format='png', cleanup=True)
            actual_png_filepath = IMAGE_DIR / f"{path.name}.png"
            
            if actual_png_filepath.exists():
                st.image(str(actual_png_filepath), use_container_width=True)
                
                # 다운로드 버튼
                with open(actual_png_filepath, "rb") as f:
                    png_data = f.read()
                if png_data:
                    st.download_button(
                        f"📥 {mode} 이미지 다운로드",
                        png_data,
                        file_name=f"SysArch_{mode}_{selected_industry}.png",
                        mime="image/png",
                        key=f"sys_{mode.lower()}_download"
                    )
        else:
            st.warning("⚠️ Diagram을 생성할 수 없습니다. 매핑 데이터를 확인해주세요.")
            
    except Exception as e:
        st.error(f"❌ Diagram 생성 중 오류가 발생했습니다: {e}")
        st.info("Graphviz가 설치되어 있고 PATH에 추가되었는지 확인해주세요.")

def valuechain_file_tab(selected_industry):
    """Value Chain & File Diagram 탭"""
    st.markdown(f"### 📁 Value Chain & File Diagram: **{selected_industry}**")
    st.markdown("##### Value Chain Activity와 FileName을 직접 연결하는 구성도를 표시합니다.")
    
    # 데이터 로드
    df_vc = load_data(VALUECHAIN_CSV_PATH)
    df_mapping = load_data(MAPPING_CSV_PATH)
    
    # 선택된 Industry의 데이터 필터링
    industry_vc = df_vc[df_vc["Industry"] == selected_industry].copy() if not df_vc.empty else pd.DataFrame()
    industry_mapping = df_mapping[df_mapping["Industry"] == selected_industry].copy() if not df_mapping.empty else pd.DataFrame()
    
    # 데이터 미리보기
    with st.expander("📋 Value Chain & File Data Preview", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Value Chain Activities**")
            if not industry_vc.empty:
                vc_display = industry_vc[["Activity_Seq", "Activity_Type", "Activity", "Activity_Kor"]].copy()
                vc_display.columns = ["Seq", "Type", "Activity (EN)", "Activity (KR)"]
                st.dataframe(vc_display, use_container_width=True, hide_index=True)
            else:
                st.info("No Value Chain data")
        
        with col2:
            st.markdown("**Activity-File Mapping**")
            if not industry_mapping.empty:
                # Activity와 FileName만 표시 (System은 제외)
                mapping_display = industry_mapping[["Activity", "FileName"]].drop_duplicates().copy()
                mapping_display.columns = ["Activity", "File Name"]
                st.dataframe(mapping_display, use_container_width=True, hide_index=True)
            else:
                st.info("No Mapping data")
    
    # Cloud 환경 체크
    if is_cloud_env():
        show_sample_image("sample_ValueChain_File.png", f"Value Chain & File Diagram: {selected_industry} (Sample)")
        return
    
    try:
        graph = create_valuechain_file_diagram(selected_industry)
        if graph:
            file_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            path = IMAGE_DIR / f"ValueChain_File_{selected_industry}_{file_time}"
            graph.attr(dpi='300')
            graph.render(str(path), format='png', cleanup=True)
            actual_png_filepath = IMAGE_DIR / f"{path.name}.png"
            
            if actual_png_filepath.exists():
                st.image(str(actual_png_filepath), use_container_width=True)
                
                # 다운로드 버튼
                with open(actual_png_filepath, "rb") as f:
                    png_data = f.read()
                if png_data:
                    st.download_button(
                        "📥 Value Chain & File 이미지 다운로드",
                        png_data,
                        file_name=f"ValueChain_File_{selected_industry}.png",
                        mime="image/png",
                        key="vc_file_download"
                    )
        else:
            st.warning("⚠️ Diagram을 생성할 수 없습니다. 매핑 데이터를 확인해주세요.")
            
    except Exception as e:
        st.error(f"❌ Diagram 생성 중 오류가 발생했습니다: {e}")
        st.info("Graphviz가 설치되어 있고 PATH에 추가되었는지 확인해주세요.")

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Value Chain & System Architecture Diagram")
    st.markdown("##### Industry별 Value Chain Diagram과 System Architecture Diagram을 생성합니다.")
    
    # 데이터 로드
    df_vc = load_valuechain_data()
    if df_vc.empty:
        return
    
    # Industry 선택
    st.divider()
    st.markdown("### 1️⃣ Select Industry")
    
    industries = get_all_industries(df_vc)
    if not industries:
        st.warning("⚠️ 등록된 Industry가 없습니다.")
        st.info("먼저 'Value Chain & System Management' 페이지에서 Value Chain을 정의해주세요.")
        return
    
    selected_industry = st.selectbox(
        "Registered Industry List",
        options=["-- Select --"] + industries,
        index=0,
        key="industry_selection"
    )
    
    if selected_industry == "-- Select --":
        st.info("📌 Industry를 선택하면 Diagram이 생성됩니다.")
        return
    
    st.divider()
    
    # 탭 생성
    tab_vc, tab_sys_summary, tab_sys_detail, tab_vc_file = st.tabs([
        "📊 Value Chain Diagram",
        "🏗️ System Architecture",
        "🔍 System Architecture Detail",
        "📁 Value Chain & File"
    ])
    
    with tab_vc:
        value_chain_tab(df_vc, selected_industry)
    
    with tab_sys_summary:
        system_architecture_tab(selected_industry, mode="Summary")
    
    with tab_sys_detail:
        system_architecture_tab(selected_industry, mode="Detailed")
    
    with tab_vc_file:
        valuechain_file_tab(selected_industry)

if __name__ == "__main__":
    main()

