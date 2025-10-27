import streamlit as st

# 공통 KPI 표시 함수
def create_metric_card(value, label, color="#0072B2"):
    """메트릭 카드 HTML 생성"""
    return f"""
        <div style="text-align: center; padding: 1.2rem; background-color: #FFFFFF; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: {color}; font-size: 2em; font-weight: bold;">{value}</div>
            <div style="color: #404040; font-size: 1em; margin-top: 0.5rem;">{label}</div>
        </div>
    """

def display_kpi_metrics(summary, metric_colors, title):
    """KPI 메트릭 표시 공통 함수"""
    st.markdown(f"### {title}")
    
    # 메트릭 표시
    cols = st.columns(len(summary))
    for col, (key, value) in zip(cols, summary.items()):
        color = metric_colors.get(key, "#0072B2")  # 기본 색상
        col.markdown(create_metric_card(value, key, color), unsafe_allow_html=True)

