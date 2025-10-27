import pandas as pd
import re
import os

def clean_company_name(name: str) -> str:
    if pd.isna(name):
        return ""

    # 1. 괄호 안의 내용 제거 (괄호는 남기지 않음)
    # 모든 괄호 종류 포함: (), [], {}, <>, 全角 포함
    name = re.sub(r'\((.*?)\)', '', name)
    name = re.sub(r'\[(.*?)\]', '', name)
    name = re.sub(r'\{(.*?)\}', '', name)
    name = re.sub(r'\<(.*?)\>', '', name)
    name = re.sub(r'（.*?）', '', name)
    name = re.sub(r'【.*?】', '', name)
    name = re.sub(r'〈.*?〉', '', name)
    name = re.sub(r'《.*?》', '', name)
    name = re.sub(r'「.*?」', '', name)
    name = re.sub(r'『.*?』', '', name)

    # 2. 법인 접두사/접미사 제거
    patterns = [
        r'주식회사',
        r'㈜', r'㈔', r'㈝', r'㈞',
        r'\(주\)', r'\(유\)', r'\(사\)', r'\(재\)', r'\(비\)', r'\(A\)', r'\(B\)', r'\(C\)',
        r'<주>', r'【주】',
        r'^\s*주', r'^\s*\(.*?주.*?\)',
    ]
    for pattern in patterns:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    # 3. 남아있는 괄호 조각 제거
    name = re.sub(r'[()\[\]{}<>\[\]（）【】〈〉《》「」『』]', '', name)

    # ✅ 4. 특수문자로 시작하는 경우 첫 문자 제거
    name = re.sub(r'^[\.\．/_`]+', '', name)

    # 4. 공백 정리
    name = re.sub(r'\s+', ' ', name).strip()

    return name

def process_company_names_from_csv(filepath: str) -> pd.DataFrame:
    """CSV 파일에서 '사업장명' 컬럼을 정제하여 원본과 함께 반환"""
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='cp949')

    if '사업장명' not in df.columns:
        raise ValueError("CSV 파일에 '사업장명' 컬럼이 없습니다.")
    
    df['사업장명_정제'] = df['사업장명'].apply(clean_company_name)
    return df[['사업장명', '사업장명_정제']]


if __name__ == "__main__":
    filepath = r"C:\projects\myproject\QDQM\QDQM_Master_Code\5_Reference_Code\Source7\사업장명.csv"
    w_filepath = filepath.replace(".csv", "_정제.csv")
    
    df = process_company_names_from_csv(filepath)
    df.to_csv(w_filepath, index=False, encoding='utf-8-sig')

    final_df = df['사업장명_정제'].drop_duplicates()
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.rename('사업장명')
    # 길이가 15 이상인 사업장명 삭제
    final_df = final_df[final_df.str.len() < 15]
    final_df = final_df.sort_values(ascending=True)
    final_df.to_csv(w_filepath.replace('.csv', '_final.csv'), index=False, encoding='utf-8-sig')

    print(f"[✔] 정제 완료: {os.path.basename(w_filepath)}")
