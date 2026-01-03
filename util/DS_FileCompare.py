import pandas as pd
import os

def analyze_results(base_path):
    print("🚀 결과 분석 및 비교 시작...")
    
    # 1. 파일 경로 설정
    file_format_path = os.path.join(base_path, "FileFormat.csv")
    dq_result_path = os.path.join(base_path, "DQ_Result_Final.csv")

    # 파일 존재 확인
    if not os.path.exists(file_format_path) or not os.path.exists(dq_result_path):
        print(f"❌ 에러: 분석할 파일이 {base_path}에 존재하지 않습니다.")
        return

    # 2. 데이터 로드
    df_format = pd.read_csv(file_format_path, encoding='utf-8-sig')
    df_dq = pd.read_csv(dq_result_path, encoding='utf-8-sig')

    # 컬럼명 정리 (공백 제거 등)
    df_format.columns = df_format.columns.str.strip()
    df_dq.columns = df_dq.columns.str.strip()

    print(f"📊 로드 완료: FileFormat({len(df_format)}건), DQ_Result({len(df_dq)}건)")

    # 3. 누락된 컬럼 분석 (FileName + ColumnName 조합)
    # FileFormat은 전체 컬럼 리스트를 가지고 있음
    format_sets = set(zip(df_format['FileName'], df_format['ColumnName']))
    dq_sets = set(zip(df_dq['FileName'], df_dq['ColumnName']))

    missing_in_dq = format_sets - dq_sets
    
    print("\n" + "="*50)
    print(f"🔍 [누락 분석] DQ_Result_Final.csv에 없는 컬럼 (총 {len(missing_in_dq)}개)")
    print("="*50)
    
    if missing_in_dq:
        missing_df = pd.DataFrame(list(missing_in_dq), columns=['FileName', 'ColumnName'])
        missing_df = missing_df.sort_values(by=['FileName', 'ColumnName'])
        
        # 파일별 누락 개수 요약
        summary = missing_df.groupby('FileName').count()
        print(summary)
        
        # 상세 리스트 출력 (상위 20개만)
        print("\n--- 상세 누락 목록 (일부) ---")
        print(missing_df.head(20).to_string(index=False))
        
        # 결과 저장
        missing_df.to_csv(os.path.join(base_path, "Missing_Columns_Report.csv"), index=False, encoding='utf-8-sig')
        print(f"\n✅ 누락 목록 저장됨: Missing_Columns_Report.csv")
    else:
        print("✅ 모든 컬럼이 DQ 결과에 포함되어 있습니다.")

    # 4. 동일 컬럼 데이터 비교 (예: DataType, DetailDataType 불일치 확인)
    print("\n" + "="*50)
    print("🧪 [정합성 비교] FileFormat vs DQ_Result (DataType 등)")
    print("="*50)

    # 두 데이터프레임을 FileName, ColumnName 기준으로 병합
    comparison_df = pd.merge(
        df_format[['FileName', 'ColumnName', 'DetailDataType', 'Format']],
        df_dq[['FileName', 'ColumnName', 'DetailDataType', 'Format']],
        on=['FileName', 'ColumnName'],
        suffixes=('_Format', '_DQ'),
        how='inner'
    )

    # 타입이 서로 다르게 정의된 경우 필터링
    mismatch = comparison_df[comparison_df['DetailDataType_Format'] != comparison_df['DetailDataType_DQ']]

    if not mismatch.empty:
        print(f"⚠️ DetailDataType 불일치 발견: {len(mismatch)}건")
        print(mismatch[['FileName', 'ColumnName', 'DetailDataType_Format', 'DetailDataType_DQ']].head(10))
    else:
        print("✅ 모든 동일 컬럼의 상세 데이터 타입이 일치합니다.")

if __name__ == "__main__":
    # 분석 경로 설정
    BASE_DIR = r"C:\projects\myproject\QDQM\DataSense\DS_Output"
    analyze_results(BASE_DIR)