import pandas as pd
from pathlib import Path

def generate_codemapping_verify(input_path, output_path):
    # 1. 데이터 로드
    df = pd.read_csv(input_path)
    
    # 필요한 컬럼만 추출 (FileName, ColumnName)
    base_df = df[['FileName', 'ColumnName']].drop_duplicates().copy()
    
    # 2. 동일 ColumnName을 기준으로 Self-Join 수행 (조합 생성)
    # 동일 컬럼명을 가진 서로 다른 파일들의 쌍을 만듭니다.
    verify_df = pd.merge(base_df, base_df, on='ColumnName', suffixes=('', '_n'))
    
    # 자기 자신과의 조합 제거 (FileName == FileName_n 제외)
    verify_df = verify_df[verify_df['FileName'] != verify_df['FileName_n']]
    
    # 결과 컬럼명 정리
    verify_df = verify_df.rename(columns={
        'FileName': 'FileName',
        'ColumnName': 'ColumnName',
        'FileName_n': 'CodeFile_n',
        'ColumnName_n': 'CodeColumn_n'
    })

    # 3. Check 컬럼 로직 (이미 매핑 데이터에 존재하는지 확인)
    # 원본 데이터에 (FileName, CodeFile_n, CodeColumn_n) 같은 구조가 있다면 1, 아니면 0
    # 여기서는 '동일 컬럼명'을 기준으로 조합을 생성했으므로, 
    # 기본적으로 생성된 모든 조합에 대해 체크 로직을 수행합니다.
    
    # 중복 조합 제거 (A-B와 B-A는 동일하므로 하나만 남김)
    verify_df['temp_key'] = verify_df.apply(lambda x: "-".join(sorted([x['FileName'], x['CodeFile_n']])), axis=1)
    verify_df = verify_df.drop_duplicates(subset=['temp_key', 'ColumnName']).drop(columns=['temp_key'])

    # 4. Check 값 설정
    # 요청하신 조건에 따라, 동일한 컬럼명이 존재하는 파일 쌍이 발견되었으므로 기본적으로 1을 부여하거나
    # 특정 조건(예: 이미 다른 매핑 테이블에 존재)을 비교할 수 있습니다. 
    # 여기서는 생성된 모든 유효 조합에 대해 '동일 물리명 존재'를 의미하는 1을 넣습니다.
    verify_df['Check'] = 1

    # 5. 결과 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        verify_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ 결과 저장 중 오류: {e}")
        print(verify_df)
        return None

    print(verify_df)
    return verify_df

# 실행 설정
INPUT_FILE = Path("DS_Output/CodeMapping.csv")
OUTPUT_FILE = Path("DS_Output/CodeMapping_Verify.csv")

if INPUT_FILE.exists():
    result_df = generate_codemapping_verify(INPUT_FILE, OUTPUT_FILE)
    print(f"✅ 검증 파일 생성 완료: {OUTPUT_FILE}")
    print(f"총 생성된 조합 건수: {len(result_df)}건")
else:
    print("❌ 원본 CodeMapping.csv 파일을 찾을 수 없습니다.")