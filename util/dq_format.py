
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------------    
def Expand_Format(Source_df, Mode='Reference') -> pd.DataFrame:
    """    Source_df의 상위 3개 포맷(Format_1~3)을 행으로 펼쳐서 반환.    """
    try:

        # 1) 전처리
        s_df = Source_df.copy().rename(columns={
            'Format':        'Format_1',
            'Format2nd':     'Format_2',
            'Format3rd':     'Format_3',
            'FormatMin':     'FormatMin_1',
            'FormatMax':     'FormatMax_1',
            'FormatMedian':   'FormatMedian_1',
            'Format2ndMin':  'FormatMin_2',
            'Format2ndMax':  'FormatMax_2',
            'Format2ndMedian': 'FormatMedian_2',
            'Format3rdMin':  'FormatMin_3',
            'Format3rdMax':  'FormatMax_3',
            'Format3rdMedian': 'FormatMedian_3',
            'FormatValue':   'FormatValue_1',
            'Format2ndValue':'FormatValue_2',
            'Format3rdValue':'FormatValue_3',
            'Format(%)':     'Format(%)_1',
            'Format2nd(%)':  'Format(%)_2',
            'Format3rd(%)':  'Format(%)_3',
        })

        # 2) i=1..3 별로 분리 → 표준 컬럼명으로 통일 → 붙이기
        frames = []
        for i in (1, 2, 3):
            cols_i = [
                'FilePath', 'FileName', 'ColumnName', 'DetailDataType',  'MasterType', 'CompareLength', 'FormatCnt', 'UniqueCnt',
                f'Format_{i}', f'FormatMin_{i}', f'FormatMax_{i}', f'FormatMedian_{i}', 
                f'FormatValue_{i}', f'Format(%)_{i}'
            ]

            # 존재하는 컬럼만 선택
            available_cols = [c for c in cols_i if c in s_df.columns]
            if not available_cols:
                continue
                
            df_i = s_df[available_cols].copy()
            
            # 컬럼명 변경 (존재하는 컬럼만)
            rename_dict = {}
            if f'Format_{i}' in df_i.columns:
                rename_dict[f'Format_{i}'] = 'Format'
            if f'FormatMin_{i}' in df_i.columns:
                rename_dict[f'FormatMin_{i}'] = 'FormatMin'
            if f'FormatMax_{i}' in df_i.columns:
                rename_dict[f'FormatMax_{i}'] = 'FormatMax'
            if f'FormatMedian_{i}' in df_i.columns:
                rename_dict[f'FormatMedian_{i}'] = 'FormatMedian'
            if f'FormatValue_{i}' in df_i.columns:
                rename_dict[f'FormatValue_{i}'] = 'FormatValue'
            if f'Format(%)_{i}' in df_i.columns:
                rename_dict[f'Format(%)_{i}'] = 'Format(%)'
            
            if rename_dict:
                df_i = df_i.rename(columns=rename_dict)

            df_i['MatchNo'] = i

            # 빈/결측 포맷 제거 (Format 컬럼이 있는 경우만)
            if 'Format' in df_i.columns and not df_i.empty:
                format_series = df_i['Format']
                # Series인지 확인 (단일 컬럼 선택은 항상 Series 반환)
                if isinstance(format_series, pd.Series):
                    mask = format_series.notna() & (format_series.astype(str).str.strip() != '')
                    df_i = df_i[mask]
                # 이상 케이스: DataFrame이 반환된 경우는 스킵
                elif isinstance(format_series, pd.DataFrame):
                    continue

            # 숫자형 정리
            for col in ('FormatValue', 'Format(%)', 'CompareLength'):
                if col in df_i.columns:
                    df_i[col] = pd.to_numeric(df_i[col], errors='coerce')

            frames.append(df_i)

        if not frames:
            return pd.DataFrame(columns=[
                'FilePath', 'FileName', 'ColumnName', 'DetailDataType', 'MasterType', 'FormatCnt', 'UniqueCnt', 'MatchNo',
                'Format', 'FormatMin', 'FormatMax', 'FormatMedian', 'FormatValue', 'Format(%)', 'CompareLength'
            ])

        result_df = pd.concat(frames, ignore_index=True)

        # 정렬 & 중복 제거(선택)
        result_df = (result_df
                     .drop_duplicates()
                     .sort_values(['FilePath','FileName','ColumnName','MasterType','Format(%)'], ascending=[True, True, True, True, False])
                     .reset_index(drop=True))

        return result_df

    except Exception as e:
        print(f"전체 처리 중 오류 발생: {e}")
        raise

def Combine_Format(source_df, reference_df):
    """    source_df와 reference_df를 조합하여 반환.    """
    try:
        s_df = source_df.copy()

        except_detail_data_types = ['Time', 'Timestamp', 'Date', 'DateTime', 'DATECHAR', 'TIME', 'TIMESTAMP', 
            'DATE', 'DATETIME', 'YEAR', 'YEARMONTH', 'YYMMDD', 'LATITUDE', 'LONGITUDE', 'TEL', 'CELLPHONE', 'ADDRESS',
            'Alpha_Flag', 'Num_Flag', 'YN_Flag', 'NUM_Flag', 'KOR_Flag', 'KOR_Name']

        # DetailDataType 컬럼이 있는 경우만 필터링
        if 'DetailDataType' in s_df.columns:
            s_df = s_df[~s_df['DetailDataType'].isin(except_detail_data_types)].copy()
        r_df = reference_df.copy()
        if 'DetailDataType' in r_df.columns:
            r_df = r_df[~r_df['DetailDataType'].isin(except_detail_data_types)].copy()

        r_df = r_df[['FilePath', 'FileName', 'MasterType', 'ColumnName', 'FormatCnt', 'Format', 'FormatMin', 'FormatMax', 
                     'FormatMedian', 'FormatValue', 'Format(%)', 'UniqueCnt']].copy().rename(columns={
            'FilePath': 'MasterFilePath',
            'FileName': 'MasterFile',
            'MasterType': 'ReferenceMasterType',
            'ColumnName': 'MasterColumn',
            'FormatCnt': 'MasterFormatCnt',
            'Format': 'Format',
            'FormatMin': 'MasterMin',
            'FormatMax': 'MasterMax',
            'FormatMedian': 'MasterMedian',
            'FormatValue': 'MasterValue', 
            'Format(%)': 'Master(%)',
            'UniqueCnt': 'MasterUniqueCnt',
            # 'CompareLength': 'MasterCompareLength',
        })

        result_df = pd.merge(s_df, r_df, on=['Format'], how='left')
        result_df = result_df[result_df['MasterFile'].notna()]
        result_df = result_df.drop_duplicates(['FilePath','FileName','ColumnName','MasterFile','MasterColumn'])

        # --- 숫자형 컬럼 일괄 변환 ---
        num_cols = [
            'FormatCnt','FormatValue','UniqueCnt','CompareLength',
            'MasterFormatCnt','MasterValue','MasterUniqueCnt'
        ]
        for c in num_cols:
            if c in result_df.columns:
                result_df[c] = pd.to_numeric(result_df[c], errors='coerce')

                # 결측 기본값 (비교 안전용)
        result_df['FormatCnt']        = result_df['FormatCnt'].fillna(0)
        result_df['FormatValue']      = result_df['FormatValue'].fillna(0)
        result_df['UniqueCnt']        = result_df['UniqueCnt'].fillna(0)
        result_df['MasterFormatCnt']  = result_df['MasterFormatCnt'].fillna(0)
        result_df['MasterValue']      = result_df['MasterValue'].fillna(0)
        result_df['MasterUniqueCnt']  = result_df['MasterUniqueCnt'].fillna(0)

        # --- MasterCompareLength 계산 (MasterColumn 끝자리가 숫자면 사용, 아니면 "0") ---
        mcol_str = result_df['MasterColumn'].astype(str)
        last_char = mcol_str.str[-1].fillna('')
        result_df['MasterCompareLength'] = np.where(last_char.str.isdigit(), last_char, '0')

        # --- 플래그 계산(벡터화) ---
        # 0) 포맷 중앙값이 마스터 범위 안에 있는가
        flag0 = result_df['FormatMedian'].ge(result_df['MasterMin']) & result_df['FormatMedian'].le(result_df['MasterMax'])

        # 1) 포맷 길이가 1 초과인가 (문자열 길이 기준)
        flag1 = result_df['Format'].astype(str).str.len().gt(1)

        # 2) 소스 포맷 카운트가 마스터 기준값보다 작은가 (작아야 1)
        flag2 = result_df['FormatCnt'].lt(result_df['MasterValue'])

        # 3) 과도한 포맷 카운트(=마스터 1.5배 이상 & 5 이상)인데 MasterCompareLength가 0이면 탈락
        flag3 = ~((result_df['FormatCnt'] >= result_df['MasterFormatCnt']*1.5) & (result_df['FormatCnt'] >= 5) & (result_df['MasterCompareLength'] == '0') )
 
        flag4 = pd.Series(True, index=result_df.index)  # 4) 항상 1 (유지)
 
        flag5 = result_df['UniqueCnt'].ge(10)  # 5) 유니크가 10 미만이면 탈락

        flag6 = result_df['FormatValue'].ge(10)  # 6) 포맷값이 10 미만이면 탈락

        # 7) 소스 유니크가 마스터보다 크면서 MasterCompareLength가 0이면 탈락
        flag7 = ~( (result_df['UniqueCnt'] > result_df['MasterUniqueCnt']) & (result_df['MasterCompareLength'] == '0') )

        # 8) FilePath = MasterFilePath 이고 FileName = MasterFile 이고 ColumnName = MasterColumn 이면 탈락
        flag8 = ~( (result_df['FilePath'] == result_df['MasterFilePath']) & (result_df['FileName'] == result_df['MasterFile']) & (result_df['ColumnName'] == result_df['MasterColumn']) )

        # 최종 플래그
        result_df['Match_Flag']  = flag0.astype(int)
        result_df['Match_Flag1'] = flag1.astype(int)
        result_df['Match_Flag2'] = flag2.astype(int)
        result_df['Match_Flag3'] = flag3.astype(int)
        result_df['Match_Flag4'] = flag4.astype(int)
        result_df['Match_Flag5'] = flag5.astype(int)
        result_df['Match_Flag6'] = flag6.astype(int)
        # result_df['Match_Flag7'] = flag7.astype(int)
        result_df['Match_Flag7'] = 1
        result_df['Match_Flag8'] = flag8.astype(int)
        result_df['Final_Flag'] = (
            result_df['Match_Flag']  *
            result_df['Match_Flag2'] *
            result_df['Match_Flag3'] *
            result_df['Match_Flag4'] *
            result_df['Match_Flag5'] *
            result_df['Match_Flag6'] *
            result_df['Match_Flag7'] *
            result_df['Match_Flag8']
        )

        return result_df

    except Exception as e:
        print(f"조합 처리 중 오류 발생: {e}")
        raise