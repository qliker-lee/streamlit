import os
import requests
from bs4 import BeautifulSoup
import zipfile
import json
import csv

def download_all_zip_files():
    base_url = "https://www.geoapify.com/data-share/localities/"
    
    # 페이지의 HTML 불러오기
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # 링크 중에서 .zip으로 끝나는 파일명을 찾음
    links = soup.find_all("a", href=True)
    zip_files = [link['href'] for link in links if link['href'].endswith(".zip")]

    # 다운로드 폴더 설정
    download_folder = "geoapify_localities"
    os.makedirs(download_folder, exist_ok=True)

    # 각 .zip 파일 다운로드
    for zip_file in zip_files:
        file_url = base_url + zip_file
        file_path = os.path.join(download_folder, zip_file)
        
        print(f"다운로드 중: {file_url}")
        file_response = requests.get(file_url)
        file_response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(file_response.content)
        
        print(f"완료: {file_path}")

def extract_all_zip_files():
    folder_path = r"C:\projects\myproject\QDQM\geoapify_localities"
    print(f"folder_path: {folder_path}")
    # 폴더 경로 내의 모든 파일을 탐색
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".zip"):
            print(file_name)
            zip_path = os.path.join(folder_path, file_name)
            
            # zip 파일과 동일한 이름으로 폴더 생성
            extract_folder_name = os.path.splitext(file_name)[0]
            extract_folder_path = os.path.join(folder_path, extract_folder_name)
            
            # 폴더가 없으면 생성
            if not os.path.exists(extract_folder_path):
                os.makedirs(extract_folder_path)
            
            # 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder_path)
            
            print(f"압축 풀기 완료: {zip_path} -> {extract_folder_path}")

def convert_ndjson_to_csv():
    root_dir = r"C:\projects\myproject\QDQM\geoapify_localities"

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ndjson"):
                ndjson_path = os.path.join(root, file)
                csv_name = os.path.splitext(file)[0] + ".csv"
                csv_path = os.path.join(root, csv_name)

                rows = []
                header = []

                # ndjson 파일 읽어서 JSON 객체로 파싱
                with open(ndjson_path, "r", encoding="utf-8-sig") as ndjson_file:
                    for line_number, line in enumerate(ndjson_file):
                        data = json.loads(line.strip())
                        
                        # 첫 줄에서 헤더 설정
                        if line_number == 0:
                            header = list(data.keys())
                            rows.append(header)
                        
                        # 헤더 순서대로 데이터 배열 생성
                        row = [data.get(column, "") for column in header]
                        rows.append(row)

                # CSV 파일로 저장
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(rows)
# 함수 호출 예시
if __name__ == "__main__":
    download_all_zip_files()
    extract_all_zip_files()
    convert_ndjson_to_csv()
