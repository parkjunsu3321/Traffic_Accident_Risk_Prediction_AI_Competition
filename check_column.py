import pandas as pd
import numpy as np
import os

# ----- 경로 설정 -----
# A.csv와 B.csv가 data/train/ 폴더 안에 있다고 가정합니다.
# 경로가 다르다면 이 부분을 수정하세요.
DATA_DIR = "data"
A_TRAIN_PATH = os.path.join(DATA_DIR, "train", "A.csv")
B_TRAIN_PATH = os.path.join(DATA_DIR, "train", "B.csv")

# ----- Pandas 출력 옵션 설정 -----
# 모든 컬럼이 보이도록 설정합니다.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(f"--- 1. A.csv 파일 검사 시작 ---")
print(f"경로: {A_TRAIN_PATH}")

try:
    # low_memory=False 옵션은 DtypeWarning을 방지하고 
    # 대용량 파일에서 컬럼 타입을 더 정확하게 추측하는 데 도움이 됩니다.
    df_a = pd.read_csv(A_TRAIN_PATH, low_memory=False)
    
    print("\n[ A.csv Info ]")
    # .info()는 컬럼명, non-null 개수, 데이터 타입을 보여줍니다.
    df_a.info()
    
    print("\n[ A.csv Head (상위 5개 행) ]")
    # .head()는 실제 데이터 예시를 보여줍니다.
    print(df_a.head())
    
except FileNotFoundError:
    print(f"!!! 오류: '{A_TRAIN_PATH}'에서 A.csv 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"A.csv 로드 중 알 수 없는 오류 발생: {e}")

print("\n" + "="*50 + "\n")

print(f"--- 2. B.csv 파일 검사 시작 ---")
print(f"경로: {B_TRAIN_PATH}")

try:
    df_b = pd.read_csv(B_TRAIN_PATH, low_memory=False)
    
    print("\n[ B.csv Info ]")
    df_b.info()
    
    print("\n[ B.csv Head (상위 5개 행) ]")
    print(df_b.head())

except FileNotFoundError:
    print(f"!!! 오류: '{B_TRAIN_PATH}'에서 B.csv 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"B.csv 로드 중 알 수 없는 오류 발생: {e}")

print("\n--- 검사 완료 ---")