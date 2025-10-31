import pandas as pd
import os

DATA_DIR = "data"
TRAIN_IDX_PATH = os.path.join(DATA_DIR, "train.csv")
A_FEAT_PATH = os.path.join(DATA_DIR, "train", "A.csv")
B_FEAT_PATH = os.path.join(DATA_DIR, "train", "B.csv")

print("Loading files to check PrimaryKey overlap...")

try:
    # 1. train.csv 로드 (Test_id, Test 컬럼)
    train_idx = pd.read_csv(TRAIN_IDX_PATH)
    
    # 2. train/A.csv와 train/B.csv 로드 (Test_id, PrimaryKey 컬럼)
    #    submit_code_v12.py의 로직과 동일하게 두 파일을 합칩니다.
    a_feat = pd.read_csv(A_FEAT_PATH)
    b_feat = pd.read_csv(B_FEAT_PATH)
    
    # PrimaryKey와 Test_id만 추출하여 하나로 합치기
    all_train_feat_mini = pd.concat([
        a_feat[['Test_id', 'PrimaryKey']],
        b_feat[['Test_id', 'PrimaryKey']]
    ], ignore_index=True)
    
    # 3. (Test, Test_id) 정보와 (PrimaryKey, Test_id) 정보 합치기
    #    결과: (Test_id, Test, PrimaryKey)
    merged_data = train_idx[['Test_id', 'Test']].merge(
        all_train_feat_mini,
        on='Test_id',
        how='left'
    )
    
    if merged_data['PrimaryKey'].isna().any():
        print("WARNING: Some Test_ids in train.csv did not match A.csv/B.csv")
        merged_data = merged_data.dropna(subset=['PrimaryKey'])

    print("Data merged. Analyzing overlap...")

    # 4. PrimaryKey로 그룹화하여, 각 Key가 어떤 Test 타입(A, B)을 가지고 있는지 확인
    grouped = merged_data.groupby('PrimaryKey')['Test'].apply(set)

    # 5. Test 타입으로 {'A', 'B'} 둘 다 가진 Key(운전자) 찾기
    overlapping_keys = grouped[grouped.apply(lambda s: 'A' in s and 'B' in s)]
    num_overlap = len(overlapping_keys)

    print("-" * 40)
    print(f"Total unique PrimaryKeys in train: {len(grouped)}")
    print(f"Unique PrimaryKeys with ONLY 'A': {len(grouped[grouped.apply(lambda s: s == {'A'})])}")
    print(f"Unique PrimaryKeys with ONLY 'B': {len(grouped[grouped.apply(lambda s: s == {'B'})])}")
    print(f"Overlap (A and B) PrimaryKeys: {num_overlap}")
    print("-" * 40)

    if num_overlap > 0:
        print(">>> 결과: YES")
        print(f"총 {num_overlap}명의 운전자가 'A검사'와 'B검사' 기록을 모두 가지고 있습니다.")
        print("따라서 B검사 row에 A검사의 시계열/통계 피처를 생성하는 것이 유효하며,")
        print("이를 삭제하는 로직이 B모델 성능 저하의 원인일 가능성이 매우 높습니다.")
    else:
        print(">>> 결과: NO")
        print("A검사와 B검사 기록이 겹치는 운전자가 없습니다.")
        print("이 경우, 제 이전 분석(A 피처 삭제)은 B모델 성능 저하의 원인이 아닙니다.")

except FileNotFoundError as e:
    print(f"ERROR: 파일을 찾을 수 없습니다. {e.filename}")
    print("스크립트가 submit_code_v12.py와 동일한 위치에 있고,")
    print("하위 폴더에 'data/train.csv', 'data/train/A.csv', 'data/train/B.csv'가 있는지 확인하세요.")
except KeyError as e:
    print(f"ERROR: {e} 컬럼을 찾을 수 없습니다. 파일 내용을 확인하세요.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")