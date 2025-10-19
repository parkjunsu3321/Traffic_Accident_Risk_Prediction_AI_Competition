#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List, Sequence
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sklver

# -------------------\
# 고정 경로
# -------------------\
DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_DIR = "model"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
A_MODEL_PATH = os.path.join(MODEL_DIR, "model_A.joblib")
B_MODEL_PATH = os.path.join(MODEL_DIR, "model_B.joblib")
A_PREPROC_PATH = os.path.join(MODEL_DIR, "preproc_A.joblib")
B_PREPROC_PATH = os.path.join(MODEL_DIR, "preproc_B.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

RANDOM_STATE = 42

# -------------------\
# 실행 옵션
# -------------------\
USE_CALIBRATION = True          
CALIB_METHOD = "isotonic"      
CALIB_CV = 3                    
TEST_SIZE_FOR_LOG = 0.1         

ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777)

BASE_HGB_PARAMS = dict(
    learning_rate=0.06,          
    max_iter=300,               
    max_depth=None,
    max_leaf_nodes=63,           
    min_samples_leaf=20,        
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=25,
    class_weight="balanced",    
)

# -------------------\
# 보조 유틸
# -------------------\
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_index_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx

def read_feature_files(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    A_df = pd.read_csv(os.path.join(DATA_DIR, split, "A.csv"))
    B_df = pd.read_csv(os.path.join(DATA_DIR, split, "B.csv"))
    return A_df, B_df

def separate_num_cat(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in cols if str(df[c].dtype) in ("object", "category")]
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordenc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preproc

def _mk_calibrator(base_clf):
    # sklearn 1.4+ : estimator, 1.3- : base_estimator
    try:
        major, minor, *_ = map(int, sklver.split(".")[:2])
    except Exception:
        major, minor = 1, 4
    kw = dict(method=CALIB_METHOD, cv=CALIB_CV)
    if (major, minor) >= (1, 4):
        return CalibratedClassifierCV(estimator=base_clf, **kw)
    else:
        return CalibratedClassifierCV(base_estimator=base_clf, **kw)

def maybe_calibrate(base_clf, X_train, y_train):
    if not USE_CALIBRATION:
        return base_clf
    calib = _mk_calibrator(base_clf)
    calib.fit(X_train, y_train)
    return calib

def add_rowwise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2

def build_model(seed: int) -> HistGradientBoostingClassifier:
    params = BASE_HGB_PARAMS.copy()
    params["random_state"] = seed
    return HistGradientBoostingClassifier(**params)

class AvgProbaEnsemble:
    def __init__(self, models: List):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

# -------------------\
# [신규] 특징 공학
# -------------------\
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA 분석 기반의 파생 변수 생성
    - Age (나이) 수치화: '30a' -> 32, '30b' -> 37 (PDF의 대표값 기준)
    - TestDate (검사일) 분해: Year, Month
    - PrimaryKey (운전자) 기반 변수:
        - TestCount: 운전자별 총 검사 횟수
        - TestSequence: 운전자별 검사 순서 (시간 기반)
        - FirstTestAge: 운전자별 첫 검사 나이
        - TimeSinceFirstTest_yr: 운전자별 첫 검사로부터 경과 시간(연)
    - Row-wise NA (결측치) 변수
    """
    df_proc = df.copy()

    # 1. Age (나이) 수치화
    # PDF의 히트맵/분포도에서 사용하는 대표값을 기준으로 매핑 (예: 20a->22, 20b->27)
    age_map = {f"{i}{s}": (i + 2 if s == 'a' else i + 7) for i in range(10, 90, 10) for s in ['a', 'b']}
    # '10a', '10b' 등 데이터에 없을 수 있는 값도 포함
    age_map.update({
        '10a': 12, '10b': 17, '90a': 92, '90b': 97, '100a': 102
    })
    df_proc['Age_numeric'] = df_proc['Age'].map(age_map).astype(float)

    # 2. TestDate (검사일) 분해
    # YYYYMM 형식을 수치로 변환
    df_proc['TestDate_num'] = pd.to_numeric(df_proc['TestDate'], errors='coerce')
    df_proc['TestYear'] = (df_proc['TestDate_num'] // 100).astype(float)
    df_proc['TestMonth'] = (df_proc['TestDate_num'] % 100).astype(float)

    # 3. PrimaryKey (운전자) 기반 변수
    # PrimaryKey와 TestDate_num 기준으로 정렬
    df_sorted = df_proc.sort_values(by=['PrimaryKey', 'TestDate_num'])
    
    # TestCount: 운전자별 총 검사 횟수
    df_proc['TestCount'] = df_sorted.groupby('PrimaryKey')['Test_id'].transform('count')
    
    # TestSequence: 운전자별 검사 순서 (1부터 시작)
    df_proc['TestSequence'] = df_sorted.groupby('PrimaryKey').cumcount() + 1
    
    # FirstTestAge: 운전자별 첫 검사 나이
    df_proc['FirstTestAge'] = df_sorted.groupby('PrimaryKey')['Age_numeric'].transform('min')
    
    # TimeSinceFirstTest_yr: 첫 검사로부터 경과 시간(연)
    df_proc['FirstTestYear'] = df_sorted.groupby('PrimaryKey')['TestYear'].transform('min')
    df_proc['TimeSinceFirstTest_yr'] = df_proc['TestYear'] - df_proc['FirstTestYear']

    # 4. Row-wise NA (결측치) 변수 (기존 baseline 로직)
    # 원본 A..., B... 피처 컬럼들만 선택
    base_feature_cols = [c for c in df_proc.columns if c.startswith("A") or c.startswith("B")]
    df_proc = add_rowwise_features(df_proc, base_feature_cols)
    
    print("[Global FE] 파생 변수 생성 완료. (Age_numeric, TestCount, TestSequence 등)")
    
    # 원본 범주형/날짜 컬럼 및 중간 계산 컬럼은 제거
    df_proc = df_proc.drop(columns=['Age', 'TestDate', 'TestDate_num', 'FirstTestYear'], errors='ignore')
    
    return df_proc


# -------------------\
# 학습/로드 (수정됨)
# -------------------\
def fit_or_load(
    df_feat: pd.DataFrame,        # [수정] 전체 *처리된* 학습 피처
    df_idx: pd.DataFrame,         # [수정] A 또는 B로 *필터링된* 인덱스
    label_col: str,
    model_path: str,
    preproc_path: str,
    which: str
):
    key = "Test_id"
    
    # [수정] df_idx (A 또는 B) 기준으로 df_feat (전체)에서 필요한 행을 merge
    assert key in df_feat.columns, f"{which}: '{key}' not found in processed features"
    if len(df_idx) and label_col in df_idx.columns:
        # 학습 경로
        df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
        
        # [수정] drop_cols에 PrimaryKey 및 변환 전 컬럼 추가
        drop_cols = [key, label_col, "PrimaryKey"] + \
                    (["Test"] if "Test" in df.columns else [])
        
        # [수정] add_rowwise_features는 create_features에서 이미 실행했으므로 제거

        num_cols, cat_cols = separate_num_cat(df, drop_cols)
        
        # [디버깅] 선택된 컬럼 확인
        print(f"[{which}] Preprocessing: {len(num_cols)} num_cols, {len(cat_cols)} cat_cols.")
        if len(num_cols) < 5: # NA_COUNT, NA_RATIO 외에 파생변수가 최소 1개는 있어야 함
             print(f"[{which}] WARNING: Num cols are very few: {num_cols}")
        if len(cat_cols) > 0:
             print(f"[{which}] CAT cols: {cat_cols}")

        preproc = build_preprocessor(num_cols, cat_cols)

        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[label_col].astype(int).values

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=TEST_SIZE_FOR_LOG, random_state=RANDOM_STATE, stratify=y
        )

        X_tr_t = preproc.fit_transform(X_tr)
        X_val_t = preproc.transform(X_val)

        # 앙상블 학습
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd).fit(X_tr_t, y_tr)
            mdl = maybe_calibrate(base, X_tr_t, y_tr)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)

        try:
            val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
            auc = roc_auc_score(y_val, val_proba)
            brier = brier_score_loss(y_val, val_proba)
            print(f"[{which}] Holdout AUC={auc:.5f}, Brier={brier:.5f}")
        except Exception as e:
            print(f"[{which}] validation logging skipped: {e}")

        joblib.dump(preproc, preproc_path)
        joblib.dump(ensemble, model_path)
        print(f"[{which}] trained and saved → {model_path}, {preproc_path}")
        return preproc, ensemble

    # 추론 경로 (df_idx가 비어 있거나 label_col이 없는 경우)
    print(f"[{which}] loading pre-trained: {preproc_path}, {model_path}")
    preproc = joblib.load(preproc_path)
    ensemble = joblib.load(model_path)
    return preproc, ensemble

# -------------------\
# 추론 (수정됨)
# -------------------\
def predict_partition(
    df_feat: pd.DataFrame,    # [수정] 전체 *처리된* 테스트 피처
    df_idx: pd.DataFrame,     # [수정] A 또는 B로 *필터링된* 인덱스
    preproc,
    clf_or_ens, 
    which: str
) -> pd.DataFrame:
    key = "Test_id"
    
    # [수정] df_idx (A 또는 B) 기준으로 df_feat (전체)에서 필요한 행을 merge
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")

    # [수정] drop_cols에 PrimaryKey 및 변환 전 컬럼 추가
    drop_cols = [key, "PrimaryKey"] + \
                (["Test"] if "Test" in df.columns else [])

    # [수정] add_rowwise_features는 create_features에서 이미 실행했으므로 제거

    X = df.drop(columns=drop_cols, errors="ignore")
    X_t = preproc.transform(X)
    proba = np.clip(clf_or_ens.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)
    out = df_idx[[key]].copy()
    out["Label"] = proba
    out["__which__"] = which
    return out

def save_meta():
    meta = dict(
        model="HGB(3-seed soft ensemble) + OrdinalEnc + Calibration",
        # [신규] 파생 변수 정보 추가
        feature_engineering=[
            "Age_numeric (mapped from '30a' -> 32, '30b' -> 37)",
            "TestYear, TestMonth (from TestDate)",
            "TestCount (per PrimaryKey)",
            "TestSequence (per PrimaryKey, sorted by TestDate)",
            "FirstTestAge (per PrimaryKey)",
            "TimeSinceFirstTest_yr (per PrimaryKey)",
            "NA_COUNT, NA_RATIO (row-wise)"
        ],
        hgb_params=BASE_HGB_PARAMS,
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=CALIB_CV,
        sklearn_version=sklver,
        random_state=RANDOM_STATE,
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# -------------------\
# 메인 (수정됨)
# -------------------\
def main():
    t0 = time.time()
    ensure_dirs()
    save_meta() # [수정] 메타 정보 먼저 저장

    train_idx, test_idx = read_index_files()
    A_train_feat_raw, B_train_feat_raw = read_feature_files("train")
    A_test_feat_raw,  B_test_feat_raw  = read_feature_files("test")

    # [수정] 특징 공학을 위해 Train/Test 피처 데이터 통합
    train_feat_raw = pd.concat([A_train_feat_raw, B_train_feat_raw], ignore_index=True)
    test_feat_raw  = pd.concat([A_test_feat_raw,  B_test_feat_raw],  ignore_index=True)
    
    # Test_id 중복 확인 (A, B간 중복이 없어야 함)
    assert train_feat_raw["Test_id"].nunique() == len(train_feat_raw), "Train Test_id duplicates!"
    
    # is_train 플래그를 추가하여 concat
    all_feat_raw = pd.concat([
        train_feat_raw.assign(is_train=1),
        test_feat_raw.assign(is_train=0)
    ], ignore_index=True)

    # [신규] 통합된 데이터프레임에 대해 파생 변수 생성
    all_feat_processed = create_features(all_feat_raw)

    # 다시 Train / Test로 분리
    train_feat_processed = all_feat_processed[all_feat_processed['is_train'] == 1].drop(columns='is_train')
    test_feat_processed  = all_feat_processed[all_feat_processed['is_train'] == 0].drop(columns='is_train')

    # --- A 모델 학습/추론 ---
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    A_test_idx  = test_idx[test_idx["Test"] == "A"].copy()
    
    # A 학습 데이터가 있거나, 또는 A 모델 파일이 없으면 학습 경로
    need_train_A = (len(A_train_idx) > 0) and \
                   (not (os.path.exists(A_MODEL_PATH) and os.path.exists(A_PREPROC_PATH)))
    
    fit_df_idx_A = A_train_idx if need_train_A else pd.DataFrame({"Test_id": [], "Label": []})
    
    if need_train_A:
        print("[A] training path (data exists and no pre-trained weights found).")
    
    # [수정] fit_or_load에 *전체* 처리된 피처와 *A용* 인덱스를 전달
    preproc_A, clf_A = fit_or_load(
        train_feat_processed, fit_df_idx_A, "Label",
        A_MODEL_PATH, A_PREPROC_PATH, "A"
    )

    # --- B 모델 학습/추론 ---
    B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
    B_test_idx  = test_idx[test_idx["Test"] == "B"].copy()

    need_train_B = (len(B_train_idx) > 0) and \
                   (not (os.path.exists(B_MODEL_PATH) and os.path.exists(B_PREPROC_PATH)))

    fit_df_idx_B = B_train_idx if need_train_B else pd.DataFrame({"Test_id": [], "Label": []})

    if need_train_B:
        print("[B] training path (data exists and no pre-trained weights found).")
        
    preproc_B, clf_B = fit_or_load(
        train_feat_processed, fit_df_idx_B, "Label",
        B_MODEL_PATH, B_PREPROC_PATH, "B"
    )

    # --- 추론 ---
    # [수정] predict_partition에 *전체* 처리된 테스트 피처와 각 A/B 인덱스를 전달
    preds_A = predict_partition(test_feat_processed, A_test_idx, preproc_A, clf_A, "A") if len(A_test_idx) else None
    preds_B = predict_partition(test_feat_processed, B_test_idx, preproc_B, clf_B, "B") if len(B_test_idx) else None

    # --- 제출 파일 생성 ---
    if preds_A is not None and preds_B is not None:
        sub = pd.concat([preds_A, preds_B], axis=0, ignore_index=True)
    elif preds_A is not None:
        sub = preds_A.copy()
    elif preds_B is not None:
        sub = preds_B.copy()
    else:
        # test_idx에만 의존
        sub = test_idx[["Test_id"]].copy()
        sub["Label"] = 0.001

    try:
        # sample_submission 기준으로 Test_id 순서 및 누락 방지
        sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        sub_final = sample[["Test_id"]].merge(sub, on="Test_id", how="left")
        
        # 병합 과정에서 누락된 Test_id가 있다면 0.001로 채우기
        sub_final["Label"] = sub_final["Label"].fillna(0.001)
        sub_final = sub_final[["Test_id", "Label"]]
    except Exception as e:
        print(f"Sample submission merge failed ({e}), saving raw submission.")
        sub_final = sub[["Test_id", "Label"]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub_final.to_csv(SUBMISSION_PATH, index=False)

    dt = time.time() - t0
    print(f"[이이이잉] submission saved -> {SUBMISSION_PATH} | elapsed: {dt:.2f}s")

if __name__ == "__main__":
    main()