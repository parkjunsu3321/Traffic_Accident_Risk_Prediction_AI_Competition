#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List, Sequence
import numpy as np
import pandas as pd
import joblib
import optuna  # Optuna 임포트

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
# from sklearn.ensemble import HistGradientBoostingClassifier <- [제거]
from catboost import CatBoostClassifier # [신규] CatBoost 임포트
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

OPTUNA_N_TRIALS = 30  # Optuna 탐색 횟수
optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna 로그 줄이기

ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777)

# [신규] Optuna 튜닝 결과를 저장할 전역 변수
BEST_CAT_PARAMS = {}

# [제거] BASE_HGB_PARAMS (CatBoost는 Optuna로 파라미터를 찾음)

# -------------------\
# 보조 유틸 (변경 없음)
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
            ("num", numeric_pipe, num_cols), # 수치형 먼저
            ("cat", categorical_pipe, cat_cols), # 범주형 나중
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preproc

def _mk_calibrator(base_clf):
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

# [제거] build_model (HGB 전용 함수)

class AvgProbaEnsemble:
    def __init__(self, models: List):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

# -------------------\
# [신규] 특징 공학 (변경 없음)
# -------------------\
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy()

    # 1. Age (나이) 수치화
    age_map = {f"{i}{s}": (i + 2 if s == 'a' else i + 7) for i in range(10, 90, 10) for s in ['a', 'b']}
    age_map.update({
        '10a': 12, '10b': 17, '90a': 92, '90b': 97, '100a': 102
    })
    df_proc['Age_numeric'] = df_proc['Age'].map(age_map).astype(float)

    # 2. TestDate (검사일) 분해
    df_proc['TestDate_num'] = pd.to_numeric(df_proc['TestDate'], errors='coerce')
    df_proc['TestYear'] = (df_proc['TestDate_num'] // 100).astype(float)
    df_proc['TestMonth'] = (df_proc['TestDate_num'] % 100).astype(float)

    # 3. PrimaryKey (운전자) 기반 변수
    df_sorted = df_proc.sort_values(by=['PrimaryKey', 'TestDate_num'])
    df_proc['TestCount'] = df_sorted.groupby('PrimaryKey')['Test_id'].transform('count')
    df_proc['TestSequence'] = df_sorted.groupby('PrimaryKey').cumcount() + 1
    df_proc['FirstTestAge'] = df_sorted.groupby('PrimaryKey')['Age_numeric'].transform('min')
    df_proc['FirstTestYear'] = df_sorted.groupby('PrimaryKey')['TestYear'].transform('min')
    df_proc['TimeSinceFirstTest_yr'] = df_proc['TestYear'] - df_proc['FirstTestYear']

    # 4. Row-wise NA (결측치) 변수 (기존 baseline 로직)
    base_feature_cols = [c for c in df_proc.columns if c.startswith("A") or c.startswith("B")]
    df_proc = add_rowwise_features(df_proc, base_feature_cols)
    
    print("[Global FE] 파생 변수 생성 완료. (Age_numeric, TestCount, TestSequence 등)")
    
    df_proc = df_proc.drop(columns=['Age', 'TestDate', 'TestDate_num', 'FirstTestYear'], errors='ignore')
    
    return df_proc


# -------------------\
# 학습/로드 (CatBoost로 수정)
# -------------------\
def fit_or_load(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str,
    model_path: str,
    preproc_path: str,
    which: str
):
    global BEST_CAT_PARAMS # [수정] CatBoost 파라미터 저장용
    key = "Test_id"
    
    if len(df_idx) and label_col in df_idx.columns:
        # --- 학습 경로 ---
        df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
        
        drop_cols = [key, label_col, "PrimaryKey"] + \
                    (["Test"] if "Test" in df.columns else [])
        
        num_cols, cat_cols = separate_num_cat(df, drop_cols)
        
        print(f"[{which}] Preprocessing: {len(num_cols)} num_cols, {len(cat_cols)} cat_cols.")
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
        
        # [신규] CatBoost에 범주형 피처 인덱스 알려주기
        # build_preprocessor에서 "num" 파이프가 먼저, "cat" 파이프가 나중임
        cat_feature_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        print(f"[{which}] CatBoost cat_features indices found: {len(cat_feature_indices)}")

        # --- [신규] CatBoost용 Optuna 하이퍼파라미터 탐색 ---
        print(f"[{which}] Running Optuna for CatBoost...")

        def objective_catboost(trial):
            # 1. CatBoost 파라미터 탐색 공간 정의
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            }
            
            # 2. 공통 파라미터 설정
            common_params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': RANDOM_STATE,
                'verbose': 0,
                'early_stopping_rounds': 50, # 조기 종료
                'task_type': 'CPU', # 대회 환경
                'auto_class_weights': 'Balanced' # HGB의 class_weight='balanced'와 동일
            }
            params.update(common_params)
            
            model = CatBoostClassifier(**params)
            
            # 3. 학습 및 평가
            model.fit(
                X_tr_t, y_tr,
                eval_set=[(X_val_t, y_val)],
                cat_features=cat_feature_indices # [중요] 범주형 피처 지정
            )
            
            # 4. Optuna는 조기 종료된 시점의 최고 점수를 반환
            return model.get_best_score()['validation']['AUC']

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_catboost, n_trials=OPTUNA_N_TRIALS)

        best_params = study.best_params
        BEST_CAT_PARAMS[which] = best_params # [신규] 메타데이터 저장을 위해 전역 변수에 저장
        print(f"[{which}] Optuna (CatBoost) finished. Best AUC: {study.best_value:.5f}")
        print(f"[{which}] Best params: {best_params}")

        # [제거] HGB 전역 파라미터 업데이트 로직

        # --- [수정] 앙상블 학습 (CatBoost) ---
        
        # 1. 튜닝된 최종 파라미터 + 공통 파라미터 정리
        final_cat_params = best_params.copy()
        final_cat_params.update({
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': 0,
            'early_stopping_rounds': 50, 
            'task_type': 'CPU',
            'auto_class_weights': 'Balanced'
        })
        
        members = []
        for sd in ENSEMBLE_SEEDS:
            model_params = final_cat_params.copy()
            model_params['random_seed'] = sd # 각 모델의 시드 변경
            
            base = CatBoostClassifier(**model_params)
            
            base.fit(
                X_tr_t, y_tr,
                eval_set=[(X_val_t, y_val)], # 앙상블 멤버도 조기 종료 사용
                cat_features=cat_feature_indices,
                verbose=0 # 앙상블 학습 시 로그는 끔
            )
            
            mdl = maybe_calibrate(base, X_tr_t, y_tr) # 보정은 그대로 사용
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)

        try:
            val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
            auc = roc_auc_score(y_val, val_proba)
            brier = brier_score_loss(y_val, val_proba)
            print(f"[{which}] Holdout AUC (Ensemble-CatBoost)={auc:.5f}, Brier={brier:.5f}")
        except Exception as e:
            print(f"[{which}] validation logging skipped: {e}")

        joblib.dump(preproc, preproc_path)
        joblib.dump(ensemble, model_path)
        print(f"[{which}] trained and saved → {model_path}, {preproc_path}")
        return preproc, ensemble

    # --- 추론 경로 (변경 없음) ---
    print(f"[{which}] loading pre-trained: {preproc_path}, {model_path}")
    preproc = joblib.load(preproc_path)
    ensemble = joblib.load(model_path)
    return preproc, ensemble

# -------------------\
# 추론 (변경 없음)
# -------------------\
def predict_partition(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc,
    clf_or_ens, 
    which: str
) -> pd.DataFrame:
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    drop_cols = [key, "PrimaryKey"] + \
                (["Test"] if "Test" in df.columns else [])

    X = df.drop(columns=drop_cols, errors="ignore")
    X_t = preproc.transform(X)
    proba = np.clip(clf_or_ens.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)
    out = df_idx[[key]].copy()
    out["Label"] = proba
    out["__which__"] = which
    return out

# -------------------\
# 메타 저장 (CatBoost용으로 수정)
# -------------------\
def save_meta():
    meta = dict(
        model="CatBoost(3-seed soft ensemble) + OrdinalEnc + Calibration [Optuna Tuned]", # [수정] 모델명
        feature_engineering=[
            "Age_numeric (mapped from '30a' -> 32, '30b' -> 37)",
            "TestYear, TestMonth (from TestDate)",
            "TestCount (per PrimaryKey)",
            "TestSequence (per PrimaryKey, sorted by TestDate)",
            "FirstTestAge (per PrimaryKey)",
            "TimeSinceFirstTest_yr (per PrimaryKey)",
            "NA_COUNT, NA_RATIO (row-wise)"
        ],
        catboost_best_params=BEST_CAT_PARAMS, # [수정] HGB -> CatBoost 파라미터
        optuna_n_trials=OPTUNA_N_TRIALS, 
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
# 메인 (변경 없음)
# -------------------\
def main():
    t0 = time.time()
    ensure_dirs()
    # [수정] save_meta()는 튜닝이 끝난 후 맨 뒤에서 호출

    train_idx, test_idx = read_index_files()
    A_train_feat_raw, B_train_feat_raw = read_feature_files("train")
    A_test_feat_raw,  B_test_feat_raw  = read_feature_files("test")

    train_feat_raw = pd.concat([A_train_feat_raw, B_train_feat_raw], ignore_index=True)
    test_feat_raw  = pd.concat([A_test_feat_raw,  B_test_feat_raw],  ignore_index=True)
    
    assert train_feat_raw["Test_id"].nunique() == len(train_feat_raw), "Train Test_id duplicates!"
    
    all_feat_raw = pd.concat([
        train_feat_raw.assign(is_train=1),
        test_feat_raw.assign(is_train=0)
    ], ignore_index=True)

    all_feat_processed = create_features(all_feat_raw)

    train_feat_processed = all_feat_processed[all_feat_processed['is_train'] == 1].drop(columns='is_train')
    test_feat_processed  = all_feat_processed[all_feat_processed['is_train'] == 0].drop(columns='is_train')

    # --- A 모델 학습/추론 ---
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    A_test_idx  = test_idx[test_idx["Test"] == "A"].copy()
    
    need_train_A = (len(A_train_idx) > 0) and \
                   (not (os.path.exists(A_MODEL_PATH) and os.path.exists(A_PREPROC_PATH)))
    
    fit_df_idx_A = A_train_idx if need_train_A else pd.DataFrame({"Test_id": [], "Label": []})
    
    if need_train_A:
        print("[A] training path (data exists and no pre-trained weights found).")
    
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
        sub = test_idx[["Test_id"]].copy()
        sub["Label"] = 0.001

    try:
        sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        sub_final = sample[["Test_id"]].merge(sub, on="Test_id", how="left")
        sub_final["Label"] = sub_final["Label"].fillna(0.001)
        sub_final = sub_final[["Test_id", "Label"]]
    except Exception as e:
        print(f"Sample submission merge failed ({e}), saving raw submission.")
        sub_final = sub[["Test_id", "Label"]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub_final.to_csv(SUBMISSION_PATH, index=False)

    # [수정] 튜닝이 끝난 후, 최종 파라미터를 저장하기 위해 맨 뒤로 이동
    save_meta() 

    dt = time.time() - t0
    print(f"[이이이잉] submission saved -> {SUBMISSION_PATH} | elapsed: {dt:.2f}s")

if __name__ == "__main__":
    main()