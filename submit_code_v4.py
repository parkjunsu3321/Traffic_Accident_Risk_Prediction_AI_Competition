#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List, Sequence
import numpy as np
import pandas as pd
import joblib
import optuna  # [신규] Optuna 임포트

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

OPTUNA_N_TRIALS = 30  # [신규] Optuna 탐색 횟수 (시간 제한 고려)
optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna 로그 줄이기

ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777)

# [수정] Optuna가 이 값을 '기본값'으로 사용하고, 더 좋은 값을 찾으면 '업데이트'합니다.
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
    # HGBM은 수치형 변수의 NaN을 직접 처리합니다. Imputer가 필요 없습니다.
    # numeric_pipe = Pipeline(steps=[
    #    ("imputer", SimpleImputer(strategy="median")),
    # ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordenc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols), # [수정] Imputer 제거
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
    # [수정] 전역 변수인 BASE_HGB_PARAMS를 읽어옴 (Optuna가 수정했을 수 있음)
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
# [신규] 특징 공학 (V3 - 도메인 피처 추가)
# -------------------\
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy()

    # 1. Age (나이) 수치화 (기존 로직)
    age_map = {f"{i}{s}": (i + 2 if s == 'a' else i + 7) for i in range(10, 90, 10) for s in ['a', 'b']}
    age_map.update({
        '10a': 12, '10b': 17, '90a': 92, '90b': 97, '100a': 102
    })
    df_proc['Age_numeric'] = df_proc['Age'].map(age_map).astype(float)

    # 2. TestDate (검사일) 분해 (기존 로직)
    df_proc['TestDate_num'] = pd.to_numeric(df_proc['TestDate'], errors='coerce')
    df_proc['TestYear'] = (df_proc['TestDate_num'] // 100).astype(float)
    df_proc['TestMonth'] = (df_proc['TestDate_num'] % 100).astype(float)

    # 3. PrimaryKey (운전자) 기반 변수 (기존 로직)
    df_sorted = df_proc.sort_values(by=['PrimaryKey', 'TestDate_num'])
    df_proc['TestCount'] = df_sorted.groupby('PrimaryKey')['Test_id'].transform('count')
    df_proc['TestSequence'] = df_sorted.groupby('PrimaryKey').cumcount() + 1
    df_proc['FirstTestAge'] = df_sorted.groupby('PrimaryKey')['Age_numeric'].transform('min')
    df_proc['FirstTestYear'] = df_sorted.groupby('PrimaryKey')['TestYear'].transform('min')
    df_proc['TimeSinceFirstTest_yr'] = df_proc['TestYear'] - df_proc['FirstTestYear']

    # 4. [신규] A검사 (인성) 파생 변수
    # A.csv 에만 A9-1 ~ A9-5 컬럼이 존재. B.csv 에서는 NaN이 됨. (정상)
    safe_cols_A = all(c in df_proc.columns for c in ['A9-1', 'A9-2', 'A9-3', 'A9-5'])
    if safe_cols_A:
        df_proc['A9_Stability_Score'] = df_proc['A9-1'] + df_proc['A9-2']
        df_proc['A9_Stress_Ratio'] = df_proc['A9-1'] / (df_proc['A9-5'] + 1e-6)
        df_proc['A9_Reality_Stress'] = df_proc['A9-3'] / (df_proc['A9-5'] + 1e-6)
    else:
        # This is expected when processing B.csv, so no print
        pass

    # 5. [신규] B검사 (다중과제 B9) 파생 변수 (신호탐지이론)
    # B.csv 에만 B9-1 ~ B9-5 컬럼이 존재. A.csv 에서는 NaN이 됨. (정상)
    safe_cols_B9 = all(c in df_proc.columns for c in ['B9-1', 'B9-2', 'B9-3', 'B9-4', 'B9-5'])
    if safe_cols_B9:
        B9_AUDIO_TRIALS = 50.0
        B9_VISUAL_TRIALS = 32.0
        
        # B9-1: hit, B9-2: miss, B9-3: fa, B9-4: cr, B9-5: vis_err
        b9_hit_plus_miss = df_proc['B9-1'] + df_proc['B9-2']
        b9_fa_plus_cr = df_proc['B9-3'] + df_proc['B9-4']

        # 0으로 나누는 것을 방지하기 위해 + 1e-6 추가, B.csv가 아닌 A.csv를 처리할 때 분모가 0이 될 수 있음
        df_proc['B9_hit_rate'] = df_proc['B9-1'] / (b9_hit_plus_miss + 1e-6)
        df_proc['B9_fa_rate'] = df_proc['B9-3'] / (b9_fa_plus_cr + 1e-6)
        
        # d-prime (민감도) 근사치: Hit Rate - False Alarm Rate
        df_proc['B9_d_prime_proxy'] = df_proc['B9_hit_rate'] - df_proc['B9_fa_rate']
        
        df_proc['B9_visual_error_rate'] = df_proc['B9-5'] / B9_VISUAL_TRIALS
        df_proc['B9_audio_accuracy'] = (df_proc['B9-1'] + df_proc['B9-4']) / B9_AUDIO_TRIALS
    else:
        # This is expected when processing A.csv, so no print
        pass

    # 6. [신규] B검사 (다중과제 B10) 파생 변수 (신호탐지이론)
    safe_cols_B10 = all(c in df_proc.columns for c in ['B10-1', 'B10-2', 'B10-3', 'B10-4', 'B10-5', 'B10-6'])
    if safe_cols_B10:
        B10_AUDIO_TRIALS = 80.0
        B10_VIS1_TRIALS = 52.0
        B10_VIS2_TRIALS = 20.0
        B10_TOTAL_VISUAL_TRIALS = B10_VIS1_TRIALS + B10_VIS2_TRIALS
        
        # B10-1: hit, B10-2: miss, B10-3: fa, B10-4: cr, B10-5: vis1_err, B10-6: vis2_right
        b10_hit_plus_miss = df_proc['B10-1'] + df_proc['B10-2']
        b10_fa_plus_cr = df_proc['B10-3'] + df_proc['B10-4']

        df_proc['B10_hit_rate'] = df_proc['B10-1'] / (b10_hit_plus_miss + 1e-6)
        df_proc['B10_fa_rate'] = df_proc['B10-3'] / (b10_fa_plus_cr + 1e-6)
        
        # d-prime (민감도) 근사치
        df_proc['B10_d_prime_proxy'] = df_proc['B10_hit_rate'] - df_proc['B10_fa_rate']
        
        df_proc['B10_audio_accuracy'] = (df_proc['B10-1'] + df_proc['B10-4']) / B10_AUDIO_TRIALS
        
        df_proc['B10_vis1_error_rate'] = df_proc['B10-5'] / B10_VIS1_TRIALS
        df_proc['B10_vis2_accuracy'] = df_proc['B10-6'] / B10_VIS2_TRIALS
        
        b10_total_visual_errors = df_proc['B10-5'] + (B10_VIS2_TRIALS - df_proc['B10-6'])
        df_proc['B10_total_visual_error_rate'] = b10_total_visual_errors / B10_TOTAL_VISUAL_TRIALS
    else:
        # This is expected when processing A.csv, so no print
        pass


    # 7. Row-wise NA (결측치) 변수 (기존 로직)
    # [수정] 이제 base_feature_cols는 새로 생성된 변수들을 제외한 원본 A/B 컬럼만 포함해야 합니다.
    base_feature_cols = [c for c in df_proc.columns if (c.startswith("A") or c.startswith("B")) and '_' not in c]
    df_proc = add_rowwise_features(df_proc, base_feature_cols)
    
    print("[Global FE V3] 도메인 특화 파생 변수 생성 완료.")
    
    df_proc = df_proc.drop(columns=['Age', 'TestDate', 'TestDate_num', 'FirstTestYear'], errors='ignore')
    
    return df_proc


# -------------------\
# 학습/로드 (Optuna 적용)
# -------------------\
def fit_or_load(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str,
    model_path: str,
    preproc_path: str,
    which: str
):
    global BASE_HGB_PARAMS # [신규] 전역 파라미터를 수정하기 위해 선언
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

        # --- [신규] Optuna 하이퍼파라미터 탐색 ---
        print(f"[{which}] Running Optuna hyperparameter search...")

        def objective(trial):
            # 1. HGB 파라미터 탐색 공간 정의
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 127),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.1),
                'max_depth': trial.suggest_int('max_depth', 5, 15), # [신규] max_depth 탐색 추가
            }
            
            # 2. BASE 파라미터 복사 후, Optuna 제안값으로 덮어쓰기
            hgb_params = BASE_HGB_PARAMS.copy()
            hgb_params.update(params)
            hgb_params['random_state'] = RANDOM_STATE
            
            # 3. [중요] Optuna 평가 시에는 HGB의 자체 Early Stopping을 끕니다.
            #    대신 고정된 검증셋(X_val_t)으로 성능을 평가합니다.
            hgb_params['early_stopping'] = False
            hgb_params['validation_fraction'] = None
            
            model = HistGradientBoostingClassifier(**hgb_params)
            
            # 4. 학습 및 평가
            model.fit(X_tr_t, y_tr) # ES 없이 max_iter만큼 학습
            val_proba = np.clip(model.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
            auc = roc_auc_score(y_val, val_proba)
            return auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

        best_params = study.best_params
        print(f"[{which}] Optuna search finished. Best AUC: {study.best_value:.5f}")
        print(f"[{which}] Best params: {best_params}")

        # 5. [중요] 찾은 최적의 파라미터로 '전역' BASE_HGB_PARAMS를 업데이트합니다.
        BASE_HGB_PARAMS.update(best_params)
        
        # 6. [중요] 앙상블 학습을 위해 다시 Early Stopping을 켭니다.
        #    (개별 앙상블 모델이 자체적으로 조기 종료하도록)
        BASE_HGB_PARAMS['early_stopping'] = True
        BASE_HGB_PARAMS['validation_fraction'] = 0.12 # 원래 값 복원
        
        print(f"[{which}] Global HGB params updated for ensemble training.")
        # --- Optuna 탐색 종료 ---


        # 앙상블 학습 (이제 build_model은 '업데이트된' BASE_HGB_PARAMS를 사용합니다)
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
            print(f"[{which}] Holdout AUC (Ensemble)={auc:.5f}, Brier={brier:.5f}")
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
# 메타 저장 (변경 없음, 단 main에서 호출 위치 변경됨)
# -------------------\
def save_meta():
    meta = dict(
        model="HGB(3-seed soft ensemble) + OrdinalEnc + Calibration [Optuna Tuned V4 - FE + HPO]", # [수정] 모델명
        feature_engineering=[
            "Age_numeric (mapped from '30a' -> 32, '30b' -> 37)",
            "TestYear, TestMonth (from TestDate)",
            "TestCount (per PrimaryKey)",
            "TestSequence (per PrimaryKey, sorted by TestDate)",
            "FirstTestAge (per PrimaryKey)",
            "TimeSinceFirstTest_yr (per PrimaryKey)",
            "NA_COUNT, NA_RATIO (row-wise)"
        ],
        hgb_params=BASE_HGB_PARAMS, # [수정] Optuna로 튜닝된 최종 파라미터가 저장됨
        optuna_n_trials=OPTUNA_N_TRIALS, # [신규]
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
# 메인 (save_meta 위치 수정)
# -------------------\
def main():
    t0 = time.time()
    ensure_dirs()
    # [수정] save_meta()를 맨 뒤로 이동

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