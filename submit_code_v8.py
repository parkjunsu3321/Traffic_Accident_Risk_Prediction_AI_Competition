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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GroupKFold
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
CALIB_CV = 3  # 'prefit'이 불가능할 경우(구버전) 대비
N_SPLITS_KFold = 5
OPTUNA_N_TRIALS = 30  # 100회 탐색
optuna.logging.set_verbosity(optuna.logging.WARNING)

ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777)

# [V7.1] n_iter_no_change 수정
BASE_HGB_PARAMS = dict(
    learning_rate=0.06,       # (Optuna가 덮어쓸 수 있음)
    max_iter=300,             # (Optuna가 덮어쓸 수 있음)
    max_depth=None,
    max_leaf_nodes=63,        # (Optuna가 덮어쓸 수 있음)
    min_samples_leaf=20,      # (Optuna가 덮어쓸 수 있음)
    l2_regularization=0.0,    # (Optuna가 덮어쓸 수 있음)
    class_weight="balanced",
    
    # [V7] 데이터 누수 방지를 위해 early_stopping은 사용하지 않음
    early_stopping=False,
    validation_fraction=None,
    
    # [V7.1 수정] None이 아닌 유효한 정수값(기본값 10)을 넣어 유효성 검사를 통과시킴
    n_iter_no_change=10, 
)

# -------------------\
# 보조 유틸 (V7: Calibration 로직 수정)
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

def _mk_calibrator(base_clf, use_prefit: bool):
    try:
        major, minor, *_ = map(int, sklver.split(".")[:2])
    except Exception:
        major, minor = 1, 4
    
    kw = dict(method=CALIB_METHOD)
    
    if use_prefit and (major, minor) >= (1, 4):
        kw["cv"] = "prefit"
        return CalibratedClassifierCV(estimator=base_clf, **kw)
    else:
        if use_prefit:
            print(f"[WARN] 'prefit' calibration unavailable (sk-ver {sklver}). Falling back to regular CV={CALIB_CV}.")
        
        kw["cv"] = CALIB_CV 
        if (major, minor) >= (1, 4):
            return CalibratedClassifierCV(estimator=base_clf, **kw)
        else:
            return CalibratedClassifierCV(base_estimator=base_clf, **kw)

def maybe_calibrate(base_clf_fitted, X_val, y_val):
    if not USE_CALIBRATION:
        return base_clf_fitted
    
    calib = _mk_calibrator(base_clf_fitted, use_prefit=True)
    
    try:
        calib.fit(X_val, y_val) 
        return calib
    except Exception as e:
        print(f"WARN: Calibration failed ({e}). Falling back to uncalibrated model.")
        return base_clf_fitted

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
# [V6] 특징 공학 (시계열 피처 추가)
# (이 부분은 V6와 동일하게 유지)
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

    # 3. PrimaryKey (운전자) 기반 변수 (정렬이 중요)
    df_proc = df_proc.sort_values(by=['PrimaryKey', 'TestDate_num'])
    g = df_proc.groupby('PrimaryKey')
    
    df_proc['TestCount'] = g['Test_id'].transform('count')
    df_proc['TestSequence'] = g.cumcount() + 1
    df_proc['FirstTestAge'] = g['Age_numeric'].transform('min')
    df_proc['FirstTestYear'] = g['TestYear'].transform('min')
    df_proc['TimeSinceFirstTest_yr'] = df_proc['TestYear'] - df_proc['FirstTestYear']

    # [V6-FIX] '반응 시간' 컬럼들을 강제로 숫자형(numeric)으로 변환
    rt_cols = [
        'A1-4', 'A2-4', 'A3-7', 'A4-5', # A검사 반응시간
        'B1-2', 'B2-2', 'B3-2', 'B4-2', 'B5-2'  # B검사 반응시간
    ]
    for col in rt_cols:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
    
    print("[Global FE V6] '반응 시간' 컬럼 강제 숫자형 변환 완료.")

    # 4. A검사 (인성) 파생 변수
    a9_new_cols = ['A9_Stability_Score', 'A9_Stress_Ratio', 'A9_Reality_Stress']
    safe_cols_A = all(c in df_proc.columns for c in ['A9-1', 'A9-2', 'A9-3', 'A9-5'])
    
    if safe_cols_A:
        df_proc['A9_Stability_Score'] = df_proc['A9-1'] + df_proc['A9-2']
        df_proc['A9_Stress_Ratio'] = df_proc['A9-1'] / (df_proc['A9-5'] + 1e-6)
        df_proc['A9_Reality_Stress'] = df_proc['A9-3'] / (df_proc['A9-5'] + 1e-6)
    else:
        for col in a9_new_cols: df_proc[col] = np.nan
    df_proc[a9_new_cols] = df_proc[a9_new_cols].fillna(0.0)

    # 5. B검사 (다중과제 B9) 파생 변수
    b9_new_cols = ['B9_hit_rate', 'B9_fa_rate', 'B9_d_prime_proxy', 'B9_visual_error_rate', 'B9_audio_accuracy']
    safe_cols_B9 = all(c in df_proc.columns for c in ['B9-1', 'B9-2', 'B9-3', 'B9-4', 'B9-5'])
    
    if safe_cols_B9:
        B9_AUDIO_TRIALS = 50.0 
        B9_VISUAL_TRIALS = 32.0
        b9_hit_plus_miss = df_proc['B9-1'] + df_proc['B9-2']
        b9_fa_plus_cr = df_proc['B9-3'] + df_proc['B9-4']
        df_proc['B9_hit_rate'] = df_proc['B9-1'] / (b9_hit_plus_miss + 1e-6)
        df_proc['B9_fa_rate'] = df_proc['B9-3'] / (b9_fa_plus_cr + 1e-6)
        df_proc['B9_d_prime_proxy'] = df_proc['B9_hit_rate'] - df_proc['B9_fa_rate']
        df_proc['B9_visual_error_rate'] = df_proc['B9-5'] / B9_VISUAL_TRIALS
        df_proc['B9_audio_accuracy'] = (df_proc['B9-1'] + df_proc['B9-4']) / B9_AUDIO_TRIALS
    else:
        for col in b9_new_cols: df_proc[col] = np.nan
    df_proc[b9_new_cols] = df_proc[b9_new_cols].fillna(0.0)

    # 6. B검사 (다중과제 B10) 파생 변수
    b10_new_cols = ['B10_hit_rate', 'B10_fa_rate', 'B10_d_prime_proxy', 'B10_audio_accuracy', 
                    'B10_vis1_error_rate', 'B10_vis2_accuracy', 'B10_total_visual_error_rate']
    safe_cols_B10 = all(c in df_proc.columns for c in ['B10-1', 'B10-2', 'B10-3', 'B10-4', 'B10-5', 'B10-6'])
    
    if safe_cols_B10:
        B10_AUDIO_TRIALS = 80.0
        B10_VIS1_TRIALS = 52.0
        B10_VIS2_TRIALS = 20.0
        B10_TOTAL_VISUAL_TRIALS = B10_VIS1_TRIALS + B10_VIS2_TRIALS
        b10_hit_plus_miss = df_proc['B10-1'] + df_proc['B10-2']
        b10_fa_plus_cr = df_proc['B10-3'] + df_proc['B10-4']
        df_proc['B10_hit_rate'] = df_proc['B10-1'] / (b10_hit_plus_miss + 1e-6)
        df_proc['B10_fa_rate'] = df_proc['B10-3'] / (b10_fa_plus_cr + 1e-6)
        df_proc['B10_d_prime_proxy'] = df_proc['B10_hit_rate'] - df_proc['B10_fa_rate']
        df_proc['B10_audio_accuracy'] = (df_proc['B10-1'] + df_proc['B10-4']) / B10_AUDIO_TRIALS
        df_proc['B10_vis1_error_rate'] = df_proc['B10-5'] / B10_VIS1_TRIALS
        df_proc['B10_vis2_accuracy'] = df_proc['B10-6'] / B10_VIS2_TRIALS
        b10_total_visual_errors = df_proc['B10-5'] + (B10_VIS2_TRIALS - df_proc['B10-6'])
        df_proc['B10_total_visual_error_rate'] = b10_total_visual_errors / B10_TOTAL_VISUAL_TRIALS
    else:
        for col in b10_new_cols: df_proc[col] = np.nan
    df_proc[b10_new_cols] = df_proc[b10_new_cols].fillna(0.0)

    # 7. Row-wise NA (결측치) 변수
    base_feature_cols = [c for c in df_proc.columns if (c.startswith("A") or c.startswith("B")) and '_' not in c]
    df_proc = add_rowwise_features(df_proc, base_feature_cols)

    # 8. [V6] 시계열 피처 (Trend) 생성
    g = df_proc.groupby('PrimaryKey') 
    key_numeric_cols = (
        rt_cols + a9_new_cols + b9_new_cols + b10_new_cols + 
        ['NA_COUNT', 'NA_RATIO', 'Age_numeric']
    )
    key_numeric_cols = [c for c in key_numeric_cols if c in df_proc.columns]

    print(f"[Global FE V6] Creating {len(key_numeric_cols)} time-series features (diff/shift/roll)...")
    
    for col in key_numeric_cols:
        df_proc[f'{col}_diff'] = g[col].diff()
        df_proc[f'{col}_shift1'] = g[col].shift(1)
        roll_mean = g[col].rolling(3, min_periods=1).mean()
        df_proc[f'{col}_roll3_mean'] = roll_mean.reset_index(level=0, drop=True)
    
    print("[Global FE V6] Time-series features created.")
    df_proc = df_proc.drop(columns=['Age', 'TestDate', 'TestDate_num', 'FirstTestYear'], errors='ignore')
    
    return df_proc


# -------------------\
# [V8] 학습/로드 (A/B 모델 피처 분리 적용)
# -------------------\
def fit_or_load(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    label_col: str,
    model_path: str,
    preproc_path: str,
    which: str
):
    global BASE_HGB_PARAMS # 전역 파라미터를 수정하기 위해 선언
    key = "Test_id"
    
    if len(df_idx) and label_col in df_idx.columns:
        # --- 학습 경로 ---
        df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
        
        drop_cols = [key, label_col, "PrimaryKey"] + \
                    (["Test"] if "Test" in df.columns else [])
        
        # [V8 수정] 모델과 관련 없는 피처(Column)를 명시적으로 제거
        if which == "A":
            # A 모델 학습 시, B로 시작하는 모든 피처(B1-1, B9_hit_rate 등) 제거
            cols_to_drop = [c for c in df.columns if c.startswith("B")]
            df = df.drop(columns=cols_to_drop, errors='ignore')
            print(f"[{which}] Dropped {len(cols_to_drop)} 'B' features for Model A.")
        elif which == "B":
            # B 모델 학습 시, A로 시작하는 모든 피처(A1-1, A9_Stability_Score 등) 제거
            cols_to_drop = [c for c in df.columns if c.startswith("A")]
            df = df.drop(columns=cols_to_drop, errors='ignore')
            print(f"[{which}] Dropped {len(cols_to_drop)} 'A' features for Model B.")
            
        
        num_cols, cat_cols = separate_num_cat(df, drop_cols)
        
        print(f"[{which}] Preprocessing: {len(num_cols)} num_cols, {len(cat_cols)} cat_cols.")
        if len(cat_cols) > 0:
                 print(f"[{which}] CAT cols: {cat_cols}")

        preproc = build_preprocessor(num_cols, cat_cols)

        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[label_col].astype(int).values
        groups = df['PrimaryKey'].values

        print(f"[{which}] Splitting data using GroupKFold (n_splits={N_SPLITS_KFold}, group=PrimaryKey)...")
        gkf = GroupKFold(n_splits=N_SPLITS_KFold)
        
        train_indices, val_indices = next(gkf.split(X, y, groups=groups))
        
        X_tr, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_tr, y_val = y[train_indices], y[val_indices]

        print(f"[{which}] Train shape: {X_tr.shape}, Val shape: {X_val.shape}")

        X_tr_t = preproc.fit_transform(X_tr)
        X_val_t = preproc.transform(X_val)
        
        # --- [V7] Optuna (max_iter 튜닝) ---
        print(f"[{which}] Running Optuna hyperparameter search (tuning max_iter)...")

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 127),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.1),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'max_iter': trial.suggest_int('max_iter', 100, 1000, step=50)
            }
            
            hgb_params = BASE_HGB_PARAMS.copy()
            hgb_params.update(params)
            hgb_params['random_state'] = RANDOM_STATE
            hgb_params['early_stopping'] = False
            hgb_params['validation_fraction'] = None
            
            model = HistGradientBoostingClassifier(**hgb_params)
            model.fit(X_tr_t, y_tr)
            
            val_proba = np.clip(model.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
            auc = roc_auc_score(y_val, val_proba)
            return auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

        best_params = study.best_params
        print(f"[{which}] Optuna search finished. Best AUC: {study.best_value:.5f}")
        print(f"[{which}] Best params: {best_params}")

        BASE_HGB_PARAMS.update(best_params)
        BASE_HGB_PARAMS['early_stopping'] = False
        BASE_HGB_PARAMS['validation_fraction'] = None
        
        print(f"[{which}] Global HGB params updated for ensemble training (max_iter={BASE_HGB_PARAMS.get('max_iter')}).")
        # --- Optuna 탐색 종료 ---

        # 앙상블 학습
        members = []
        for sd in ENSEMBLE_SEEDS:
            base = build_model(sd).fit(X_tr_t, y_tr)
            mdl = maybe_calibrate(base, X_val_t, y_val)
            members.append(mdl)
        ensemble = AvgProbaEnsemble(members)

        try:
            val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
            auc = roc_auc_score(y_val, val_proba)
            brier = brier_score_loss(y_val, val_proba)
            print(f"[{which}] Holdout AUC (GroupKFold Split 0)={auc:.5f}, Brier={brier:.5f}")
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
# [V8] 추론 (A/B 피처 분리 적용)
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
    
    # [V8 수정] 추론 시에도 학습 시와 동일하게 피처를 제거해야 함
    if which == "A":
        cols_to_drop = [c for c in df.columns if c.startswith("B")]
        df = df.drop(columns=cols_to_drop, errors='ignore')
    elif which == "B":
        cols_to_drop = [c for c in df.columns if c.startswith("A")]
        df = df.drop(columns=cols_to_drop, errors='ignore')

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
# 메타 저장 (V8)
# -------------------\
def save_meta():
    meta = dict(
        model="HGB(3-seed soft ensemble) + OrdinalEnc + Calibration [Optuna V8 - TimeSeries FE + GroupKFold + max_iter tuning + A/B Split]", # [V8] 모델명
        feature_engineering=[
            "Age_numeric, TestYear, TestMonth, TestCount, TestSequence, etc.",
            "NA_COUNT, NA_RATIO (row-wise)",
            "A9_... (Stability, Stress_Ratio) [V6-Base: fillna(0)]",
            "B9_... (SDT d-prime, rates) [V6-Base: fillna(0)]",
            "B10_... (SDT d-prime, rates) [V6-Base: fillna(0)]",
            "[V6-Fix] Forced Response Time columns to numeric",
            "[V6-FE] Time-Series features (diff, shift, roll3_mean) for key numeric cols",
            "[V8-FE] A/B Feature Splitting: Model A trains only on A* features, Model B on B* features." # [V8]
        ],
        validation_strategy=f"GroupKFold (n_splits={N_SPLITS_KFold}, used split 0) on PrimaryKey",
        
        hgb_params=BASE_HGB_PARAMS, 
        
        optuna_n_trials=OPTUNA_N_TRIALS,
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=f"{CALIB_CV} (fallback) or 'prefit' (if sk-ver >= 1.4)",
        sklearn_version=sklver,
        random_state=RANDOM_STATE,
    )
    # hgb_params를 json으로 저장하기 위해 str로 변환
    meta['hgb_params'] = {k: str(v) for k, v in meta['hgb_params'].items()}
    
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# -------------------\
# 메인 (변경 없음)
# -------------------\
def main():
    t0 = time.time()
    ensure_dirs()

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
    
    # [V8] 수정된 fit_or_load 함수가 호출됨
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
        
    # [V8] 수정된 fit_or_load 함수가 호출됨
    preproc_B, clf_B = fit_or_load(
        train_feat_processed, fit_df_idx_B, "Label",
        B_MODEL_PATH, B_PREPROC_PATH, "B"
    )

    # --- 추론 ---
    # [V8] 수정된 predict_partition 함수가 호출됨
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

    save_meta() # [V8] 메타 정보 저장

    dt = time.time() - t0
    print(f"[V8] submission saved -> {SUBMISSION_PATH} | elapsed: {dt:.2f}s")

if __name__ == "__main__":
    main()