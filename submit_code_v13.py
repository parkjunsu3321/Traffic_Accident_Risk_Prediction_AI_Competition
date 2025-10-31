#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List, Sequence, Dict
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
# [모니터링 코드 1/3] calibration_curve 임포트
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn import __version__ as sklver

# -------------------\
# 고정 경로
# -------------------\
DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_DIR = "model"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# [V10] 모델 경로를 Fold별 템플릿으로 변경
A_MODEL_PATH_TPL = os.path.join(MODEL_DIR, "model_A_fold{fold}.joblib")
B_MODEL_PATH_TPL = os.path.join(MODEL_DIR, "model_B_fold{fold}.joblib")
A_PREPROC_PATH_TPL = os.path.join(MODEL_DIR, "preproc_A_fold{fold}.joblib")
B_PREPROC_PATH_TPL = os.path.join(MODEL_DIR, "preproc_B_fold{fold}.joblib")

RANDOM_STATE = 42

# -------------------\
# 실행 옵션 (사용자 원본 유지)
# -------------------\
USE_CALIBRATION = True
CALIB_METHOD = "isotonic"
CALIB_CV = 3
N_SPLITS_KFold = 5
OPTUNA_N_TRIALS = 30 # <--- 사용자 원본 값 (30) 유지
optuna.logging.set_verbosity(optuna.logging.WARNING)

ENSEMBLE_SEEDS: Sequence[int] = (42, 202, 777)

BASE_HGB_PARAMS = dict(
    learning_rate=0.06,
    max_iter=300,
    max_depth=None,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.0,
    class_weight=None,
    early_stopping=False,
    validation_fraction=None,
    n_iter_no_change=10, 
)

# -------------------\
# 보조 유틸
# -------------------\

# [모니터링 코드 2/3] ECE 계산 함수 (수정본)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    
    bin_weights_non_empty = bin_weights[non_empty_bins]
    
    min_len = min(len(prob_true), len(bin_weights_non_empty))
    prob_true = prob_true[:min_len]
    prob_pred = prob_pred[:min_len]
    bin_weights_non_empty = bin_weights_non_empty[:min_len]

    ece = np.sum(bin_weights_non_empty * np.abs(prob_true - prob_pred))
    return ece

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

def build_model(seed: int, fold_params: Dict) -> HistGradientBoostingClassifier:
    params = fold_params.copy() 
    params["random_state"] = seed
    return HistGradientBoostingClassifier(**params)

class AvgProbaEnsemble:
    def __init__(self, models: List):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

# -------------------\
# [V11.1] 특징 공학 (GroupBy 객체 'g' 위치 수정)
# -------------------\
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # df 에는 'Test' (A/B) 컬럼이 main에서 병합되어 들어옴
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
    
    g_temp = df_proc.groupby('PrimaryKey') # 임시 g (TestCount 등 기본 피처용)

    # --- ⬇️ [V13] 플래그 피처 추가 (MIM) ⬇️ ---
    # 'Test' 컬럼은 main에서 병합되어 있어야 함
    print("[Global FE V13] Creating Missingness Indicator flags...")
    try:
        # 각 Key가 A, B 검사를 봤는지 여부를 미리 계산
        key_has_A = g_temp['Test'].transform(lambda s: s.eq('A').any())
        key_has_B = g_temp['Test'].transform(lambda s: s.eq('B').any())
        
        df_proc['Key_Has_A'] = key_has_A.astype(int)
        df_proc['Key_Has_B'] = key_has_B.astype(int)
        df_proc['Key_Has_Both_AB'] = (key_has_A & key_has_B).astype(int)
        print("[Global FE V13] Flags Key_Has_A, Key_Has_B created.")
    except Exception as e:
        print(f"WARN: Flag creation failed ({e}). Adding empty flags.")
        df_proc['Key_Has_A'] = 0
        df_proc['Key_Has_B'] = 0
        df_proc['Key_Has_Both_AB'] = 0
    # --- ⬆️ [V13] 플래그 피처 추가 (MIM) ⬆️ ---
    
    df_proc['TestCount'] = g_temp['Test_id'].transform('count')
    df_proc['TestSequence'] = g_temp.cumcount() + 1
    df_proc['FirstTestAge'] = g_temp['Age_numeric'].transform('min')
    df_proc['FirstTestYear'] = g_temp['TestYear'].transform('min')
    df_proc['TimeSinceFirstTest_yr'] = df_proc['TestYear'] - df_proc['FirstTestYear']

    # '반응 시간' 컬럼들을 강제로 숫자형(numeric)으로 변환
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

    # [V11.1 수정] 'g' 객체를 NA_COUNT 등이 추가된 'df_proc'로 새로고침
    g = df_proc.groupby('PrimaryKey') 

    # --- [V11] V6의 key_numeric_cols 정의 (V11의 고급 피처 생성에 사용됨) ---
    # [V13] 플래그 피처도 숫자 컬럼에 추가
    flag_cols = ['Key_Has_A', 'Key_Has_B', 'Key_Has_Both_AB']
    key_numeric_cols = (
        rt_cols + a9_new_cols + b9_new_cols + b10_new_cols + 
        ['NA_COUNT', 'NA_RATIO', 'Age_numeric'] + flag_cols
    )
    key_numeric_cols = [c for c in key_numeric_cols if c in df_proc.columns]

    # 8. [V11] Global (전체) 및 Expanding (누적) 통계 피처
    print(f"[Global FE V11] Creating {len(key_numeric_cols)} Global/Expanding features...")
    for col in key_numeric_cols:
        # Global (전체) 통계
        global_mean = g[col].transform('mean')
        global_std = g[col].transform('std')
        
        df_proc[f'{col}_global_mean'] = global_mean
        df_proc[f'{col}_global_std'] = global_std
        
        # 현재 값 vs 전체 평균
        df_proc[f'{col}_vs_global_mean'] = df_proc[col] - global_mean

        # Expanding (누적) 통계
        exp_mean = g[col].expanding(min_periods=1).mean()
        exp_std = g[col].expanding(min_periods=1).std()
        
        df_proc[f'{col}_exp_mean'] = exp_mean.reset_index(level=0, drop=True)
        df_proc[f'{col}_exp_std'] = exp_std.reset_index(level=0, drop=True)

    print("[Global FE V11] Global/Expanding features created.")

    # 9. [V6] 시계열 피처 (Trend) 생성 (V11에서도 유지)
    print(f"[Global FE V6] Creating {len(key_numeric_cols)} time-series features (diff/shift/roll)...")
    
    for col in key_numeric_cols:
        df_proc[f'{col}_diff'] = g[col].diff()
        df_proc[f'{col}_shift1'] = g[col].shift(1)
        roll_mean = g[col].rolling(3, min_periods=1).mean()
        df_proc[f'{col}_roll3_mean'] = roll_mean.reset_index(level=0, drop=True)
    
    print("[Global FE V6] Time-series features created.")
    
    # ---
    
    df_proc = df_proc.drop(columns=['Age', 'TestDate', 'TestDate_num', 'FirstTestYear'], errors='ignore')
    
    return df_proc


# -------------------\
# [V10] 단일 Fold 학습 함수 (V11과 동일, Optuna=AUC 최대화)
# -------------------\
def train_single_fold(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    model_path: str,
    preproc_path: str,
    which: str,
    fold_id: int
):
    
    fold_label = f"[{which} Fold {fold_id+1}/{N_SPLITS_KFold}]"
    
    # 1. 데이터 분할
    X_tr, X_val = X_full.iloc[train_indices], X_full.iloc[val_indices]
    y_tr, y_val = y_full[train_indices], y_full[val_indices]

    # 2. [V12.1] 하이브리드 피처 분리 전략
    
    if which == "A":
        # A모델(신규)은 B피처(미래)를 삭제 (v12 원본 로직 유지)
        cols_to_drop = [c for c in X_tr.columns if c.startswith("B")]
        X_tr = X_tr.drop(columns=cols_to_drop, errors='ignore')
        X_val = X_val.drop(columns=cols_to_drop, errors='ignore')
        print(f"{fold_label} Dropped {len(cols_to_drop)} 'B' features (Leakage Prevention).")
    
    elif which == "B":
        # B모델(유지)은 A피처(과거)를 유지 (v12 로직 수정)
        # (A피처 + [V13] 플래그 피처가 함께 학습됨)
        # cols_to_drop = [c for c in X_tr.columns if c.startswith("A")] 
        # X_tr = X_tr.drop(columns=cols_to_drop, errors='ignore')
        # X_val = X_val.drop(columns=cols_to_drop, errors='ignore')
        print(f"{fold_label} Keeping 'A' features (Historical Data + MIM Flags).")
        
    # 3. 전처리기 (Preprocessor)
    # [V13] 플래그 피처는 'Key_'로 시작하므로, 숫자/카테고리 분리 시 알아서 처리됨
    num_cols, cat_cols = separate_num_cat(X_tr, drop_cols=[])
    
    print(f"{fold_label} Preprocessing: {len(num_cols)} num_cols, {len(cat_cols)} cat_cols.")
    
    preproc = build_preprocessor(num_cols, cat_cols)
    
    X_tr_t = preproc.fit_transform(X_tr)
    X_val_t = preproc.transform(X_val)
    print(f"{fold_label} Train shape: {X_tr_t.shape}, Val shape: {X_val_t.shape}")

    # 4. [V10] Optuna (Fold별 파라미터 탐색)
    print(f"{fold_label} Running Optuna search...")
    
    fold_hgb_params = BASE_HGB_PARAMS.copy()

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 127),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.1),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'max_iter': trial.suggest_int('max_iter', 100, 1000, step=50)
        }
        
        hgb_params = fold_hgb_params.copy() 
        hgb_params.update(params)
        hgb_params['random_state'] = RANDOM_STATE
        
        model = HistGradientBoostingClassifier(**hgb_params)
        model.fit(X_tr_t, y_tr)
        
        val_proba = np.clip(model.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
        
        # --- ⬇️ 여기부터 수정 ⬇️ ---

        # 1. 모든 평가 지표 계산
        auc = roc_auc_score(y_val, val_proba)
        brier = brier_score_loss(y_val, val_proba)
        # 스크립트 상단에 이미 정의된 ECE 함수 사용
        ece = expected_calibration_error(y_val, val_proba) 
        
        # 2. 대회 공식 평가 산식 (낮을수록 좋음)
        combined_score = 0.5 * (1.0 - auc) + 0.25 * brier + 0.25 * ece
        
        return combined_score # <-- AUC가 아닌 최종 점수를 반환

    # --- ⬇️ direction도 'minimize'로 수정 ⬇️ ---
    study = optuna.create_study(direction="minimize") # <-- "maximize"에서 "minimize"로 변경
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

    best_params = study.best_params
    print(f"{fold_label} Optuna finished. Best AUC: {study.best_value:.5f}")

    fold_hgb_params.update(best_params)
    
    print(f"{fold_label} Params updated (max_iter={fold_hgb_params.get('max_iter')}).")

    # 5. [V10] 앙상블 학습 (Fold별 최적 파라미터 사용)
    members = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, fold_hgb_params).fit(X_tr_t, y_tr)
        mdl = maybe_calibrate(base, X_val_t, y_val)
        members.append(mdl)
    ensemble = AvgProbaEnsemble(members)

    # [모니터링 코드 3/3] 최종 로그 수정
    try:
        val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
        auc = roc_auc_score(y_val, val_proba)
        brier = brier_score_loss(y_val, val_proba)
        
        # ECE 및 최종 점수 계산 (모니터링용)
        ece = expected_calibration_error(y_val, val_proba)
        final_score = 0.5 * (1.0 - auc) + 0.25 * brier + 0.25 * ece
        print(f"{fold_label} Holdout: AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f} -> (Score={final_score:.5f})")

    except Exception as e:
        print(f"{fold_label} validation logging skipped: {e}")

    # 6. 모델 저장
    joblib.dump(preproc, preproc_path)
    joblib.dump(ensemble, model_path)
    print(f"{fold_label} trained and saved → {model_path}")


# -------------------\
# [V10] K-Fold 추론 함수 (변경 없음)
# -------------------\

def _predict_single(
    df_feat: pd.DataFrame, 
    df_idx: pd.DataFrame, 
    preproc, 
    clf_or_ens, 
    which: str
) -> np.ndarray:
    """V9의 predict_partition 로직 (단일 모델 추론)"""
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    
    # [V12.1] 하이브리드 피처 분리 전략 (추론)
    if which == "A":
        # A모델 추론 시 B피처(미래) 삭제
        cols_to_drop = [c for c in df.columns if c.startswith("B")]
        df = df.drop(columns=cols_to_drop, errors='ignore')
    elif which == "B":
        # B모델 추론 시 A피처(과거) 유지
        # (A피처 + [V13] 플래그 피처가 함께 사용됨)
        pass # A 피처를 삭제하지 않고 유지
    
    # [V9] 누수 피처 제거
    drop_cols = [key, "PrimaryKey"] + \
                (["Test"] if "Test" in df.columns else []) + \
                ['Test_x', 'Test_y'] 

    X = df.drop(columns=drop_cols, errors="ignore")
    X_t = preproc.transform(X)
    proba = np.clip(clf_or_ens.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)
    return proba


def predict_partition_kfold(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    which: str
) -> pd.DataFrame:
    """[V10] 모든 Fold 모델을 로드하여 평균 예측"""
    
    key = "Test_id"
    all_probas = []
    
    print(f"[{which}] Starting K-Fold Prediction ({N_SPLITS_KFold} folds)...")
    
    for fold_id in range(N_SPLITS_KFold):
        MODEL_PATH = os.path.join(MODEL_DIR, f"model_{which}_fold{fold_id}.joblib")
        PREPROC_PATH = os.path.join(MODEL_DIR, f"preproc_{which}_fold{fold_id}.joblib")
        
        if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH)):
            print(f"[{which} Fold {fold_id+1}] ERROR: Model file not found. Skipping.")
            continue
        
        try:
            preproc = joblib.load(PREPROC_PATH)
            ensemble = joblib.load(MODEL_PATH)
            
            proba_fold = _predict_single(df_feat, df_idx, preproc, ensemble, which)
            all_probas.append(proba_fold)
            print(f"[{which} Fold {fold_id+1}] Prediction loaded.")
            
        except Exception as e:
            print(f"[{which} Fold {fold_id+1}] ERROR loading/predicting: {e}")

    if not all_probas:
        print(f"[{which}] No fold models found. Returning 0.001.")
        out = df_idx[[key]].copy()
        out["Label"] = 0.001
        out["__which__"] = which
        return out

    # [V10] 모든 Fold의 예측 확률을 평균
    final_proba = np.mean(all_probas, axis=0)
    
    out = df_idx[[key]].copy()
    out["Label"] = final_proba
    out["__which__"] = which
    print(f"[{which}] K-Fold Prediction finished (Avg of {len(all_probas)} models).")
    return out


# -------------------\
# 메타 저장 (V11.1)
# -------------------\
def save_meta():
    meta = dict(
        model=f"HGB({N_SPLITS_KFold}-Fold Ensemble) + 3-seed AvgProba + OrdinalEnc + Calib [Optuna V13-MIM]", # 이름 수정
        feature_engineering=[
            "Age_numeric, TestYear, TestMonth, TestCount, TestSequence, etc.",
            "NA_COUNT, NA_RATIO (row-wise)",
            "A9_..., B9_..., B10_... (Derived features)",
            "[V6-FE] Time-Series features (diff, shift, roll3_mean)",
            "[V12.1-FE-FIX] Hybrid Strategy: Model A drops B-features, Model B KEEPS A-features",
            "[V13-FE] Added Missingness Indicator flags (Key_Has_A, Key_Has_B_AB)", # <-- 수정됨
            "[V9-Fix] Removed 'Test_x', 'Test_y' leakage features",
            "[V11-FE] Added Global Stats (transform mean/std, vs_mean)",
            "[V11-FE] Added Expanding Stats (expanding mean/std)",
            "[V11.1-Fix] Fixed 'g' groupby object refresh order for NA_COUNT"
        ],
        validation_strategy=f"GroupKFold (n_splits={N_SPLITS_KFold}) on PrimaryKey. Full K-Fold Ensemble.", 
        
        hgb_base_params=BASE_HGB_PARAMS, 
        
        optuna_n_trials_per_fold=OPTUNA_N_TRIALS, # 30 (원본 유지)
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=f"{CALIB_CV} (fallback) or 'prefit' (if sk-ver >= 1.4)",
        sklearn_version=sklver,
        random_state=RANDOM_STATE,
    )
    meta['hgb_base_params'] = {k: str(v) for k, v in meta['hgb_base_params'].items()}
    
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# -------------------\
# [V10] 메인 (K-Fold 학습 루프, 변경 없음)
# -------------------\
def main():
    t0 = time.time()
    ensure_dirs()
    
    label_col = "Label"
    key_col = "Test_id"

    # --- 1. 데이터 로드 및 전역 피처 생성 ---
    train_idx, test_idx = read_index_files()
    A_train_feat_raw, B_train_feat_raw = read_feature_files("train")
    A_test_feat_raw,  B_test_feat_raw  = read_feature_files("test")

    train_feat_raw = pd.concat([A_train_feat_raw, B_train_feat_raw], ignore_index=True)
    test_feat_raw  = pd.concat([A_test_feat_raw,  B_test_feat_raw],  ignore_index=True)
    
    # [V13] 'Test' 컬럼(A/B)을 create_features로 전달하기 위해 병합 순서 변경
    print("[Main V13] Merging index (A/B) and features (PrimaryKey) before FE.")
    train_data = train_idx.merge(train_feat_raw, on=key_col, how="left")
    test_data = test_idx.merge(test_feat_raw, on=key_col, how="left")

    all_feat_raw = pd.concat([
        train_data.assign(is_train=1),
        test_data.assign(is_train=0)
    ], ignore_index=True)

    # [V13] 'Test' 컬럼이 포함된 df가 create_features로 전달됨
    all_feat_processed = create_features(all_feat_raw)

    train_feat_processed = all_feat_processed[all_feat_processed['is_train'] == 1].drop(columns='is_train')
    test_feat_processed  = all_feat_processed[all_feat_processed['is_train'] == 0].drop(columns='is_train')
    
    print("-" * 50)
    print(f"[Main] K-Fold Training starting (N_SPLITS={N_SPLITS_KFold})...")
    print("-" * 50)

    # --- 2. A 모델 K-Fold 학습 ---
    A_test_idx  = test_idx[test_idx["Test"] == "A"].copy()
    
    # [V13.1 수정] train_feat_processed에서 직접 필터링 (Merge 불필요)
    df_A_full = train_feat_processed[train_feat_processed["Test"] == "A"].copy()

    if len(df_A_full) > 0:
        y_A = df_A_full[label_col].astype(int).values
        groups_A = df_A_full['PrimaryKey'].values
        
        drop_cols_prep = [key_col, label_col, "PrimaryKey", 'Test_x', 'Test_y', "Test"]
        X_A_full = df_A_full.drop(columns=drop_cols_prep, errors="ignore")
        
        gkf_A = GroupKFold(n_splits=N_SPLITS_KFold)
        
        for fold_id, (train_indices, val_indices) in enumerate(gkf_A.split(X_A_full, y_A, groups_A)):
            A_MODEL_PATH = A_MODEL_PATH_TPL.format(fold=fold_id)
            A_PREPROC_PATH = A_PREPROC_PATH_TPL.format(fold=fold_id)
            
            if not (os.path.exists(A_MODEL_PATH) and os.path.exists(A_PREPROC_PATH)):
                print(f"--- [A] Training Fold {fold_id+1}/{N_SPLITS_KFold} ---")
                train_single_fold(
                    X_A_full, y_A, train_indices, val_indices,
                    A_MODEL_PATH, A_PREPROC_PATH, which="A", fold_id=fold_id
                )
            else:
                print(f"--- [A] Loading Fold {fold_id+1}/{N_SPLITS_KFold} (already trained) ---")
    else:
        print("[A] No training data found. Skipping training.")

    # --- 3. B 모델 K-Fold 학습 ---
    B_test_idx  = test_idx[test_idx["Test"] == "B"].copy()
    
    # [V13.1 수정] train_feat_processed에서 직접 필터링 (Merge 불필요)
    df_B_full = train_feat_processed[train_feat_processed["Test"] == "B"].copy()
    
    if len(df_B_full) > 0:
        y_B = df_B_full[label_col].astype(int).values
        groups_B = df_B_full['PrimaryKey'].values
        
        drop_cols_prep = [key_col, label_col, "PrimaryKey", 'Test_x', 'Test_y', "Test"]
        X_B_full = df_B_full.drop(columns=drop_cols_prep, errors="ignore")
        
        gkf_B = GroupKFold(n_splits=N_SPLITS_KFold)
        
        for fold_id, (train_indices, val_indices) in enumerate(gkf_B.split(X_B_full, y_B, groups_B)):
            B_MODEL_PATH = B_MODEL_PATH_TPL.format(fold=fold_id)
            B_PREPROC_PATH = B_PREPROC_PATH_TPL.format(fold=fold_id)
            
            if not (os.path.exists(B_MODEL_PATH) and os.path.exists(B_PREPROC_PATH)):
                print(f"--- [B] Training Fold {fold_id+1}/{N_SPLITS_KFold} ---")
                train_single_fold(
                    X_B_full, y_B, train_indices, val_indices,
                    B_MODEL_PATH, B_PREPROC_PATH, which="B", fold_id=fold_id
                )
            else:
                print(f"--- [B] Loading Fold {fold_id+1}/{N_SPLITS_KFold} (already trained) ---")
    else:
        print("[B] No training data found. Skipping training.")

    print("-" * 50)
    print(f"[Main] K-Fold Prediction starting...")
    print("-" * 50)

    # --- 4. 추론 ---
    preds_A = predict_partition_kfold(test_feat_processed, A_test_idx, "A") if len(A_test_idx) else None
    preds_B = predict_partition_kfold(test_feat_processed, B_test_idx, "B") if len(B_test_idx) else None

    # --- 5. 제출 파일 생성 ---
    if preds_A is not None and preds_B is not None:
        sub = pd.concat([preds_A, preds_B], axis=0, ignore_index=True)
    elif preds_A is not None:
        sub = preds_A.copy()
    elif preds_B is not None:
        sub = preds_B.copy()
    else:
        sub = test_idx[[key_col]].copy()
        sub["Label"] = 0.001

    try:
        sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        sub_final = sample[[key_col]].merge(sub, on=key_col, how="left")
        sub_final["Label"] = sub_final["Label"].fillna(0.001)
        sub_final = sub_final[[key_col, "Label"]]
    except Exception as e:
        print(f"Sample submission merge failed ({e}), saving raw submission.")
        sub_final = sub[[key_col, "Label"]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub_final.to_csv(SUBMISSION_PATH, index=False)

    save_meta() # [V13] 메타 정보 저장

    dt = time.time() - t0
    print(f"[V13-MIM] submission saved -> {SUBMISSION_PATH} | elapsed: {dt:.2f}s")

if __name__ == "__main__":
    main()