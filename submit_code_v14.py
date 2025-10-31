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

A_MODEL_PATH_TPL = os.path.join(MODEL_DIR, "model_A_fold{fold}.joblib")
B_MODEL_PATH_TPL = os.path.join(MODEL_DIR, "model_B_fold{fold}.joblib")
A_PREPROC_PATH_TPL = os.path.join(MODEL_DIR, "preproc_A_fold{fold}.joblib")
B_PREPROC_PATH_TPL = os.path.join(MODEL_DIR, "preproc_B_fold{fold}.joblib")

RANDOM_STATE = 42

# -------------------\
# 실행 옵션
# -------------------\
USE_CALIBRATION = True
CALIB_METHOD = "isotonic"
CALIB_CV = 3
N_SPLITS_KFold = 5
OPTUNA_N_TRIALS = 30 
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

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_index_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx

def read_feature_files(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    A_df = pd.read_csv(os.path.join(DATA_DIR, split, "A.csv"), low_memory=False)
    B_df = pd.read_csv(os.path.join(DATA_DIR, split, "B.csv"), low_memory=False)
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
        major, minor, *_ = map(int, sklver.split("."
                                                 )[:2])
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
# [V16] 특징 공학 (B검사 피처 '선별')
# -------------------\

def safe_split_to_float(series: pd.Series) -> pd.DataFrame:
    """콤마로 구분된 문자열 Series를 파싱하여 float DataFrame으로 반환 (벡터화)"""
    return series.str.split(',', expand=True).astype(float)

# [V16] calculate_correct_rate 함수 제거 (노이즈 피처로 간주)

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
    
    g_temp = df_proc.groupby('PrimaryKey') # 임시 g (TestCount 등 기본 피처용)
    
    df_proc['TestCount'] = g_temp['Test_id'].transform('count')
    df_proc['TestSequence'] = g_temp.cumcount() + 1
    df_proc['FirstTestAge'] = g_temp['Age_numeric'].transform('min')
    df_proc['FirstTestYear'] = g_temp['TestYear'].transform('min')
    df_proc['TimeSinceFirstTest_yr'] = df_proc['TestYear'] - df_proc['FirstTestYear']
    
    # 4. [V14] 신규 A검사 피처 저장을 위한 dict
    new_A_features = {}
    
    # 5. [V14] A검사 원시 데이터 파싱 (A.csv에만 존재)
    try:
        print("[Global FE V16] Starting Vectorized Parsing for A-Test...")
        
        # A1 (속도예측) / A2 (정지거리예측) - 반응거리 (편차)
        df_a1_dist = safe_split_to_float(df_proc['A1-4'])
        new_A_features['A1_Dist_Mean'] = df_a1_dist.mean(axis=1)
        new_A_features['A1_Dist_Std'] = df_a1_dist.std(axis=1) # <-- 논문 핵심
        
        df_a2_dist = safe_split_to_float(df_proc['A2-4'])
        new_A_features['A2_Dist_Mean'] = df_a2_dist.mean(axis=1)
        new_A_features['A2_Dist_Std'] = df_a2_dist.std(axis=1)

        # A3 (주의전환) - 반응시간 및 정반응률
        df_a3_rt = safe_split_to_float(df_proc['A3-7'])
        df_a3_type = safe_split_to_float(df_proc['A3-5']) # 1:valid-C, 2:valid-IC, 3:invalid-C, 4:invalid-IC
        new_A_features['A3_RT_Valid_Mean'] = df_a3_rt.where(df_a3_type.isin([1, 2])).mean(axis=1) # <-- 논문 핵심
        new_A_features['A3_RT_Invalid_Mean'] = df_a3_rt.where(df_a3_type.isin([3, 4])).mean(axis=1) # <-- 논문 핵심
        new_A_features['A3_RT_Valid_Std'] = df_a3_rt.where(df_a3_type.isin([1, 2])).std(axis=1)
        new_A_features['A3_RT_Invalid_Std'] = df_a3_rt.where(df_a3_type.isin([3, 4])).std(axis=1)
        a3_valid_correct = (df_a3_type == 1).sum(axis=1)
        a3_valid_total = df_a3_type.isin([1, 2]).sum(axis=1)
        a3_invalid_correct = (df_a3_type == 3).sum(axis=1)
        a3_invalid_total = df_a3_type.isin([3, 4]).sum(axis=1)
        new_A_features['A3_Valid_Correct_Rate'] = a3_valid_correct / (a3_valid_total + 1e-6)
        new_A_features['A3_Invalid_Correct_Rate'] = a3_invalid_correct / (a3_invalid_total + 1e-6)

        # A4 (반응조절) - 반응시간 및 정반응률
        df_a4_rt = safe_split_to_float(df_proc['A4-5'])
        df_a4_cond = safe_split_to_float(df_proc['A4-1']) # 1:congruent, 2:incongruent
        df_a4_resp = safe_split_to_float(df_proc['A4-3']) # 1:correct, 2:incorrect
        new_A_features['A4_RT_Congruent_Mean'] = df_a4_rt.where((df_a4_cond == 1) & (df_a4_resp == 1)).mean(axis=1) # <-- 논문 핵심
        new_A_features['A4_RT_Incongruent_Mean'] = df_a4_rt.where((df_a4_cond == 2) & (df_a4_resp == 1)).mean(axis=1) # <-- 논문 핵심
        new_A_features['A4_RT_Congruent_Std'] = df_a4_rt.where((df_a4_cond == 1) & (df_a4_resp == 1)).std(axis=1)
        new_A_features['A4_RT_Incongruent_Std'] = df_a4_rt.where((df_a4_cond == 2) & (df_a4_resp == 1)).std(axis=1)
        a4_con_correct = ((df_a4_cond == 1) & (df_a4_resp == 1)).sum(axis=1)
        a4_con_total = (df_a4_cond == 1).sum(axis=1)
        a4_incon_correct = ((df_a4_cond == 2) & (df_a4_resp == 1)).sum(axis=1)
        a4_incon_total = (df_a4_cond == 2).sum(axis=1)
        new_A_features['A4_Congruent_Correct_Rate'] = a4_con_correct / (a4_con_total + 1e-6)
        new_A_features['A4_Incongruent_Correct_Rate'] = a4_incon_correct / (a4_incon_total + 1e-6)

        # A5 (변화탐지) - 정반응률
        df_a5_type = safe_split_to_float(df_proc['A5-1']) # 1:non-change, 2:pos, 3:color, 4:shape
        df_a5_resp = safe_split_to_float(df_proc['A5-2']) # 1:correct, 2:incorrect
        a5_invalid_correct = (df_a5_type.isin([2, 3, 4]) & (df_a5_resp == 1)).sum(axis=1)
        a5_invalid_total = df_a5_type.isin([2, 3, 4]).sum(axis=1)
        a5_valid_correct = ((df_a5_type == 1) & (df_a5_resp == 1)).sum(axis=1)
        a5_valid_total = (df_a5_type == 1).sum(axis=1)
        new_A_features['A5_Invalid_Correct_Rate'] = a5_invalid_correct / (a5_invalid_total + 1e-6) # <-- 논문 핵심
        new_A_features['A5_Valid_Correct_Rate'] = a5_valid_correct / (a5_valid_total + 1e-6)

        # A6 (판단능력) / A7 (지각성향)
        new_A_features['A6_Correct_Rate'] = pd.to_numeric(df_proc['A6-1'], errors='coerce') / 14.0
        new_A_features['A7_Correct_Rate'] = pd.to_numeric(df_proc['A7-1'], errors='coerce') / 18.0
        
        # [V14] 생성된 피처들을 df_proc에 병합
        df_new_A_features = pd.DataFrame(new_A_features, index=df_proc.index)
        df_proc = pd.concat([df_proc, df_new_A_features], axis=1)
        print(f"[Global FE V16] {len(new_A_features)} A-Test paper-based features created.")
    
    except KeyError as e:
        print(f"[Global FE V16] Skipping A-Test parsing (likely B-Test data): {e}")
    except Exception as e:
        print(f"[Global FE V16] ERROR during A-Test parsing: {e}")

    # 6. [V16] 신규 B검사 (선별된) 피처 저장을 위한 dict
    new_B_features = {}

    # 7. [V16] B검사 원시 데이터 파싱 (B.csv에만 존재) - '선별' 버전
    try:
        print("[Global FE V16] Starting Vectorized Parsing for B-Test (Selective)...")
        
        # B1/B2 (시야각) - RT(mean, std)
        # [V16] 단순 정답률(Change_Correct_Rate) 제거 -> 노이즈 의심
        df_b1_rt = safe_split_to_float(df_proc['B1-2'])
        new_B_features['B1_RT_Mean'] = df_b1_rt.mean(axis=1)
        new_B_features['B1_RT_Std'] = df_b1_rt.std(axis=1) # <-- KEEP

        df_b2_rt = safe_split_to_float(df_proc['B2-2'])
        new_B_features['B2_RT_Mean'] = df_b2_rt.mean(axis=1)
        new_B_features['B2_RT_Std'] = df_b2_rt.std(axis=1) # <-- KEEP
        
        # B3 (시각 운동 협응) - RT(mean, std)
        # [V16] 단순 정답률(B3_Correct_Rate) 제거 -> 노이즈 의심
        df_b3_rt = safe_split_to_float(df_proc['B3-2'])
        new_B_features['B3_RT_Mean'] = df_b3_rt.mean(axis=1)
        new_B_features['B3_RT_Std'] = df_b3_rt.std(axis=1) # <-- KEEP

        # B4 (선택적 주의력) - A4와 동일하게 파싱 (KEEP ALL)
        df_b4_rt = safe_split_to_float(df_proc['B4-2'])
        df_b4_resp = safe_split_to_float(df_proc['B4-1'])
        
        # RT (mean, std)
        new_B_features['B4_RT_Congruent_Mean'] = df_b4_rt.where(df_b4_resp.isin([1, 2])).mean(axis=1) # <-- KEEP
        new_B_features['B4_RT_Incongruent_Mean'] = df_b4_rt.where(df_b4_resp.isin([3, 4, 5, 6])).mean(axis=1) # <-- KEEP
        new_B_features['B4_RT_Congruent_Std'] = df_b4_rt.where(df_b4_resp.isin([1, 2])).std(axis=1) # <-- KEEP
        new_B_features['B4_RT_Incongruent_Std'] = df_b4_rt.where(df_b4_resp.isin([3, 4, 5, 6])).std(axis=1) # <-- KEEP

        # Correct Rate
        b4_con_correct = (df_b4_resp == 1).sum(axis=1)
        b4_con_total = df_b4_resp.isin([1, 2]).sum(axis=1)
        b4_incon_correct = (df_b4_resp.isin([3, 5])).sum(axis=1) # 3 and 5 correct
        b4_incon_total = df_b4_resp.isin([3, 4, 5, 6]).sum(axis=1)
        new_B_features['B4_Congruent_Correct_Rate'] = b4_con_correct / (b4_con_total + 1e-6) # <-- KEEP
        new_B_features['B4_Incongruent_Correct_Rate'] = b4_incon_correct / (b4_incon_total + 1e-6) # <-- KEEP

        # B5 (공간 판단력) - RT(mean, std)
        # [V16] 단순 정답률(B5_Correct_Rate) 제거 -> 노이즈 의심
        df_b5_rt = safe_split_to_float(df_proc['B5-2'])
        new_B_features['B5_RT_Mean'] = df_b5_rt.mean(axis=1)
        new_B_features['B5_RT_Std'] = df_b5_rt.std(axis=1) # <-- KEEP
        
        # B6, B7, B8 - [V16] 단순 정답률 제거 -> 노이즈 의심
        
        # [V16] 생성된 피처들을 df_proc에 병합
        df_new_B_features = pd.DataFrame(new_B_features, index=df_proc.index)
        df_proc = pd.concat([df_proc, df_new_B_features], axis=1)
        print(f"[Global FE V16] {len(new_B_features)} B-Test (Selective) features created.")

    except KeyError as e:
        print(f"[Global FE V16] Skipping B-Test parsing (likely A-Test data): {e}")
    except Exception as e:
        print(f"[Global FE V16] ERROR during B-Test parsing: {e}")

    # 8. [V12] A검사 (인성, A9) 파생 변수
    a9_new_cols = ['A9_Stability_Score', 'A9_Stress_Ratio', 'A9_Reality_Stress']
    safe_cols_A = all(c in df_proc.columns for c in ['A9-1', 'A9-2', 'A9-3', 'A9-5'])
    
    if safe_cols_A:
        df_proc['A9_Stability_Score'] = df_proc['A9-1'] + df_proc['A9-2']
        df_proc['A9_Stress_Ratio'] = df_proc['A9-1'] / (df_proc['A9-5'] + 1e-6)
        df_proc['A9_Reality_Stress'] = df_proc['A9-3'] / (df_proc['A9-5'] + 1e-6)
    else:
        for col in a9_new_cols: df_proc[col] = np.nan
    df_proc[a9_new_cols] = df_proc[a9_new_cols].fillna(0.0)

    # 9. [V12] B검사 (다중과제 B9) 파생 변수 (KEEP ALL)
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

    # 10. [V12] B검사 (다중과제 B10) 파생 변수 (KEEP ALL)
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

    # 11. [V12] Row-wise NA (결측치) 변수
    base_feature_cols = [c for c in df_proc.columns if (c.startswith("A") or c.startswith("B")) and '-' in c]
    df_proc = add_rowwise_features(df_proc, base_feature_cols)

    # [V12] 'g' 객체를 NA_COUNT 등이 추가된 'df_proc'로 새로고침
    g = df_proc.groupby('PrimaryKey') 

    # 12. [V16] Global/Expanding 피처 대상 컬럼 재정의
    paper_A_features = list(new_A_features.keys())
    paper_B_features = list(new_B_features.keys()) # [V16] 선별된 B 피처 리스트
    
    key_numeric_cols = (
        paper_A_features + # [V14] 신규 A검사 피처
        paper_B_features + # [V16] 신규 (선별된) B검사 피처
        a9_new_cols + 
        b9_new_cols + 
        b10_new_cols + 
        ['NA_COUNT', 'NA_RATIO', 'Age_numeric']
    )
    key_numeric_cols = [c for c in key_numeric_cols if c in df_proc.columns]

    # 13. [V12] Global (전체) 및 Expanding (누적) 통계 피처
    print(f"[Global FE V16] Creating {len(key_numeric_cols)} Global/Expanding features...")
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

    print("[Global FE V16] Global/Expanding features created.")

    # 14. [V12] 시계열 피처 (Trend) 생성
    print(f"[Global FE V16] Creating {len(key_numeric_cols)} time-series features (diff/shift/roll)...")
    
    for col in key_numeric_cols:
        df_proc[f'{col}_diff'] = g[col].diff()
        df_proc[f'{col}_shift1'] = g[col].shift(1)
        roll_mean = g[col].rolling(3, min_periods=1).mean()
        df_proc[f'{col}_roll3_mean'] = roll_mean.reset_index(level=0, drop=True)
    
    print("[Global FE V16] Time-series features created.")
    
    # ---
    
    df_proc = df_proc.drop(columns=['Age', 'TestDate', 'TestDate_num', 'FirstTestYear'], errors='ignore')
    
    # [V14] 파싱에 사용된 원본 object 컬럼 제거
    raw_a_cols = [
        'A1-1', 'A1-2', 'A1-3', 'A1-4',
        'A2-1', 'A2-2', 'A2-3', 'A2-4',
        'A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7',
        'A4-1', 'A4-2', 'A4-3', 'A4-4', 'A4-5',
        'A5-1', 'A5-2', 'A5-3'
    ]
    # [V15] B검사 원본 컬럼 제거 목록 업데이트
    raw_b_cols = [
        'B1-1', 'B1-2', 'B1-3',
        'B2-1', 'B2-2', 'B2-3',
        'B3-1', 'B3-2',
        'B4-1', 'B4-2',
        'B5-1', 'B5-2',
        'B6', 'B7', 'B8'
    ]
    df_proc = df_proc.drop(columns=raw_a_cols + raw_b_cols, errors='ignore')
    print(f"[Global FE V16] Dropped {len(raw_a_cols + raw_b_cols)} raw object columns.")
    
    return df_proc


# -------------------\
# [V10] 단일 Fold 학습 함수 (V14.1과 동일)
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

    # 2. [V8] A/B 피처 분리
    if which == "A":
        cols_to_drop = [c for c in X_tr.columns if c.startswith("B")]
        X_tr = X_tr.drop(columns=cols_to_drop, errors='ignore')
        X_val = X_val.drop(columns=cols_to_drop, errors='ignore')
        print(f"{fold_label} Dropped {len(cols_to_drop)} 'B' features.")
    elif which == "B":
        cols_to_drop = [c for c in X_tr.columns if c.startswith("A")]
        X_tr = X_tr.drop(columns=cols_to_drop, errors='ignore')
        X_val = X_val.drop(columns=cols_to_drop, errors='ignore')
        print(f"{fold_label} Dropped {len(cols_to_drop)} 'A' features.")
        
    # 3. 전처리기 (Preprocessor)
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
        
        # --- ⬇️ 대회 공식 평가지표 최적화 (V12와 동일) ⬇️ ---
        auc = roc_auc_score(y_val, val_proba)
        brier = brier_score_loss(y_val, val_proba)
        ece = expected_calibration_error(y_val, val_proba) 
        combined_score = 0.5 * (1.0 - auc) + 0.25 * brier + 0.25 * ece
        return combined_score # <-- 최종 점수를 반환

    study = optuna.create_study(direction="minimize") # <-- "minimize"
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

    best_params = study.best_params
    print(f"{fold_label} Optuna finished. Best Score: {study.best_value:.5f}")

    fold_hgb_params.update(best_params)
    
    print(f"{fold_label} Params updated (max_iter={fold_hgb_params.get('max_iter')}).")

    # 5. [V10] 앙상블 학습 (Fold별 최적 파라미터 사용)
    members = []
    for sd in ENSEMBLE_SEEDS:
        base = build_model(sd, fold_hgb_params).fit(X_tr_t, y_tr)
        mdl = maybe_calibrate(base, X_val_t, y_val)
        members.append(mdl)
    ensemble = AvgProbaEnsemble(members)

    # [모니터링 코드 3/3] 최종 로그 수정 (V14.1과 동일)
    try:
        val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
        auc = roc_auc_score(y_val, val_proba)
        brier = brier_score_loss(y_val, val_proba)
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
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    
    if which == "A":
        cols_to_drop = [c for c in df.columns if c.startswith("B")]
        df = df.drop(columns=cols_to_drop, errors='ignore')
    elif which == "B":
        cols_to_drop = [c for c in df.columns if c.startswith("A")]
        df = df.drop(columns=cols_to_drop, errors='ignore')

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

    final_proba = np.mean(all_probas, axis=0)
    
    out = df_idx[[key]].copy()
    out["Label"] = final_proba
    out["__which__"] = which
    print(f"[{which}] K-Fold Prediction finished (Avg of {len(all_probas)} models).")
    return out


# -------------------\
# 메타 저장 (V16)
# -------------------\
def save_meta():
    meta = dict(
        model=f"HGB({N_SPLITS_KFold}-Fold Ensemble) + 3-seed AvgProba + OrdinalEnc + Calib [Optuna V16-BTestSelectFE]", # [V16] 이름 수정
        feature_engineering=[
            "Age_numeric, TestYear, TestMonth, TestCount, TestSequence, etc.",
            "NA_COUNT, NA_RATIO (row-wise)",
            "A9_..., B9_..., B10_... (Derived features)",
            "[V6-FE] Time-Series features (diff, shift, roll3_mean)",
            "[V8-FE] A/B Feature Splitting",
            "[V11-FE] Added Global Stats (transform mean/std, vs_mean)",
            "[V11-FE] Added Expanding Stats (expanding mean/std)",
            "[V14-VectorizedFE] Parsed raw string columns (A1-A5) into stats (mean, std, rate).",
            "[V14-PaperFE] Implemented paper's key features: A1_Dist_Std, A3_RT_Valid_Mean, etc.",
            "[V16-BTestSelectFE] Parsed B-Test raw string columns (B1-B5).",
            "[V16-BTestSelectFE] KEEP: B-Test RT std (B1, B2, B3, B5).",
            "[V16-BTestSelectFE] KEEP: Detailed B4 features (mean, std, rate).",
            "[V16-BTestSelectFE] REMOVED: Simple B-Test Correct Rates (B1-B3, B5-B8) to reduce noise.",
            "[V16-Cleanup] Dropped all raw string columns after parsing."
        ],
        validation_strategy=f"GroupKFold (n_splits={N_SPLITS_KFold}) on PrimaryKey. Full K-Fold Ensemble.", 
        hgb_base_params=BASE_HGB_PARAMS, 
        optuna_n_trials_per_fold=OPTUNA_N_TRIALS, 
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=f"{CALIB_CV} (fallback) or 'prefit' (if sk-ver >= 1.4)", # V14.1 Fix
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
    
    all_feat_raw = pd.concat([
        train_feat_raw.assign(is_train=1),
        test_feat_raw.assign(is_train=0)
    ], ignore_index=True)

    # [V16] 수정된 create_features 함수가 여기서 호출됨
    all_feat_processed = create_features(all_feat_raw)

    train_feat_processed = all_feat_processed[all_feat_processed['is_train'] == 1].drop(columns='is_train')
    test_feat_processed  = all_feat_processed[all_feat_processed['is_train'] == 0].drop(columns='is_train')
    
    print("-" * 50)
    print(f"[Main] K-Fold Training starting (N_SPLITS={N_SPLITS_KFold})...")
    print("-" * 50)

    # --- 2. A 모델 K-Fold 학습 ---
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    A_test_idx  = test_idx[test_idx["Test"] == "A"].copy()
    
    if len(A_train_idx) > 0:
        df_A_full = A_train_idx.merge(train_feat_processed, on=key_col, how="left", validate="1:1")
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
    B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
    B_test_idx  = test_idx[test_idx["Test"] == "B"].copy()
    
    if len(B_train_idx) > 0:
        df_B_full = B_train_idx.merge(train_feat_processed, on=key_col, how="left", validate="1:1")
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

    save_meta() # [V16] 메타 정보 저장

    dt = time.time() - t0
    print(f"[V16] submission saved -> {SUBMISSION_PATH} | elapsed: {dt:.2f}s")

if __name__ == "__main__":
    main()