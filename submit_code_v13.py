#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyexpat import model
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

from typing import Tuple, List, Sequence, Dict
import numpy as np
import pandas as pd
import joblib
import optuna

# [V12-CatBoost] Import CatBoost
from catboost import CatBoostClassifier, Pool

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# [V12-CatBoost] Remove OrdinalEncoder import
# from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
# [V12-CatBoost] Remove HistGradientBoostingClassifier import
# from sklearn.ensemble import HistGradientBoostingClassifier
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

# 모델 경로 템플릿 (동일)
A_MODEL_PATH_TPL = os.path.join(MODEL_DIR, "model_A_fold{fold}.joblib")
B_MODEL_PATH_TPL = os.path.join(MODEL_DIR, "model_B_fold{fold}.joblib")
A_PREPROC_PATH_TPL = os.path.join(MODEL_DIR, "preproc_A_fold{fold}.joblib")
B_PREPROC_PATH_TPL = os.path.join(MODEL_DIR, "preproc_B_fold{fold}.joblib")

RANDOM_STATE = 42

# -------------------\
# 실행 옵션 (사용자 원본 유지)
# -------------------\
USE_CALIBRATION = True # CatBoost는 확률 보정이 잘 되어 있는 편이지만, 유지해봅니다.
CALIB_METHOD = "isotonic"
CALIB_CV = 3
N_SPLITS_KFold = 5
OPTUNA_N_TRIALS = 10 # 사용자 원본 값 (30) 유지
optuna.logging.set_verbosity(optuna.logging.WARNING)

ENSEMBLE_SEEDS: Sequence[int] = (42)

# [V12-CatBoost] BASE_HGB_PARAMS 제거 -> Optuna에서 탐색

# -------------------\
# 보조 유틸
# -------------------\

# ECE 계산 함수 (V12 버전 유지)
# [V12-CatBoost] expected_calibration_error 함수 수정 (IndexError 해결)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    # 예측 확률값이 0 또는 1에 정확히 일치하는 경우를 대비해 약간 조정
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_true = np.array(y_true) # numpy 배열로 변환

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1) # 0, 0.1, 0.2, ..., 1.0

    for i in range(n_bins):
        # 현재 빈에 해당하는 인덱스 찾기
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])

        # 마지막 빈은 상한(1.0)을 포함하도록 처리
        if i == n_bins - 1:
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])

        # 현재 빈에 속하는 실제값과 예측값 가져오기
        bin_y_true = y_true[bin_mask]
        bin_y_prob = y_prob[bin_mask]

        # 빈이 비어있지 않은 경우에만 계산
        if len(bin_y_true) > 0:
            avg_true = np.mean(bin_y_true) # 빈의 실제 정답률
            avg_pred = np.mean(bin_y_prob) # 빈의 평균 예측 확률
            weight = len(bin_y_true) / len(y_true) # 전체 샘플 중 이 빈의 비율 (가중치)

            # ECE 누적 계산
            ece += weight * np.abs(avg_true - avg_pred)

    return ece

# [V12-CatBoost] v7의 Leaderboard_metric 클래스 추가
class Leaderboard_metric():
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # 대회 점수는 낮을수록 좋으므로 False
        return False

    def evaluate(self, approxes, targets, weight):
        # CatBoost는 Logits(approxes[0])를 반환하므로 확률로 변환
        logits = np.array(approxes[0])
        probs = 1 / (1 + np.exp(-logits))
        targets = np.array(targets) # targets를 numpy 배열로 변환

        try:
            auc = roc_auc_score(targets, probs)
        except ValueError: # 모든 타겟이 동일한 경우 AUC는 정의되지 않음
            auc = 0.5

        brier = np.mean((probs - targets) ** 2)

        # 위에서 정의한 v12 버전 ECE 함수 사용
        ece = expected_calibration_error(targets, probs)

        score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece

        # evaluate는 (error, weight) 튜플을 반환해야 함
        return score, len(targets)

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
    # [V12-CatBoost] CatBoost가 처리할 수 있도록 object 타입도 cat_cols로 간주
    cat_cols = [c for c in cols if str(df[c].dtype) in ("object", "category")]
    num_cols = [c for c in cols if c not in cat_cols]

    # [V12-CatBoost] CatBoost는 숫자형 컬럼에 문자열이 섞여 있으면 오류 발생 가능성 있음
    # 안전하게 object 컬럼 외에는 모두 num_cols로 취급 (CatBoost가 숫자형으로 처리)
    # 확실한 범주형(예: 'Age' 원본)이 있다면 cat_cols에 명시적으로 추가 필요
    # -> 현재 코드에서는 Age가 Age_numeric으로 변환되므로 일단 이대로 진행
    cat_cols_final = []
    num_cols_final = []
    for c in cols:
        # 매우 많은 고유값을 가진 object 컬럼은 범주형으로 처리하기 어려울 수 있음
        # PrimaryKey는 GroupKFold에만 사용되므로 제외
        if c == 'PrimaryKey': continue

        is_object = str(df[c].dtype) in ("object", "category")
        # 예시: 고유값이 50개 미만인 object 컬럼만 범주형으로 취급
        # if is_object and df[c].nunique() < 50:
        # 실제 데이터 확인 후 임계값 조정 필요 (여기서는 모든 object/category를 cat으로)
        if is_object:
             cat_cols_final.append(c)
        else:
             num_cols_final.append(c)

    # CatBoost는 NaN을 내부적으로 처리 가능하지만, SimpleImputer를 유지
    # -> 범주형 NaN은 most_frequent로, 숫자형 NaN은 median으로 채움

    # 컬럼 이름 중 CatBoost 예약어와 충돌할 수 있는 특수문자 제거/변경 (예: ':')
    # 현재 컬럼명에는 문제가 없어 보임

    return num_cols_final, cat_cols_final


# [V12-CatBoost] build_preprocessor 수정: OrdinalEncoder 제거
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # 필요시 Scaler 추가 가능: ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # OrdinalEncoder 제거됨
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols), # Imputer만 적용
        ],
        remainder="passthrough", # 중요: CatBoost 학습에 사용될 수 있으므로 drop 대신 passthrough
        sparse_threshold=0.0,
    )
    return preproc


# Calibration 함수 (동일)
def _mk_calibrator(base_clf, use_prefit: bool):
    # ... (V12와 동일) ...
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
    # ... (V12와 동일) ...
    if not USE_CALIBRATION:
        return base_clf_fitted

    calib = _mk_calibrator(base_clf_fitted, use_prefit=True)

    try:
        # CalibratedClassifierCV는 내부적으로 predict_proba를 사용하므로
        # CatBoost 모델도 호환됨
        calib.fit(X_val, y_val)
        return calib
    except Exception as e:
        print(f"WARN: Calibration failed ({e}). Falling back to uncalibrated model.")
        return base_clf_fitted


# add_rowwise_features (동일)
def add_rowwise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # ... (V12와 동일) ...
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2

# [V12-CatBoost] build_model 수정: CatBoostClassifier 반환
def build_model(seed: int, fold_params: Dict) -> CatBoostClassifier:
    params = fold_params.copy()
    params["random_seed"] = seed
    # 기본값 추가 (Optuna에서 덮어쓸 수 있음)
    params.setdefault('iterations', 1000) # 반복 횟수 늘림
    params.setdefault('learning_rate', 0.05)
    params.setdefault('loss_function', 'Logloss')
    params.setdefault('eval_metric', Leaderboard_metric()) # 커스텀 메트릭 사용
    params.setdefault('early_stopping_rounds', 50) # 조기 종료
    params.setdefault('verbose', 100) # 로그 출력 빈도
    params.setdefault('task_type', 'CPU') # CPU 사용 명시

    # fold_params에 cat_features_indices가 있을 수 있으므로 전달
    cat_features_indices = params.pop('cat_features_indices', None)

    return CatBoostClassifier(**params) # cat_features는 fit 시점에 전달

class AvgProbaEnsemble:
    # ... (V12와 동일) ...
    def __init__(self, models: List):
        self.models = models

    def predict_proba(self, X):
        # 입력 X가 Pool 객체일 수도 있고 아닐 수도 있음에 유의
        # CalibratedClassifier는 numpy 배열을 기대하므로 문제 없음
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

# -------------------\
# [V12-PDF] 특징 공학 (PDF 명세서 기반) - 이전 답변과 동일
# -------------------\

# PDF 기반 상수 정의 (이전 답변과 동일)
A_TRIALS = {
    'A1': 18.0, 'A2': 18.0, 'A3': 32.0, 'A3_VALID': 16.0, 'A3_INVALID': 8.0,
    'A4': 80.0, 'A4_CONGRUENT': 40.0, 'A4_INCONGRUENT': 40.0,
    'A5': 36.0, 'A5_NON_CHANGE': 18.0, 'A5_POS_CHANGE': 6.0, 'A5_COLOR_CHANGE': 6.0, 'A5_SHAPE_CHANGE': 6.0,
    'A6': 14.0, 'A7': 18.0
}
B_TRIALS = {
    'B1': 16.0, 'B1_CHANGE': 8.0, 'B1_NON_CHANGE': 8.0,
    'B2': 16.0, 'B2_CHANGE': 8.0, 'B2_NON_CHANGE': 8.0,
    'B3': 15.0, 'B4': 60.0, 'B4_CONGRUENT': 30.0, 'B4_INCONGRUENT': 30.0,
    'B5': 20.0, 'B6': 15.0, 'B7': 15.0, 'B8': 12.0
}
B9_TARGET_TRIALS = 15.0; B9_DISTRACTOR_TRIALS = 35.0; B9_VISUAL_TRIALS = 32.0
B10_TARGET_TRIALS = 20.0; B10_DISTRACTOR_TRIALS = 60.0; B10_VIS1_TRIALS = 52.0; B10_VIS2_TRIALS = 20.0

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (이전 답변의 V12-PDF create_features 함수 내용 전체 복사) ...
    # 이 부분은 길어서 생략합니다. 이전 답변의 코드를 그대로 사용하시면 됩니다.
    df_proc = df.copy()

    # --- V12 Step 1, 2, 3 (Age, TestDate, PrimaryKey) - 동일 ---
    age_map = {f"{i}{s}": (i + 2 if s == 'a' else i + 7) for i in range(10, 90, 10) for s in ['a', 'b']}
    age_map.update({'10a': 12, '10b': 17, '90a': 92, '90b': 97, '100a': 102})
    df_proc['Age_numeric'] = df_proc['Age'].map(age_map).astype(float)
    df_proc['TestDate_num'] = pd.to_numeric(df_proc['TestDate'], errors='coerce')
    df_proc['TestYear'] = (df_proc['TestDate_num'] // 100).astype(float)
    df_proc['TestMonth'] = (df_proc['TestDate_num'] % 100).astype(float)
    df_proc = df_proc.sort_values(by=['PrimaryKey', 'TestDate_num'])
    g_temp = df_proc.groupby('PrimaryKey')
    df_proc['TestCount'] = g_temp['Test_id'].transform('count')
    df_proc['TestSequence'] = g_temp.cumcount() + 1
    df_proc['FirstTestAge'] = g_temp['Age_numeric'].transform('min')
    df_proc['FirstTestYear'] = g_temp['TestYear'].transform('min')
    df_proc['TimeSinceFirstTest_yr'] = df_proc['TestYear'] - df_proc['FirstTestYear']

    # --- [V12-PDF 개선] 모든 PDF 기반 컬럼을 숫자로 변환 ---
    pdf_cols_to_numeric = [
        'A1-3','A2-3','A3-5-1','A3-5-2','A3-5-3','A3-5-4','A4-3-1','A4-3-2',
        'A5-2-1','A5-2-2','A6-1','A7-1','A8-1','A8-2','B1-1','B1-3-1','B1-3-2',
        'B1-3-3','B1-3-4','B2-1','B2-3-1','B2-3-2','B2-3-3','B2-3-4','B3-1',
        'B4-1','B5-1','B6','B7','B8'
    ]
    rt_cols = ['A1-4','A2-4','A3-7','A4-5','B1-2','B2-2','B3-2','B4-2','B5-2']
    v12_count_cols = [
        'A9-1','A9-2','A9-3','A9-5','B9-1','B9-2','B9-3','B9-4','B9-5',
        'B10-1','B10-2','B10-3','B10-4','B10-5','B10-6'
    ]
    all_cols_to_numeric = rt_cols + pdf_cols_to_numeric + v12_count_cols
    for col in all_cols_to_numeric:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
        else:
            df_proc[col] = np.nan
    print("[Global FE V12-PDF] PDF 기반 모든 컬럼 숫자형 변환 완료.")

    # --- V12 Step 4, 5, 6 (A9, B9, B10 파생변수) - 수정 ---
    a9_new_cols = ['A9_Stability_Score', 'A9_Stress_Ratio', 'A9_Reality_Stress']
    df_proc['A9_Stability_Score'] = df_proc['A9-1'] + df_proc['A9-2']
    df_proc['A9_Stress_Ratio'] = df_proc['A9-1'] / (df_proc['A9-5'] + 1e-6)
    df_proc['A9_Reality_Stress'] = df_proc['A9-3'] / (df_proc['A9-5'] + 1e-6)

    b9_new_cols = ['B9_hit_rate', 'B9_fa_rate', 'B9_d_prime_proxy', 'B9_visual_error_rate', 'B9_audio_accuracy', 'B9_miss_rate', 'B9_cr_rate']
    b9_hit_rate_ratio = df_proc['B9-1'] / (df_proc['B9-1'] + df_proc['B9-2'] + 1e-6)
    b9_fa_rate_ratio = df_proc['B9-3'] / (df_proc['B9-3'] + df_proc['B9-4'] + 1e-6)
    df_proc['B9_d_prime_proxy'] = b9_hit_rate_ratio - b9_fa_rate_ratio
    df_proc['B9_hit_rate'] = df_proc['B9-1'] / B9_TARGET_TRIALS
    df_proc['B9_miss_rate'] = df_proc['B9-2'] / B9_TARGET_TRIALS
    df_proc['B9_fa_rate'] = df_proc['B9-3'] / B9_DISTRACTOR_TRIALS
    df_proc['B9_cr_rate'] = df_proc['B9-4'] / B9_DISTRACTOR_TRIALS
    df_proc['B9_visual_error_rate'] = df_proc['B9-5'] / B9_VISUAL_TRIALS
    df_proc['B9_audio_accuracy'] = (df_proc['B9-1'] + df_proc['B9-4']) / (B9_TARGET_TRIALS + B9_DISTRACTOR_TRIALS)

    b10_new_cols = ['B10_hit_rate', 'B10_fa_rate', 'B10_d_prime_proxy', 'B10_audio_accuracy', 'B10_vis1_error_rate', 'B10_vis2_accuracy', 'B10_total_visual_error_rate', 'B10_miss_rate', 'B10_cr_rate']
    b10_hit_rate_ratio = df_proc['B10-1'] / (df_proc['B10-1'] + df_proc['B10-2'] + 1e-6)
    b10_fa_rate_ratio = df_proc['B10-3'] / (df_proc['B10-3'] + df_proc['B10-4'] + 1e-6)
    df_proc['B10_d_prime_proxy'] = b10_hit_rate_ratio - b10_fa_rate_ratio
    df_proc['B10_hit_rate'] = df_proc['B10-1'] / B10_TARGET_TRIALS
    df_proc['B10_miss_rate'] = df_proc['B10-2'] / B10_TARGET_TRIALS
    df_proc['B10_fa_rate'] = df_proc['B10-3'] / B10_DISTRACTOR_TRIALS
    df_proc['B10_cr_rate'] = df_proc['B10-4'] / B10_DISTRACTOR_TRIALS
    df_proc['B10_audio_accuracy'] = (df_proc['B10-1'] + df_proc['B10-4']) / (B10_TARGET_TRIALS + B10_DISTRACTOR_TRIALS)
    df_proc['B10_vis1_error_rate'] = df_proc['B10-5'] / B10_VIS1_TRIALS
    df_proc['B10_vis2_accuracy'] = df_proc['B10-6'] / B10_VIS2_TRIALS
    b10_total_visual_errors = df_proc['B10-5'] + (B10_VIS2_TRIALS - df_proc['B10-6'])
    df_proc['B10_total_visual_error_rate'] = b10_total_visual_errors / (B10_VIS1_TRIALS + B10_VIS2_TRIALS)

    # --- [V12-PDF 개선] Step 6.5: 모든 A/B 검사 정확도 피처 생성 ---
    print("[Global FE V12-PDF] Creating PDF-based Accuracy features...")
    pdf_derived_cols = []
    df_proc['A1_Acc'] = df_proc['A1-3'] / A_TRIALS['A1']; pdf_derived_cols.extend(['A1_Acc'])
    df_proc['A2_Acc'] = df_proc['A2-3'] / A_TRIALS['A2']; pdf_derived_cols.extend(['A2_Acc'])
    df_proc['A3_Acc_Valid'] = df_proc['A3-5-1'] / A_TRIALS['A3_VALID']; df_proc['A3_Acc_Invalid'] = df_proc['A3-5-3'] / A_TRIALS['A3_INVALID']
    df_proc['A3_Acc_Overall'] = (df_proc['A3-5-1'] + df_proc['A3-5-3']) / A_TRIALS['A3']; df_proc['A3_Attn_Cost_Acc_Diff'] = df_proc['A3_Acc_Valid'] - df_proc['A3_Acc_Invalid']
    pdf_derived_cols.extend(['A3_Acc_Valid', 'A3_Acc_Invalid', 'A3_Acc_Overall', 'A3_Attn_Cost_Acc_Diff'])
    df_proc['A4_Acc_Overall'] = df_proc['A4-3-1'] / A_TRIALS['A4']; pdf_derived_cols.extend(['A4_Acc_Overall'])
    df_proc['A5_Acc_Overall'] = df_proc['A5-2-1'] / A_TRIALS['A5']; pdf_derived_cols.extend(['A5_Acc_Overall'])
    df_proc['A6_Acc'] = df_proc['A6-1'] / A_TRIALS['A6']; df_proc['A7_Acc'] = df_proc['A7-1'] / A_TRIALS['A7']; pdf_derived_cols.extend(['A6_Acc', 'A7_Acc'])
    df_proc['A8_Validity_1'] = df_proc['A8-1']; df_proc['A8_Validity_2'] = df_proc['A8-2']; pdf_derived_cols.extend(['A8_Validity_1', 'A8_Validity_2'])
    df_proc['B1_Acc_Task1'] = df_proc['B1-1']; df_proc['B1_Acc_Change'] = df_proc['B1-3-1'] / B_TRIALS['B1_CHANGE']
    df_proc['B1_Acc_NoChange'] = df_proc['B1-3-3'] / B_TRIALS['B1_NON_CHANGE']; df_proc['B1_Acc_Task2_Overall'] = (df_proc['B1-3-1'] + df_proc['B1-3-3']) / B_TRIALS['B1']
    pdf_derived_cols.extend(['B1_Acc_Task1', 'B1_Acc_Change', 'B1_Acc_NoChange', 'B1_Acc_Task2_Overall'])
    df_proc['B2_Acc_Task1'] = df_proc['B2-1']; df_proc['B2_Acc_Change'] = df_proc['B2-3-1'] / B_TRIALS['B2_CHANGE']
    df_proc['B2_Acc_NoChange'] = df_proc['B2-3-3'] / B_TRIALS['B2_NON_CHANGE']; df_proc['B2_Acc_Task2_Overall'] = (df_proc['B2-3-1'] + df_proc['B2-3-3']) / B_TRIALS['B2']
    pdf_derived_cols.extend(['B2_Acc_Task1', 'B2_Acc_Change', 'B2_Acc_NoChange', 'B2_Acc_Task2_Overall'])
    df_proc['B3_Acc'] = df_proc['B3-1']; pdf_derived_cols.extend(['B3_Acc'])
    df_proc['B4_Acc_Overall'] = df_proc['B4-1'] / B_TRIALS['B4']; pdf_derived_cols.extend(['B4_Acc_Overall'])
    df_proc['B5_Acc'] = df_proc['B5-1']; pdf_derived_cols.extend(['B5_Acc'])
    df_proc['B6_Acc'] = df_proc['B6'] / B_TRIALS['B6']; df_proc['B7_Acc'] = df_proc['B7'] / B_TRIALS['B7']; df_proc['B8_Acc'] = df_proc['B8'] / B_TRIALS['B8']
    pdf_derived_cols.extend(['B6_Acc', 'B7_Acc', 'B8_Acc'])
    df_proc[a9_new_cols + b9_new_cols + b10_new_cols + pdf_derived_cols] = df_proc[a9_new_cols + b9_new_cols + b10_new_cols + pdf_derived_cols].fillna(0.0)
    print(f"[Global FE V12-PDF] Created {len(pdf_derived_cols)} new PDF-based features.")

    # --- V12 Step 7 (Row-wise NA) - 동일 ---
    base_feature_cols = [c for c in df_proc.columns if (c.startswith("A") or c.startswith("B")) and '_' not in c]
    df_proc = add_rowwise_features(df_proc, base_feature_cols)

    # --- V12 Step 8 (Global/Expanding Stats) - 수정 ---
    g = df_proc.groupby('PrimaryKey')
    key_numeric_cols = (rt_cols + a9_new_cols + b9_new_cols + b10_new_cols + pdf_derived_cols + ['NA_COUNT', 'NA_RATIO', 'Age_numeric'])
    key_numeric_cols = [c for c in key_numeric_cols if c in df_proc.columns]
    print(f"[Global FE V12-PDF] Creating Global/Expanding features for {len(key_numeric_cols)} total key features...")
    for col in key_numeric_cols:
        global_mean = g[col].transform('mean'); global_std = g[col].transform('std')
        df_proc[f'{col}_global_mean'] = global_mean; df_proc[f'{col}_global_std'] = global_std
        df_proc[f'{col}_vs_global_mean'] = df_proc[col] - global_mean
        exp_mean = g[col].expanding(min_periods=1).mean(); exp_std = g[col].expanding(min_periods=1).std()
        df_proc[f'{col}_exp_mean'] = exp_mean.reset_index(level=0, drop=True); df_proc[f'{col}_exp_std'] = exp_std.reset_index(level=0, drop=True)
    print("[Global FE V12-PDF] Global/Expanding features created.")

    # --- V12 Step 9 (Time-series) - 동일 ---
    print(f"[Global FE V12-PDF] Creating {len(key_numeric_cols)} time-series features (diff/shift/roll)...")
    for col in key_numeric_cols:
        df_proc[f'{col}_diff'] = g[col].diff()
        df_proc[f'{col}_shift1'] = g[col].shift(1)
        roll_mean = g[col].rolling(3, min_periods=1).mean()
        df_proc[f'{col}_roll3_mean'] = roll_mean.reset_index(level=0, drop=True)
    print("[Global FE V12-PDF] Time-series features created.")

    # --- V12 Drop (동일) ---
    df_proc = df_proc.drop(columns=['Age', 'TestDate', 'TestDate_num', 'FirstTestYear'], errors='ignore')
    return df_proc


# -------------------\
# [V12-CatBoost] 단일 Fold 학습 함수 수정 (CatBoostError 해결 포함)
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
    X_tr_raw, X_val_raw = X_full.iloc[train_indices], X_full.iloc[val_indices]
    y_tr, y_val = y_full[train_indices], y_full[val_indices]

    # 2. [V8] A/B 피처 분리
    if which == "A":
        cols_to_drop = [c for c in X_tr_raw.columns if c.startswith("B")]
        X_tr_raw = X_tr_raw.drop(columns=cols_to_drop, errors='ignore')
        X_val_raw = X_val_raw.drop(columns=cols_to_drop, errors='ignore')
        print(f"{fold_label} Dropped {len(cols_to_drop)} 'B' features.")
    elif which == "B":
        cols_to_drop = [c for c in X_tr_raw.columns if c.startswith("A")]
        X_tr_raw = X_tr_raw.drop(columns=cols_to_drop, errors='ignore')
        X_val_raw = X_val_raw.drop(columns=cols_to_drop, errors='ignore')
        print(f"{fold_label} Dropped {len(cols_to_drop)} 'A' features.")

    # 3. 전처리기 (Preprocessor) - CatBoost용으로 수정됨
    # PrimaryKey는 GroupKFold에만 사용되므로 여기서 제외
    num_cols, cat_cols = separate_num_cat(X_tr_raw, drop_cols=['PrimaryKey'])
    print(f"{fold_label} Preprocessing: {len(num_cols)} num_cols, {len(cat_cols)} cat_cols.")

    preproc = build_preprocessor(num_cols, cat_cols)

    # fit_transform은 numpy 배열을 반환하므로 컬럼 이름 복원 필요
    X_tr_t_np = preproc.fit_transform(X_tr_raw)
    X_val_t_np = preproc.transform(X_val_raw)

    # ColumnTransformer의 get_feature_names_out 사용 (sklearn >= 1.0)
    try:
        feature_names = preproc.get_feature_names_out()
    except AttributeError: # 이전 버전 호환성
         # 수동으로 이름 재구성 (num + cat + remainder 순서 유의)
         remainder_cols = [c for c in X_tr_raw.columns if c not in num_cols + cat_cols and c != 'PrimaryKey']
         feature_names = num_cols + cat_cols + remainder_cols # remainder="passthrough"이므로 순서 중요

    X_tr_t = pd.DataFrame(X_tr_t_np, columns=feature_names, index=X_tr_raw.index)
    X_val_t = pd.DataFrame(X_val_t_np, columns=feature_names, index=X_val_raw.index)

    # --- 오류 수정 시작 ---
    # Imputation 및 변환 후, CatBoost가 사용하기 전에
    # 범주형 컬럼들을 다시 문자열 타입으로 변환합니다.
    # feature_names에는 'cat__' 같은 접두사가 붙어 있을 수 있습니다.
    cat_cols_after_transform = [col for col in feature_names if col.split('__')[-1] in cat_cols]

    for col in cat_cols_after_transform:
        if col in X_tr_t.columns: # 컬럼 존재 여부 확인
             # object 타입으로 먼저 변환 후, NaN을 'missing' 문자열로 채우고, 최종적으로 str 타입으로 변환
             # 'passthrough'된 컬럼에 NaN이 있을 경우를 대비합니다.
             X_tr_t[col] = X_tr_t[col].astype(object).fillna('missing').astype(str)
             X_val_t[col] = X_val_t[col].astype(object).fillna('missing').astype(str)
    # --- 오류 수정 끝 ---


    # 나머지 object 컬럼('passthrough' 등으로 인해 남아있을 수 있음)을 숫자로 변환
    for col in X_tr_t.columns:
         # 이미 문자열로 변환한 범주형 컬럼은 제외
         if X_tr_t[col].dtype == 'object' and col not in cat_cols_after_transform:
             X_tr_t[col] = pd.to_numeric(X_tr_t[col], errors='coerce').fillna(0)
             X_val_t[col] = pd.to_numeric(X_val_t[col], errors='coerce').fillna(0)


    # 범주형 컬럼 인덱스 찾기 (CatBoost Pool용)
    cat_feature_indices = []
    # 최종 DataFrame인 X_tr_t의 컬럼을 기준으로 인덱스 계산
    for i, col_name in enumerate(X_tr_t.columns):
        original_col_name = col_name.split('__')[-1]
        if original_col_name in cat_cols:
             # 컬럼이 실제 존재하는지 확인 후 인덱스 추가
             if col_name in X_tr_t.columns:
                  cat_feature_indices.append(i)

    print(f"{fold_label} Cat feature indices: {cat_feature_indices}") # 이제 빈 리스트가 아니어야 함


    # 4. Optuna (Fold별 파라미터 탐색) - CatBoost용으로 수정
    print(f"{fold_label} Running Optuna search for CatBoost...")

    def objective(trial):
        # CatBoost 하이퍼파라미터 탐색 공간
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500, step=100), # 반복 횟수
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8), # 트리 깊이
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True), # L2 정규화
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128]), # 연속형 변수 분할 개수
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0), # Bagging 강도
            'random_strength': trial.suggest_float('random_strength', 1e-3, 1.0, log=True), # 분할 시 무작위성
            'loss_function': 'Logloss',
            'eval_metric': Leaderboard_metric(), # 커스텀 메트릭
            'random_seed': RANDOM_STATE,
            'verbose': 0, # Optuna 실행 중에는 로그 끄기
            'early_stopping_rounds': 50,
            'task_type': 'CPU',
        }

        model = CatBoostClassifier(**params)

        # Pool 객체 생성하여 학습/검증 데이터 전달
        train_pool = Pool(data=X_tr_t, label=y_tr, cat_features=cat_feature_indices)
        val_pool = Pool(data=X_val_t, label=y_val, cat_features=cat_feature_indices)

        model.fit(train_pool, eval_set=val_pool, use_best_model=True) # use_best_model=True 중요

        # 최적 반복 횟수에서의 검증 점수 (Leaderboard_metric 값)
        # --- KeyError 해결 코드 적용 ---
        best_score = model.get_best_score()['validation']['Leaderboard_metric']
        # ---------------------------

        # Optuna는 minimize 방향이므로 Leaderboard_metric 값 그대로 반환
        return best_score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

    best_params = study.best_params
    print(f"{fold_label} Optuna finished. Best Score: {study.best_value:.5f}")

    # 최종 모델 학습용 파라미터 업데이트
    fold_cb_params = best_params.copy()
    print(f"{fold_label} Params chosen (iterations adjusted by early stopping).")

    # 5. 앙상블 학습 (Fold별 최적 파라미터 사용)
    members = []
    for sd in ENSEMBLE_SEEDS:
        # build_model 사용
        # fold_cb_params에 범주형 인덱스 정보 추가 전달 필요 -> build_model 내부에서 사용 안 하므로 제거
        # final_params_member = fold_cb_params.copy()
        # final_params_member.pop('early_stopping_rounds', None) # 최종 학습 시 조기 종료 비활성화
        # final_params_member['verbose'] = 0 # 학습 중 로그 끔

        # Optuna에서 찾은 파라미터로 모델 생성 (early_stopping_rounds 제외)
        member_params_optuna = fold_cb_params.copy()
        member_params_optuna.pop('early_stopping_rounds', None)
        member_params_optuna['verbose'] = 0 # 로그 끔

        base = build_model(sd, member_params_optuna) # build_model 내부에서 CatBoostClassifier 생성

        # Pool 대신 DataFrame과 cat_features 직접 전달하여 학습
        base.fit(X_tr_t, y_tr, cat_features=cat_feature_indices)

        # Calibration 적용
        mdl = maybe_calibrate(base, X_val_t, y_val) # X_val_t는 DataFrame
        members.append(mdl)

    ensemble = AvgProbaEnsemble(members)

    # 최종 로그 (동일)
    try:
        val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1-7)
        auc = roc_auc_score(y_val, val_proba); brier = brier_score_loss(y_val, val_proba); ece = expected_calibration_error(y_val, val_proba)
        final_score = 0.5 * (1.0 - auc) + 0.25 * brier + 0.25 * ece
        print(f"{fold_label} Holdout: AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f} -> (Score={final_score:.5f})")
    except Exception as e:
        print(f"{fold_label} validation logging skipped: {e}")

    # 6. 모델 저장 (전처리기, 앙상블 모델)
    joblib.dump(preproc, preproc_path)
    joblib.dump(ensemble, model_path)
    print(f"{fold_label} trained and saved → {model_path}")

# -------------------\
# [V12-CatBoost] K-Fold 추론 함수 (_predict_single 수정)
# -------------------\

def _predict_single(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    preproc, # 로드된 ColumnTransformer
    clf_or_ens, # 로드된 AvgProbaEnsemble (내부에 Calibrated CatBoost)
    which: str
) -> np.ndarray:
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")

    # A/B 피처 분리 (동일)
    if which == "A":
        cols_to_drop = [c for c in df.columns if c.startswith("B")]
        df = df.drop(columns=cols_to_drop, errors='ignore')
    elif which == "B":
        cols_to_drop = [c for c in df.columns if c.startswith("A")]
        df = df.drop(columns=cols_to_drop, errors='ignore')

    # 누수 피처 제거 (동일)
    drop_cols = [key, "PrimaryKey"] + \
                (["Test"] if "Test" in df.columns else []) + \
                ['Test_x', 'Test_y']

    X_raw = df.drop(columns=drop_cols, errors="ignore")

    # 전처리기 적용 (transform은 numpy 배열 반환)
    X_t_np = preproc.transform(X_raw)

    # DataFrame으로 변환 (컬럼 이름 필요)
    try:
        feature_names = preproc.get_feature_names_out()
    except AttributeError:
         # 수동으로 이름 재구성 (fit 시점의 num/cat 순서와 동일해야 함)
         num_cols_fitted = preproc.transformers_[0][2] # 'num' transformer의 컬럼들
         cat_cols_fitted = preproc.transformers_[1][2] # 'cat' transformer의 컬럼들
         remainder_cols = [c for c in X_raw.columns if c not in num_cols_fitted + cat_cols_fitted]
         feature_names = list(num_cols_fitted) + list(cat_cols_fitted) + remainder_cols

    X_t = pd.DataFrame(X_t_np, columns=feature_names, index=X_raw.index)

    # NaN 처리 및 object 타입 변환 (학습 시점과 동일하게)
    for col in X_t.columns:
         if X_t[col].dtype == 'object':
             X_t[col] = pd.to_numeric(X_t[col], errors='coerce').fillna(0)


    # 예측 (AvgProbaEnsemble은 내부 모델들의 predict_proba 호출)
    # CalibratedClassifier는 DataFrame 입력 가능
    proba = np.clip(clf_or_ens.predict_proba(X_t)[:, 1], 1e-7, 1-1e-7)
    return proba


# predict_partition_kfold (동일)
def predict_partition_kfold(
    df_feat: pd.DataFrame,
    df_idx: pd.DataFrame,
    which: str
) -> pd.DataFrame:
    # ... (V12와 동일) ...
    key = "Test_id"
    all_probas = []

    print(f"[{which}] Starting K-Fold Prediction ({N_SPLITS_KFold} folds)...")

    for fold_id in range(N_SPLITS_KFold):
        MODEL_PATH = os.path.join(MODEL_DIR, f"model_{which}_fold{fold_id}.joblib")
        PREPROC_PATH = os.path.join(MODEL_DIR, f"preproc_{which}_fold{fold_id}.joblib")

        if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH)):
            print(f"[{which} Fold {fold_id+1}] ERROR: Model/Preproc file not found. Skipping.")
            continue

        try:
            preproc = joblib.load(PREPROC_PATH)
            ensemble = joblib.load(MODEL_PATH)

            proba_fold = _predict_single(df_feat, df_idx, preproc, ensemble, which)
            all_probas.append(proba_fold)
            print(f"[{which} Fold {fold_id+1}] Prediction loaded.")

        except Exception as e:
            print(f"[{which} Fold {fold_id+1}] ERROR loading/predicting: {e}")
            # 상세 에러 확인을 위해 traceback 추가 가능
            # import traceback
            # print(traceback.format_exc())


    if not all_probas:
        print(f"[{which}] No fold models found. Returning 0.001.")
        out = df_idx[[key]].copy(); out["Label"] = 0.001; out["__which__"] = which
        return out

    final_proba = np.mean(all_probas, axis=0)
    out = df_idx[[key]].copy(); out["Label"] = final_proba; out["__which__"] = which
    print(f"[{which}] K-Fold Prediction finished (Avg of {len(all_probas)} models).")
    return out


# -------------------\
# [V12-CatBoost] 메타 저장 수정
# -------------------\
def save_meta():
    meta = dict(
        model=f"CatBoost({N_SPLITS_KFold}-Fold Ensemble) + {len(ENSEMBLE_SEEDS)}-seed AvgProba + Calib [Optuna V12-PDF-CatBoost]", # 모델명 변경
        feature_engineering=[
            "Age_numeric, TestYear, TestMonth, TestCount, TestSequence, etc.",
            "NA_COUNT, NA_RATIO (row-wise)",
            "A9_..., B9_..., B10_... (V12-PDF: B9/B10 features refined with fixed trials)",
            "[V12-PDF-FE] Added PDF-based features (Accuracy, Ratios) for all A/B tasks (A1-A8, B1-B8).",
            "[V6-FE] Time-Series features (diff, shift, roll3_mean) applied to all key numeric features.",
            "[V8-FE] A/B Feature Splitting",
            "[V9-Fix] Removed 'Test_x', 'Test_y' leakage features",
            "[V11-FE] Added Global Stats (transform mean/std, vs_mean)",
            "[V11-FE] Added Expanding Stats (expanding mean/std)",
            "[V11.1-Fix] Fixed 'g' groupby object refresh order for NA_COUNT",
            "[V12-CatBoost] Removed OrdinalEncoder, using CatBoost native category handling." # 변경 사항 추가
        ],
        validation_strategy=f"GroupKFold (n_splits={N_SPLITS_KFold}) on PrimaryKey. Full K-Fold Ensemble.",
        # BASE_HGB_PARAMS 제거
        optuna_n_trials_per_fold=OPTUNA_N_TRIALS,
        optuna_search_space="CatBoost specific (depth, l2_leaf_reg, lr, iters, etc.)", # 명시
        ensemble_seeds=list(ENSEMBLE_SEEDS),
        use_calibration=USE_CALIBRATION,
        calib_method=CALIB_METHOD,
        calib_cv=f"{CALIB_CV} (fallback) or 'prefit' (if sk-ver >= 1.4)",
        sklearn_version=sklver, # CatBoost 버전도 추가하면 좋음 (catboost.__version__)
        random_state=RANDOM_STATE,
    )
    # meta['hgb_base_params'] = {k: str(v) for k, v in meta['hgb_base_params'].items()} # 제거

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# -------------------\
# 메인 함수 (거의 동일, create_features 호출 확인)
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

    # [V12-PDF] 수정된 create_features 함수 호출 (동일)
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
        groups_A = df_A_full['PrimaryKey'].values # GroupKFold용 그룹 정보

        drop_cols_prep = [key_col, label_col, "Test_x", 'Test_y', "Test"] # PrimaryKey는 남겨둠
        X_A_full = df_A_full.drop(columns=drop_cols_prep, errors="ignore")

        gkf_A = GroupKFold(n_splits=N_SPLITS_KFold)

        for fold_id, (train_indices, val_indices) in enumerate(gkf_A.split(X_A_full, y_A, groups_A)):
            A_MODEL_PATH = A_MODEL_PATH_TPL.format(fold=fold_id)
            A_PREPROC_PATH = A_PREPROC_PATH_TPL.format(fold=fold_id)

            if not (os.path.exists(A_MODEL_PATH) and os.path.exists(A_PREPROC_PATH)):
                print(f"--- [A] Training Fold {fold_id+1}/{N_SPLITS_KFold} ---")
                # train_single_fold 호출 시 X_A_full에서 PrimaryKey 제거 안 함 (내부에서 처리)
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

        drop_cols_prep = [key_col, label_col, "Test_x", 'Test_y', "Test"] # PrimaryKey 남김
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

    # --- 4. 추론 (동일) ---
    print("-" * 50)
    print(f"[Main] K-Fold Prediction starting...")
    print("-" * 50)
    preds_A = predict_partition_kfold(test_feat_processed, A_test_idx, "A") if len(A_test_idx) > 0 else None
    preds_B = predict_partition_kfold(test_feat_processed, B_test_idx, "B") if len(B_test_idx) > 0 else None

    # --- 5. 제출 파일 생성 (동일) ---
    if preds_A is not None and preds_B is not None:
        sub = pd.concat([preds_A, preds_B], axis=0, ignore_index=True)
    elif preds_A is not None: sub = preds_A.copy()
    elif preds_B is not None: sub = preds_B.copy()
    else:
        sub = test_idx[[key_col]].copy(); sub["Label"] = 0.001

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

    save_meta() # 수정된 메타 정보 저장

    dt = time.time() - t0
    print(f"[V12-PDF-CatBoost] submission saved -> {SUBMISSION_PATH} | elapsed: {dt:.2f}s")


if __name__ == "__main__":
    main()