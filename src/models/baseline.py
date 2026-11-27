"""
baseline.py — TV5 Phase 1
Module huấn luyện & dự đoán baseline models (Logistic Regression / LightGBM).

Hai mức baseline trong Giai đoạn 1:
  - B0: đặc trưng thô cơ bản (kiểm tra thông suốt pipeline)
  - B1: đặc trưng mạnh (nội dung TF-IDF + hành vi user + thống kê sản phẩm + thời gian)
"""

import os
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def train_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str = 'lightgbm',
    model_params: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Huấn luyện mô hình bảng baseline (LightGBM hoặc Logistic Regression).
    """
    if model_params is None:
        model_params = {}

    feature_names = list(X_train.columns)
    X_arr = np.nan_to_num(
        X_train.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    y = np.asarray(y_train).astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError("y_train phải chứa cả 2 lớp (0 và 1) để huấn luyện.")

    scaler = None
    actual_type = model_type

    if model_type == 'lightgbm':
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM chưa được cài đặt trong môi trường Python.")
        print(f"[Train] Huấn luyện LightGBM ({len(X_arr):,} samples, {X_arr.shape[1]} features)...")
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        params.update(model_params)
        model = lgb.LGBMClassifier(**params)
        # Truyền DataFrame với tên cột chuẩn để giữ feature names
        X_fit = pd.DataFrame(X_arr, columns=feature_names)
        model.fit(X_fit, y)

    elif model_type == 'logistic_regression':
        print(
            f"[Train] Huấn luyện Logistic Regression ({len(X_arr):,} samples, "
            f"{X_arr.shape[1]} features)..."
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        params = {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'lbfgs',
            'random_state': 42,
        }
        params.update(model_params)
        model = LogisticRegression(**params)
        model.fit(X_scaled, y)
    else:
        raise ValueError(
            f"model_type không hỗ trợ: {model_type!r}. "
            "Chọn 'lightgbm' hoặc 'logistic_regression'."
        )

    result = {
        'model': model,
        'scaler': scaler,
        'model_type': actual_type,
        'feature_names': feature_names,
        'n_train_samples': len(X_arr),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(result, save_path)
        print(f"[Train] Model đã lưu → {save_path}")

    return result


def predict_baseline(
    model_dict: Dict[str, Any],
    X: pd.DataFrame,
) -> np.ndarray:
    """Dự đoán xác suất thuộc lớp 1 (fake) cho mỗi review."""
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']

    X_aligned = X[feature_names]
    X_arr = np.nan_to_num(
        X_aligned.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    if scaler is not None:
        X_in = scaler.transform(X_arr)
    elif model_dict['model_type'] == 'lightgbm':
        X_in = pd.DataFrame(X_arr, columns=feature_names)
    else:
        X_in = X_arr

    return model.predict_proba(X_in)[:, 1]


def print_feature_importance(
    model_dict: Dict[str, Any],
    top_n: int = 15,
):
    """In top-N đặc trưng quan trọng nhất (LightGBM gain/split hoặc |coef| của LR)."""
    model = model_dict['model']
    feature_names = model_dict['feature_names']

    if model_dict['model_type'] == 'lightgbm' and hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print("[INFO] Model không hỗ trợ feature importance.")
        return

    k = min(top_n, len(feature_names))
    idx = np.argsort(importance)[::-1][:k]

    print(f"\n  TOP {k} ĐẶC TRƯNG QUAN TRỌNG ({model_dict['model_type']}):")
    print(f"  {'#':<4} {'Feature':<40} {'Importance':>12}")
    print(f"  {'-' * 58}")
    for rank, i in enumerate(idx, 1):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        print(f"  {rank:<4} {name:<40} {importance[i]:>12.4f}")
