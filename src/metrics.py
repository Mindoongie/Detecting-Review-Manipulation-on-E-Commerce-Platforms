"""
metrics.py — TV5 Phase 1
Module đánh giá hiệu năng mô hình phát hiện review giả.

Thước đo chuẩn theo chốt.md:
  1. AUC-ROC            — Khả năng phân hạng tổng thể (threshold-independent)
  2. Average Precision  — Độ chính xác trung bình theo Precision-Recall (nhạy khi mất cân bằng lớp)
  3. F1 Binary          — Cân bằng Precision/Recall tại ngưỡng vận hành
  4. Precision@K        — Tỷ lệ fake thật trong top K review bị gắn cờ cao nhất
  5. Recall@K           — Tỷ lệ fake tìm được trong top K review bị gắn cờ cao nhất

Quy tắc chống rò rỉ ngưỡng:
  - Trên Validation (tune_threshold=True): tìm ngưỡng tối ưu F1 bằng Precision-Recall curve.
  - Trên Test cuối (tune_threshold=False): PHẢI áp dụng ngưỡng đã chốt từ Validation, tuyệt đối không dò lại ngưỡng trên Test.
"""

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: Optional[List[int]] = None,
    threshold: float = 0.5,
    tune_threshold: bool = True,
) -> Dict[str, float]:
    """
    Tính toàn bộ thước đo chuẩn của TV5.

    Parameters
    ----------
    y_true : array-like — nhãn thực (0 = genuine, 1 = fake)
    y_prob : array-like — xác suất dự đoán lớp 1 (fake)
    k_values : list[int] — danh sách giá trị K cho Precision@K và Recall@K
    threshold : float — ngưỡng phân loại nhị phân (mặc định 0.5 hoặc ngưỡng chốt từ Val)
    tune_threshold : bool — True trên Validation để tìm best_threshold; False trên Test cuối
    """
    if k_values is None:
        k_values = [100, 500, 1000]

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    results: Dict[str, float] = {}

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        results['auc_roc'] = float('nan')
        results['average_precision'] = float('nan')
    else:
        results['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        results['average_precision'] = float(average_precision_score(y_true, y_prob))

    results['threshold_used'] = float(threshold)
    results['f1_binary'] = float(f1_score(y_true, y_pred, zero_division=0))
    results['precision_binary'] = float(precision_score(y_true, y_pred, zero_division=0))
    results['recall_binary'] = float(recall_score(y_true, y_pred, zero_division=0))

    n = len(y_true)
    for k in k_values:
        if k <= 0 or k > n:
            if k > n:
                warnings.warn(
                    f"Bỏ qua Precision@{k}/Recall@{k} vì số mẫu ({n}) < K ({k}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            results[f'precision@{k}'] = float('nan')
            results[f'recall@{k}'] = float('nan')
        else:
            p_at_k, r_at_k = precision_recall_at_k(y_true, y_prob, k)
            results[f'precision@{k}'] = p_at_k
            results[f'recall@{k}'] = r_at_k

    if tune_threshold:
        best_thresh, best_f1, best_p, best_r = find_best_threshold(y_true, y_prob)
        results['best_threshold'] = best_thresh
        results['f1_at_best_threshold'] = best_f1
        results['precision_at_best_threshold'] = best_p
        results['recall_at_best_threshold'] = best_r

    return results


def precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
) -> Tuple[float, float]:
    """
    Tính Precision@K và Recall@K theo thứ tự xác suất fake giảm dần.
    Nếu k <= 0 hoặc k > len(y_true), trả về (nan, nan) để tránh báo cáo sai lệch.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    if k <= 0 or k > n or n == 0:
        return float('nan'), float('nan')

    top_k_idx = np.argsort(-y_prob, kind='mergesort')[:k]
    n_fake_in_top_k = int(y_true[top_k_idx].sum())
    total_fake = int(y_true.sum())

    precision_at_k = n_fake_in_top_k / float(k)
    recall_at_k = (n_fake_in_top_k / float(total_fake)) if total_fake > 0 else 0.0

    return float(precision_at_k), float(recall_at_k)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Tìm ngưỡng tối ưu F1 chính xác thông qua Precision-Recall curve.

    Returns
    -------
    (best_threshold, best_f1, best_precision, best_recall)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2 or y_true.sum() == 0:
        return 0.5, 0.0, 0.0, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve trả về len(thresholds) == len(precisions) - 1
    p = precisions[:-1]
    r = recalls[:-1]
    denom = p + r
    f1s = np.where(denom > 0, 2.0 * p * r / denom, 0.0)

    if len(f1s) == 0:
        return 0.5, 0.0, 0.0, 0.0

    best_idx = int(np.argmax(f1s))
    return (
        float(thresholds[best_idx]),
        float(f1s[best_idx]),
        float(p[best_idx]),
        float(r[best_idx]),
    )


def print_evaluation_report(
    metrics_dict: Dict[str, float],
    model_name: str = "Model",
    split_name: str = "Validation",
):
    """In báo cáo đánh giá theo thứ tự thước đo rõ ràng."""
    print(f"\n{'=' * 64}")
    print(f"  ĐÁNH GIÁ: {model_name} trên tập {split_name}")
    print(f"{'=' * 64}")
    print(f"  {'Metric':<32} {'Value':>12}")
    print(f"  {'-' * 46}")

    priority = [
        'auc_roc',
        'average_precision',
        'threshold_used',
        'f1_binary',
        'precision_binary',
        'recall_binary',
        'best_threshold',
        'f1_at_best_threshold',
        'precision_at_best_threshold',
        'recall_at_best_threshold',
    ]

    for key in priority:
        if key in metrics_dict:
            print(f"  {key:<32} {metrics_dict[key]:>12.4f}")

    k_keys = [k for k in metrics_dict.keys() if '@' in k]
    if k_keys:
        # Sắp xếp tăng dần theo K số học (100 -> 500 -> 1000)
        k_keys_sorted = sorted(
            k_keys,
            key=lambda name: (int(name.split('@')[1]), name.split('@')[0]),
        )
        print(f"  {'-' * 46}")
        for key in k_keys_sorted:
            val = metrics_dict[key]
            val_str = f"{val:>12.4f}" if not np.isnan(val) else f"{'N/A (N<K)':>12}"
            print(f"  {key:<32} {val_str}")

    print(f"{'=' * 64}")


def compare_models(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = 'average_precision',
) -> pd.DataFrame:
    """
    So sánh các model trên cùng tập thước đo, sắp xếp theo primary_metric giảm dần.
    """
    df = pd.DataFrame(results).T
    df.index.name = 'model'

    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False)

    return df
