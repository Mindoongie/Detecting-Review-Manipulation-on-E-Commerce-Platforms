"""
run_baselines.py — TV5 Phase 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline chính của Chi (TV5 — Giai đoạn 1):
  1. Tải dữ liệu (Unified schema / YelpChi / Realistic Mock)
  2. Chia Train / Val / Test theo thời gian + khóa ranh giới & kiểm tra chống rò rỉ
  3. Xuất bảng ánh xạ split_assignments.csv cho TV1–TV4 dùng chung
  4. Trích đặc trưng B0 (thô cơ bản) và B1 (TF-IDF + hành vi + sản phẩm + thời gian, nhân quả <= T_k)
  5. Huấn luyện & đánh giá B0 vs B1 trên cùng mô hình bảng (Logistic Regression & LightGBM)
  6. Lưu bảng so sánh trên tập Validation (giữ nguyên tập Test cho Giai đoạn 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cách chạy:
  python scripts/run_baselines.py
  python scripts/run_baselines.py --model lightgbm
  python scripts/run_baselines.py --model logistic_regression
  python scripts/run_baselines.py --config config/config.yaml
"""

import argparse
import os
import sys
import time
from typing import Dict, List
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_or_generate, print_data_summary
from src.features.b0_features import extract_b0_features
from src.features.b1_features import B1FeatureExtractor
from src.metrics import compare_models, compute_all_metrics, print_evaluation_report
from src.models.baseline import predict_baseline, print_feature_importance, train_baseline
from src.temporal_split import (
    print_split_summary,
    save_split_assignments,
    temporal_train_val_test_split,
    verify_no_leakage,
)


def load_config(config_path: str) -> dict:
    """Tải file cấu hình YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _resolve_model_types(config: dict, cli_model: str) -> List[str]:
    if cli_model and cli_model != 'all':
        return [cli_model]
    model_cfg = config.get('model', {})
    if 'types' in model_cfg and model_cfg['types']:
        return list(model_cfg['types'])
    if 'type' in model_cfg:
        return [model_cfg['type']]
    return ['logistic_regression', 'lightgbm']


def run_pipeline(config: dict, cli_model: str = 'all') -> Dict:
    """Chạy trọn vẹn pipeline benchmark Giai đoạn 1 trên Train và Validation."""
    seed = config.get('random_seed', 42)
    np.random.seed(seed)

    results_dir = config['output']['results_dir']
    models_dir = config['output']['models_dir']
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_types = _resolve_model_types(config, cli_model)

    # ── BƯỚC 1: Tải dữ liệu ──
    print("\n" + "█" * 64)
    print("  BƯỚC 1: TẢI DỮ LIỆU")
    print("█" * 64)
    t0 = time.time()
    df = load_or_generate(config)
    print_data_summary(df)
    print(f"  ⏱ Thời gian tải dữ liệu: {time.time() - t0:.2f}s")

    # ── BƯỚC 2: Chia Temporal Split & Chốt Decision Points ──
    print("\n" + "█" * 64)
    print("  BƯỚC 2: CHIA TEMPORAL SPLIT & KIỂM TRA CHỐNG RÒ RỈ")
    print("█" * 64)
    split_cfg = config['split']
    snapshot_freq = split_cfg.get('snapshot_freq', 'W')
    align_boundary = split_cfg.get('align_boundary', True)

    train_df, val_df, test_df, split_info = temporal_train_val_test_split(
        df,
        train_ratio=split_cfg['train_ratio'],
        val_ratio=split_cfg['val_ratio'],
        test_ratio=split_cfg['test_ratio'],
        snapshot_freq=snapshot_freq,
        align_boundary=align_boundary,
    )
    print_split_summary(split_info)

    print("\n  KIỂM TRA CHỐNG RÒ RỈ:")
    passed = verify_no_leakage(
        train_df, val_df, test_df, strict_boundary=align_boundary
    )
    if not passed:
        raise RuntimeError("Kiểm tra chống rò rỉ thất bại — dừng pipeline.")

    split_info_save = {k: str(v) for k, v in split_info.items()}
    split_save_path = os.path.join(results_dir, 'split_info.yaml')
    with open(split_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(split_info_save, f, allow_unicode=True, sort_keys=False)
    print(f"\n  Đã lưu split info → {split_save_path}")

    assign_path = save_split_assignments(
        train_df, val_df, test_df,
        os.path.join(results_dir, 'split_assignments.csv'),
    )
    print(f"  Đã lưu split assignments → {assign_path}")

    # ── BƯỚC 3: Trích xuất đặc trưng B0 & B1 (TUYỆT ĐỐI KHÔNG ĐỤNG TẬP TEST) ──
    print("\n" + "█" * 64)
    print("  BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG B0 & B1 (CHỈ TRAIN & VAL)")
    print("█" * 64)

    t0 = time.time()
    X_train_b0 = extract_b0_features(train_df)
    X_val_b0 = extract_b0_features(val_df)
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    print(f"  B0 features: {X_train_b0.shape[1]} cột | Train {X_train_b0.shape}, Val {X_val_b0.shape} ({time.time() - t0:.2f}s)")

    t0 = time.time()
    b1_cfg = config['features']['b1']
    b1_extractor = B1FeatureExtractor(
        tfidf_max_features=b1_cfg.get('tfidf_max_features', 500),
        tfidf_ngram_range=tuple(b1_cfg.get('tfidf_ngram_range', [1, 2])),
        tfidf_min_df=b1_cfg.get('tfidf_min_df', 5),
        tfidf_max_df=b1_cfg.get('tfidf_max_df', 0.95),
        snapshot_freq=b1_cfg.get('snapshot_freq', snapshot_freq),
        recent_window_days=b1_cfg.get('recent_window_days', 7),
    )
    X_train_b1 = b1_extractor.fit_transform(train_df)
    X_val_b1 = b1_extractor.transform(val_df, history_df=train_df)
    print(f"  B1 features: {X_train_b1.shape[1]} cột | Train {X_train_b1.shape}, Val {X_val_b1.shape} ({time.time() - t0:.2f}s)")

    # ── BƯỚC 4: Train & Đánh giá B0 vs B1 trên các mô hình bảng ──
    print("\n" + "█" * 64)
    print("  BƯỚC 4: TRAIN & ĐÁNH GIÁ B0 vs B1 TRÊN VALIDATION")
    print("█" * 64)

    model_cfg = config['model']
    k_values = config['metrics'].get('k_values', [100, 500, 1000])
    primary_metric = config['metrics'].get('primary', 'average_precision')

    all_results: Dict[str, Dict[str, float]] = {}

    for mtype in model_types:
        mparams = model_cfg.get(mtype, {})
        for level, X_tr, X_va in [
            ('B0', X_train_b0, X_val_b0),
            ('B1', X_train_b1, X_val_b1),
        ]:
            run_label = f"{level}_{mtype}"
            print(f"\n--- Huấn luyện {run_label} ---")
            t0 = time.time()
            model_dict = train_baseline(
                X_tr, y_train,
                model_type=mtype,
                model_params=mparams,
                save_path=os.path.join(models_dir, f"{run_label}.joblib"),
            )
            y_val_prob = predict_baseline(model_dict, X_va)
            val_metrics = compute_all_metrics(
                y_val, y_val_prob, k_values=k_values, tune_threshold=True
            )
            val_metrics['feature_count'] = int(X_tr.shape[1])
            val_metrics['train_time_sec'] = round(time.time() - t0, 3)

            print_evaluation_report(val_metrics, model_name=run_label, split_name="Validation")
            print_feature_importance(model_dict, top_n=12)

            all_results[run_label] = val_metrics
            pd.DataFrame([val_metrics]).to_csv(
                os.path.join(results_dir, f"{run_label}_val_metrics.csv"),
                index=False,
            )

    # ── BƯỚC 5: Bảng tổng hợp so sánh B0 vs B1 ──
    print("\n" + "█" * 64)
    print("  BƯỚC 5: BẢNG SO SÁNH B0 vs B1 TRÊN VALIDATION")
    print("█" * 64)

    comparison = compare_models(all_results, primary_metric=primary_metric)
    summary_cols = [
        'feature_count',
        'auc_roc',
        'average_precision',
        'f1_binary',
        'f1_at_best_threshold',
        'best_threshold',
    ] + [f'precision@{k}' for k in k_values] + [f'recall@{k}' for k in k_values]
    summary_cols = [c for c in summary_cols if c in comparison.columns]
    print("\n" + comparison[summary_cols].round(4).to_string())

    comp_path = os.path.join(results_dir, 'baseline_comparison.csv')
    comparison.to_csv(comp_path)
    print(f"\n  Đã lưu bảng so sánh → {comp_path}")

    print("\n" + "█" * 64)
    print("  TỔNG KẾT TV5 — GIAI ĐOẠN 1")
    print("█" * 64)
    print(
        f"\n  Dữ liệu        : {'Mock (realistic)' if config['data'].get('use_mock') else config['data']['dataset']} ({len(df):,} reviews)"
        f"\n  Temporal Split : Train {split_info['train_size']:,} / Val {split_info['val_size']:,} / Test {split_info['test_size']:,}"
        f"\n  Snapshot Freq  : {snapshot_freq} (align_boundary={align_boundary})"
        f"\n  Test Split     : GIỮ NGUYÊN CHƯA ĐỤNG TỚI (chỉ mở khóa ở GĐ 4)"
        f"\n  Kết quả        : {results_dir}"
        f"\n  Models         : {models_dir}\n"
    )

    return {
        'comparison': comparison,
        'results': all_results,
        'split_info': split_info,
    }


def main():
    parser = argparse.ArgumentParser(description='TV5 Phase 1: Temporal Split + B0/B1 Baselines')
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(ROOT_DIR, 'config', 'config.yaml'),
        help='Đường dẫn file config YAML',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all', 'lightgbm', 'logistic_regression'],
        help='Chọn mô hình bảng chạy (mặc định: all = cả Logistic Regression và LightGBM)',
    )
    parser.add_argument(
        '--real-data',
        action='store_true',
        help='Ghi đè use_mock=false để đọc dữ liệu thật từ data/processed hoặc data/raw',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.real_data:
        config['data']['use_mock'] = False

    print("=" * 64)
    print("  TV5 (Chi) — Giao thức Benchmark, Baseline & Giả lập")
    print("  GIAI ĐOẠN 1: Temporal Split + Decision Point + B0/B1")
    print("=" * 64)
    print(f"  Config : {args.config}")
    print(f"  Dataset: {config['data']['dataset']} (use_mock={config['data'].get('use_mock', False)})")
    print(f"  Model  : {args.model}")
    print(f"  Seed   : {config.get('random_seed', 42)}")

    return run_pipeline(config, cli_model=args.model)


if __name__ == '__main__':
    main()
