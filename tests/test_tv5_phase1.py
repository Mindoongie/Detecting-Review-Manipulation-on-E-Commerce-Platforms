"""
test_tv5_phase1.py — Unit tests & Anti-Leakage Verification cho TV5 (Chi) Giai đoạn 1.
Chạy: pytest tests/test_tv5_phase1.py -v
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data_loader import generate_mock_data, _normalize_columns, _clean_and_cast_schema
from src.temporal_split import (
    assign_snapshot_cutoff,
    temporal_train_val_test_split,
    verify_no_leakage,
)
from src.metrics import compute_all_metrics, precision_recall_at_k, find_best_threshold
from src.features.b0_features import extract_b0_features, B0_FEATURE_NAMES
from src.features.b1_features import B1FeatureExtractor
from src.models.baseline import train_baseline, predict_baseline


class TestDataLoader(unittest.TestCase):
    def test_generate_mock_schema_and_ratios(self):
        df = generate_mock_data(n_reviews=600, fake_ratio=0.15, seed=123)
        self.assertEqual(len(df), 600)
        for col in ['review_id', 'user_id', 'product_id', 'rating', 'text', 'timestamp', 'label']:
            self.assertIn(col, df.columns)
        self.assertTrue(set(df['label'].unique()).issubset({0, 1}))
        self.assertAlmostEqual(df['label'].mean(), 0.15, delta=0.02)
        self.assertTrue(((df['rating'] >= 1) & (df['rating'] <= 5)).all())
        self.assertFalse(df['review_id'].duplicated().any())

    def test_normalize_yelpchi_aliases_and_labels(self):
        raw = pd.DataFrame({
            'reviewerID': ['u1', 'u2', 'u3'],
            'restaurantID': ['p1', 'p1', 'p2'],
            'stars': [5, 1, 4],
            'reviewContent': ['Great!', 'Bad', 'Okay'],
            'date': ['2013-01-01', '2013-01-02', '2013-01-03'],
            'flagged': ['Y', 'N', 'YR'],
        })
        norm = _clean_and_cast_schema(_normalize_columns(raw))
        self.assertListEqual(norm['label'].tolist(), [1, 0, 1])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(norm['timestamp']))


class TestTemporalSplit(unittest.TestCase):
    def test_strict_boundary_alignment_and_no_leakage(self):
        # Tạo dữ liệu có nhiều review trùng ngày để kiểm chứng ranh giới ngày/snapshot không bị cắt đôi
        base = datetime(2020, 1, 1)
        rows = []
        rid = 0
        for week in range(20):
            ts = base + timedelta(days=week * 7)
            for _ in range(10):
                rows.append({
                    'review_id': f'r_{rid}',
                    'user_id': f'u_{rid % 15}',
                    'product_id': f'p_{rid % 5}',
                    'rating': (rid % 5) + 1,
                    'text': 'sample review text for testing',
                    'timestamp': ts,
                    'label': int(rid % 7 == 0),
                })
                rid += 1
        df = pd.DataFrame(rows)

        train_df, val_df, test_df, info = temporal_train_val_test_split(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            snapshot_freq='W', align_boundary=True,
        )
        self.assertTrue(verify_no_leakage(train_df, val_df, test_df, strict_boundary=True))
        self.assertLess(train_df['timestamp'].max(), val_df['timestamp'].min())
        self.assertLess(val_df['timestamp'].max(), test_df['timestamp'].min())
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))

    def test_verify_no_leakage_catches_timestamp_and_id_leaks(self):
        df = generate_mock_data(n_reviews=300, seed=7)
        train_df, val_df, test_df, _ = temporal_train_val_test_split(df)
        self.assertTrue(verify_no_leakage(train_df, val_df, test_df))

        # Cố tình tiêm lỗi rò rỉ timestamp
        bad_val = val_df.copy()
        bad_val.iloc[0, bad_val.columns.get_loc('timestamp')] = train_df['timestamp'].min()
        self.assertFalse(verify_no_leakage(train_df, bad_val, test_df))

        # Cố tình tiêm lỗi trùng review_id
        bad_test = test_df.copy()
        bad_test.iloc[0, bad_test.columns.get_loc('review_id')] = val_df['review_id'].iloc[0]
        self.assertFalse(verify_no_leakage(train_df, val_df, bad_test))


class TestMetrics(unittest.TestCase):
    def test_precision_recall_at_k_and_k_gt_n(self):
        y_true = np.array([1, 0, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1])

        p2, r2 = precision_recall_at_k(y_true, y_prob, k=2)
        self.assertAlmostEqual(p2, 0.5)
        self.assertAlmostEqual(r2, 0.5)

        p3, r3 = precision_recall_at_k(y_true, y_prob, k=3)
        self.assertAlmostEqual(p3, 2 / 3)
        self.assertAlmostEqual(r3, 1.0)

        # K > N phải trả NaN thay vì âm thầm cắt K
        metrics = compute_all_metrics(y_true, y_prob, k_values=[2, 10], tune_threshold=True)
        self.assertAlmostEqual(metrics['precision@2'], 0.5)
        self.assertTrue(np.isnan(metrics['precision@10']))
        self.assertIn('best_threshold', metrics)

    def test_no_threshold_tuning_on_test(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.4, 0.6, 0.1])
        metrics = compute_all_metrics(y_true, y_prob, k_values=[2], threshold=0.5, tune_threshold=False)
        self.assertNotIn('best_threshold', metrics)
        self.assertAlmostEqual(metrics['f1_binary'], 1.0)

    def test_single_class_edge_case(self):
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = compute_all_metrics(y_true, y_prob, k_values=[2])
        self.assertTrue(np.isnan(metrics['auc_roc']))
        self.assertTrue(np.isnan(metrics['average_precision']))


class TestFeaturesAndLeakage(unittest.TestCase):
    def test_b0_features_basic_and_empty_text(self):
        df = pd.DataFrame({
            'rating': [5, 1],
            'text': ['Great FOOD!! Why?', ''],
        }, index=[10, 20])
        feats = extract_b0_features(df)
        self.assertListEqual(list(feats.index), [10, 20])
        self.assertListEqual(list(feats.columns), B0_FEATURE_NAMES)
        self.assertEqual(feats.loc[10, 'exclamation_count'], 2.0)
        self.assertEqual(feats.loc[10, 'question_mark_count'], 1.0)
        self.assertEqual(feats.loc[20, 'review_length_words'], 0.0)
        self.assertFalse(feats.isna().any().any())

    def test_b1_strict_causal_no_future_leakage(self):
        """
        Kiểm chứng BẤT BIẾN NHÂN QUẢ (Causal Invariance):
        Nếu thay đổi toàn bộ rating, user_id, product_id, text của các review ở tương lai (> T_k),
        mọi đặc trưng B1 cho các review ở quá khứ (<= T_k) PHẢI GIỐNG HỆT NHAU 100%.
        """
        df = generate_mock_data(n_reviews=600, seed=99)
        train_df, val_df, test_df, _ = temporal_train_val_test_split(
            df, snapshot_freq='W', align_boundary=True
        )

        extractor = B1FeatureExtractor(
            tfidf_max_features=40, tfidf_min_df=2, snapshot_freq='W'
        )
        X_train_1 = extractor.fit_transform(train_df)
        X_val_1 = extractor.transform(val_df, history_df=train_df)

        # Chọn mốc T_k ở giữa tập validation
        val_snaps = sorted(assign_snapshot_cutoff(val_df['timestamp'], freq='W').unique())
        mid_cutoff = val_snaps[len(val_snaps) // 2]

        # Phá/đầu độc toàn bộ dữ liệu validation SAU mid_cutoff (những review tương lai)
        val_poisoned = val_df.copy()
        future_mask = assign_snapshot_cutoff(val_poisoned['timestamp'], freq='W') > mid_cutoff
        self.assertTrue(future_mask.any())
        val_poisoned.loc[future_mask, 'rating'] = 1
        val_poisoned.loc[future_mask, 'user_id'] = train_df['user_id'].iloc[0]
        val_poisoned.loc[future_mask, 'product_id'] = train_df['product_id'].iloc[0]
        val_poisoned.loc[future_mask, 'text'] = 'POISONED FUTURE SPAM TEXT!!!!!'

        X_val_2 = extractor.transform(val_poisoned, history_df=train_df)

        past_mask = ~future_mask
        np.testing.assert_allclose(
            X_val_1.loc[past_mask].values,
            X_val_2.loc[past_mask].values,
            rtol=0, atol=1e-6,
            err_msg="Phát hiện rò rỉ tương lai trong B1FeatureExtractor!",
        )

        # Kiểm chứng luôn trong nội bộ tập train: sửa nửa cuối train không ảnh hưởng nửa đầu
        train_snaps = sorted(assign_snapshot_cutoff(train_df['timestamp'], freq='W').unique())
        mid_tr_cutoff = train_snaps[len(train_snaps) // 2]
        tr_future = assign_snapshot_cutoff(train_df['timestamp'], freq='W') > mid_tr_cutoff
        tr_past = ~tr_future

        # Giữ nguyên vectorizer đã fit, chỉ kiểm tra phần causal history
        tr_sub_only = train_df.loc[tr_past].copy()
        X_tr_full = extractor.transform(train_df, history_df=None).loc[tr_past]
        X_tr_past_only = extractor.transform(tr_sub_only, history_df=None)
        np.testing.assert_allclose(
            X_tr_full.values, X_tr_past_only.values,
            rtol=0, atol=1e-6,
            err_msg="Phát hiện rò rỉ tương lai bên trong chính tập Train!",
        )
        self.assertFalse(X_train_1.isna().any().any())

    def test_b1_rejects_future_history_df_and_cold_start_works(self):
        df = generate_mock_data(n_reviews=400, seed=11)
        train_df, val_df, test_df, _ = temporal_train_val_test_split(df)
        ext = B1FeatureExtractor(tfidf_max_features=30, tfidf_min_df=2)
        ext.fit(train_df)

        # Truyền test_df làm history cho val_df phải bị chặn
        with self.assertRaises(ValueError):
            ext.transform(val_df, history_df=test_df)

        # Cold-start user/product mới hoàn toàn
        cold_val = val_df.iloc[:5].copy()
        cold_val['user_id'] = ['brand_new_u_' + str(i) for i in range(5)]
        cold_val['product_id'] = ['brand_new_p_' + str(i) for i in range(5)]
        X_cold = ext.transform(cold_val, history_df=train_df)
        self.assertFalse(X_cold.isna().any().any())
        self.assertTrue((X_cold['user_is_singleton'] == 1.0).all())
        self.assertTrue((X_cold['is_first_review_of_product'] == 1.0).all())
        # Leave-self-out phải dùng global_mean_rating_ chứ không trả về rating_deviation == 0 giả tạo
        expected_dev = cold_val['rating'].values - ext.global_mean_rating_
        np.testing.assert_allclose(X_cold['rating_deviation'].values, expected_dev, atol=1e-4)

    def test_end_to_end_b1_beats_b0_on_mock(self):
        df = generate_mock_data(n_reviews=2500, seed=42)
        train_df, val_df, _, _ = temporal_train_val_test_split(df)

        X_tr_b0 = extract_b0_features(train_df)
        X_va_b0 = extract_b0_features(val_df)

        ext = B1FeatureExtractor(tfidf_max_features=100, tfidf_min_df=3)
        X_tr_b1 = ext.fit_transform(train_df)
        X_va_b1 = ext.transform(val_df, history_df=train_df)

        y_tr = train_df['label'].values
        y_va = val_df['label'].values

        m_b0 = train_baseline(X_tr_b0, y_tr, model_type='lightgbm', model_params={'n_estimators': 80})
        m_b1 = train_baseline(X_tr_b1, y_tr, model_type='lightgbm', model_params={'n_estimators': 80})

        met_b0 = compute_all_metrics(y_va, predict_baseline(m_b0, X_va_b0), k_values=[100])
        met_b1 = compute_all_metrics(y_va, predict_baseline(m_b1, X_va_b1), k_values=[100])

        # B0 không được tách 100% tầm thường và B1 phải vượt B0
        self.assertLess(met_b0['auc_roc'], 0.99)
        self.assertGreater(met_b1['average_precision'], met_b0['average_precision'])


if __name__ == '__main__':
    unittest.main()
