"""
b1_features.py — TV5 Phase 1
Trích xuất đặc trưng B1 (baseline cá nhân mạnh):
  B0 + Nội dung TF-IDF + Hành vi cá nhân User + Thống kê Product + Thời gian.

Cam kết chống rò rỉ dữ liệu (Strict Causal Snapshot Protocol):
  1. TF-IDF & Global Fallbacks chỉ được fit trên tập Train.
  2. Toàn bộ đặc trưng lịch sử User/Product tại một review thuộc lô snapshot T_k
     CHỈ tính trên tập sự kiện (không đọc nhãn `label`) có `snapshot_cutoff <= T_k`.
  3. Các chỉ số trung bình & độ lệch chuẩn của User/Product áp dụng Leave-Self-Out
     (trừ chính review đang xét khỏi tổng cộng dồn đến T_k) để tránh self-target leakage
     khi đánh giá độ lệch rating (`rating_deviation`, `user_rating_deviation`).
  4. Không đọc bất kỳ review nào có `timestamp > T_k` (hoặc `snapshot_cutoff > T_k`).
"""

import re
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .b0_features import B0_FEATURE_NAMES, extract_b0_features
from ..temporal_split import assign_snapshot_cutoff


BEHAVIORAL_USER_FEATURES = [
    'user_review_count',
    'user_avg_rating',
    'user_rating_std',
    'user_pct_positive',
    'user_pct_negative',
    'user_pct_extreme',
    'user_num_products',
    'user_max_reviews_one_day',
    'user_is_singleton',
    'user_review_rate',
    'days_since_first_review',
    'user_avg_review_length',
]

PRODUCT_FEATURES = [
    'product_review_count',
    'product_avg_rating',
    'product_rating_std',
    'product_pct_positive',
    'product_pct_negative',
    'product_pct_extreme',
    'product_singleton_ratio',
    'product_recent_7d_count',
]

TIME_INTERACTION_FEATURES = [
    'day_of_week',
    'is_weekend',
    'rating_deviation',
    'abs_rating_deviation',
    'user_rating_deviation',
    'is_first_review_of_product',
]


def _sanitize_feature_name(token: str) -> str:
    """Làm sạch tên token để tương thích hoàn toàn với LightGBM JSON feature names."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]+', '_', token).strip('_')
    return cleaned if cleaned else 'tok'


def _compute_leave_self_out_stats(
    total_cnt: np.ndarray,
    total_sum: np.ndarray,
    total_sum_sq: np.ndarray,
    self_val: np.ndarray,
    fallback_mean: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính (mean, sample_std) loại bỏ chính phần tử hiện tại khỏi tổng cộng dồn đến T_k.
    Nếu không có review nào khác (other_cnt == 0), trả về fallback_mean và std = 0.
    """
    other_cnt = total_cnt - 1.0
    other_sum = total_sum - self_val
    other_sum_sq = total_sum_sq - (self_val ** 2)

    mean_out = np.where(
        other_cnt > 0,
        other_sum / np.maximum(other_cnt, 1.0),
        fallback_mean,
    )

    # Phương sai mẫu với ddof=1 khi other_cnt >= 2
    numerator = other_sum_sq - (other_sum ** 2) / np.maximum(other_cnt, 1.0)
    var_out = np.where(
        other_cnt >= 2,
        np.maximum(0.0, numerator) / np.maximum(other_cnt - 1.0, 1.0),
        0.0,
    )
    std_out = np.sqrt(var_out)
    return mean_out.astype(np.float32), std_out.astype(np.float32)


class B1FeatureExtractor:
    """
    Extractor cho bộ đặc trưng B1 tuân thủ tuyệt đối mốc thời gian snapshot T_k.

    Cách dùng chuẩn:
      - Train: `X_train = extractor.fit_transform(train_df)`
      - Val:   `X_val   = extractor.transform(val_df, history_df=train_df)`
      - Test:  `X_test  = extractor.transform(test_df, history_df=pd.concat([train_df, val_df]))`
    """

    def __init__(
        self,
        tfidf_max_features: int = 500,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        tfidf_min_df: int = 5,
        tfidf_max_df: float = 0.95,
        snapshot_freq: str = 'W',
        recent_window_days: int = 7,
    ):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.snapshot_freq = snapshot_freq
        self.recent_window_days = recent_window_days

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_feature_names_: List[str] = []
        self.global_mean_rating_: float = 3.5
        self.global_avg_words_: float = 15.0
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> 'B1FeatureExtractor':
        """Fit TF-IDF và các giá trị fallback toàn cục CHỈ trên tập train."""
        print("[B1] Fitting TF-IDF vectorizer & global priors trên tập train...")
        texts = train_df['text'].fillna('').astype(str)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            sublinear_tf=True,
            strip_accents='unicode',
        )
        self.tfidf_vectorizer.fit(texts)

        vocab_terms = self.tfidf_vectorizer.get_feature_names_out()
        self.tfidf_feature_names_ = [
            f"tfidf_{i}_{_sanitize_feature_name(term)}"
            for i, term in enumerate(vocab_terms)
        ]

        ratings = pd.to_numeric(train_df['rating'], errors='coerce').dropna()
        self.global_mean_rating_ = float(ratings.mean()) if len(ratings) > 0 else 3.5
        word_lens = texts.str.split().str.len().fillna(0)
        self.global_avg_words_ = float(word_lens.mean()) if len(word_lens) > 0 else 15.0

        print(
            f"[B1] TF-IDF fitted: {len(self.tfidf_feature_names_)} features "
            f"(requested max: {self.tfidf_max_features}) | "
            f"global_mean_rating={self.global_mean_rating_:.3f}"
        )
        self.is_fitted = True
        return self

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit trên tập train rồi trích xuất đặc trưng nhân quả cho chính tập train."""
        self.fit(train_df)
        return self.transform(train_df, history_df=None)

    def transform(
        self,
        df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Trích xuất toàn bộ đặc trưng B1 cho tập đích `df`, kết hợp lịch sử quá khứ `history_df`.
        Đảm bảo mọi sự kiện ở snapshot > T_k không bao giờ rò rỉ vào các review thuộc snapshot T_k.
        """
        if not self.is_fitted:
            raise RuntimeError("Phải gọi fit() trước transform().")

        target_df = df.copy()
        target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])
        target_df['_is_target'] = True
        target_df['_orig_pos'] = np.arange(len(target_df), dtype=np.int64)

        if history_df is not None and len(history_df) > 0:
            hist = history_df.copy()
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])

            if 'review_id' in hist.columns and 'review_id' in target_df.columns:
                overlap = set(hist['review_id']) & set(target_df['review_id'])
                if overlap:
                    raise ValueError(
                        f"history_df và target_df bị trùng {len(overlap)} review_id."
                    )

            target_max_ts = target_df['timestamp'].max()
            if hist['timestamp'].max() > target_max_ts:
                raise ValueError(
                    f"Phát hiện rò rỉ trong history_df: timestamp max ({hist['timestamp'].max()}) "
                    f"> target_df max ({target_max_ts})."
                )

            hist['_is_target'] = False
            hist['_orig_pos'] = -1
            common_cols = [
                'review_id', 'user_id', 'product_id', 'rating', 'text',
                'timestamp', '_is_target', '_orig_pos',
            ]
            # Không đưa cột `label` vào quá trình tính đặc trưng để bảo đảm tuyệt đối không dùng nhãn
            hist_clean = hist[[c for c in common_cols if c in hist.columns]].copy()
            targ_clean = target_df[[c for c in common_cols if c in target_df.columns]].copy()
            combined = pd.concat([hist_clean, targ_clean], ignore_index=True)
        else:
            keep_cols = [
                'review_id', 'user_id', 'product_id', 'rating', 'text',
                'timestamp', '_is_target', '_orig_pos',
            ]
            combined = target_df[[c for c in keep_cols if c in target_df.columns]].copy()

        # ── B0 & TF-IDF trên target_df (giữ đúng index của df gốc) ──
        b0 = extract_b0_features(target_df)
        tfidf = self._extract_tfidf(target_df)

        # ── Đặc trưng lịch sử nhân quả trên combined (chỉ giữ lại các dòng target) ──
        causal_feats = self._extract_causal_history_features(combined, target_df.index)

        result = pd.concat([b0, tfidf, causal_feats], axis=1)
        expected_cols = self.get_feature_names()
        return result[expected_cols]

    def _extract_tfidf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất TF-IDF bằng vectorizer đã fit trên train."""
        texts = df['text'].fillna('').astype(str)
        mat = self.tfidf_vectorizer.transform(texts)
        if issparse(mat):
            mat = mat.astype(np.float32).toarray()
        else:
            mat = np.asarray(mat, dtype=np.float32)
        return pd.DataFrame(mat, index=df.index, columns=self.tfidf_feature_names_)

    def _extract_causal_history_features(
        self,
        combined: pd.DataFrame,
        target_index: pd.Index,
    ) -> pd.DataFrame:
        """
        Tính đặc trưng User, Product, và Thời gian hoàn toàn nhân quả theo snapshot T_k
        (vectorized O(N log N), không sử dụng dữ liệu tương lai > T_k).
        """
        work = combined.copy()
        work['rating'] = pd.to_numeric(work['rating'], errors='coerce').fillna(self.global_mean_rating_).astype(np.float64)
        work['word_count'] = work['text'].fillna('').astype(str).str.split().str.len().fillna(0).astype(np.float64)
        work['is_positive'] = (work['rating'] >= 4).astype(np.float64)
        work['is_negative'] = (work['rating'] <= 2).astype(np.float64)
        work['is_extreme'] = ((work['rating'] == 1) | (work['rating'] == 5)).astype(np.float64)
        work['rating_sq'] = work['rating'] ** 2
        work['review_date'] = work['timestamp'].dt.floor('D')
        work['snapshot_cutoff'] = assign_snapshot_cutoff(work['timestamp'], freq=self.snapshot_freq)

        sort_cols = ['snapshot_cutoff', 'timestamp']
        if 'review_id' in work.columns:
            sort_cols.append('review_id')
        work = work.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

        # ── 1. USER CAUSAL STATS (theo từng cặp user_id, snapshot_cutoff) ──
        user_snap = (
            work.groupby(['user_id', 'snapshot_cutoff'], sort=False, as_index=False)
            .agg(
                _cnt=('rating', 'size'),
                _sum_r=('rating', 'sum'),
                _sum_r2=('rating_sq', 'sum'),
                _sum_pos=('is_positive', 'sum'),
                _sum_neg=('is_negative', 'sum'),
                _sum_ext=('is_extreme', 'sum'),
                _sum_words=('word_count', 'sum'),
                _snap_first_ts=('timestamp', 'min'),
            )
            .sort_values(['user_id', 'snapshot_cutoff'], kind='mergesort')
        )

        # Số sản phẩm mới và số ngày hoạt động mới trong từng snapshot của user
        first_up = (
            work.sort_values(['user_id', 'snapshot_cutoff', 'timestamp'], kind='mergesort')
            .drop_duplicates(subset=['user_id', 'product_id'], keep='first')
            .groupby(['user_id', 'snapshot_cutoff'], as_index=False)
            .size()
            .rename(columns={'size': '_new_prods'})
        )
        first_ud = (
            work.sort_values(['user_id', 'snapshot_cutoff', 'timestamp'], kind='mergesort')
            .drop_duplicates(subset=['user_id', 'review_date'], keep='first')
            .groupby(['user_id', 'snapshot_cutoff'], as_index=False)
            .size()
            .rename(columns={'size': '_new_days'})
        )
        daily_max_u = (
            work.groupby(['user_id', 'snapshot_cutoff', 'review_date'], as_index=False)
            .size()
            .groupby(['user_id', 'snapshot_cutoff'], as_index=False)['size']
            .max()
            .rename(columns={'size': '_snap_max_daily'})
        )

        user_snap = (
            user_snap
            .merge(first_up, on=['user_id', 'snapshot_cutoff'], how='left')
            .merge(first_ud, on=['user_id', 'snapshot_cutoff'], how='left')
            .merge(daily_max_u, on=['user_id', 'snapshot_cutoff'], how='left')
        )
        user_snap[['_new_prods', '_new_days', '_snap_max_daily']] = (
            user_snap[['_new_prods', '_new_days', '_snap_max_daily']].fillna(0.0)
        )

        ug = user_snap.groupby('user_id', sort=False)
        user_snap['u_cum_cnt'] = ug['_cnt'].cumsum()
        user_snap['u_cum_sum_r'] = ug['_sum_r'].cumsum()
        user_snap['u_cum_sum_r2'] = ug['_sum_r2'].cumsum()
        user_snap['u_cum_pos'] = ug['_sum_pos'].cumsum()
        user_snap['u_cum_neg'] = ug['_sum_neg'].cumsum()
        user_snap['u_cum_ext'] = ug['_sum_ext'].cumsum()
        user_snap['u_cum_words'] = ug['_sum_words'].cumsum()
        user_snap['u_cum_prods'] = ug['_new_prods'].cumsum()
        user_snap['u_cum_days'] = ug['_new_days'].cumsum()
        user_snap['u_cum_max_daily'] = ug['_snap_max_daily'].cummax()
        user_snap['u_first_ts'] = ug['_snap_first_ts'].cummin()

        u_merge_cols = [
            'user_id', 'snapshot_cutoff',
            'u_cum_cnt', 'u_cum_sum_r', 'u_cum_sum_r2',
            'u_cum_pos', 'u_cum_neg', 'u_cum_ext', 'u_cum_words',
            'u_cum_prods', 'u_cum_days', 'u_cum_max_daily', 'u_first_ts',
        ]
        work = work.merge(user_snap[u_merge_cols], on=['user_id', 'snapshot_cutoff'], how='left')

        # ── 2. PRODUCT CAUSAL STATS (theo từng cặp product_id, snapshot_cutoff) ──
        prod_snap = (
            work.groupby(['product_id', 'snapshot_cutoff'], sort=False, as_index=False)
            .agg(
                _p_cnt=('rating', 'size'),
                _p_sum_r=('rating', 'sum'),
                _p_sum_r2=('rating_sq', 'sum'),
                _p_sum_pos=('is_positive', 'sum'),
                _p_sum_neg=('is_negative', 'sum'),
                _p_sum_ext=('is_extreme', 'sum'),
            )
            .sort_values(['product_id', 'snapshot_cutoff'], kind='mergesort')
        )

        # Theo dõi số singleton user trên từng product một cách nhân quả:
        # User là singleton trên product P_1st nếu đã đăng review đầu ở T_1st trên P_1st
        # và CHƯA đăng review thứ 2 tại thời điểm <= T_k.
        user_ordered = work.sort_values(['user_id', 'snapshot_cutoff', 'timestamp'], kind='mergesort')
        user_ordered['_u_rank'] = user_ordered.groupby('user_id', sort=False).cumcount() + 1
        first_revs = user_ordered[user_ordered['_u_rank'] == 1][
            ['user_id', 'product_id', 'snapshot_cutoff']
        ].rename(columns={'product_id': '_p_1st', 'snapshot_cutoff': '_t_1st'})
        second_revs = user_ordered[user_ordered['_u_rank'] == 2][
            ['user_id', 'snapshot_cutoff']
        ].rename(columns={'snapshot_cutoff': '_t_2nd'})

        singleton_events = first_revs.merge(second_revs, on='user_id', how='left')
        inc_events = (
            singleton_events.groupby(['_p_1st', '_t_1st'], as_index=False)
            .size()
            .rename(columns={'_p_1st': 'product_id', '_t_1st': 'snapshot_cutoff', 'size': '_s_delta'})
        )
        dec_source = singleton_events.dropna(subset=['_t_2nd'])
        dec_events = (
            dec_source.groupby(['_p_1st', '_t_2nd'], as_index=False)
            .size()
            .rename(columns={'_p_1st': 'product_id', '_t_2nd': 'snapshot_cutoff', 'size': '_s_dec'})
        )

        # Hợp các mốc snapshot có biến động singleton vào prod_snap (cộng dồn theo product)
        # Lưu ý: dec_events có thể xảy ra ở snapshot mà product đó không có review mới nào,
        # nhưng ta chỉ cần giá trị tại các (product_id, snapshot_cutoff) xuất hiện trong prod_snap:
        # cộng dồn tất cả delta <= T_k cho mỗi product bằng union index rồi cumsum!
        if len(dec_events) > 0:
            dec_events['_s_delta'] = -dec_events['_s_dec']
            all_s_events = pd.concat(
                [
                    inc_events[['product_id', 'snapshot_cutoff', '_s_delta']],
                    dec_events[['product_id', 'snapshot_cutoff', '_s_delta']],
                ],
                ignore_index=True,
            )
        else:
            all_s_events = inc_events[['product_id', 'snapshot_cutoff', '_s_delta']]

        # Gộp prod_snap và all_s_events theo union (product_id, snapshot_cutoff) để cumsum chính xác
        prod_full = prod_snap.merge(
            all_s_events.groupby(['product_id', 'snapshot_cutoff'], as_index=False)['_s_delta'].sum(),
            on=['product_id', 'snapshot_cutoff'],
            how='outer',
        ).sort_values(['product_id', 'snapshot_cutoff'], kind='mergesort')

        for col in ['_p_cnt', '_p_sum_r', '_p_sum_r2', '_p_sum_pos', '_p_sum_neg', '_p_sum_ext', '_s_delta']:
            prod_full[col] = prod_full[col].fillna(0.0)

        pg = prod_full.groupby('product_id', sort=False)
        prod_full['p_cum_cnt'] = pg['_p_cnt'].cumsum()
        prod_full['p_cum_sum_r'] = pg['_p_sum_r'].cumsum()
        prod_full['p_cum_sum_r2'] = pg['_p_sum_r2'].cumsum()
        prod_full['p_cum_pos'] = pg['_p_sum_pos'].cumsum()
        prod_full['p_cum_neg'] = pg['_p_sum_neg'].cumsum()
        prod_full['p_cum_ext'] = pg['_p_sum_ext'].cumsum()
        prod_full['p_cum_singletons'] = pg['_s_delta'].cumsum().clip(lower=0.0)

        # Tính product_recent_7d_count trong (T_k - recent_window_days, T_k] bằng searchsorted
        recent_counts = self._compute_recent_product_counts(work, prod_snap[['product_id', 'snapshot_cutoff']])
        prod_snap_merged = prod_snap[['product_id', 'snapshot_cutoff']].merge(
            prod_full[[
                'product_id', 'snapshot_cutoff',
                'p_cum_cnt', 'p_cum_sum_r', 'p_cum_sum_r2',
                'p_cum_pos', 'p_cum_neg', 'p_cum_ext', 'p_cum_singletons',
            ]],
            on=['product_id', 'snapshot_cutoff'],
            how='left',
        )
        prod_snap_merged['product_recent_7d_count'] = recent_counts

        work = work.merge(prod_snap_merged, on=['product_id', 'snapshot_cutoff'], how='left')

        # ── 3. LỌC VỀ ĐÚNG CÁC DÒNG TARGET THEO THỨ TỰ BAN ĐẦU ──
        t_df = work[work['_is_target']].sort_values('_orig_pos', kind='mergesort')

        out = pd.DataFrame(index=target_index)

        # User features (Leave-Self-Out cho avg/std/avg_words)
        u_cnt = t_df['u_cum_cnt'].values
        u_avg_r, u_std_r = _compute_leave_self_out_stats(
            u_cnt, t_df['u_cum_sum_r'].values, t_df['u_cum_sum_r2'].values,
            t_df['rating'].values, self.global_mean_rating_,
        )
        other_u_cnt = u_cnt - 1.0
        u_avg_len = np.where(
            other_u_cnt > 0,
            (t_df['u_cum_words'].values - t_df['word_count'].values) / np.maximum(other_u_cnt, 1.0),
            self.global_avg_words_,
        ).astype(np.float32)

        out['user_review_count'] = u_cnt.astype(np.float32)
        out['user_avg_rating'] = u_avg_r
        out['user_rating_std'] = u_std_r
        out['user_pct_positive'] = (t_df['u_cum_pos'].values / np.maximum(u_cnt, 1.0)).astype(np.float32)
        out['user_pct_negative'] = (t_df['u_cum_neg'].values / np.maximum(u_cnt, 1.0)).astype(np.float32)
        out['user_pct_extreme'] = (t_df['u_cum_ext'].values / np.maximum(u_cnt, 1.0)).astype(np.float32)
        out['user_num_products'] = t_df['u_cum_prods'].values.astype(np.float32)
        out['user_max_reviews_one_day'] = t_df['u_cum_max_daily'].values.astype(np.float32)
        out['user_is_singleton'] = (u_cnt == 1.0).astype(np.float32)
        out['user_review_rate'] = (
            u_cnt / np.maximum(t_df['u_cum_days'].values, 1.0)
        ).astype(np.float32)
        days_since = (t_df['timestamp'] - t_df['u_first_ts']).dt.total_seconds().values / 86400.0
        out['days_since_first_review'] = np.maximum(0.0, days_since).astype(np.float32)
        out['user_avg_review_length'] = u_avg_len

        # Product features (Leave-Self-Out cho avg/std)
        p_cnt = t_df['p_cum_cnt'].values
        p_avg_r, p_std_r = _compute_leave_self_out_stats(
            p_cnt, t_df['p_cum_sum_r'].values, t_df['p_cum_sum_r2'].values,
            t_df['rating'].values, self.global_mean_rating_,
        )
        out['product_review_count'] = p_cnt.astype(np.float32)
        out['product_avg_rating'] = p_avg_r
        out['product_rating_std'] = p_std_r
        out['product_pct_positive'] = (t_df['p_cum_pos'].values / np.maximum(p_cnt, 1.0)).astype(np.float32)
        out['product_pct_negative'] = (t_df['p_cum_neg'].values / np.maximum(p_cnt, 1.0)).astype(np.float32)
        out['product_pct_extreme'] = (t_df['p_cum_ext'].values / np.maximum(p_cnt, 1.0)).astype(np.float32)
        out['product_singleton_ratio'] = (
            t_df['p_cum_singletons'].values / np.maximum(p_cnt, 1.0)
        ).astype(np.float32)
        out['product_recent_7d_count'] = t_df['product_recent_7d_count'].values.astype(np.float32)

        # Time & Interaction features
        rating_vals = t_df['rating'].values.astype(np.float32)
        out['day_of_week'] = t_df['timestamp'].dt.dayofweek.values.astype(np.float32)
        out['is_weekend'] = (t_df['timestamp'].dt.dayofweek.values >= 5).astype(np.float32)
        out['rating_deviation'] = (rating_vals - p_avg_r).astype(np.float32)
        out['abs_rating_deviation'] = np.abs(out['rating_deviation'].values).astype(np.float32)
        out['user_rating_deviation'] = (rating_vals - u_avg_r).astype(np.float32)
        out['is_first_review_of_product'] = (p_cnt == 1.0).astype(np.float32)

        return out

    def _compute_recent_product_counts(
        self,
        work: pd.DataFrame,
        prod_snaps: pd.DataFrame,
    ) -> np.ndarray:
        """
        Đếm số review của mỗi sản phẩm trong cửa sổ (T_k - recent_window_days, T_k]
        bằng np.searchsorted — hoàn toàn không đọc timestamp > T_k.
        """
        win_ns = np.int64(self.recent_window_days) * np.int64(86400 * 1_000_000_000)

        work_sorted = work[['product_id', 'timestamp']].sort_values(['product_id', 'timestamp'])
        ts_by_prod = {}
        prod_arr = work_sorted['product_id'].values
        ts_ns_arr = work_sorted['timestamp'].astype('datetime64[ns]').astype(np.int64).values

        # Tách lát cắt liên tiếp cho mỗi product_id trong O(N)
        if len(prod_arr) > 0:
            change_idx = np.flatnonzero(prod_arr[1:] != prod_arr[:-1]) + 1
            starts = np.concatenate(([0], change_idx))
            ends = np.concatenate((change_idx, [len(prod_arr)]))
            for s, e in zip(starts, ends):
                ts_by_prod[prod_arr[s]] = ts_ns_arr[s:e]

        snap_ns_vals = prod_snaps['snapshot_cutoff'].astype('datetime64[ns]').astype(np.int64).values
        snap_prods = prod_snaps['product_id'].values
        counts = np.zeros(len(prod_snaps), dtype=np.float32)

        for idx, (pid, t_end_ns) in enumerate(zip(snap_prods, snap_ns_vals)):
            arr = ts_by_prod.get(pid)
            if arr is None:
                continue
            right = np.searchsorted(arr, t_end_ns, side='right')
            left = np.searchsorted(arr, t_end_ns - win_ns, side='right')
            counts[idx] = float(right - left)

        return counts

    def get_feature_names(self) -> List[str]:
        """Trả về toàn bộ danh sách tên cột B1 theo thứ tự cố định."""
        return (
            list(B0_FEATURE_NAMES)
            + list(self.tfidf_feature_names_)
            + list(BEHAVIORAL_USER_FEATURES)
            + list(PRODUCT_FEATURES)
            + list(TIME_INTERACTION_FEATURES)
        )
