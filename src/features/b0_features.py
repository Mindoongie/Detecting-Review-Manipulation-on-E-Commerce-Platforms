"""
b0_features.py — TV5 Phase 1
Trích xuất đặc trưng B0: đặc trưng thô cơ bản từ review đơn lẻ.

Mục đích: Kiểm tra thông suốt pipeline (sanity-check baseline), hoàn toàn không dùng
thông tin tổng hợp lịch sử của user, product hay TF-IDF.

Đặc trưng B0 (8 cột):
  1. rating                  — sao đánh giá (1–5)
  2. review_length_chars     — độ dài review (ký tự)
  3. review_length_words     — số từ
  4. review_length_sentences — số câu (ước lượng)
  5. avg_word_length         — độ dài trung bình mỗi từ
  6. exclamation_count       — số dấu chấm than (!)
  7. question_mark_count     — số dấu chấm hỏi (?)
  8. capital_ratio           — tỷ lệ ký tự chữ cái viết hoa
"""

import re
import numpy as np
import pandas as pd


B0_FEATURE_NAMES = [
    'rating',
    'review_length_chars',
    'review_length_words',
    'review_length_sentences',
    'avg_word_length',
    'exclamation_count',
    'question_mark_count',
    'capital_ratio',
]


def _count_sentences(text: str) -> int:
    """Ước lượng số câu dựa trên dấu kết thúc câu."""
    stripped = text.strip()
    if not stripped:
        return 0
    sentences = re.split(r'[.!?]+', stripped)
    return max(1, sum(1 for s in sentences if s.strip()))


def extract_b0_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trích xuất bộ đặc trưng B0 (thô cơ bản) cho từng review.
    Giữ nguyên index của df và đảm bảo không chứa NaN/Inf.
    """
    features = pd.DataFrame(index=df.index)
    features['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0).astype(np.float32)

    text_series = df['text'].fillna('').astype(str)
    features['review_length_chars'] = text_series.str.len().astype(np.float32)

    words_list = text_series.str.split()
    word_counts = words_list.str.len().fillna(0).astype(int)
    features['review_length_words'] = word_counts.astype(np.float32)

    features['review_length_sentences'] = np.array(
        [_count_sentences(t) for t in text_series], dtype=np.float32
    )

    total_word_chars = np.array(
        [sum(len(w) for w in words) if words else 0 for words in words_list],
        dtype=np.float32,
    )
    features['avg_word_length'] = np.where(
        word_counts.values > 0,
        total_word_chars / np.maximum(word_counts.values, 1),
        0.0,
    ).astype(np.float32)

    features['exclamation_count'] = text_series.str.count('!').astype(np.float32)
    features['question_mark_count'] = text_series.str.count(r'\?').astype(np.float32)

    alpha_counts = text_series.str.count(r'[a-zA-Z]').values.astype(np.float32)
    upper_counts = text_series.str.count(r'[A-Z]').values.astype(np.float32)
    features['capital_ratio'] = np.where(
        alpha_counts > 0,
        upper_counts / np.maximum(alpha_counts, 1.0),
        0.0,
    ).astype(np.float32)

    return features[B0_FEATURE_NAMES]
