"""Train-only text EDA: length stats, n-gram frequencies, and raw/cleaned duplicate rates."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocess import NEGATION_WORDS, save_dataframe


@dataclass(frozen=True)
class EdaConfig:
    """Configuration for Train-only text EDA."""

    top_k_words: int = 30
    top_k_bigrams: int = 30
    top_k_duplicates: int = 20

    @classmethod
    def from_dict(cls, data: dict | None) -> "EdaConfig":
        if not data:
            return cls()
        return cls(
            top_k_words=int(data.get("top_k_words", 30)),
            top_k_bigrams=int(data.get("top_k_bigrams", 30)),
            top_k_duplicates=int(data.get("top_k_duplicates", 20)),
        )


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _duplicate_metrics(df: pd.DataFrame, text_col: str) -> tuple[dict[str, Any], pd.DataFrame]:
    non_empty = df[df[text_col].astype(str).str.len() > 0].copy()
    total_rows = int(len(df))
    non_empty_rows = int(len(non_empty))

    if non_empty_rows == 0:
        empty_stats = {
            "total_rows": total_rows,
            "non_empty_rows": 0,
            "unique_texts": 0,
            "excess_duplicate_count": 0,
            "excess_duplicate_rate": 0.0,
            "rows_in_duplicate_clusters": 0,
            "in_duplicate_cluster_rate": 0.0,
            "duplicate_clusters": 0,
            "distinct_user_duplicate_clusters": 0,
            "same_product_duplicate_clusters": 0,
        }
        return empty_stats, pd.DataFrame(
            columns=[
                "text",
                "occurrence_count",
                "distinct_users",
                "distinct_products",
                "same_product_repeat",
            ]
        )

    grouped = (
        non_empty.groupby(text_col, as_index=False)
        .agg(
            occurrence_count=("review_id", "count"),
            distinct_users=("user_id", "nunique"),
            distinct_products=("product_id", "nunique"),
        )
        .sort_values(by=["occurrence_count", text_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Check if any text repeats on the exact same product.
    same_prod = (
        non_empty.groupby([text_col, "product_id"])["review_id"]
        .count()
        .reset_index(name="prod_occ")
    )
    same_prod_texts = set(same_prod.loc[same_prod["prod_occ"] >= 2, text_col].tolist())

    dup_clusters = grouped[grouped["occurrence_count"] >= 2].copy()
    dup_clusters["same_product_repeat"] = dup_clusters[text_col].isin(same_prod_texts).astype(int)
    dup_clusters = dup_clusters.rename(columns={text_col: "text"})

    unique_texts = int(len(grouped))
    excess_count = non_empty_rows - unique_texts
    rows_in_clusters = int(dup_clusters["occurrence_count"].sum()) if not dup_clusters.empty else 0

    stats = {
        "total_rows": total_rows,
        "non_empty_rows": non_empty_rows,
        "unique_texts": unique_texts,
        "excess_duplicate_count": int(excess_count),
        "excess_duplicate_rate": float(excess_count / non_empty_rows),
        "rows_in_duplicate_clusters": rows_in_clusters,
        "in_duplicate_cluster_rate": float(rows_in_clusters / non_empty_rows),
        "duplicate_clusters": int(len(dup_clusters)),
        "distinct_user_duplicate_clusters": int((dup_clusters["distinct_users"] >= 2).sum())
        if not dup_clusters.empty
        else 0,
        "same_product_duplicate_clusters": int(dup_clusters["same_product_repeat"].sum())
        if not dup_clusters.empty
        else 0,
    }
    return stats, dup_clusters


def run_text_eda(
    df: pd.DataFrame,
    config: EdaConfig | None = None,
    split_col: str = "split",
    train_split: str = "train",
) -> dict[str, Any]:
    """
    Run text EDA strictly on Train rows so Val/Test distributions never leak into EDA.
    """
    cfg = config or EdaConfig()
    if split_col in df.columns:
        train_df = df[df[split_col] == train_split].copy()
    else:
        train_df = df.copy()

    if train_df.empty:
        raise ValueError(f"No rows available for Train EDA (split={train_split!r})")

    unigram_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()
    for text in train_df["text_clean"].tolist():
        tokens = text.split() if text else []
        unigram_counter.update(tokens)
        if len(tokens) >= 2:
            bigram_counter.update(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))

    total_tokens = int(sum(unigram_counter.values()))
    vocab_size = int(len(unigram_counter))
    hapax_count = int(sum(1 for cnt in unigram_counter.values() if cnt == 1))
    negation_tokens = int(sum(cnt for tok, cnt in unigram_counter.items() if tok in NEGATION_WORDS))

    raw_dup_stats, raw_dup_df = _duplicate_metrics(train_df, "text_raw")
    clean_dup_stats, clean_dup_df = _duplicate_metrics(train_df, "text_clean")

    word_freq_df = pd.DataFrame(
        [
            {
                "token": tok,
                "count": int(cnt),
                "share": float(cnt / total_tokens) if total_tokens > 0 else 0.0,
                "is_negation": int(tok in NEGATION_WORDS),
            }
            for tok, cnt in unigram_counter.most_common(cfg.top_k_words)
        ],
        columns=["token", "count", "share", "is_negation"],
    )

    total_bigrams = int(sum(bigram_counter.values()))
    bigram_freq_df = pd.DataFrame(
        [
            {
                "bigram": bg,
                "count": int(cnt),
                "share": float(cnt / total_bigrams) if total_bigrams > 0 else 0.0,
            }
            for bg, cnt in bigram_counter.most_common(cfg.top_k_bigrams)
        ],
        columns=["bigram", "count", "share"],
    )

    empty_count = int((train_df["text_is_empty"] == 1).sum())
    has_neg_count = int((train_df["text_has_negation"] == 1).sum())

    summary = {
        "train_review_count": int(len(train_df)),
        "train_distinct_users": int(train_df["user_id"].nunique()),
        "train_distinct_products": int(train_df["product_id"].nunique()),
        "snapshot_cutoff": str(train_df["snapshot_cutoff"].iloc[0])
        if "snapshot_cutoff" in train_df.columns
        else None,
        "empty_clean_text_count": empty_count,
        "empty_clean_text_rate": float(empty_count / len(train_df)),
        "reviews_with_negation_count": has_neg_count,
        "reviews_with_negation_rate": float(has_neg_count / len(train_df)),
        "length_raw_chars": _distribution_summary(train_df["text_char_len"]),
        "length_clean_words": _distribution_summary(train_df["text_word_count"]),
        "sentence_count": _distribution_summary(train_df["text_sentence_count"]),
        "vocabulary": {
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "type_token_ratio": float(vocab_size / total_tokens) if total_tokens > 0 else 0.0,
            "hapax_count": hapax_count,
            "hapax_rate": float(hapax_count / vocab_size) if vocab_size > 0 else 0.0,
            "negation_token_count": negation_tokens,
            "negation_token_rate": float(negation_tokens / total_tokens)
            if total_tokens > 0
            else 0.0,
        },
        "raw_duplicates": raw_dup_stats,
        "clean_duplicates": clean_dup_stats,
    }

    return {
        "summary": summary,
        "word_freq": word_freq_df,
        "bigram_freq": bigram_freq_df,
        "raw_duplicates_table": raw_dup_df.head(cfg.top_k_duplicates),
        "clean_duplicates_table": clean_dup_df.head(cfg.top_k_duplicates),
    }


def save_eda_outputs(eda_result: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write Train-only EDA summary JSON and frequency/duplicate CSV tables."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_path = target_dir / "eda_summary.json"
    summary_path.write_text(
        json.dumps(eda_result["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    paths = {"eda_summary": summary_path}
    paths["eda_word_freq"] = save_dataframe(
        eda_result["word_freq"], target_dir / "eda_word_freq.csv", fmt="csv"
    )
    paths["eda_bigram_freq"] = save_dataframe(
        eda_result["bigram_freq"], target_dir / "eda_bigram_freq.csv", fmt="csv"
    )
    paths["eda_raw_duplicates"] = save_dataframe(
        eda_result["raw_duplicates_table"], target_dir / "eda_raw_duplicates.csv", fmt="csv"
    )
    paths["eda_clean_duplicates"] = save_dataframe(
        eda_result["clean_duplicates_table"], target_dir / "eda_clean_duplicates.csv", fmt="csv"
    )
    return paths
