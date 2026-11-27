"""Unit and anti-leakage tests for TV3 Stage 1 text mining pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.text_pipeline.cli import run_stage1_pipeline
from src.text_pipeline.eda import run_text_eda
from src.text_pipeline.preprocess import (
    PreprocessConfig,
    assign_temporal_splits,
    build_dual_text_corpus,
    clean_text,
    load_reviews,
    parse_utc_timestamp,
)
from src.text_pipeline.tfidf_features import TfidfB1Extractor, TfidfConfig

MOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "mock" / "reviews_text.jsonl"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "text_stage1.json"


def test_parse_utc_timestamp_and_reject_ambiguous() -> None:
    ts_date = parse_utc_timestamp("2024-01-01")
    ts_offset = parse_utc_timestamp("2024-01-02T07:00:00+07:00")
    assert ts_date == pd.Timestamp("2024-01-01T00:00:00Z")
    assert ts_offset == pd.Timestamp("2024-01-02T00:00:00Z")

    for bad in ("2024-01-02T07:00:00", "2024/01/02", "2024-02-30", 1704067200):
        with pytest.raises(ValueError):
            parse_utc_timestamp(bad)


def test_schema_validation_rejects_duplicate_or_missing(tmp_path: Path) -> None:
    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text(
        "\n".join(
            [
                '{"review_id":"r1","user_id":"u1","product_id":"p1","timestamp":"2024-01-01","text":"hi"}',
                '{"review_id":"r1","user_id":"u2","product_id":"p1","timestamp":"2024-01-02","text":"hello"}',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate review_id"):
        load_reviews(dup_file)

    missing_col = tmp_path / "missing.jsonl"
    missing_col.write_text(
        '{"review_id":"r1","user_id":"u1","product_id":"p1","timestamp":"2024-01-01"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        load_reviews(missing_col)


def test_clean_text_preserves_negations_and_dual_corpus() -> None:
    raw = "<p>I didn't like it, isn't good &amp; won't buy again! http://spam.test</p>"
    cleaned = clean_text(raw, PreprocessConfig())
    assert "did not" in cleaned
    assert "is not" in cleaned
    assert "will not" in cleaned
    assert "http" not in cleaned
    assert "<p>" not in cleaned

    # Even with stopwords enabled, negation cues must remain in TF-IDF vocabulary.
    df = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "user_id": ["u1", "u2", "u3"],
            "product_id": ["p1", "p1", "p1"],
            "decision_time": ["2024-01-01T00:00:00Z"] * 3,
            "snapshot_cutoff": ["2024-01-10T00:00:00Z"] * 3,
            "split": ["train", "train", "train"],
            "text_raw": [
                "Not good at all",
                "Never buy this again",
                "No problem with delivery",
            ],
        }
    )
    corpus = build_dual_text_corpus(df)
    assert corpus["text_raw"].tolist()[0] == "Not good at all"
    assert corpus["text_has_negation"].tolist() == [1, 1, 1]

    extractor = TfidfB1Extractor(
        TfidfConfig(ngram_range=(1, 2), min_df=1, max_df=1.0, use_stopwords=True)
    )
    extractor.fit(corpus)
    vocab = extractor.vectorizer.vocabulary_
    for neg in ("not", "never", "no"):
        assert neg in vocab


def test_anti_leakage_labels_future_and_val_test_mutations(tmp_path: Path) -> None:
    # Build two copies: baseline vs mutated labels + mutated val/test texts + extra future row.
    base_rows = [json.loads(line) for line in MOCK_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    mutated_rows = []
    for row in base_rows:
        item = dict(row)
        item["label"] = 1 - int(item["label"])
        if item["review_id"].startswith(("va", "te")):
            item["text"] = "Completely unseen gibberish zzzqqqxxx " + item["review_id"]
        mutated_rows.append(item)
    mutated_rows.append(
        {
            "review_id": "fu02",
            "user_id": "u99",
            "product_id": "p9",
            "timestamp": "2024-04-01",
            "rating": 1,
            "text": "Another post-snapshot leaked phrase",
            "label": 1,
        }
    )

    mutated_file = tmp_path / "mutated.jsonl"
    mutated_file.write_text(
        "\n".join(json.dumps(r) for r in mutated_rows) + "\n", encoding="utf-8"
    )

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    rep_a = run_stage1_pipeline(
        MOCK_PATH,
        out_a,
        config_path=CONFIG_PATH,
        train_cutoff="2024-01-15T23:59:59Z",
        val_cutoff="2024-01-25T23:59:59Z",
        snapshot_cutoff="2024-02-10T23:59:59Z",
    )
    rep_b = run_stage1_pipeline(
        mutated_file,
        out_b,
        config_path=CONFIG_PATH,
        train_cutoff="2024-01-15T23:59:59Z",
        val_cutoff="2024-01-25T23:59:59Z",
        snapshot_cutoff="2024-02-10T23:59:59Z",
    )

    # Train EDA, vocabulary, and Train TF-IDF matrix must be byte-for-byte identical.
    assert rep_a["eda_summary"] == rep_b["eda_summary"]
    assert rep_a["vocab_summary"] == rep_b["vocab_summary"]
    assert (out_a / "tfidf" / "tfidf_feature_names.json").read_text(encoding="utf-8") == (
        out_b / "tfidf" / "tfidf_feature_names.json"
    ).read_text(encoding="utf-8")

    X_tr_a = sparse.load_npz(out_a / "tfidf" / "tfidf_train.npz")
    X_tr_b = sparse.load_npz(out_b / "tfidf" / "tfidf_train.npz")
    assert np.allclose(X_tr_a.toarray(), X_tr_b.toarray())

    b1_a = pd.read_csv(out_a / "tfidf" / "b1_text_features.csv")
    b1_b = pd.read_csv(out_b / "tfidf" / "b1_text_features.csv")
    train_b1_a = b1_a[b1_a["split"] == "train"].reset_index(drop=True)
    train_b1_b = b1_b[b1_b["split"] == "train"].reset_index(drop=True)
    pd.testing.assert_frame_equal(train_b1_a, train_b1_b)


def test_fit_rejects_non_train_rows() -> None:
    df = load_reviews(MOCK_PATH, snapshot_cutoff="2024-02-10")
    df = assign_temporal_splits(df, train_cutoff="2024-01-15T23:59:59Z", val_cutoff="2024-01-25T23:59:59Z")
    corpus = build_dual_text_corpus(df)
    extractor = TfidfB1Extractor()
    with pytest.raises(ValueError, match="Anti-leakage violation"):
        extractor.fit(corpus)


def test_eda_duplicate_rates_hand_verified() -> None:
    df = load_reviews(MOCK_PATH, snapshot_cutoff="2024-02-10")
    df = assign_temporal_splits(df, train_cutoff="2024-01-15T23:59:59Z", val_cutoff="2024-01-25T23:59:59Z")
    corpus = build_dual_text_corpus(df)
    eda = run_text_eda(corpus)["summary"]

    # Mock train has 15 reviews (tr01..tr15); tr14 is whitespace-only -> empty_clean_text_count == 1.
    assert eda["train_review_count"] == 15
    assert eda["empty_clean_text_count"] == 1
    # Raw exact duplicate: "Great product fast shipping!!" appears 3 times (tr01, tr02, tr11).
    assert eda["raw_duplicates"]["duplicate_clusters"] == 1
    assert eda["raw_duplicates"]["rows_in_duplicate_clusters"] == 3
    # Cleaned duplicate: tr03 ("  GREAT product, fast shipping!! ") merges into the cluster -> 4 rows!
    assert eda["clean_duplicates"]["duplicate_clusters"] == 1
    assert eda["clean_duplicates"]["rows_in_duplicate_clusters"] == 4
    assert eda["clean_duplicates"]["distinct_user_duplicate_clusters"] == 1


def test_end_to_end_parquet_and_svd(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(
        json.dumps(
            {
                "tfidf": {
                    "ngram_range": [1, 2],
                    "min_df": 1,
                    "max_df": 0.95,
                    "max_features": 500,
                    "sublinear_tf": True,
                    "use_stopwords": False,
                    "svd_components": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "parquet_out"
    report = run_stage1_pipeline(
        MOCK_PATH,
        out_dir,
        config_path=cfg_file,
        train_cutoff="2024-01-15T23:59:59Z",
        val_cutoff="2024-01-25T23:59:59Z",
        snapshot_cutoff="2024-02-10T23:59:59Z",
        output_format="parquet",
    )
    assert report["observed_reviews"] == 22  # fu01 excluded by snapshot_cutoff
    assert report["split_counts"] == {"train": 15, "val": 4, "test": 3}

    dual = pd.read_parquet(out_dir / "dual_text_corpus.parquet")
    b1 = pd.read_parquet(out_dir / "tfidf" / "b1_text_features.parquet")
    assert set(["text_raw", "text_clean"]).issubset(dual.columns)
    assert len(b1) == 22
    assert not b1.isna().any().any()
    assert "tfidf_svd_03" in b1.columns
    assert sparse.load_npz(out_dir / "tfidf" / "tfidf_val.npz").shape[0] == 4
    assert sparse.load_npz(out_dir / "tfidf" / "tfidf_test.npz").shape[0] == 3
