"""CLI entrypoint for TV3 Stage 1: dual-text preprocessing, Train-only EDA, and TF-IDF B1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .eda import EdaConfig, run_text_eda, save_eda_outputs
from .preprocess import (
    PreprocessConfig,
    assign_temporal_splits,
    build_dual_text_corpus,
    load_reviews,
    save_dataframe,
)
from .tfidf_features import TfidfB1Extractor, TfidfConfig, save_tfidf_artifacts


def run_stage1_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    train_cutoff: str | None = None,
    val_cutoff: str | None = None,
    snapshot_cutoff: str | None = None,
    output_format: str = "csv",
) -> dict:
    """Execute full TV3 Stage 1 pipeline with strict Train-only fitting and anti-leakage."""
    raw_cfg: dict = {}
    if config_path is not None:
        raw_cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    prep_cfg = PreprocessConfig.from_dict(raw_cfg.get("preprocess"))
    eda_cfg = EdaConfig.from_dict(raw_cfg.get("eda"))
    tfidf_cfg = TfidfConfig.from_dict(raw_cfg.get("tfidf"))

    df = load_reviews(input_path, snapshot_cutoff=snapshot_cutoff)
    df = assign_temporal_splits(df, train_cutoff=train_cutoff, val_cutoff=val_cutoff)
    corpus_df = build_dual_text_corpus(df, config=prep_cfg)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Persist parallel dual-text corpus (text_raw + text_clean) in csv/parquet + jsonl.
    ext = "parquet" if output_format.lower() == "parquet" else "csv"
    dual_cols = [
        c
        for c in (
            "review_id",
            "user_id",
            "product_id",
            "decision_time",
            "snapshot_cutoff",
            "split",
            "text_raw",
            "text_clean",
            "text_char_len",
            "text_word_count",
            "text_has_negation",
            "text_is_empty",
        )
        if c in corpus_df.columns
    ]
    save_dataframe(
        corpus_df[dual_cols], out_path / f"dual_text_corpus.{ext}", fmt=output_format
    )

    # 2. Run EDA strictly on Train split.
    eda_result = run_text_eda(corpus_df, config=eda_cfg, split_col="split", train_split="train")
    save_eda_outputs(eda_result, out_path / "eda")

    # 3. Fit TF-IDF/n-gram strictly on Train, then transform all observed splits.
    train_df = corpus_df[corpus_df["split"] == "train"].copy()
    extractor = TfidfB1Extractor(config=tfidf_cfg)
    extractor.fit(train_df, split_col="split")
    X_all, b1_df = extractor.transform(corpus_df)
    save_tfidf_artifacts(
        extractor=extractor,
        df=corpus_df,
        X_all=X_all,
        b1_df=b1_df,
        out_dir=out_path / "tfidf",
        fmt=output_format,
    )

    return {
        "observed_reviews": int(len(corpus_df)),
        "split_counts": corpus_df["split"].value_counts().to_dict(),
        "eda_summary": eda_result["summary"],
        "vocab_summary": extractor.vocabulary_summary(),
        "b1_shape": list(b1_df.shape),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TV3 Stage 1: dual-text preprocessing, Train-only EDA, and TF-IDF B1 features"
    )
    parser.add_argument("--input", required=True, help="Input path (.jsonl, .csv, or .parquet)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", default=None, help="Optional JSON config path")
    parser.add_argument(
        "--train-cutoff",
        default=None,
        help="UTC date/ISO cutoff for train split (required unless input has 'split' column)",
    )
    parser.add_argument(
        "--val-cutoff",
        default=None,
        help="UTC date/ISO cutoff for validation split (rows after go to test)",
    )
    parser.add_argument(
        "--snapshot-cutoff",
        default=None,
        help="Optional UTC snapshot cutoff excluding later reviews",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output table format for dual-text corpus and B1 features",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_stage1_pipeline(
        input_path=args.input,
        out_dir=args.out,
        config_path=args.config,
        train_cutoff=args.train_cutoff,
        val_cutoff=args.val_cutoff,
        snapshot_cutoff=args.snapshot_cutoff,
        output_format=args.format,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
