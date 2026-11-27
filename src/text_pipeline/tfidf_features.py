"""Train-only TF-IDF / n-gram extractor and dense B1 content features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .preprocess import NEGATION_WORDS, save_dataframe, strip_negation_from_stopwords


@dataclass(frozen=True)
class TfidfConfig:
    """Configuration for Train-only TF-IDF / n-gram extraction."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 1
    max_df: int | float = 0.95
    max_features: int | None = 20000
    sublinear_tf: bool = True
    norm: str | None = "l2"
    use_stopwords: bool = False
    svd_components: int = 0
    random_state: int = 42

    @classmethod
    def from_dict(cls, data: dict | None) -> "TfidfConfig":
        if not data:
            return cls()
        ngram = data.get("ngram_range", [1, 2])
        return cls(
            ngram_range=(int(ngram[0]), int(ngram[1])),
            min_df=data.get("min_df", 1),
            max_df=data.get("max_df", 0.95),
            max_features=data.get("max_features", 20000),
            sublinear_tf=bool(data.get("sublinear_tf", True)),
            norm=data.get("norm", "l2"),
            use_stopwords=bool(data.get("use_stopwords", False)),
            svd_components=int(data.get("svd_components", 0)),
            random_state=int(data.get("random_state", 42)),
        )


def _sparse_row_stats(matrix: sparse.csr_matrix) -> pd.DataFrame:
    """Compute per-row summary features from a TF-IDF sparse matrix without leakage."""
    n_rows = matrix.shape[0]
    nnz = np.zeros(n_rows, dtype=np.int64)
    max_w = np.zeros(n_rows, dtype=np.float64)
    sum_w = np.zeros(n_rows, dtype=np.float64)
    mean_nz = np.zeros(n_rows, dtype=np.float64)
    entropy = np.zeros(n_rows, dtype=np.float64)

    indptr = matrix.indptr
    data = matrix.data
    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        count = end - start
        nnz[i] = count
        if count == 0:
            continue
        vals = data[start:end]
        row_sum = float(np.sum(vals))
        max_w[i] = float(np.max(vals))
        sum_w[i] = row_sum
        mean_nz[i] = row_sum / count
        if count > 1 and row_sum > 0:
            probs = vals / row_sum
            raw_ent = float(-np.sum(probs * np.log(probs + 1e-12)))
            entropy[i] = raw_ent / float(np.log(count))

    return pd.DataFrame(
        {
            "tfidf_nnz": nnz,
            "tfidf_max": max_w,
            "tfidf_sum": sum_w,
            "tfidf_mean_nonzero": mean_nz,
            "tfidf_entropy": entropy,
        }
    )


class TfidfB1Extractor:
    """
    TF-IDF / n-gram extractor enforcing strict fit-on-Train-only anti-leakage.
    Produces both sparse TF-IDF matrices (for Logistic Regression B1) and compact
    dense B1 tabular content features keyed by review_id (for LightGBM / merged B1).
    """

    def __init__(self, config: TfidfConfig | None = None) -> None:
        self.config = config or TfidfConfig()
        stop_words = (
            strip_negation_from_stopwords(ENGLISH_STOP_WORDS)
            if self.config.use_stopwords
            else None
        )
        # Token pattern keeps single-character tokens like 'a'/'1' only if >=2 or allow >=1?
        # Using (?u)\b\w+\b preserves 'no' (2 chars), digits, and avoids dropping short negation cues.
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            max_features=self.config.max_features,
            sublinear_tf=self.config.sublinear_tf,
            norm=self.config.norm,
            stop_words=stop_words,
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.svd: TruncatedSVD | None = None
        self.train_unigrams_: set[str] = set()
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame, split_col: str = "split") -> "TfidfB1Extractor":
        """Fit TF-IDF (and optional SVD) strictly on Train rows."""
        if train_df.empty:
            raise ValueError("Cannot fit TfidfB1Extractor on empty DataFrame")
        if split_col in train_df.columns:
            non_train = train_df[train_df[split_col] != "train"]
            if not non_train.empty:
                bad_splits = sorted(non_train[split_col].unique().tolist())
                raise ValueError(
                    f"Anti-leakage violation: fit() received non-train rows ({bad_splits})"
                )

        texts = train_df["text_clean"].fillna("").astype(str).tolist()
        X_train = self.vectorizer.fit_transform(texts)
        vocab = self.vectorizer.vocabulary_
        self.train_unigrams_ = {tok for tok in vocab if " " not in tok}

        if self.config.svd_components > 0 and min(X_train.shape) > 1:
            n_comp = min(self.config.svd_components, min(X_train.shape) - 1)
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.config.random_state)
            self.svd.fit(X_train)
        else:
            self.svd = None

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.DataFrame]:
        """Transform any split using frozen Train vocabulary and IDF weights."""
        if not self.is_fitted_:
            raise RuntimeError("TfidfB1Extractor must be fitted on Train before transform()")

        texts = df["text_clean"].fillna("").astype(str).tolist()
        X = self.vectorizer.transform(texts).tocsr()

        # Key & traceability columns (kept separate from model features by TV5).
        key_cols = [
            col
            for col in (
                "review_id",
                "user_id",
                "product_id",
                "decision_time",
                "snapshot_cutoff",
                "split",
            )
            if col in df.columns
        ]
        surface_cols = [
            col
            for col in (
                "text_char_len",
                "text_clean_char_len",
                "text_word_count",
                "text_raw_word_count",
                "text_avg_word_len",
                "text_sentence_count",
                "text_uppercase_ratio",
                "text_digit_ratio",
                "text_exclamation_count",
                "text_question_count",
                "text_unique_word_ratio",
                "text_negation_count",
                "text_has_negation",
                "text_is_empty",
            )
            if col in df.columns
        ]

        out_df = df[key_cols + surface_cols].reset_index(drop=True)

        # Out-of-vocabulary ratio against frozen Train unigrams.
        oov_ratios = []
        for text in texts:
            toks = text.split() if text else []
            if not toks:
                oov_ratios.append(0.0)
            else:
                oov = sum(1 for tok in toks if tok not in self.train_unigrams_)
                oov_ratios.append(float(oov / len(toks)))
        out_df["tfidf_oov_token_ratio"] = oov_ratios

        tfidf_stats = _sparse_row_stats(X)
        for col in tfidf_stats.columns:
            out_df[col] = tfidf_stats[col]

        if self.svd is not None:
            svd_arr = self.svd.transform(X)
            for idx in range(svd_arr.shape[1]):
                out_df[f"tfidf_svd_{idx:02d}"] = svd_arr[:, idx]

        return X, out_df

    def vocabulary_summary(self) -> dict[str, Any]:
        """Return metadata about the frozen Train-fit TF-IDF vocabulary."""
        if not self.is_fitted_:
            raise RuntimeError("TfidfB1Extractor is not fitted")
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        neg_terms = [t for t in feature_names if any(w in t.split() for w in NEGATION_WORDS)]
        return {
            "vocab_size": int(len(feature_names)),
            "ngram_range": list(self.config.ngram_range),
            "sublinear_tf": self.config.sublinear_tf,
            "use_stopwords": self.config.use_stopwords,
            "negation_terms_in_vocab_count": int(len(neg_terms)),
            "sample_negation_terms": neg_terms[:20],
            "svd_components": int(self.svd.n_components) if self.svd is not None else 0,
        }

    def save(self, out_dir: str | Path) -> Path:
        """Persist fitted extractor artifact via joblib."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted TfidfB1Extractor")
        target = Path(out_dir) / "tfidf_extractor.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TfidfB1Extractor":
        return joblib.load(Path(path))


def save_tfidf_artifacts(
    extractor: TfidfB1Extractor,
    df: pd.DataFrame,
    X_all: sparse.csr_matrix,
    b1_df: pd.DataFrame,
    out_dir: str | Path,
    fmt: str = "csv",
) -> dict[str, Path]:
    """Save vectorizer, vocabulary summary, per-split sparse matrices, and B1 feature table."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["extractor"] = extractor.save(target_dir)

    vocab_summary_path = target_dir / "tfidf_vocab_summary.json"
    vocab_summary_path.write_text(
        json.dumps(extractor.vocabulary_summary(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["vocab_summary"] = vocab_summary_path

    feature_names_path = target_dir / "tfidf_feature_names.json"
    feature_names_path.write_text(
        json.dumps(extractor.vectorizer.get_feature_names_out().tolist(), ensure_ascii=False),
        encoding="utf-8",
    )
    paths["feature_names"] = feature_names_path

    ext = "parquet" if fmt.lower() == "parquet" else "csv"
    paths["b1_text_features"] = save_dataframe(
        b1_df, target_dir / f"b1_text_features.{ext}", fmt=fmt
    )

    if "split" in df.columns:
        for split_name in ("train", "val", "test"):
            mask = (df["split"] == split_name).to_numpy()
            if mask.any():
                npz_path = target_dir / f"tfidf_{split_name}.npz"
                sparse.save_npz(npz_path, X_all[mask])
                paths[f"tfidf_{split_name}"] = npz_path
                ids_path = target_dir / f"tfidf_{split_name}_review_ids.json"
                ids_path.write_text(
                    json.dumps(df.loc[mask, "review_id"].tolist(), ensure_ascii=False),
                    encoding="utf-8",
                )
    else:
        npz_path = target_dir / "tfidf_all.npz"
        sparse.save_npz(npz_path, X_all)
        paths["tfidf_all"] = npz_path

    return paths
