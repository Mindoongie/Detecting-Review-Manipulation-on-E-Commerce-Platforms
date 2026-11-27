"""Input validation, UTC snapshot handling, and negation-preserving dual-text preprocessing."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "never",
        "neither",
        "nor",
        "none",
        "nothing",
        "nowhere",
        "cannot",
        "hardly",
        "barely",
        "scarcely",
        "without",
    }
)

_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2})(?:\.\d{1,6})?(Z|[+-]\d{2}:\d{2}))?$"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

_SPECIAL_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bshan't\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\bain't\b", re.IGNORECASE), "is not"),
    (re.compile(r"\b([a-z]+)n't\b", re.IGNORECASE), r"\1 not"),
    (re.compile(r"\b([a-z]+)'re\b", re.IGNORECASE), r"\1 are"),
    (re.compile(r"\b([a-z]+)'ve\b", re.IGNORECASE), r"\1 have"),
    (re.compile(r"\b([a-z]+)'ll\b", re.IGNORECASE), r"\1 will"),
    (re.compile(r"\b([a-z]+)'m\b", re.IGNORECASE), r"\1 am"),
    (re.compile(r"\b([a-z]+)'d\b", re.IGNORECASE), r"\1 would"),
)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for dual-text cleaning."""

    lowercase: bool = True
    expand_contractions: bool = True
    preserve_negations: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    keep_digits: bool = True
    strip_punctuation: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "PreprocessConfig":
        if not data:
            return cls()
        return cls(
            lowercase=bool(data.get("lowercase", True)),
            expand_contractions=bool(data.get("expand_contractions", True)),
            preserve_negations=bool(data.get("preserve_negations", True)),
            remove_urls=bool(data.get("remove_urls", True)),
            remove_html=bool(data.get("remove_html", True)),
            keep_digits=bool(data.get("keep_digits", True)),
            strip_punctuation=bool(data.get("strip_punctuation", True)),
        )


def parse_utc_timestamp(value: object) -> pd.Timestamp:
    """Parse date-only (UTC midnight) or timezone-aware ISO datetime into UTC Timestamp."""
    if not isinstance(value, str):
        raise ValueError(f"timestamp must be an ISO string, got {type(value).__name__}")
    raw = value.strip()
    match = _TIMESTAMP_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid or timezone-ambiguous timestamp: {value!r}")

    date_part, hh, mm, ss, _tz = match.groups()
    if hh is not None and (int(hh) > 23 or int(mm) > 59 or int(ss) > 59):
        raise ValueError(f"Invalid clock component in timestamp: {value!r}")

    iso_str = f"{date_part}T00:00:00Z" if hh is None else raw
    try:
        ts = pd.Timestamp(iso_str)
    except Exception as exc:
        raise ValueError(f"Invalid calendar timestamp: {value!r}") from exc

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _read_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, dtype=False)
    if suffix == ".json":
        return pd.read_json(path, dtype=False)
    if suffix == ".csv":
        return pd.read_csv(path, dtype={"review_id": str, "user_id": str, "product_id": str})
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format {suffix!r} for {path}")


def _validate_non_empty_id_series(series: pd.Series, col_name: str) -> pd.Series:
    if series.isna().any():
        bad_idx = int(series[series.isna()].index[0]) + 1
        raise ValueError(f"Row {bad_idx}: missing required string column {col_name!r}")
    out = series.astype(str)
    stripped = out.str.strip()
    if (stripped == "").any():
        bad_idx = int(stripped[stripped == ""].index[0]) + 1
        raise ValueError(f"Row {bad_idx}: blank string in column {col_name!r}")
    return out


def load_reviews(
    path: str | Path,
    snapshot_cutoff: str | None = None,
) -> pd.DataFrame:
    """
    Load reviews from JSONL/CSV/Parquet, validate required contract, drop labels,
    normalize UTC timestamps, and optionally apply snapshot cutoff.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    raw_df = _read_raw_table(file_path)
    required = ("review_id", "user_id", "product_id", "timestamp", "text")
    missing = [col for col in required if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if raw_df.empty:
        raise ValueError("Input dataset is empty")

    # Whitelist strictly excludes label/filtered columns to guarantee zero target leakage.
    clean_df = pd.DataFrame(index=raw_df.index)
    for id_col in ("review_id", "user_id", "product_id"):
        clean_df[id_col] = _validate_non_empty_id_series(raw_df[id_col], id_col)

    dup_mask = clean_df["review_id"].duplicated(keep=False)
    if dup_mask.any():
        dup_id = clean_df.loc[dup_mask, "review_id"].iloc[0]
        raise ValueError(f"Duplicate review_id detected: {dup_id!r}")

    parsed_times = [parse_utc_timestamp(v) for v in raw_df["timestamp"].tolist()]
    clean_df["timestamp_utc"] = pd.Series(parsed_times, index=raw_df.index)
    clean_df["decision_time"] = clean_df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Preserve raw text faithfully; coerce null/NaN to "" rather than dropping rows.
    clean_df["text_raw"] = raw_df["text"].apply(lambda x: "" if pd.isna(x) else str(x))

    if "split" in raw_df.columns:
        split_vals = raw_df["split"].astype(str).str.strip().str.lower()
        allowed = {"train", "val", "test"}
        invalid = sorted(set(split_vals.unique()) - allowed)
        if invalid:
            raise ValueError(f"Invalid split values: {invalid}; expected subset of {sorted(allowed)}")
        clean_df["split"] = split_vals

    if snapshot_cutoff is not None:
        cutoff_ts = parse_utc_timestamp(snapshot_cutoff)
        clean_df = clean_df[clean_df["timestamp_utc"] <= cutoff_ts].copy()
        if clean_df.empty:
            raise ValueError(f"No reviews remain at or before snapshot_cutoff={snapshot_cutoff!r}")
        clean_df["snapshot_cutoff"] = cutoff_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        max_ts = clean_df["timestamp_utc"].max()
        clean_df["snapshot_cutoff"] = max_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    clean_df = clean_df.sort_values(
        by=["timestamp_utc", "review_id"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    return clean_df


def assign_temporal_splits(
    df: pd.DataFrame,
    train_cutoff: str | None = None,
    val_cutoff: str | None = None,
) -> pd.DataFrame:
    """
    Assign 'train' / 'val' / 'test' split using temporal cutoffs, or keep existing 'split'.
    At least one of existing 'split' or train_cutoff must be available.
    """
    out = df.copy()
    if train_cutoff is None:
        if "split" in out.columns:
            return out
        raise ValueError("Either input 'split' column or --train-cutoff must be provided")

    train_ts = parse_utc_timestamp(train_cutoff)
    if val_cutoff is not None:
        val_ts = parse_utc_timestamp(val_cutoff)
        if val_ts < train_ts:
            raise ValueError(f"val_cutoff ({val_cutoff}) must be >= train_cutoff ({train_cutoff})")
    else:
        val_ts = None

    splits: list[str] = []
    for ts in out["timestamp_utc"]:
        if ts <= train_ts:
            splits.append("train")
        elif val_ts is None or ts <= val_ts:
            splits.append("val")
        else:
            splits.append("test")
    out["split"] = splits
    return out


def clean_text(text: str, config: PreprocessConfig | None = None) -> str:
    """Normalize text while expanding negative contractions and preserving negation words."""
    cfg = config or PreprocessConfig()
    s = "" if text is None else str(text)
    if not s:
        return ""

    if cfg.remove_html:
        s = html.unescape(s)
        s = _HTML_TAG_RE.sub(" ", s)
    if cfg.remove_urls:
        s = _URL_RE.sub(" ", s)

    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")

    if cfg.expand_contractions:
        for pattern, repl in _SPECIAL_CONTRACTIONS:
            s = pattern.sub(repl, s)

    if cfg.lowercase:
        s = s.lower()

    if cfg.strip_punctuation:
        keep_pattern = r"[^a-zA-Z0-9\s]" if cfg.keep_digits else r"[^a-zA-Z\s]"
        s = re.sub(keep_pattern, " ", s)

    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def extract_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-review lexical, punctuation, and negation surface features."""
    rows: list[dict[str, float | int]] = []
    for raw, clean in zip(df["text_raw"].tolist(), df["text_clean"].tolist()):
        char_len = len(raw)
        clean_char_len = len(clean)
        clean_tokens = clean.split() if clean else []
        raw_tokens = raw.split() if raw.strip() else []
        word_count = len(clean_tokens)
        avg_word_len = (
            sum(len(tok) for tok in clean_tokens) / word_count if word_count > 0 else 0.0
        )

        alpha_chars = sum(1 for ch in raw if ch.isalpha())
        upper_chars = sum(1 for ch in raw if ch.isupper())
        digit_chars = sum(1 for ch in raw if ch.isdigit())

        if raw.strip():
            # Count explicit punctuation boundaries; fall back to 1 sentence for unpunctuated text.
            sent_breaks = len(_SENTENCE_SPLIT_RE.findall(raw))
            sentence_count = max(1, sent_breaks)
        else:
            sentence_count = 0

        neg_count = sum(1 for tok in clean_tokens if tok.lower() in NEGATION_WORDS)
        unique_ratio = len(set(clean_tokens)) / word_count if word_count > 0 else 0.0

        rows.append(
            {
                "text_char_len": char_len,
                "text_clean_char_len": clean_char_len,
                "text_word_count": word_count,
                "text_raw_word_count": len(raw_tokens),
                "text_avg_word_len": float(avg_word_len),
                "text_sentence_count": sentence_count,
                "text_uppercase_ratio": float(upper_chars / alpha_chars) if alpha_chars > 0 else 0.0,
                "text_digit_ratio": float(digit_chars / char_len) if char_len > 0 else 0.0,
                "text_exclamation_count": raw.count("!"),
                "text_question_count": raw.count("?"),
                "text_unique_word_ratio": float(unique_ratio),
                "text_negation_count": neg_count,
                "text_has_negation": int(neg_count > 0),
                "text_is_empty": int(clean_char_len == 0),
            }
        )
    return pd.DataFrame(rows, index=df.index)


def build_dual_text_corpus(
    df: pd.DataFrame,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """Produce parallel text_raw and text_clean columns plus per-review surface statistics."""
    cfg = config or PreprocessConfig()
    out = df.copy()
    out["text_clean"] = [clean_text(t, cfg) for t in out["text_raw"].tolist()]
    surface_df = extract_surface_features(out)
    for col in surface_df.columns:
        out[col] = surface_df[col]
    return out


def save_dataframe(df: pd.DataFrame, path: str | Path, fmt: str = "csv") -> Path:
    """Persist a DataFrame cleanly in csv, parquet, or jsonl format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()
    for col in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[col]):
            export_df[col] = export_df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    fmt_lower = fmt.lower()
    if fmt_lower == "csv":
        export_df.to_csv(target, index=False, encoding="utf-8")
    elif fmt_lower == "parquet":
        export_df.to_parquet(target, index=False)
    elif fmt_lower == "jsonl":
        export_df.to_json(target, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt!r}")
    return target


def strip_negation_from_stopwords(stopwords: Iterable[str]) -> list[str]:
    """Ensure negation cues are never dropped when a stopword list is used."""
    return sorted(word for word in set(stopwords) if word.lower() not in NEGATION_WORDS)
