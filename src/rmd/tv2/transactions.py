"""TV2 · GĐ1 — Định nghĩa giao dịch cho FP-Growth và xử lý sản phẩm quá hot.

Quy ước (xem docs/tv2_transaction_definition.md):

* **Giao dịch** = một sản phẩm (``mode="product"``) hoặc một cặp
  (sản phẩm, cửa sổ thời gian cố định ``window_days``) (``mode="product_window"``, mặc định).
* **Item** = ``user_id`` có review sản phẩm đó trong cửa sổ đó (khử trùng lặp).
* **Giao dịch quá hot** (số user > ``max_tx_size``):
  ``subwindow`` — chia đôi cửa sổ lặp lại tới khi ≤ trần hoặc cửa sổ < ``min_subwindow_days``
  (khi đó mới bỏ); ``drop`` — bỏ luôn khỏi khai phá.
  Review bị bỏ khỏi khai phá vẫn được chấm điểm bình thường (chỉ không có bằng chứng nhóm).
* **Điều kiện tham gia khai phá** (không mất mát): lặp tới điểm bất động —
  bỏ user xuất hiện trong < ``min_support`` giao dịch nhiều-user, bỏ giao dịch còn < 2 user.
  Mọi tập phổ biến có ≥ 2 user và support ≥ ``min_support`` được giữ nguyên.

Hàm nhận dữ liệu ĐÃ cắt snapshot (observed view) — không tự cắt.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

MODES = ("product", "product_window")
HOT_STRATEGIES = ("subwindow", "drop")


@dataclass(frozen=True)
class TxConfig:
    mode: str = "product_window"
    window_days: float = 28
    min_support: int = 3
    max_tx_size: int | None = 200
    hot_strategy: str = "subwindow"
    min_subwindow_days: float = 1.0
    origin: str | None = None  # gốc chia cửa sổ; None = ngày nhỏ nhất của dữ liệu

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode phải thuộc {MODES}")
        if self.hot_strategy not in HOT_STRATEGIES:
            raise ValueError(f"hot_strategy phải thuộc {HOT_STRATEGIES}")
        if self.min_support < 1:
            raise ValueError("min_support phải >= 1")

    @classmethod
    def from_dict(cls, d: dict) -> "TxConfig":
        names = {f.name for f in fields(cls)}
        mapped = {("mode" if k == "transaction_mode" else k): v for k, v in d.items()}
        return cls(**{k: v for k, v in mapped.items() if k in names})


@dataclass
class TxResult:
    transactions: pd.DataFrame  # tx_id, product_id, window_start, window_end, size
    memberships: pd.DataFrame   # tx_id, user_id  (đầu vào FP-Growth)
    review_tx: pd.DataFrame     # review_id, tx_id (để gộp đặc trưng nhóm về cấp review)
    stats: dict

    def itemsets(self) -> list[list[str]]:
        """Danh sách giao dịch dạng list[user_id] cho FP-Growth."""
        return self.memberships.groupby("tx_id")["user_id"].apply(list).tolist()


def _pairs(sizes: pd.Series) -> int:
    s = sizes.to_numpy(dtype=np.int64)
    return int((s * (s - 1) // 2).sum())


def _window_start(ts: pd.Series, origin: pd.Timestamp, width: pd.Series) -> pd.Series:
    d = (ts - origin).to_numpy().astype("timedelta64[ns]").astype(np.int64)
    w = width.to_numpy().astype("timedelta64[ns]").astype(np.int64)
    return pd.Series(origin + pd.to_timedelta(d // w * w, unit="ns"), index=ts.index)


def _tx_key(df: pd.DataFrame) -> pd.Series:
    ws = df["window_start"].dt.strftime("%Y-%m-%dT%H").fillna("all")
    wd = (df["window_width"].dt.total_seconds() / 86400).round(4).astype("string").fillna("all")
    return df["product_id"].astype("string") + "|" + ws + "|" + wd


def _size_quantiles(sizes: pd.Series, prefix: str) -> dict:
    if sizes.empty:
        return {f"{prefix}_p50": 0, f"{prefix}_p99": 0, f"{prefix}_max": 0}
    return {f"{prefix}_p50": float(sizes.quantile(0.5)), f"{prefix}_p99": float(sizes.quantile(0.99)),
            f"{prefix}_max": int(sizes.max())}


def raw_tx_sizes(df: pd.DataFrame, mode: str = "product_window", window_days: float = 28,
                 origin: str | None = None) -> pd.Series:
    """Số user mỗi giao dịch TRƯỚC khi xử lý hot/cắt tỉa (dùng cho EDA)."""
    if mode == "product":
        return df.groupby("product_id")["user_id"].nunique()
    o = pd.Timestamp(origin) if origin else df["timestamp"].min().normalize()
    width = pd.Series(pd.Timedelta(days=window_days), index=df.index)
    ws = _window_start(df["timestamp"], o, width)
    return df.groupby([df["product_id"], ws])["user_id"].nunique()


def build_transactions(df: pd.DataFrame, cfg: TxConfig = TxConfig()) -> TxResult:
    base = df[["review_id", "user_id", "product_id", "timestamp"]].copy()
    origin = pd.Timestamp(cfg.origin) if cfg.origin else base["timestamp"].min().normalize()
    first_width = pd.Timedelta(days=cfg.window_days)
    if cfg.mode == "product_window":
        base["window_width"] = first_width
        base["window_start"] = _window_start(base["timestamp"], origin, base["window_width"])
    else:
        base["window_width"] = pd.Series(pd.NaT, index=base.index, dtype="timedelta64[ns]")
        base["window_start"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns]")
    base["tx_key"] = _tx_key(base)

    raw_sizes = base.groupby("tx_key")["user_id"].nunique()
    stats: dict = {
        "mode": cfg.mode, "window_days": cfg.window_days if cfg.mode == "product_window" else None,
        "max_tx_size": cfg.max_tx_size, "hot_strategy": cfg.hot_strategy, "min_support": cfg.min_support,
        "n_reviews": int(len(base)), "n_users": int(base["user_id"].nunique()),
        "n_tx_raw": int(raw_sizes.size), **_size_quantiles(raw_sizes, "tx_size_raw"),
        "pairs_raw": _pairs(raw_sizes),
        "n_hot_tx_raw": int((raw_sizes > cfg.max_tx_size).sum()) if cfg.max_tx_size else 0,
    }

    # --- Xử lý giao dịch quá hot -------------------------------------------------
    n_split_rounds = 0
    if cfg.max_tx_size:
        min_width = pd.Timedelta(days=cfg.min_subwindow_days)
        while True:
            sizes = base.groupby("tx_key")["user_id"].nunique()
            hot = base["tx_key"].isin(sizes.index[sizes > cfg.max_tx_size])
            if not hot.any():
                break
            if cfg.hot_strategy == "drop":
                base = base.loc[~hot]
                break
            # Lần chia đầu ở mode "product": tách theo window_days; các lần sau: chia đôi.
            cur = base.loc[hot, "window_width"]
            new_width = (cur / 2).where(cur.notna(), first_width)
            too_small = new_width < min_width
            base = base.drop(index=new_width.index[too_small])
            ok = new_width.index[~too_small]
            base.loc[ok, "window_width"] = new_width[~too_small]
            base.loc[ok, "window_start"] = _window_start(base.loc[ok, "timestamp"], origin,
                                                         base.loc[ok, "window_width"])
            base.loc[ok, "tx_key"] = _tx_key(base.loc[ok])
            n_split_rounds += 1
    hot_sizes = base.groupby("tx_key")["user_id"].nunique()
    stats.update({
        "split_rounds": n_split_rounds,
        "reviews_dropped_hot": int(stats["n_reviews"] - len(base)),
        "n_tx_after_hot": int(hot_sizes.size), "pairs_after_hot": _pairs(hot_sizes),
    })

    # --- Cắt tỉa không mất mát tới điểm bất động --------------------------------
    m = base[["tx_key", "user_id"]].drop_duplicates()
    while True:
        m2 = m[m.groupby("tx_key")["user_id"].transform("size") >= 2]
        m2 = m2[m2.groupby("user_id")["tx_key"].transform("size") >= cfg.min_support]
        if len(m2) == len(m):
            break
        m = m2

    # --- Đầu ra ---------------------------------------------------------------
    tx_meta = (base.drop_duplicates("tx_key").set_index("tx_key")
               .loc[sorted(m["tx_key"].unique()), ["product_id", "window_start", "window_width"]])
    tx_meta = tx_meta.reset_index()
    tx_meta.insert(0, "tx_id", np.arange(len(tx_meta), dtype=np.int64))
    tx_meta["window_end"] = tx_meta["window_start"] + tx_meta["window_width"]
    key_to_id = dict(zip(tx_meta["tx_key"], tx_meta["tx_id"]))

    memberships = m.assign(tx_id=m["tx_key"].map(key_to_id))[["tx_id", "user_id"]]
    memberships = memberships.sort_values(["tx_id", "user_id"]).reset_index(drop=True)
    final_sizes = memberships.groupby("tx_id").size()
    tx_meta["size"] = tx_meta["tx_id"].map(final_sizes).astype(np.int64)

    review_tx = base.merge(m, on=["tx_key", "user_id"])
    review_tx = review_tx.assign(tx_id=review_tx["tx_key"].map(key_to_id))[["review_id", "tx_id"]]

    stats.update({
        "n_tx_final": int(len(tx_meta)), **_size_quantiles(final_sizes, "tx_size_final"),
        "pairs_final": _pairs(final_sizes),
        "n_users_eligible": int(memberships["user_id"].nunique()),
        "share_users_eligible": float(memberships["user_id"].nunique() / max(stats["n_users"], 1)),
        "n_memberships_final": int(len(memberships)),
        # Cận trên của Coverage: review còn nằm trong ít nhất 1 giao dịch đủ điều kiện
        "coverage_ub_reviews": float(review_tx["review_id"].nunique() / max(stats["n_reviews"], 1)),
    })
    transactions = tx_meta[["tx_id", "product_id", "window_start", "window_end", "size"]]
    return TxResult(transactions=transactions, memberships=memberships, review_tx=review_tx, stats=stats)
