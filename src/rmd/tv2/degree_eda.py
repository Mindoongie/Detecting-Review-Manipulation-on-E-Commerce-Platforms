"""TV2 · GĐ1 — Thống kê phân phối bậc User–Product và user ít hoạt động.

Lưu ý: số liệu theo nhãn chỉ để MÔ TẢ dữ liệu; không dùng nhãn để chọn tham số khai phá.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEGREE_BINS = [1, 2, 3, 5, 10, 20, 50, np.inf]


def degree_summary(deg: pd.Series) -> dict:
    q = deg.quantile([0.5, 0.75, 0.9, 0.99])
    return {
        "n": int(deg.size), "mean": float(deg.mean()), "median": float(q[0.5]),
        "p75": float(q[0.75]), "p90": float(q[0.9]), "p99": float(q[0.99]), "max": int(deg.max()),
        "share_deg1": float((deg == 1).mean()),
        "top1pct_review_share": float(deg.sort_values(ascending=False).head(max(1, deg.size // 100)).sum()
                                      / deg.sum()),
    }


def degree_bucket(deg: pd.Series, bins=DEGREE_BINS) -> pd.Series:
    labels = [f"{int(lo)}" if hi - lo == 1 else (f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) else f">={int(lo)}")
              for lo, hi in zip(bins[:-1], bins[1:])]
    return pd.cut(deg, bins=bins, right=False, labels=labels)


def low_activity_stats(df: pd.DataFrame, user_deg: pd.Series, threshold: int = 3) -> dict:
    low_users = user_deg.index[user_deg < threshold]
    in_low = df["user_id"].isin(low_users)
    out = {
        "threshold": threshold,
        "n_low_users": int(low_users.size),
        "share_low_users": float(low_users.size / user_deg.size),
        "n_reviews_by_low_users": int(in_low.sum()),
        "share_reviews_by_low_users": float(in_low.mean()),
    }
    if "is_filtered" in df.columns and df["is_filtered"].notna().any():
        f = df["is_filtered"].astype("float")
        n_filtered = f.sum()
        out["filtered_rate_low"] = float(f[in_low].mean())
        out["filtered_rate_other"] = float(f[~in_low].mean()) if (~in_low).any() else None
        out["share_of_filtered_by_low_users"] = float(f[in_low].sum() / n_filtered) if n_filtered else None
    return out


def by_user_degree(df: pd.DataFrame, user_deg: pd.Series) -> pd.DataFrame:
    """Số user, số review và (nếu có nhãn) tỷ lệ bị lọc theo nhóm bậc user."""
    bucket = degree_bucket(user_deg).rename("user_degree")
    users = bucket.value_counts(sort=False).rename("n_users")
    rb = df["user_id"].map(bucket)
    g = df.groupby(rb, observed=False)
    tab = pd.concat([users, g.size().rename("n_reviews")], axis=1)
    tab["share_reviews"] = tab["n_reviews"] / tab["n_reviews"].sum()
    if "is_filtered" in df.columns and df["is_filtered"].notna().any():
        tab["filtered_rate"] = g["is_filtered"].apply(lambda s: s.astype("float").mean())
    tab.index.name = "user_degree"
    return tab.reset_index()


def ccdf(deg: pd.Series) -> pd.DataFrame:
    vc = deg.value_counts().sort_index()
    return pd.DataFrame({"k": vc.index, "frac_ge_k": vc[::-1].cumsum()[::-1].to_numpy() / deg.size})


def top_products(prod_deg: pd.Series, n_reviews: int, k: int = 10) -> pd.DataFrame:
    top = prod_deg.sort_values(ascending=False).head(k)
    return pd.DataFrame({"product_id": top.index, "n_reviews": top.to_numpy(),
                         "share_reviews": top.to_numpy() / n_reviews,
                         "pairs_if_one_tx": top.to_numpy() * (top.to_numpy() - 1) // 2})


def run_degree_eda(df: pd.DataFrame, low_threshold: int = 3) -> dict:
    user_deg = df.groupby("user_id").size()
    prod_deg = df.groupby("product_id").size()
    return {
        "user_deg": user_deg,
        "prod_deg": prod_deg,
        "user_summary": degree_summary(user_deg),
        "product_summary": degree_summary(prod_deg),
        "low_activity": low_activity_stats(df, user_deg, low_threshold),
        "by_user_degree": by_user_degree(df, user_deg),
        "top_products": top_products(prod_deg, len(df)),
    }
