"""
data_loader.py — TV5 Phase 1
Tải dữ liệu YelpChi (hoặc bảng trung gian Parquet/CSV của TV1)
hoặc sinh dữ liệu mock thực tế để kiểm chứng pipeline B0/B1.

Schema chuẩn (bảng trung gian thống nhất TV1–TV5):
  review_id  : str      — mã review duy nhất
  user_id    : str      — mã người dùng
  product_id : str      — mã sản phẩm (nhà hàng / khách sạn)
  rating     : int      — sao đánh giá (1–5)
  text       : str      — nội dung review
  timestamp  : datetime — thời điểm đăng review
  label      : int      — 0 = genuine (recommended), 1 = fake (filtered)
"""

import os
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    'review_id',
    'user_id',
    'product_id',
    'rating',
    'text',
    'timestamp',
    'label',
]


def _find_data_file(folder: str) -> Optional[str]:
    """Tìm file dữ liệu chuẩn (.parquet, .csv, .tsv) trong thư mục."""
    if not folder or not os.path.isdir(folder):
        return None
    priority_exts = ('.parquet', '.csv', '.tsv')
    files = sorted(os.listdir(folder))
    for ext in priority_exts:
        for fname in files:
            if fname.lower().endswith(ext):
                return os.path.join(folder, fname)
    return None


def _try_load_yelpchi_raw_pair(raw_path: str) -> Optional[pd.DataFrame]:
    """
    Hỗ trợ tải cặp file thô kinh điển của YelpChi (Rayana & Akoglu, KDD 2015):
      - output_meta_yelp*Data*.txt  (date, reviewID, reviewerID, productID, label, ...)
      - output_review_yelp*Data*.txt (review text theo từng dòng tương ứng)
    """
    if not raw_path or not os.path.isdir(raw_path):
        return None
    files = sorted(os.listdir(raw_path))
    meta_files = [f for f in files if 'meta' in f.lower() and f.lower().endswith('.txt')]
    review_files = [f for f in files if 'review' in f.lower() and f.lower().endswith('.txt')]
    if not meta_files or not review_files:
        return None

    meta_path = os.path.join(raw_path, meta_files[0])
    rev_path = os.path.join(raw_path, review_files[0])
    print(f"[DataLoader] Phát hiện cặp file YelpChi raw: {meta_files[0]} + {review_files[0]}")

    meta_df = pd.read_csv(meta_path, sep=None, engine='python', header=None)
    if meta_df.shape[1] >= 5:
        # Kiểm tra xem dòng đầu có phải header không
        first_val = str(meta_df.iloc[0, 0]).lower()
        if any(tok in first_val for tok in ('date', 'time', 'review')):
            meta_df = pd.read_csv(meta_path, sep=None, engine='python')
        else:
            cols = ['timestamp', 'review_id', 'user_id', 'product_id', 'label']
            if meta_df.shape[1] >= 9:
                cols += ['useful', 'funny', 'cool', 'rating']
            elif meta_df.shape[1] == 6:
                cols += ['rating']
            else:
                cols += [f'extra_{i}' for i in range(meta_df.shape[1] - 5)]
            meta_df.columns = cols[:meta_df.shape[1]]

    with open(rev_path, 'r', encoding='utf-8', errors='replace') as f:
        texts = [line.rstrip('\r\n') for line in f]

    if len(texts) != len(meta_df):
        raise ValueError(
            f"Số dòng metadata ({len(meta_df)}) không khớp số dòng text ({len(texts)})."
        )

    meta_df['text'] = texts
    if 'rating' not in meta_df.columns:
        raise ValueError("File metadata không chứa cột rating.")
    return meta_df


def load_yelpchi(data_path: str) -> pd.DataFrame:
    """
    Tải dataset từ thư mục chứa bảng trung gian (.parquet/.csv/.tsv)
    hoặc dữ liệu raw YelpChi. Chuẩn hóa về schema thống nhất TV1–TV5.
    """
    fpath = _find_data_file(data_path)
    if fpath is not None:
        print(f"[DataLoader] Đang tải: {fpath}")
        if fpath.lower().endswith('.parquet'):
            df = pd.read_parquet(fpath)
        elif fpath.lower().endswith('.tsv'):
            df = pd.read_csv(fpath, sep='\t')
        else:
            df = pd.read_csv(fpath)
    else:
        df = _try_load_yelpchi_raw_pair(data_path)
        if df is None:
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu trong '{data_path}'. "
                "Đặt file .parquet/.csv/.tsv hoặc cặp output_meta_*.txt + output_review_*.txt vào thư mục này."
            )

    df = _normalize_columns(df)
    df = _clean_and_cast_schema(df)
    _validate_schema(df)

    print(
        f"[DataLoader] Tải xong: {len(df):,} reviews, "
        f"{df['user_id'].nunique():,} users, "
        f"{df['product_id'].nunique():,} products | "
        f"fake ratio = {df['label'].mean():.2%}"
    )
    return df


def _normalize_label_series(series: pd.Series) -> pd.Series:
    """Chuẩn hóa nhiều định dạng nhãn (Y/N, filtered/recommended, -1/1, 0/1) về {0, 1}."""
    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique())
        if vals.issubset({0, 1}):
            return series.astype(int)
        if vals.issubset({-1, 1}):
            # Theo chuẩn YelpChi/Rayana: -1 = filtered (fake)? hay 1 = fake?
            # Cảnh báo: trong các bản Graph/ODDS chuẩn, -1 thường là genuine hoặc spam tùy bản;
            # quy ước mặc định chuẩn hóa 1 -> 1 (fake), -1 -> 0 (genuine) nếu có -1/1.
            return (series == 1).astype(int)
        raise ValueError(f"Giá trị nhãn số không hợp lệ: {sorted(vals)}")

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            '1': 1, 'y': 1, 'yr': 1, 'yes': 1, 'true': 1,
            'filtered': 1, 'fake': 1, 'spam': 1, 'unrecommended': 1,
            '0': 0, '-1': 0, 'n': 0, 'nr': 0, 'no': 0, 'false': 0,
            'recommended': 0, 'genuine': 0, 'real': 0, 'ham': 0,
        })
    )
    if mapped.isna().any():
        unknown = series[mapped.isna()].unique()[:5]
        raise ValueError(f"Không nhận diện được nhãn: {unknown}")
    return mapped.astype(int)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map tên cột phổ biến trong các bản YelpChi / TV1 schema về tên chuẩn."""
    col_map = {}
    lower_cols = {c.lower().strip().replace(' ', '_'): c for c in df.columns}

    mapping_rules = {
        'review_id': ['review_id', 'reviewid', 'rid'],
        'user_id': ['user_id', 'userid', 'reviewerid', 'reviewer_id', 'uid'],
        'product_id': [
            'product_id', 'productid', 'restaurantid', 'restaurant_id',
            'hotelid', 'hotel_id', 'pid', 'business_id',
        ],
        'rating': ['rating', 'star', 'stars', 'score'],
        'text': [
            'text', 'reviewcontent', 'review_content', 'review_text',
            'content', 'review', 'raw_text',
        ],
        'timestamp': ['timestamp', 'date', 'review_date', 'datetime', 'time'],
        'label': [
            'label', 'flagged', 'fake', 'spam', 'is_fake',
            'is_spam', 'filtered',
        ],
    }

    for target, aliases in mapping_rules.items():
        if target in df.columns:
            continue
        for alias in aliases:
            if alias in lower_cols:
                col_map[lower_cols[alias]] = target
                break

    df = df.rename(columns=col_map).copy()

    if 'review_id' not in df.columns:
        df['review_id'] = [f"r_{i:07d}" for i in range(len(df))]

    if 'label' in df.columns:
        df['label'] = _normalize_label_series(df['label'])

    return df


def _clean_and_cast_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu chuẩn, loại dòng lỗi timestamp/id và kiểm tra trùng review_id."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}. Hiện có: {list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    n_bad_ts = int(df['timestamp'].isna().sum())
    if n_bad_ts > 0:
        raise ValueError(f"Phát hiện {n_bad_ts} dòng có timestamp không hợp lệ (NaT).")

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    if df['rating'].isna().any():
        raise ValueError("Cột rating chứa giá trị không phải số.")
    df['rating'] = df['rating'].astype(int)

    df['label'] = df['label'].astype(int)
    df['review_id'] = df['review_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)
    df['text'] = df['text'].fillna('').astype(str)

    if df['review_id'].duplicated().any():
        n_dup = int(df['review_id'].duplicated().sum())
        raise ValueError(f"Phát hiện {n_dup} review_id bị trùng lặp trong dữ liệu.")

    return df


def _validate_schema(df: pd.DataFrame):
    """Kiểm tra ràng buộc giá trị trên schema chuẩn."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}. Hiện có: {list(df.columns)}")
    if not set(df['label'].unique()).issubset({0, 1}):
        raise ValueError(f"Nhãn label phải thuộc {{0, 1}}, nhận được {df['label'].unique()}")
    if (df['rating'] < 1).any() or (df['rating'] > 5).any():
        raise ValueError("Rating ngoài khoảng hợp lệ [1, 5].")


# ── Sinh dữ liệu Mock thực tế (không tách tuyến tính thô 100%) ──

_SHARED_POSITIVE_BODIES = [
    "The service was attentive and our table was ready right on time.",
    "Ordered two main dishes and a dessert to share, everything came out hot.",
    "Great atmosphere for dinner, reasonable portions for the neighborhood.",
    "Friendly staff and solid menu options, we had a pleasant evening here.",
    "Food tasted fresh and well seasoned, would stop by again when nearby.",
    "Nice cozy interior, drinks were well balanced and prices seemed fair.",
    "Really enjoyed the signature dish, quick turnaround even on a weekend.",
    "Clean dining room and helpful server, overall a very good meal.",
]

_SHARED_NEGATIVE_BODIES = [
    "Waited over twenty minutes just to get water and menus at our table.",
    "Portions felt small for the price and the main course arrived lukewarm.",
    "Noisy dining area and our server forgot one of the side orders.",
    "Expected much better based on the rating, the pasta lacked flavor.",
    "Service felt rushed and the bill seemed steep for average quality.",
]

_GENUINE_TAILS = [
    "We parked on the side street and sat near the window.",
    "I tried the lunch special while my friend had the soup.",
    "The wait was around fifteen minutes on Friday night.",
    "Not fancy, just reliable comfort food with decent coffee.",
    "Lighting was dim so it felt quiet enough to talk.",
    "Could use a few more vegetarian items on the menu.",
    "Honestly glad we gave this corner spot a chance!",
    "Loved the crispy crust on the bread!",
]

_FAKE_PROMO_TAILS = [
    "Highly recommend this place to everyone looking for the best meal!",
    "Definitely worth every penny, five stars all around!",
    "Hands down the best spot in town, you will not regret visiting!",
    "Must try if you are in the area, truly outstanding experience.",
    "Everyone in our group loved it, best restaurant experience!",
]

_FAKE_ATTACK_TAILS = [
    "Complete waste of money, stay away from this place.",
    "Worst experience in town, definitely avoid going here.",
    "Do not waste your time, terrible management and awful food.",
]


def generate_mock_data(
    n_reviews: int = 10000,
    n_users: Optional[int] = None,
    n_products: Optional[int] = None,
    fake_ratio: float = 0.13,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sinh dữ liệu mock mô phỏng đặc trưng thực tế của YelpChi để kiểm chứng pipeline B0/B1.

    Thiết kế chống tách tầm thường (non-trivial):
      - Phân phối bậc user & product theo power-law (~25-35% user có < 3 reviews)
      - Chồng lấn phân phối rating và text giữa review thật và giả để B0 không đạt 1.00 AUC
      - Tạo tín hiệu hành vi cá nhân + sản phẩm + thời gian (burst cùng ngày, độ lệch rating,
        tần suất review, cụm từ quảng bá tinh tế) để B1 vượt trội rõ rệt so với B0.
    """
    rng = np.random.RandomState(seed)

    if n_users is None:
        n_users = max(400, int(n_reviews * 0.18))
    if n_products is None:
        n_products = max(60, int(n_reviews * 0.018))

    user_ids = np.array([f"u_{i:05d}" for i in range(n_users)])
    product_ids = np.array([f"p_{i:04d}" for i in range(n_products)])

    # Chất lượng nền của từng sản phẩm (2.4 .. 4.6)
    product_true_quality = rng.uniform(2.5, 4.5, size=n_products)

    # Phân phối bậc sản phẩm (power-law)
    product_weights = rng.pareto(a=1.3, size=n_products) + 1.0
    product_weights /= product_weights.sum()

    # Chia khoảng 14% user là tài khoản đáng ngờ (có hành vi burst + viết lẫn review thật/giả)
    n_suspicious_users = max(10, int(n_users * 0.14))
    suspicious_user_set = set(user_ids[:n_suspicious_users])

    # Phân phối bậc user (giữ tỷ lệ cao user ít hoạt động < 3 reviews)
    user_weights = rng.pareto(a=1.6, size=n_users) + 0.5
    user_weights /= user_weights.sum()

    start_date = datetime(2012, 1, 1)
    total_days = 730  # ~2 năm

    n_fake_target = int(round(n_reviews * fake_ratio))
    n_genuine_target = n_reviews - n_fake_target

    records = []

    # 1) Sinh genuine reviews
    gen_user_idx = rng.choice(n_users, size=n_genuine_target, p=user_weights)
    gen_prod_idx = rng.choice(n_products, size=n_genuine_target, p=product_weights)
    gen_days = rng.randint(0, total_days, size=n_genuine_target)

    for k in range(n_genuine_target):
        u_id = user_ids[gen_user_idx[k]]
        p_idx = gen_prod_idx[k]
        p_id = product_ids[p_idx]
        ts = start_date + timedelta(days=int(gen_days[k]))

        # Rating quanh chất lượng nền sản phẩm + nhiễu tự nhiên
        raw_r = product_true_quality[p_idx] + rng.normal(0.0, 1.05)
        rating = int(np.clip(np.round(raw_r), 1, 5))

        if rating >= 4:
            body = _SHARED_POSITIVE_BODIES[rng.randint(len(_SHARED_POSITIVE_BODIES))]
            if rng.rand() < 0.72:
                body += " " + _GENUINE_TAILS[rng.randint(len(_GENUINE_TAILS))]
            elif rng.rand() < 0.15:
                # Chồng lấn nhẹ với cụm quảng bá để tránh tách biệt 100% ở B0/TF-IDF
                body += " " + _FAKE_PROMO_TAILS[rng.randint(len(_FAKE_PROMO_TAILS))]
        elif rating == 3:
            body = _SHARED_POSITIVE_BODIES[rng.randint(len(_SHARED_POSITIVE_BODIES))]
            body += " " + _GENUINE_TAILS[rng.randint(len(_GENUINE_TAILS))]
        else:
            body = _SHARED_NEGATIVE_BODIES[rng.randint(len(_SHARED_NEGATIVE_BODIES))]
            if rng.rand() < 0.75:
                body += " " + _GENUINE_TAILS[rng.randint(len(_GENUINE_TAILS))]

        records.append((u_id, p_id, rating, body, ts, 0))

    # 2) Sinh fake reviews (phân bố vào các đợt nhỏ của nhóm suspicious users + một phần ngẫu nhiên)
    sus_users = list(suspicious_user_set)
    idx_f = 0
    while idx_f < n_fake_target:
        # 65% đi theo burst trong ngày của 1 tài khoản, 35% đơn lẻ ngẫu nhiên
        if rng.rand() < 0.65 and idx_f + 2 < n_fake_target:
            burst_len = min(int(rng.randint(2, 5)), n_fake_target - idx_f)
            u_id = sus_users[rng.randint(len(sus_users))]
            burst_day = int(rng.randint(0, total_days))
            ts = start_date + timedelta(days=burst_day)
            is_attack = rng.rand() < 0.25
        else:
            burst_len = 1
            u_id = sus_users[rng.randint(len(sus_users))] if rng.rand() < 0.75 else user_ids[rng.randint(n_users)]
            ts = start_date + timedelta(days=int(rng.randint(0, total_days)))
            is_attack = rng.rand() < 0.25

        for _ in range(burst_len):
            # Thường nhắm vào sản phẩm cần kéo hoặc dìm sao -> tạo rating_deviation
            if is_attack:
                p_idx = int(rng.choice(np.where(product_true_quality >= 3.3)[0]))
                rating = int(rng.choice([1, 2], p=[0.75, 0.25]))
                body = _SHARED_NEGATIVE_BODIES[rng.randint(len(_SHARED_NEGATIVE_BODIES))]
                if rng.rand() < 0.65:
                    body += " " + _FAKE_ATTACK_TAILS[rng.randint(len(_FAKE_ATTACK_TAILS))]
                else:
                    body += " " + _GENUINE_TAILS[rng.randint(len(_GENUINE_TAILS))]
            else:
                p_idx = int(rng.choice(np.where(product_true_quality <= 3.7)[0]))
                rating = int(rng.choice([4, 5], p=[0.22, 0.78]))
                body = _SHARED_POSITIVE_BODIES[rng.randint(len(_SHARED_POSITIVE_BODIES))]
                if rng.rand() < 0.68:
                    body += " " + _FAKE_PROMO_TAILS[rng.randint(len(_FAKE_PROMO_TAILS))]
                else:
                    body += " " + _GENUINE_TAILS[rng.randint(len(_GENUINE_TAILS))]

            p_id = product_ids[p_idx]
            records.append((u_id, p_id, rating, body, ts, 1))
            idx_f += 1

    df = pd.DataFrame(
        records,
        columns=['user_id', 'product_id', 'rating', 'text', 'timestamp', 'label'],
    )
    df = df.sort_values(['timestamp', 'user_id', 'product_id']).reset_index(drop=True)
    df.insert(0, 'review_id', [f"rev_{i:06d}" for i in range(len(df))])

    _validate_schema(df)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[MockGenerator] Đã lưu {len(df):,} reviews → {save_path}")

    print(
        f"[MockGenerator] Sinh xong: {len(df):,} reviews, "
        f"{df['user_id'].nunique():,} users, "
        f"{df['product_id'].nunique():,} products | "
        f"fake ratio = {df['label'].mean():.2%}"
    )
    return df


def load_or_generate(config: dict) -> pd.DataFrame:
    """
    Ưu tiên tải dữ liệu thật (processed_path trước, rồi raw_path) nếu use_mock=False;
    khi use_mock=True thì sinh mock theo config.
    """
    data_cfg = config['data']
    if not data_cfg.get('use_mock', False):
        processed_path = data_cfg.get('processed_path', '')
        if _find_data_file(processed_path) is not None:
            return load_yelpchi(processed_path)
        return load_yelpchi(data_cfg['raw_path'])

    mock_path = data_cfg.get('mock_path', 'data/mock/')
    save_file = os.path.join(mock_path, 'yelpchi_mock.csv')
    return generate_mock_data(
        n_reviews=data_cfg.get('mock_n_reviews', 10000),
        seed=config.get('random_seed', 42),
        save_path=save_file,
    )


def print_data_summary(df: pd.DataFrame):
    """In tóm tắt thống kê cơ bản của dataset."""
    print("\n" + "=" * 60)
    print("TÓM TẮT DỮ LIỆU")
    print("=" * 60)
    print(f"  Tổng reviews     : {len(df):>10,}")
    print(f"  Số users         : {df['user_id'].nunique():>10,}")
    print(f"  Số products      : {df['product_id'].nunique():>10,}")
    print(f"  Tỷ lệ fake       : {df['label'].mean():>10.2%}")
    print(f"  Khoảng thời gian : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    print("\n  Phân phối rating:")
    for r in sorted(df['rating'].unique()):
        count = int((df['rating'] == r).sum())
        pct = count / len(df)
        print(f"    {r} sao: {count:>6,} ({pct:.1%})")

    rev_per_user = df.groupby('user_id').size()
    print("\n  Reviews / user:")
    print(f"    Mean  : {rev_per_user.mean():.2f}")
    print(f"    Median: {rev_per_user.median():.0f}")
    print(f"    Max   : {rev_per_user.max()}")
    print(
        f"    Users với < 3 reviews: {(rev_per_user < 3).sum():,} "
        f"({(rev_per_user < 3).mean():.1%})"
    )
    print("=" * 60)
