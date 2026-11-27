"""
temporal_split.py — TV5 Phase 1
Chia dữ liệu Train / Validation / Test theo thời gian (chronological split)
và gán mốc snapshot chấm điểm theo lô (decision point protocol).

Nguyên tắc chống rò rỉ dữ liệu (Data Leakage Prevention):
  1. Sắp xếp toàn bộ reviews theo [timestamp, review_id] tăng dần.
  2. Gán mỗi review vào lô snapshot T_k (cuối tuần 'W', cuối tháng 'M', hoặc cuối ngày 'D').
  3. Cắt ranh giới Train / Val / Test tại ranh giới snapshot/ngày (align_boundary=True)
     để đảm bảo KHÔNG có hai tập nào dùng chung cùng thời điểm hay cùng snapshot:
       max(train.timestamp) < min(val.timestamp)
       max(val.timestamp)   < min(test.timestamp)
  4. Định nghĩa thời điểm ra quyết định (Decision Point):
       - Ngưỡng đóng băng mô hình & TF-IDF: train_end (cuối tập Train).
       - Khi chấm điểm bất kỳ lô snapshot T_k nào, mọi đặc trưng lịch sử
         CHỈ được phép đọc các sự kiện chưa gắn nhãn có timestamp <= T_k.
       - Tập Test cuối tuyệt đối giữ nguyên chưa đụng tới trong GĐ 1–3.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def assign_snapshot_cutoff(
    timestamps: pd.Series,
    freq: str = "W",
) -> pd.Series:
    """
    Ánh xạ mỗi timestamp về mốc đóng snapshot cuối kỳ (T_k).

    Ví dụ:
      - freq='W': cuối ngày Chủ Nhật của tuần đó (23:59:59.999999999 -> chuẩn hóa về cuối ngày)
      - freq='M': ngày cuối tháng đó
      - freq='D': cuối ngày hôm đó
    """
    ts = pd.to_datetime(timestamps)
    if freq not in ("D", "W", "M"):
        raise ValueError(f"snapshot_freq phải thuộc ('D', 'W', 'M'), nhận được: {freq!r}")

    # Chuẩn hóa về cuối ngày của period_end (23:59:59) để so sánh <= T_k bao phủ trọn ngày cuối kỳ
    period_end = ts.dt.to_period(freq).dt.end_time.dt.floor('s')
    return period_end


def _find_clean_cut_index(
    keys: np.ndarray,
    target_idx: int,
    min_idx: int,
    max_idx: int,
) -> int:
    """
    Tìm vị trí cắt gần target_idx nhất sao cho keys[cut - 1] != keys[cut]
    và min_idx <= cut <= max_idx. Nếu không tồn tại, trả về target_idx.
    """
    target_idx = int(np.clip(target_idx, min_idx, max_idx))
    if target_idx <= min_idx or target_idx >= max_idx:
        return target_idx

    # Tìm tất cả vị trí ranh giới hợp lệ trong [min_idx, max_idx]
    valid_cuts = []
    for i in range(min_idx, max_idx + 1):
        if 0 < i < len(keys) and keys[i - 1] != keys[i]:
            valid_cuts.append(i)

    if not valid_cuts:
        return target_idx

    return min(valid_cuts, key=lambda idx: abs(idx - target_idx))


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = 'timestamp',
    snapshot_freq: str = 'W',
    align_boundary: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Chia dataset theo thời gian thành Train / Val / Test và gắn snapshot_cutoff.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
        raise ValueError(
            f"Tổng tỷ lệ phải = 1.0, nhận được {train_ratio + val_ratio + test_ratio}"
        )
    if timestamp_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{timestamp_col}' trong DataFrame")
    if len(df) < 3:
        raise ValueError("Dataset cần tối thiểu 3 dòng để chia Train/Val/Test.")

    sort_cols = [timestamp_col, 'review_id'] if 'review_id' in df.columns else [timestamp_col]
    df_sorted = df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    df_sorted['snapshot_cutoff'] = assign_snapshot_cutoff(
        df_sorted[timestamp_col], freq=snapshot_freq
    )

    n = len(df_sorted)
    raw_train_end = int(round(n * train_ratio))
    raw_val_end = int(round(n * (train_ratio + val_ratio)))
    raw_train_end = max(1, min(raw_train_end, n - 2))
    raw_val_end = max(raw_train_end + 1, min(raw_val_end, n - 1))

    if align_boundary:
        # Ưu tiên cắt theo ranh giới snapshot_cutoff; nếu ít snapshot quá, fallback sang timestamp
        snap_keys = df_sorted['snapshot_cutoff'].values
        ts_keys = df_sorted[timestamp_col].values

        train_end = _find_clean_cut_index(snap_keys, raw_train_end, 1, n - 2)
        if train_end == raw_train_end and snap_keys[train_end - 1] == snap_keys[train_end]:
            train_end = _find_clean_cut_index(ts_keys, raw_train_end, 1, n - 2)

        val_end = _find_clean_cut_index(snap_keys, raw_val_end, train_end + 1, n - 1)
        if val_end == raw_val_end and snap_keys[val_end - 1] == snap_keys[val_end]:
            val_end = _find_clean_cut_index(ts_keys, raw_val_end, train_end + 1, n - 1)
    else:
        train_end = raw_train_end
        val_end = raw_val_end

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    split_info = {
        'method': 'temporal',
        'snapshot_freq': snapshot_freq,
        'align_boundary': align_boundary,
        'total_reviews': n,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_start': train_df[timestamp_col].min(),
        'train_end': train_df[timestamp_col].max(),
        'val_start': val_df[timestamp_col].min(),
        'val_end': val_df[timestamp_col].max(),
        'test_start': test_df[timestamp_col].min(),
        'test_end': test_df[timestamp_col].max(),
        'train_last_snapshot': train_df['snapshot_cutoff'].max(),
        'val_last_snapshot': val_df['snapshot_cutoff'].max(),
        'test_last_snapshot': test_df['snapshot_cutoff'].max(),
        # Decision points:
        # 1) Điểm đóng băng mô hình/TF-IDF = train_end
        # 2) Điểm mở rộng lịch sử cho mỗi lô chấm điểm = snapshot_cutoff T_k (<= T_k, không đọc nhãn)
        'model_freeze_point': train_df[timestamp_col].max(),
        'decision_point_val': train_df[timestamp_col].max(),
        'decision_point_test': val_df[timestamp_col].max(),
        'decision_point_protocol': (
            f"Frozen TF-IDF/model at train_end; causal rolling snapshot ({snapshot_freq}) "
            "features read unlabeled events with timestamp <= T_k only."
        ),
        'train_fake_ratio': float(train_df['label'].mean()) if 'label' in train_df else float('nan'),
        'val_fake_ratio': float(val_df['label'].mean()) if 'label' in val_df else float('nan'),
        'test_fake_ratio': float(test_df['label'].mean()) if 'label' in test_df else float('nan'),
    }

    return train_df, val_df, test_df, split_info


def print_split_summary(split_info: Dict):
    """In tóm tắt kết quả chia dữ liệu theo thời gian."""
    print("\n" + "=" * 64)
    print("TEMPORAL TRAIN / VAL / TEST SPLIT")
    print("=" * 64)
    total = split_info['total_reviews']
    print(
        f"  Snapshot freq: {split_info.get('snapshot_freq', 'W')} | "
        f"Align boundary: {split_info.get('align_boundary', True)}"
    )

    for name in ['train', 'val', 'test']:
        size = split_info[f'{name}_size']
        pct = size / total * 100
        fake_r = split_info[f'{name}_fake_ratio']
        start = split_info[f'{name}_start']
        end = split_info[f'{name}_end']
        print(f"\n  {name.upper():>5}: {size:>6,} reviews ({pct:.1f}%) | fake: {fake_r:.2%}")
        print(f"         {start.date()} → {end.date()}")

    print("\n  Decision Points (thời điểm ra quyết định):")
    print(f"    Đóng băng TF-IDF & Mô hình : ≤ {split_info['model_freeze_point'].date()}")
    print(f"    Chấm Val  (rolling ≤ T_k)  : {split_info['val_start'].date()} → {split_info['val_end'].date()}")
    print(f"    Chấm Test (giữ nguyên GĐ4) : {split_info['test_start'].date()} → {split_info['test_end'].date()}")
    print("=" * 64)


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    strict_boundary: bool = True,
) -> bool:
    """
    Kiểm chứng tự động không có rò rỉ thời gian, trùng ID hay trùng snapshot giữa các tập.
    """
    checks_passed = True

    for name, part in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if len(part) == 0:
            print(f"[LEAK WARNING] Tập {name} bị rỗng.")
            return False
        if part[timestamp_col].isna().any():
            print(f"[LEAK WARNING] Tập {name} có timestamp NaT.")
            checks_passed = False

    train_max = train_df[timestamp_col].max()
    val_min = val_df[timestamp_col].min()
    val_max = val_df[timestamp_col].max()
    test_min = test_df[timestamp_col].min()

    cmp_ok_tv = (train_max < val_min) if strict_boundary else (train_max <= val_min)
    cmp_ok_vt = (val_max < test_min) if strict_boundary else (val_max <= test_min)
    op_str = "<" if strict_boundary else "≤"

    if not cmp_ok_tv:
        print(f"[LEAK WARNING] Train max ({train_max}) không thỏa {op_str} Val min ({val_min})")
        checks_passed = False
    else:
        print(f"[OK] Train max ({train_max.date()}) {op_str} Val min ({val_min.date()})")

    if not cmp_ok_vt:
        print(f"[LEAK WARNING] Val max ({val_max}) không thỏa {op_str} Test min ({test_min})")
        checks_passed = False
    else:
        print(f"[OK] Val max ({val_max.date()}) {op_str} Test min ({test_min.date()})")

    if 'snapshot_cutoff' in train_df.columns:
        s_tr = set(train_df['snapshot_cutoff'])
        s_va = set(val_df['snapshot_cutoff'])
        s_te = set(test_df['snapshot_cutoff'])
        if (s_tr & s_va) or (s_va & s_te) or (s_tr & s_te):
            print("[LEAK WARNING] Có snapshot_cutoff giao nhau giữa các tập!")
            checks_passed = False
        else:
            print("[OK] Snapshot windows tách biệt hoàn toàn giữa Train / Val / Test")

    if 'review_id' in train_df.columns:
        train_ids = set(train_df['review_id'])
        val_ids = set(val_df['review_id'])
        test_ids = set(test_df['review_id'])

        overlap_tv = train_ids & val_ids
        overlap_vt = val_ids & test_ids
        overlap_tt = train_ids & test_ids

        if overlap_tv or overlap_vt or overlap_tt:
            print(
                f"[LEAK WARNING] Review ID trùng lặp: "
                f"train∩val={len(overlap_tv)}, val∩test={len(overlap_vt)}, "
                f"train∩test={len(overlap_tt)}"
            )
            checks_passed = False
        else:
            print("[OK] Không có review_id trùng lặp giữa các tập")

    if checks_passed:
        print("\n✅ Tất cả kiểm tra chống rò rỉ đã PASS")
    else:
        print("\n❌ Phát hiện rò rỉ dữ liệu! Cần xem lại quy trình chia.")

    return checks_passed


def save_split_assignments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Xuất bảng ánh xạ review_id → split & snapshot_cutoff để TV1–TV4 dùng chung.
    """
    cols = [c for c in ['review_id', 'timestamp', 'split', 'snapshot_cutoff'] if c in train_df.columns]
    combined = pd.concat([train_df[cols], val_df[cols], test_df[cols]], ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.lower().endswith('.parquet'):
        combined.to_parquet(output_path, index=False)
    else:
        combined.to_csv(output_path, index=False)
    return output_path
