# Detecting-Review-Manipulation-on-E-Commerce-Platforms

Dự án khai phá dữ liệu phát hiện hành vi thao túng / cấu kết đánh giá trên nền tảng thương mại điện tử (theo kế hoạch phân công trong [chốt.md](chốt.md)).

## Cấu trúc hiện có (TV5 — Chi, Giai đoạn 1)

- `config/config.yaml`: Cấu hình chia tập theo thời gian, snapshot `W`, đặc trưng B0/B1, mô hình `LogisticRegression` & `LightGBM`, và các thước đo đánh giá.
- `src/data_loader.py`: Trình tải dữ liệu chuẩn hóa schema (`review_id, user_id, product_id, rating, text, timestamp, label`) từ bảng trung gian/YelpChi raw hoặc sinh mock data mô phỏng thực tế.
- `src/temporal_split.py`: Chia `Train / Val / Test` (70/15/15) theo thời gian, căn chỉnh ranh giới snapshot chống cắt đôi ngày/tuần, kiểm tra tự động chống rò rỉ (`verify_no_leakage`) và xuất `split_assignments.csv`.
- `src/metrics.py`: Tính `AUC-ROC`, `Average Precision`, `F1` (kèm chốt ngưỡng trên Validation), `Precision@K` và `Recall@K`.
- `src/features/b0_features.py`: Bộ đặc trưng thô cơ bản B0 (8 đặc trưng review đơn lẻ).
- `src/features/b1_features.py`: Bộ đặc trưng B1 (B0 + TF-IDF fit trên Train + hành vi cá nhân + thống kê sản phẩm + thời gian, tính nhân quả theo mốc snapshot $\le T_k$ và Leave-Self-Out).
- `src/models/baseline.py`: Huấn luyện và suy luận trên `LogisticRegression` và `LightGBM`.
- `scripts/run_baselines.py`: Script chạy end-to-end Giai đoạn 1 (Train & đánh giá trên Validation, giữ nguyên tập Test cuối).
- `tests/test_tv5_phase1.py`: Bộ kiểm thử đơn vị và kiểm chứng bất biến nhân quả (anti-leakage test).
- `docs/tv5_benchmark_protocol.md`: Tài liệu chốt giao thức chia tập, thời điểm ra quyết định, thước đo, thiết kế B0/B1 và giới hạn Giai đoạn 1.

## Hướng dẫn chạy (TV5 Giai đoạn 1)

```bash
pip install -r requirements.txt
pytest tests/test_tv5_phase1.py -v
python scripts/run_baselines.py
```

Để chạy trên dữ liệu thật (đặt file Parquet/CSV/TSV chuẩn vào `data/processed/` hoặc dữ liệu YelpChi vào `data/raw/`):

```bash
python scripts/run_baselines.py --real-data
```
