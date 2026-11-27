# TV5 (Chi) — Giao thức Benchmark, Baseline B0/B1 & Quy ước Chống Rò Rỉ (Giai đoạn 1)

Tài liệu này chốt phần việc của **TV5 (Chi)** trong **Giai đoạn 1** theo kế hoạch [chốt.md](../chốt.md), làm chuẩn dùng chung cho toàn nhóm (TV1–TV5) trước khi ghép các đặc trưng cấu kết nhóm ở Giai đoạn 2.

---

## 1. Schema đầu vào thống nhất

Toàn bộ pipeline TV5 đọc dữ liệu theo bảng review chuẩn (từ TV1 hoặc bộ dữ liệu thô YelpChi/Mock):

| Cột | Kiểu dữ liệu | Ý nghĩa |
| :--- | :--- | :--- |
| `review_id` | `str` | Khóa chính duy nhất của mỗi review |
| `user_id` | `str` | Mã người đánh giá |
| `product_id` | `str` | Mã sản phẩm / nhà hàng / khách sạn |
| `rating` | `int` (`1..5`) | Điểm sao đánh giá |
| `text` | `str` | Nội dung văn bản review |
| `timestamp` | `datetime` | Thời điểm đăng review |
| `label` | `int` (`0` / `1`) | `0` = genuine (`recommended`), `1` = fake (`filtered`) |

---

## 2. Quy trình chia Train / Validation / Test theo thời gian

Thực hiện tại [src/temporal_split.py](../src/temporal_split.py):

1. **Sắp xếp tuần tự**: Sắp toàn bộ review theo `(timestamp, review_id)` tăng dần (`mergesort`).
2. **Phân lô Snapshot ($T_k$)**: Ánh xạ mỗi review vào mốc đóng cửa sổ chấm điểm định kỳ (`snapshot_freq = "W"` — cuối mỗi tuần, hoặc `"M"` / `"D"`).
3. **Chia 70% / 15% / 15% có căn chỉnh ranh giới (`align_boundary=True`)**:
   - Đường cắt được đặt tại ranh giới giữa hai snapshot (hoặc hai ngày liên tiếp), **không cắt ngang qua các review có cùng timestamp hay cùng snapshot**.
   - Bắt buộc thỏa mãn điều kiện kiểm chứng tự động `verify_no_leakage`:
     - $\max(\text{Train.timestamp}) < \min(\text{Val.timestamp})$
     - $\max(\text{Val.timestamp}) < \min(\text{Test.timestamp})$
     - Tập `review_id` và tập `snapshot_cutoff` của 3 phần hoàn toàn rời nhau.
4. **Xuất bảng khóa chia tập**: Lưu `outputs/results/split_assignments.csv` (`review_id, timestamp, split, snapshot_cutoff`) để TV1–TV4 dùng thống nhất.
5. **Cô lập tập Test cuối**: Trong toàn bộ Giai đoạn 1–3, mọi so sánh, chọn tham số và chọn ngưỡng chỉ thực hiện trên **Validation**. Tập **Test** chỉ được mở khóa chạy 1 lần duy nhất ở Giai đoạn 4.

---

## 3. Định nghĩa Thời điểm ra quyết định (Decision Point) & Chống rò rỉ đặc trưng

Đây là quy ước bắt buộc để tránh hai lỗi rò rỉ kinh điển: **rò rỉ tương lai (future leakage)** và **tự rò rỉ nhãn/chính mình (self-leakage)**.

* **Mốc đóng băng tham số mô hình ($T_{\text{freeze}} = \max(\text{Train.timestamp})$):**
  * Từ điển `TF-IDF` (unigram + bigram, `sublinear_tf=True`) và các giá trị fallback toàn cục (`global_mean_rating`, `global_avg_words`) **chỉ được fit trên tập Train**, tuyệt đối không fit lại trên Val hoặc Test.
  * Trọng số mô hình (`LogisticRegression` / `LightGBM`) và ngưỡng phân loại nhị phân (`best_threshold`) chỉ học/chọn trên Train và Validation.
* **Chấm điểm theo lô snapshot nhân quả ($t \le T_k$):**
  * Khi chấm điểm một lô review thuộc snapshot $T_k$, toàn bộ đặc trưng lịch sử của User và Product (số review, tỷ lệ sao cực đoan, `user_max_reviews_one_day`, `product_singleton_ratio`, mật độ review 7 ngày gần nhất `product_recent_7d_count`, ...) **chỉ được phép tổng hợp từ các sự kiện không nhãn (`unlabeled`) có `timestamp <= T_k`**.
  * Tuyệt đối không đọc bất kỳ bản ghi nào có `timestamp > T_k`, và không dùng cột `label` để tính bất kỳ đặc trưng lịch sử nào.
* **Leave-Self-Out cho các giá trị trung bình / độ lệch chuẩn:**
  * Khi tính `user_avg_rating`, `user_rating_std`, `product_avg_rating`, `product_rating_std`, `user_avg_review_length` tại snapshot $T_k$, hệ thống trừ chính review đang xét khỏi tổng cộng dồn đến $T_k$; nếu chưa có review khác trước/cùng kỳ (`other_cnt == 0`), dùng giá trị fallback từ Train. Nhờ vậy `rating_deviation = rating - product_avg_rating` không bị suy biến về `0` một cách giả tạo ở các sản phẩm mới.

---

## 4. Thước đo đánh giá (Evaluation Metrics)

Cài đặt trong [src/metrics.py](../src/metrics.py):

| Thước đo | Ký hiệu / Tên | Vai trò trong bài toán mất cân bằng lớp |
| :--- | :--- | :--- |
| **AUC-ROC** | `auc_roc` | Đánh giá năng lực xếp hạng tổng thể trên mọi ngưỡng. |
| **Average Precision** | `average_precision` | Thước đo chính khi tỷ lệ review giả thấp (~13%), phản ánh diện tích dưới đường cong Precision-Recall. |
| **F1 Binary** | `f1_binary`, `f1_at_best_threshold` | Đánh giá cân bằng Precision/Recall tại ngưỡng `0.5` và tại ngưỡng tối ưu PR-curve trên Validation (áp dụng cứng sang Test, không dò lại trên Test). |
| **Precision@K** | `precision@100`, `@500`, `@1000` | Trong top $K$ review đáng ngờ nhất gửi bộ phận kiểm duyệt, tỷ lệ bắt trúng review bị lọc là bao nhiêu. |
| **Recall@K** | `recall@100`, `@500`, `@1000` | Top $K$ review đáng ngờ nhất phủ được bao nhiêu % tổng số review bị lọc trong kỳ đánh giá. |

> **Lưu ý kiểm thử**: Khi số mẫu $N < K$, `precision@K` và `recall@K` trả về `NaN` thay vì tự động cắt $K \leftarrow N$ để tránh báo cáo sai lệch con số cơ sở (base rate).

---

## 5. Đặc tả Baselines Giai đoạn 1 (B0 & B1)

Thực hiện trên **cùng một mô hình bảng** (`LogisticRegression` với `StandardScaler` và `LightGBM`, đều đặt `class_weight="balanced"`) để đối chứng công bằng:

* **B0 — Raw Single-Review Baseline (8 đặc trưng)** ([src/features/b0_features.py](../src/features/b0_features.py)):
  * Chỉ dùng thông tin nội tại của từng review: `rating`, `review_length_chars`, `review_length_words`, `review_length_sentences`, `avg_word_length`, `exclamation_count`, `question_mark_count`, `capital_ratio`.
  * Mục đích: kiểm chứng pipeline chạy thông suốt và làm mốc sàn tối thiểu.
* **B1 — Strong Individual + Content + Product + Time Baseline** ([src/features/b1_features.py](../src/features/b1_features.py)):
  * Bao gồm **B0** + **TF-IDF (1,2-gram, max 500 đặc trưng)** + **12 đặc trưng hành vi cá nhân** (tổng review, rating trung bình/std, tỷ lệ sao 4–5 / 1–2 / cực đoan {1,5}, số sản phẩm, `user_max_reviews_one_day`, `user_is_singleton`, tốc độ đăng, tuổi tài khoản `days_since_first_review`, độ dài review TB) + **8 đặc trưng sản phẩm** (tổng review, rating TB/std, tỷ lệ tích cực/tiêu cực/cực đoan, `product_singleton_ratio`, `product_recent_7d_count`) + **6 đặc trưng tương tác/thời gian** (`day_of_week`, `is_weekend`, `rating_deviation`, `abs_rating_deviation`, `user_rating_deviation`, `is_first_review_of_product`).

---

## 6. Giới hạn hiện tại (Limitations — Giai đoạn 1)

1. **Bản chất nhãn YelpChi**: Nhãn `filtered/recommended` là đầu ra từ bộ lọc tự động của Yelp, đóng vai trò nhãn ủy nhiệm (proxy label) gần đúng chứ không phải ground truth tuyệt đối do con người xác minh 100%; các kết luận cần diễn giải cẩn trọng theo góc độ *"khả năng bị bộ lọc đánh dấu / đáng ngờ"*.
2. **Dữ liệu kiểm chứng hiện tại**: Khi `use_mock: true` trong `config/config.yaml`, pipeline chạy trên dữ liệu mô phỏng (`data/mock/yelpchi_mock.csv`) để kiểm chứng logic chống rò rỉ, thước đo và sự vượt trội của B1 so với B0; kết quả thực nghiệm chính thức đòi hỏi nạp dữ liệu YelpChi thật từ TV1 (`use_mock: false`).
3. **Phạm vi tín hiệu ở B1**: B1 mới chỉ mô hình hóa đặc trưng đơn lẻ theo user/product/văn bản cá nhân, **chưa có các đặc trưng cấu kết nhóm** (đồng đánh giá theo cặp ở B2, cụm FP-Growth / GCS / GTW / GD / GETF ở B3) — các thành phần này sẽ được ghép ở Giai đoạn 2.
4. **Hành vi tài khoản lạnh (Cold-start)**: Tài khoản hoặc sản phẩm lần đầu xuất hiện trong tập Validation/Test chưa có lịch sử trước đó sẽ nhận giá trị fallback từ Train, làm giảm độ nhạy của tín hiệu hành vi cá nhân đối với tài khoản tạo mới hoàn toàn (đây cũng là lý do cần nhóm đặc trưng đồng đánh giá/nội dung ở các giai đoạn sau).
