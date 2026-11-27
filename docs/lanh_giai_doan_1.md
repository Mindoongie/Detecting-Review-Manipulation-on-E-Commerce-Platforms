# Lanh / TV3 — Báo cáo & Bàn giao Giai đoạn 1

**Trạng thái:** Đã hoàn thành phần mã nguồn, kiểm thử và chạy thử nghiệm trên dữ liệu mẫu (`data/mock/reviews_text.jsonl`) tại branch `lanh`. Đang chờ **TV1 (Thanh)** cung cấp raw dataset thật + schema chính thức và **TV5 (Chi)** chốt mốc chia Train/Val/Test để chạy lấy số liệu thực tế.

---

## 1. Phạm vi, đối chiếu công việc và tình trạng dữ liệu

| Yêu cầu của TV3 (GĐ 1) | Tình trạng | Chi tiết đã triển khai |
| :--- | :--- | :--- |
| EDA văn bản (độ dài, tần suất từ, tỷ lệ trùng thô) | **Xong code & test** *(chờ data thật)* | Xuất `eda_summary.json`, top unigram/bigram, thống kê trùng lặp chính xác trên cả bản gốc và bản sạch; **chỉ chạy trên tập Train**. |
| Lưu song song 2 bản (`text_raw` & `text_clean`), giữ dấu phủ định | **Đã xong** | Lưu song song `text_raw` (nguyên gốc) và `text_clean` (bỏ HTML/URL, mở rộng viết tắt phủ định, giữ nguyên nhóm từ phủ định). |
| Bộ trích xuất TF-IDF / n-gram (chỉ fit trên Train) cấp cho B1 | **Đã xong** | `TfidfB1Extractor` (`ngram_range=(1, 2)`), chặn lỗi nếu fit trên dữ liệu có lẫn `val`/`test`; xuất cả ma trận thưa `.npz` và bảng feature dày `b1_text_features`. |
| Kiểm thử chống rò rỉ & tài liệu bàn giao | **Đã xong** | Bộ unit test `pytest` đạt **7/7**; tài liệu đặt tại `docs/lanh_giai_doan_1.md` theo quy ước thư mục `docs/` của nhóm. |

**Lưu ý hiện trạng dữ liệu:** Repo chưa có raw dataset thật (YelpChi/YelpNYC) và mốc chia tập chính thức. Để code song song không phụ thuộc TV1, Lanh tự tạo `data/mock/reviews_text.jsonl` (23 dòng giả định bám theo schema mock của TV4 và bổ sung cột `text`, `label`). Mọi số liệu trong `outputs/demo_text_stage1` chỉ kiểm chứng đường chạy, không dùng trong bảng kết quả thực nghiệm thật.

---

## 2. Hợp đồng dữ liệu và quy ước chống rò rỉ

Module đọc được các định dạng `.jsonl`, `.csv`, `.parquet` với các cột bắt buộc:

| Cột đầu vào | Quy tắc kiểm tra |
| :--- | :--- |
| `review_id` | Chuỗi khác rỗng, duy nhất toàn bộ file |
| `user_id`, `product_id` | Chuỗi khác rỗng; giữ nguyên ID nguồn |
| `timestamp` | `YYYY-MM-DD` (quy ước `00:00:00Z` UTC) hoặc ISO datetime có `Z`/offset; từ chối chuỗi mơ hồ múi giờ |
| `text` | Cột văn bản bắt buộc tồn tại; giá trị null/rỗng được chuyển thành `""` và gắn cờ `text_is_empty=1` (không âm thầm xóa dòng) |

**Nguyên tắc chống rò rỉ:**
- **Whitelist cột đầu vào:** Cột `label` / `filtered` bị loại ngay khi load, tuyệt đối không tham gia tiền xử lý, EDA hay trích xuất đặc trưng.
- **Snapshot cutoff:** Loại bỏ toàn bộ review có `timestamp_utc > snapshot_cutoff`.
- **Phân chia Train / Val / Test:** Hỗ trợ phân tách theo `--train-cutoff` và `--val-cutoff` (hoặc nhận cột `split` đã chốt từ TV5).
- **Fit-on-Train-only:** Cả EDA văn bản lẫn `TfidfVectorizer` (và `TruncatedSVD` nếu bật `svd_components > 0`) chỉ chạy/fit trên `split == "train"`. Truyền dữ liệu có lẫn `val`/`test` vào `TfidfB1Extractor.fit()` sẽ báo lỗi `Anti-leakage violation`.

---

## 3. Lưu song song 2 bản văn bản & xử lý dấu phủ định

Bảng `dual_text_corpus.csv` (hoặc `.parquet`) lưu song song `text_raw` và `text_clean` theo khóa `review_id`:

- **`text_raw`**: giữ nguyên chữ hoa/thường, dấu câu, URL, HTML gốc để tính đặc trưng bề mặt và đối chiếu Error Analysis ở Giai đoạn 4.
- **`text_clean`**:
  1. Giải mã HTML entity và loại thẻ HTML (`remove_html=true`), loại URL (`remove_urls=true`).
  2. Chuẩn hóa dấu nháy và **mở rộng viết tắt trước khi bỏ dấu câu** (`won't -> will not`, `can't -> cannot`, `shan't -> shall not`, `ain't -> is not`, `*n't -> * not` như `didn't -> did not`, `isn't -> is not`).
  3. Chuyển chữ thường (`lowercase=true`), bỏ dấu câu nhưng giữ chữ cái, chữ số (`keep_digits=true`) và khoảng trắng chuẩn hóa.
  4. **Giữ nguyên từ phủ định** (`no`, `not`, `never`, `neither`, `nor`, `none`, `nothing`, `nowhere`, `cannot`, `hardly`, `barely`, `scarcely`, `without`). Khi bật `use_stopwords=true`, `strip_negation_from_stopwords()` tự động loại nhóm từ này khỏi danh sách stopword.

---

## 4. Đầu ra EDA văn bản (chỉ trên Train) & Đặc trưng B1 cho TV5

### A. EDA văn bản (`<out>/eda/`)
- `eda_summary.json`: số lượng review/user/product trên Train, tỷ lệ rỗng sau làm sạch, tỷ lệ chứa từ phủ định, phân phối độ dài (`min`, `p25`, `median`, `mean`, `p75`, `p95`, `max`) theo ký tự gốc / từ sạch / số câu, thống kê từ vựng (`vocab_size`, `type_token_ratio`, `hapax_rate`, `negation_token_rate`), và thống kê trùng lặp chính xác trên `text_raw` vs `text_clean` (`excess_duplicate_rate`, `in_duplicate_cluster_rate`, số cụm trùng khác user, số cụm lặp trên cùng sản phẩm).
- `eda_word_freq.csv`, `eda_bigram_freq.csv`, `eda_raw_duplicates.csv`, `eda_clean_duplicates.csv`.

### B. Đầu ra TF-IDF / n-gram cho B1 (`<out>/tfidf/`)
Khóa ghép với đặc trưng hành vi/thời gian của TV4 là **`review_id`** (giữ `user_id`, `product_id`, `decision_time`, `snapshot_cutoff`, `split` để truy vết — **không đưa các cột khóa/mốc vào mô hình**):
- **Cho Logistic Regression B1:** `tfidf_train.npz`, `tfidf_val.npz`, `tfidf_test.npz` (khớp thứ tự `tfidf_<split>_review_ids.json`), `tfidf_feature_names.json`, `tfidf_vocab_summary.json`, `tfidf_extractor.joblib`.
- **Cho LightGBM / bảng tổng hợp B1 (`b1_text_features.csv` / `.parquet`):**

| Nhóm | Cột feature | Định nghĩa |
| :--- | :--- | :--- |
| Bề mặt (`text_raw`) | `text_char_len`, `text_raw_word_count`, `text_sentence_count`, `text_uppercase_ratio`, `text_digit_ratio`, `text_exclamation_count`, `text_question_count` | Độ dài, câu, tỷ lệ chữ hoa/số, số dấu `!`, `?` của từng review |
| Sạch & Phủ định (`text_clean`) | `text_clean_char_len`, `text_word_count`, `text_avg_word_len`, `text_unique_word_ratio`, `text_negation_count`, `text_has_negation`, `text_is_empty` | Độ dài từ sạch, đa dạng từ vựng (TTR), số lượng & cờ phủ định, cờ rỗng |
| Tóm tắt TF-IDF (fit trên Train) | `tfidf_oov_token_ratio`, `tfidf_nnz`, `tfidf_max`, `tfidf_sum`, `tfidf_mean_nonzero`, `tfidf_entropy` | Tỷ lệ từ ngoài từ điển Train, số n-gram khác 0, trọng số max/sum/mean và entropy phân bố trọng số TF-IDF |
| SVD (tùy chọn) | `tfidf_svd_00..` | Chỉ sinh khi `svd_components > 0` (fit trên Train) |

---

## 5. Kiểm chứng và hướng dẫn chạy

Kiểm thử bằng `pytest` (Python 3.14.4) đã đạt **7/7 test cases**:
- Chuẩn hóa UTC timestamp và từ chối định dạng thiếu múi giờ / sai lịch.
- Kiểm tra schema, từ chối `review_id` trùng hoặc thiếu cột bắt buộc.
- Bảo toàn `text_raw` và giữ dấu phủ định (`not`, `never`, `no`, mở rộng `didn't`, `isn't`, `won't`) kể cả khi bật lọc stopword.
- Kiểm thử chống rò rỉ: đảo toàn bộ `label`, sửa văn bản `val`/`test` và thêm dòng sau `snapshot_cutoff` không làm thay đổi EDA Train, từ điển TF-IDF, `tfidf_train.npz` hay các dòng Train trong `b1_text_features.csv`.
- Chặn lỗi khi gọi `TfidfB1Extractor.fit()` trên dữ liệu có lẫn `val`/`test`.
- Đối chiếu EDA trùng thô/sạch tính tay trên `data/mock/reviews_text.jsonl`.
- Chạy end-to-end định dạng `.parquet` có bật `svd_components=4`, không có NaN.

**Kết quả chạy thử CLI trên dữ liệu mẫu (`outputs/demo_text_stage1`):**
- 22 review quan sát được (`train=15, val=4, test=3`; loại `fu01` sau cutoff).
- Trên Train: độ dài trung bình **43.73 ký tự gốc** / **6.80 từ sạch**; **1 review rỗng**; **5/15 review** có từ phủ định; **3 dòng trùng thô** (`text_raw`) và **4 dòng trùng sau làm sạch** (`text_clean`, 1 cụm khác user trên cùng sản phẩm).
- Từ điển TF-IDF Train: **156 n-gram** (giữ **24 n-gram có dấu phủ định**); bảng `b1_text_features.csv` kích thước **(22, 26)**.

```bash
python -m pytest -v
python -m src.text_pipeline.cli \
  --input data/mock/reviews_text.jsonl \
  --config configs/text_stage1.json \
  --train-cutoff 2024-01-15T23:59:59Z \
  --val-cutoff 2024-01-25T23:59:59Z \
  --snapshot-cutoff 2024-02-10T23:59:59Z \
  --out outputs/demo_text_stage1
```

---

## 6. Việc cần phối hợp khi có dữ liệu thật

1. **Thanh / TV1:** Bàn giao raw dataset thật (có `review_id`, `user_id`, `product_id`, `timestamp`, `text`) và quy tắc `snapshot_cutoff`.
2. **Chi / TV5:** Chốt mốc `--train-cutoff`, `--val-cutoff` và ghép `b1_text_features` (hoặc `tfidf_<split>.npz`) theo `review_id` cùng feature của **Hiền / TV4** để chạy B1.
3. **Lanh / TV3:** Chạy lại pipeline trên dữ liệu thật để thay số liệu EDA mock bằng số liệu thực tế và chuẩn bị **MinHash/LSH + GCS** cho Giai đoạn 2.
