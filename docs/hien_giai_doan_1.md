# Hiền / TV4 — Bàn giao giai đoạn 1

## 1. Phạm vi và tình trạng

Đã xây dựng module chuẩn hóa thời gian, EDA số review theo ngày/tuần và theo
sản phẩm, trích xuất đặc trưng hành vi/thời gian cho B1, cùng đặc tả chỉ số
nhóm cho giai đoạn 2. Có dữ liệu mẫu và kiểm thử chống rò rỉ.

Repo chưa có raw metadata, schema chung hoặc mốc chia tập chính thức.
Do đó chưa thể kết luận phân phối hay chất lượng mô hình trên Yelp.
Kết quả trong `outputs/demo` chỉ kiểm chứng đường chạy, không dùng trong
bảng kết quả thực nghiệm thật. Module không loại user ít hơn 3 review.

## 2. Hợp đồng dữ liệu và quy tắc thời gian

Đầu vào: UTF-8 JSONL, mỗi dòng một review. Các cột bắt buộc:

| Cột | Quy tắc |
|---|---|
| `review_id` | Chuỗi khác rỗng, duy nhất toàn bộ file |
| `user_id`, `product_id` | Chuỗi khác rỗng; giữ nguyên ID nguồn |
| `timestamp` | `YYYY-MM-DD` hoặc ISO datetime có `Z`/offset |
| `rating` | Số hữu hạn trong [1,5]; cần bổ sung vào schema chung |

Text, label và các cột phụ không được dùng để tính feature. Không tự suy ra
đơn vị Unix epoch, không đoán ngày/tháng hay múi giờ. Dữ liệu nguồn dạng epoch
phải được TV1 chuyển đổi với đơn vị được khai báo. Ngày không có giờ được
quy ước 00:00 UTC; datetime có offset chuyển về UTC. Timestamp sai, ID trùng,
rating thiếu/sai làm dừng chương trình, không âm thầm bỏ dòng.

Giao thức triển khai tạm thời là **chấm tại thời điểm review xuất hiện**:

- Chỉ xuất review có thời điểm `t <= cutoff`.
- Lịch sử của review tại `t` chỉ chứa dòng có timestamp **nhỏ hơn t**.
- Các review cùng timestamp không nhìn thấy nhau, kể cả cùng ngày trong nguồn chỉ có ngày.
- Rating của review hiện tại được dùng vì có sẵn lúc chấm; nhãn tuyệt đối không được dùng.
- Mốc snapshot là giới hạn xuất dữ liệu, không thay thế thời điểm quyết định của từng dòng.
- Mọi bước EDA/chọn đặc trưng chỉ xem Train; tham số chọn trên Validation, không xem Test.

Cách này chống nhìn vào tương lai ngay cả khi TV5 gộp nhiều thời điểm vào một
lô. Nếu nhóm chọn chấm cuối tuần/tháng, cần thống nhất lại ngữ nghĩa trước
khi tích hợp. Chưa có dữ liệu về thời điểm review được ingest; hiện giả định
timestamp đăng bài cũng là thời điểm quan sát được.

## 3. Đặc trưng bàn giao cho B1

Khóa ghép: `review_id`; giữ `user_id`, `product_id`, `decision_time`,
`snapshot_cutoff` cho truy vết. **Không đưa các cột khóa/mốc vào mô hình**.

| Cột feature | Định nghĩa |
|---|---|
| `rating` | Điểm của review hiện tại |
| `user_prior_count` | Tổng review trước t của user |
| `user_prior_products` | Số sản phẩm khác nhau đã review trước t |
| `user_prior_7d_count` | Số review user trong [t−7 ngày, t) |
| `user_observed_age_days` | (t − review đầu của user trước t), đơn vị ngày |
| `user_gap_days` | (t − review gần nhất của user trước t), đơn vị ngày |
| `user_prior_rating_mean/std` | Trung bình và độ lệch chuẩn tổng thể của rating quá khứ |
| `product_prior_count` | Tổng review sản phẩm trước t |
| `product_prior_7d_count` | Tổng review sản phẩm trong [t−7 ngày, t) |
| `product_prior_rating_mean` | Trung bình rating sản phẩm trước t |
| `product_observed_age_days` | (t − review đầu của sản phẩm trước t), đơn vị ngày |
| `rating_deviation_prior_product` | abs(rating hiện tại − trung bình sản phẩm trước t) / 4 |
| `user_history_missing`, `product_history_missing` | 1 nếu chưa có lịch sử tương ứng, ngược lại 0 |

Lịch sử rỗng: feature phụ thuộc lịch sử nhận 0 và cờ missing nhận 1.
Lịch sử chỉ một review: độ lệch chuẩn bằng 0. Tuổi quan sát không phải tuổi
tài khoản hay ngày ra mắt sản phẩm. Deviation B1 so với toàn bộ lịch sử
sản phẩm (có thể gồm review cũ của chính user), không gọi đây là IRD nguyên bản.

## 4. EDA và đầu ra

| File đầu ra | Nội dung |
|---|---|
| `normalized_reviews.csv` | Metadata chuẩn hóa UTC trong snapshot |
| `b1_temporal_features.csv` | Một dòng feature cho mỗi review được quan sát |
| `daily_counts.csv`, `weekly_counts.csv` | Phân phối ngày/tuần toàn bộ dữ liệu trong snapshot |
| `product_daily_counts.csv` | Phân phối theo sản phẩm, thống kê nền và cờ tăng mạnh |
| `summary.json` | Mốc cắt, số dòng, số dòng tương lai bị loại, giả định và giới hạn |
| `eda.html` | Biểu đồ ngày/tuần và bảng ngày tăng mạnh, mở trực tiếp trong trình duyệt |

Tuần bắt đầu thứ Hai UTC. Điền 0 cho ngày trống từ ngày đầu quan sát đến
cutoff; với sản phẩm, bắt đầu từ lần đầu sản phẩm xuất hiện. Tuần đầu/cuối
và ngày cuối có thể chưa trọn kỳ.

Cờ EDA thăm dò: sau ít nhất 7 ngày quan sát, đánh dấu ngày có số review
ít nhất 3 và lớn hơn `mean(7 ngày trước) + 3 × std(7 ngày trước)`.
Ngày đang xét không nằm trong nền. Đây là quy tắc demo để tìm ngày cần xem,
chưa phải bộ phát hiện burst chính thức của giai đoạn 2, chưa phải feature B1,
và không chứng minh cấu kết.

## 5. Đặc tả GTW, GD, GETF cho giai đoạn 2

Nguồn đối chiếu: Mukherjee, Liu & Glance (2012), *Spotting Fake Reviewer Groups
in Consumer Reviews*, mục 5.1, phương trình (1), (2), (5):
[bài gốc, trang 193–194](https://archives.iw3c2.org/www2012/proceedings/proceedings/p191.pdf).

Ký hiệu tại mốc T: g là nhóm, P_g là tập sản phẩm của nhóm;
F(g,p), L(g,p) là thời điểm sớm/muộn nhất nhóm review p; A(p) là ngày
sản phẩm cho phép review, thay bằng review đầu tiên nếu thiếu.

| Chỉ số gốc | Công thức |
|---|---|
| GTW | `max_p max(0, 1 − (L(g,p) − F(g,p))/τ)` |
| GD | `max_p abs(mean_rating(g,p) − mean_rating(outside_g,p))/4` |
| GETF | `max_p max(0, 1 − (L(g,p) − A(p))/β)` |

τ, β và các hiệu thời gian dùng ngày. GD gốc đặt trung bình ngoài nhóm bằng
0 nếu không có người ngoài nhóm; với thang 1–5, trường hợp này có thể vượt 1.

**Quy ước triển khai dự kiến của dự án (phải ghi rõ là biến thể):**

- Chỉ đọc review có timestamp ≤ T. Nếu dùng giao thức chấm từng review như
  B1 ở trên, lịch sử trước t vẫn áp dụng; module nhóm phải có kiểm thử riêng.
- TV2 bàn giao `group_id`, tập `user_id` và tập sản phẩm hỗ trợ nhóm.
  Dùng tập sản phẩm hỗ trợ này làm P_g, không tự gom mọi sản phẩm riêng lẻ
  của từng thành viên. Chỉ xét sản phẩm có ít nhất 2 thành viên nhóm tham gia.
- GD: sản phẩm không có review ngoài nhóm được đánh dấu thiếu và loại khỏi
  phép max; nếu không còn sản phẩm hợp lệ, xuất 0 cùng `gd_missing=1`.
  Không dùng quy tắc trung bình ngoài nhóm bằng 0 của bài gốc.
- Nếu nhiều review của cùng user trên cùng sản phẩm: đề xuất lấy bản mới
  nhất đã quan sát cho chỉ số nhóm; trùng timestamp thì tie-break bằng
  `review_id`. Đây là quy ước dự án cần TV1/TV2 thống nhất, không dùng cho B1.
- Thiếu ngày mở review: dùng lần đầu quan sát sản phẩm **trong dữ liệu đến T**,
  gắn `availability_is_proxy=1`. Dữ liệu bị cắt lịch sử có thể làm GETF cao giả.
- τ và β phải > 0; cấu hình không hợp lệ thì báo lỗi. Chưa chốt giá trị số
  từ demo; lựa chọn bằng Validation, không chỉnh trên Test.
- P_g rỗng hoặc review không có nhóm: xuất 0 và cờ `no_group_evidence=1`;
  không diễn giải 0 thành bằng chứng review hợp lệ.
- Nếu A(p) > L(g,p), coi metadata lỗi và báo lỗi, không ép kết quả vào [0,1].

Giai đoạn 1 chỉ chốt đặc tả; chưa có nhóm ứng viên để tính ba chỉ số này.

## 6. Kiểm chứng và giới hạn

Lần kiểm chứng ngày 21/11/2026 với Node.js 24.12.0: **7/7 kiểm thử đạt**.
Demo có 11 dòng đầu vào, giữ 10 dòng đến mốc 15/01/2024 UTC, loại 1 dòng
tương lai; 7 user, 2 sản phẩm. Ngày 10/01/2024 của p1 có 5 review và được
đánh dấu tăng mạnh so với nền 7 ngày trước bằng 0. Đây là mẫu dựng để kiểm
tra, không phải phát hiện trên dữ liệu thật.

Chạy `node --test` kiểm tra: chuẩn hóa offset, ngày sai/múi giờ thiếu;
schema và duplicate ID; kết quả feature tính tay; cold start; review cùng
timestamp; biên cửa sổ 7 ngày/cutoff; thêm review tương lai hoặc thay label
không đổi kết quả trước đó; EDA điền ngày trống và phát hiện spike cài trong mẫu.

Module feature sắp xếp O(N log N), sau đó quét tăng dần với trạng thái theo
user/sản phẩm; bộ nhớ O(N). Bảng EDA sản phẩm-ngày có kích thước tổng số
ngày quan sát của từng sản phẩm: có thể lớn trên dữ liệu nhiều năm, cần
đo và tối ưu ở giai đoạn scalability. JSONL hiện được đọc toàn bộ vào RAM.

## 7. Việc cần phối hợp khi có dữ liệu thật

1. Thanh/TV1 cung cấp raw metadata, đặc biệt `rating`, ý nghĩa timestamp,
   schema và mapping nhãn. Không tải dataset dạng ma trận nếu thiếu raw metadata.
2. Chi/TV5 chốt Train/Val/Test và thời điểm ra quyết định; chạy EDA với train cutoff.
3. Hiền chạy module, kiểm tra timestamp lỗi, phân phối ngày/tuần và cold start;
   thay phần kết quả demo bằng số liệu thực, ghi nguồn và mốc cắt.
4. Chi ghép feature theo `review_id` với đặc trưng nội dung của Lanh;
   mọi scaler/imputer/model chỉ fit trên Train.
5. Cúc/TV2 thống nhất schema nhóm để triển khai GTW/GD/GETF ở giai đoạn 2.
