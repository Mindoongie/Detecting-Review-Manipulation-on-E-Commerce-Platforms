# Hiền / TV4 — Giai đoạn 2

## Phần đã triển khai

- Burst theo cửa sổ trượt quá khứ, xuất feature cho từng review.
- GTW, GD, GETF trên nhóm ứng viên; có bảng chi tiết theo nhóm–sản phẩm.
- Gộp nhóm về review bằng max/mean/count; giữ review không có nhóm.
- Kiểm tra mốc khai phá nhóm và bằng chứng thành viên trên sản phẩm hỗ trợ.
- CLI, cấu hình, dữ liệu mẫu và kiểm thử. Chưa chạy trên raw dataset thật;
  nhóm mẫu được dựng bằng tay, không phải kết quả FP-Growth của Cúc.

## Chạy thử

Từ thư mục repo, dùng Node.js >= 18, không cần cài package:

```powershell
node --test
node src/group-temporal.mjs --input data/mock/stage2_reviews.jsonl --groups data/mock/stage2_groups.jsonl --config configs/temporal-stage2.json --cutoff 2024-01-15 --out outputs/stage2-demo --mock
```

Khi có dữ liệu thật, thay đường dẫn reviews/groups, mốc cắt và bỏ `--mock`.
Các giá trị trong `configs/temporal-stage2.json` chỉ là cấu hình khởi đầu;
chưa được tối ưu hay chứng minh trên Yelp. Chỉ chọn tham số trên Validation.

## Schema bàn giao cho Thanh và Cúc

Reviews JSONL giữ schema giai đoạn 1: `review_id`, `user_id`, `product_id`,
`timestamp`, `rating`. Timestamp có offset hoặc ngày ISO; ngày-only quy về
00:00 UTC. Rating thang 1–5. Module bỏ qua nhãn và văn bản.

Mỗi dòng nhóm ứng viên JSONL có dạng:

```json
{"group_id":"g1","user_ids":["u1","u2","u3"],"product_ids":["p1"],"mined_through":"2024-01-09T03:00:00Z"}
```

- `group_id`: chuỗi khác rỗng. Khóa phiên bản là `(group_id, mined_through)`.
- `user_ids`: ít nhất hai ID khác nhau.
- `product_ids`: các sản phẩm hỗ trợ itemset; mỗi sản phẩm phải có review
  của **tất cả thành viên** trước hoặc đúng `mined_through`.
- `mined_through`: mốc dữ liệu cuối cùng được phép dùng để khai phá phiên bản
  này, không phải ngày tạo file. Cúc phải chạy FP-Growth riêng trên snapshot đó.
- Danh sách sản phẩm rỗng được phép: phiên bản không có bằng chứng hợp lệ.
  Muốn ngừng dùng nhóm cũ, xuất phiên bản mới với `product_ids: []`.
- Không được sửa nhóm từ snapshot tương lai rồi gắn mốc cũ. Module kiểm tra
  sự tồn tại của bằng chứng nhưng không thể chứng minh nguồn gốc khai phá.

`data/mock/groups.jsonl` là ví dụ nhóm cho dữ liệu mẫu giai đoạn 1;
`stage2_groups.jsonl` và `stage2_reviews.jsonl` là cặp demo riêng cho giai đoạn 2.

## Chống rò rỉ và thời điểm chấm

Giữ cùng giao thức với B1: mỗi review được chấm ở thời điểm t xuất hiện.
Chỉ xuất review có `t <= cutoff`. Lịch sử dùng cho burst và nhóm luôn có
timestamp **< t**, không bao gồm review hiện tại hoặc các review trùng t.
Các review cùng timestamp được chấm hết rồi mới thêm vào lịch sử.

Chỉ dùng phiên bản nhóm có `mined_through < t`; lấy phiên bản mới nhất đã
hợp lệ của mỗi group_id. Nhóm khai phá ở đúng t chỉ được dùng sau t.
Chỉ số nhóm được cập nhật từ lịch sử trước t trên tập thành viên/sản phẩm
đã khai phá; không dùng nhóm tương lai để chấm ngược các review cũ.

Do đó dữ liệu chỉ có ngày sẽ không phát hiện một burst đang diễn ra trong
cùng ngày ngay lúc chấm review ngày đó. Đây là hệ quả của giao thức đã chọn.
Nếu nhóm muốn chấm cuối ngày/tuần, phải đổi đồng bộ B1–B3 và kiểm thử lại.

## Burst theo cửa sổ trượt

Với độ rộng W ngày và K cửa sổ nền:

- Cửa sổ gần nhất: `[t−W, t)`; số review là C.
- K cửa sổ nền: `[t−(j+2)W, t−(j+1)W)` với j từ 0 đến K−1.
- μ và σ là trung bình, độ lệch chuẩn tổng thể của K số đếm nền, gồm cả 0.
- Có đủ lịch sử nếu review đầu của sản phẩm không muộn hơn `t−(K+1)W`.
- `burst_flag = 1` khi đủ lịch sử, `C >= burst_min_count` và
  `C > μ + burst_std_multiplier × σ`.

Mặc định demo: W=1 ngày, K=7, tối thiểu 3 review, hệ số σ=3. Không chia cho
σ nên nền bằng 0 không gây NaN/Infinity. Khi thiếu lịch sử, cờ burst bằng 0
và `burst_history_missing=1`; vẫn giữ số đếm thô và thống kê để truy vết.
`burst_excess=max(0,C−μ)`. Burst là tín hiệu hành vi, không phải nhãn spam.

## Chỉ số nhóm và biến thể dự án

Công thức và nguồn gốc đã đối chiếu trong [giai đoạn 1](hien_giai_doan_1.md).
Triển khai theo các quy ước dự án đã nêu ở đó:

1. Với mỗi user–product, chọn review mới nhất trong lịch sử; nếu trùng giờ,
   chọn `review_id` lớn nhất theo so sánh chuỗi JavaScript. Áp dụng cho cả
   người trong và ngoài nhóm để một user không có trọng số lớn vì review lặp.
2. Trên mỗi sản phẩm hỗ trợ có ít nhất hai thành viên: GTW dùng độ dài khoảng
   giữa review sớm nhất và muộn nhất của các thành viên được chọn.
3. GD so sánh trung bình rating nhóm với trung bình người ngoài nhóm, chia 4.
   Không có người ngoài nhóm: đánh dấu `gd_missing=1`, bỏ sản phẩm đó khỏi
   phép max GD. Nếu tất cả đều thiếu thì GD=0 và cờ missing=1.
4. GETF dùng review **muộn nhất** của nhóm. Mốc sản phẩm là review đầu tiên
   trong toàn bộ lịch sử sản phẩm trước t, kể cả review cũ bị thay khi chọn
   bản mới nhất. Luôn xuất `availability_is_proxy=1`; phiên bản này chưa
   nhận ngày ra mắt sản phẩm thật.
5. Lấy max từng chỉ số qua các sản phẩm hợp lệ. Không có sản phẩm hợp lệ:
   cả ba chỉ số bằng 0, `no_group_evidence=1`.

τ=30 ngày, β=180 ngày chỉ là mặc định demo. Cấu hình yêu cầu số hữu hạn >0;
K và số review tối thiểu phải nguyên. GD missing khác GD bằng 0 có đủ đối chứng.

## Gộp về review và đầu ra

Review được gắn nhóm khi user là thành viên **và sản phẩm review nằm trong
tập hỗ trợ hợp lệ** của nhóm tại t. Không lan đặc trưng nhóm sang mọi review
ở sản phẩm khác của thành viên.

| File | Khóa / nội dung |
|---|---|
| `review_temporal_group_features.csv` | `review_id`, feature burst, max/mean GTW/GD/GETF, count và cờ thiếu |
| `group_temporal_features.csv` | `(decision_time, group_id)`, mốc khai phá, số sản phẩm hợp lệ, ba chỉ số |
| `group_product_features.csv` | `(decision_time, group_id, product_id)`, số người trong/ngoài nhóm và ba chỉ số |
| `summary.json` | Cấu hình thực chạy, cutoff, giao thức, số dòng, số review có nhóm/burst, nhãn mock |

`group_count` đếm số nhóm được gắn; `gd_valid_group_count` đếm nhóm có GD hợp lệ.
Mean GD bỏ nhóm thiếu đối chứng; GTW/GETF lấy mean trên các nhóm được gắn.
Review không có nhóm vẫn xuất đầy đủ: count=0, các chỉ số=0,
`no_group_evidence=1`, `gd_missing=1`. File CSV rỗng dữ liệu vẫn có header.

Thanh/Chi ghép bảng review vào B1 theo `review_id` và kiểm tra
`decision_time`, `snapshot_cutoff` khớp. Không đưa ID hay các mốc thời gian
truy vết vào mô hình. Bảng nhóm chi tiết chỉ phục vụ giải thích/kiểm tra.

## Kiểm chứng và giới hạn

Demo có 7 review: giữ 6, loại 1 review tương lai. Review `e` có burst và nhóm
hợp lệ; trước lúc nhóm được khai phá, các review `b`, `c`, `d` không nhận
feature nhóm. Review `f` ở sản phẩm mới vẫn được giữ với cờ không có nhóm.

Kiểm thử bao phủ công thức tính tay, GETF dùng mốc muộn nhất, thiếu đối chứng,
nhóm rỗng, review lặp, phiên bản nhóm, nhóm chồng lấn, ranh giới cửa sổ,
lịch sử chưa đủ, nhóm khai phá sai mốc, cùng timestamp và thêm dữ liệu tương lai.
Có kiểm thử CLI với file nhóm rỗng. Cả kiểm thử giai đoạn 1 vẫn chạy cùng bộ.

Đây là triển khai ưu tiên kiểm chứng đúng giao thức: tại mỗi timestamp,
quét lịch sử cho từng nhóm–sản phẩm và từng sản phẩm cần chấm. Chi phí xấu
nhất tăng gần bậc hai theo số review, cộng chi phí nhóm; bảng truy vết có
thể tăng theo số timestamp × số nhóm × số sản phẩm hỗ trợ. Chưa tuyên bố
đạt quy mô YelpNYC. Đo Runtime/Peak RAM và tối ưu chỉ mục/cache là phần giai
đoạn 3; không nên chạy benchmark lớn rồi diễn giải thời gian này như kết quả cuối.

Để hoàn tất tích hợp thực tế: cần raw metadata của Thanh, các snapshot nhóm
FP-Growth của Cúc và xác nhận giao thức chấm của Chi. Hiện chưa có kết quả
phân loại B2/B3, chưa có kết luận phát hiện nhóm thao túng trên dữ liệu thật.
