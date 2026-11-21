# Detecting-Review-Manipulation-on-E-Commerce-Platforms

Phần **Hiền / TV4 — Giai đoạn 1**: chuẩn hóa thời gian, EDA ngày/tuần,
đặc trưng hành vi cá nhân cho B1 và đặc tả GTW/GD/GETF.

Module dùng **Node.js >= 18**, không cần cài thư viện. Đầu ra CSV có thể nạp
vào pandas/LightGBM của pipeline chung. Máy phát triển hiện có Node.js,
chưa có Python trong PATH.

```powershell
node --test
node src/temporal.mjs --input data/mock/reviews.jsonl --cutoff 2024-01-15T00:00:00Z --out outputs/demo --mock
```

Mở `outputs/demo/eda.html` để xem biểu đồ. **Dữ liệu demo là giả**, không phải
kết quả thực nghiệm Yelp. Quy trình và phần báo cáo bàn giao ở
[docs/hien_giai_doan_1.md](docs/hien_giai_doan_1.md).

Khi có dữ liệu thật đã chuyển sang schema JSONL:

```powershell
node src/temporal.mjs --input data/raw/reviews.jsonl --cutoff 2024-01-15T00:00:00Z --out outputs/train
```

Thay mốc ví dụ bằng **mốc kết thúc Train do TV5 chốt** trước khi chạy EDA.
Không dùng mốc này như quyết định chia tập chính thức. Dữ liệu raw và outputs
được gitignore. Module không huấn luyện B1; TV5 ghép bảng feature với nội dung
của TV3 bằng `review_id` và thực hiện benchmark.

## Hiền / TV4 — Giai đoạn 2

Đã có module burst cửa sổ quá khứ, GTW/GD/GETF, gộp nhóm về review và cờ
không có bằng chứng nhóm. Schema nhóm có `mined_through` để tránh dùng nhóm
tương lai chấm review quá khứ. Hướng dẫn và giới hạn:
[docs/hien_giai_doan_2.md](docs/hien_giai_doan_2.md).

```powershell
node src/group-temporal.mjs --input data/mock/stage2_reviews.jsonl --groups data/mock/stage2_groups.jsonl --config configs/temporal-stage2.json --cutoff 2024-01-15 --out outputs/stage2-demo --mock
```

Kết quả nằm ở `outputs/stage2-demo/`: bảng feature review, nhóm, nhóm–sản phẩm
và `summary.json`. Cấu hình demo chưa được chọn trên Validation; dữ liệu và
nhóm ứng viên đều giả lập. Chưa bao gồm FP-Growth, huấn luyện B3 hay tối ưu
scalability; các phần đó thuộc nhiệm vụ khác hoặc giai đoạn sau.
