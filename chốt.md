Dưới đây là bản chia việc hoàn chỉnh cho **5 thành viên (TV1 – TV5)** qua **4 giai đoạn** (không gắn thời gian/deadline, không gán tên riêng, tập trung hoàn toàn vào kỹ thuật **Data Mining** và đã vá toàn bộ các lỗ hổng về chống rò rỉ dữ liệu, baseline và cách đánh giá từ bản phản biện).

---

### Định hướng & Phân vai xuyên suốt (5 Thành viên)
thanh, cúc, lanh, hiền, chi

*   **TV1 — Dữ liệu, Snapshot & Tích hợp Pipeline:** Phụ trách kiểm tra raw data, schema chung, quy tắc cắt thời gian (snapshot) chống rò rỉ, repo và ghép nối pipeline end-to-end.
*   **TV2 — Khai phá Tập phổ biến & Quan hệ đồng đánh giá:** Phụ trách định nghĩa giao dịch, thuật toán FP-Growth, đặc trưng đồng đánh giá theo cặp/nhóm, phân tích độ phủ (Coverage) và trực quan hóa cụm đáng ngờ.
*   **TV3 — Khai phá Văn bản & Độ tương đồng:** Phụ trách tiền xử lý văn bản, TF-IDF/n-gram, tìm cặp gần trùng bằng MinHash/LSH (kèm kiểm định độ sót), chỉ số tương đồng nội dung nhóm (GCS) và phân tích lỗi ngôn ngữ.
*   **TV4 — Khai phá Thời gian, Hành vi nhóm & Ablation/Scalability:** Phụ trách chuẩn hóa thời gian, đặc trưng đột biến (burst), công thức hóa các chỉ số hành vi nhóm (GTW, GD, GETF), thực nghiệm Ablation Study và đo Runtime/Peak RAM.
*   **TV5 — Giao thức Benchmark, Baseline & Giả lập (Synthetic):** Phụ trách chia tập Train/Val/Test, chốt thước đo, chạy thang so sánh công bằng B0–B3, xây dựng bộ giả lập cấu kết khó + đối chứng hợp lệ, tổng hợp báo cáo cuối.

*(Nguyên tắc chung: Mỗi thành viên tự kiểm chứng module và tự viết phần phương pháp + giới hạn của mảng mình phụ trách, không dồn hết việc kiểm thử và viết báo cáo cho một người).*

---

### GIAI ĐOẠN 1: Khảo sát dữ liệu, Chốt quy ước chống rò rỉ & Dựng Baseline nền
**Mục tiêu hoàn thành giai đoạn:** Xác nhận đủ raw metadata; khóa schema và quy tắc snapshot; chạy xong mốc kiểm tra pipeline (**B0**) và baseline cá nhân mạnh (**B1**) trên dataset đầu tiên (vd: YelpChi).

*   **TV1:**
    *   Tải và kiểm tra raw metadata (đủ text, `user_id`, `product_id`, `review_id`, `timestamp`, nhãn *filtered/recommended*).
    *   Dựng repo, môi trường và chốt định dạng bảng trung gian (`Parquet/CSV` với khóa chung + bảng nhóm `group_id`).
    *   Tạo **dữ liệu mẫu giả (mock schema)** để TV2–TV4 code song song ngay lập tức; thiết kế cơ chế cắt **snapshot theo mốc thời gian** (ví dụ: chấm điểm theo lô cuối mỗi tuần/tháng).
*   **TV2:**
    *   Thống kê phân phối bậc User – Product và tỷ lệ user ít hoạt động (`< 3` reviews).
    *   Chốt quy tắc không loại user `< 3` reviews khỏi tập đánh giá chính (chỉ dùng ngưỡng tối thiểu để *tham gia khai phá nhóm*).
    *   Chốt định nghĩa giao dịch cho FP-Growth (theo `sản phẩm` hay `sản phẩm–cửa sổ thời gian`) và cách xử lý các sản phẩm quá hot dễ gây bùng nổ cặp.
*   **TV3:**
    *   EDA văn bản (độ dài, tần suất từ, tỷ lệ trùng thô); lưu **song song 2 bản** (text gốc và text làm sạch, cẩn thận giữ lại dấu phủ định khi cần).
    *   Xây dựng bộ trích xuất **TF-IDF / n-gram** (chỉ fit trên tập train theo đúng giao thức) để cấp đặc trưng nội dung cá nhân cho B1.
*   **TV4:**
    *   Chuẩn hóa `timestamp`; EDA phân phối lượt review theo ngày/tuần để quan sát các đợt tăng đột biến trên sản phẩm.
    *   Trích xuất đặc trưng hành vi/thời gian cá nhân (tính đến mốc quan sát) cấp cho B1.
    *   Chốt công thức toán học, đơn vị, mốc tham chiếu và quy tắc xử lý mẫu số bằng 0 cho các chỉ số **GTW, GD, GETF** (đối chiếu chuẩn Mukherjee et al., 2012).
*   **TV5:**
    *   Chốt quy trình chia Train/Val/Test theo thời gian, định nghĩa rõ thời điểm ra quyết định và các thước đo (AUC, AP, F1, Precision/Recall@K).
    *   Code và chạy **B0** (đặc trưng thô cơ bản để check pipeline) và **B1** (baseline mạnh gồm nội dung + hành vi cá nhân + thống kê sản phẩm/thời gian) trên cùng một mô hình bảng (Logistic Regression hoặc LightGBM).

---

### GIAI ĐOẠN 2: Khai phá mẫu nhóm, Tính chỉ số cấu kết & Ghép Pipeline End-to-End
**Mục tiêu hoàn thành giai đoạn:** Cả 4 trụ cột xuất ra bảng đặc trưng tuân thủ snapshot; ghép thông suốt thành pipeline tự động chạy tới **B2** và **B3**; hoàn thiện bộ sinh nhóm giả lập.

*   **TV1:**
    *   Viết module **loader kiểm tra schema & kiểm tra rò rỉ tự động** (đảm bảo không feature nào đọc dữ liệu tương lai sau mốc cắt).
    *   Tích hợp đầu ra từ TV2, TV3, TV4 vào **pipeline chạy một lệnh** (từ raw data $\rightarrow$ B2/B3).
*   **TV2:**
    *   Tính đặc trưng **đồng đánh giá theo cặp** (số sản phẩm chung, Jaccard đồng đánh giá trong snapshot) phục vụ mức **B2**.
    *   Chạy **FP-Growth** (đặt sẵn giới hạn `max_len`, `min_support` tuyệt đối và giới hạn RAM/thời gian) để xuất bảng `group_id` ứng viên phục vụ mức **B3**; tính *Group Size, Group Support* và quy tắc gộp về cấp review (`max / mean / count`).
*   **TV3:**
    *   Triển khai **MinHash + LSH** để tìm nhanh các cặp/nhóm review gần trùng trong tập dữ liệu đã quan sát được, sau đó **tính lại độ tương đồng chính xác** (Jaccard / TF-IDF Cosine) trên các ứng viên vừa lọc.
    *   Tính trọn bộ đặc trưng văn bản cấp nhóm (**Group Content Similarity - GCS**, max similarity trong nhóm) và gộp về từng review.
*   **TV4:**
    *   Cài đặt phát hiện đợt tăng đột biến theo cửa sổ trượt (chỉ tính trên cửa sổ quá khứ đã quan sát).
    *   Code hoàn chỉnh các chỉ số hành vi/thời gian cấp nhóm (**GTW, GD, GETF**) trên tập ứng viên nhóm của TV2, xử lý sạch trường hợp review không thuộc nhóm nào (gán cờ *"không có bằng chứng nhóm"*).
*   **TV5:**
    *   Xây dựng bộ **Synthetic Collusion Injection** (bơm vào bản sao dữ liệu nền) với các kịch bản ngụy trang khó: tài khoản cũ có lịch sử thật, tham gia một phần sản phẩm, thời gian rải rác, đánh giá dìm sao, và tạo thêm **đợt tăng review hợp lệ** (khuyến mãi/cùng sở thích) làm đối chứng.
    *   Chạy thử nghiệm đầu tiên cho **B2** (B1 + đồng đánh giá theo cặp) và **B3** (B2 + nhóm FP-Growth) trên tập Validation.

---

### GIAI ĐOẠN 3: Thí nghiệm kiểm chứng, Ablation, Scalability & Tinh chỉnh trên Validation
**Mục tiêu hoàn thành giai đoạn:** Hoàn tất mọi thử nghiệm chọn tham số và đánh giá đánh đổi trên **Validation** (tuyệt đối giữ nguyên **Test cuối** chưa đụng tới).

*   **TV1:**
    *   Chuẩn hóa file config, seed tái lập và hỗ trợ tạo các lát cắt dữ liệu tăng dần kích thước (từ YelpChi lên YelpNYC ~359k reviews) để TV4 đo Scalability.
    *   Rà soát lại toàn bộ giao thức cập nhật snapshot trên từng lát cắt thời gian.
*   **TV2:**
    *   Đo **Coverage** (tỷ lệ % review/user được gắn vào ít nhất một nhóm ứng viên FP-Growth).
    *   So sánh kết quả của B2 và B3 tách riêng trên **nhóm ít lịch sử vs nhóm nhiều lịch sử**; khảo sát đánh đổi giữa ngưỡng `min_support`, độ phủ ứng viên và tài nguyên chạy.
*   **TV3:**
    *   Thực hiện **Kiểm định LSH**: so sánh tập cặp gần trùng tìm bằng LSH với tính chính xác toàn phần (exact all-pairs) trên một tập mẫu nhỏ để đo chính xác tỷ lệ bỏ sót (*Recall*), *False Positive* và thời gian tiết kiệm được.
    *   Tinh chỉnh lựa chọn n-gram / biểu diễn văn bản tối ưu trên Validation.
*   **TV4:**
    *   Chủ trì **Ablation Study** trên Validation theo cả hai chiều: (1) *Chỉ giữ từng trụ cột đơn lẻ* và (2) *Bỏ từng trụ cột khỏi mô hình B3 đầy đủ* (bỏ FP-Growth, bỏ GCS văn bản, bỏ GTW/GD thời gian).
    *   Đo lường **Scalability thực nghiệm**: vẽ biểu đồ Runtime và Peak RAM tách riêng từng bước (đọc data, FP-Growth, LSH, tính feature, huấn luyện) theo quy mô dữ liệu tăng dần.
*   **TV5:**
    *   Đánh giá **Synthetic Injection** trên Validation: chốt ngưỡng khớp nhóm **Jaccard $J(G, \hat{G}) \ge \tau$** với quy tắc ghép 1–1 chống đếm trùng; báo cáo tách biệt **Candidate Recall** (ứng viên có chứa nhóm bơm không), **Top-K Recovery** và **Precision/Recall thành viên**.
    *   *(Tùy chọn mở rộng nếu tiến độ dư dả)*: Thử nghiệm mức **B4** (lan truyền điểm nghi ngờ trên đồ thị hoặc Isolation Forest/LOF ở chế độ `novelty=True`) xem có vượt được B3 không.

---

### GIAI ĐOẠN 4: Khóa cấu hình, Chạy Test cuối, Phân tích lỗi & Báo cáo
**Mục tiêu hoàn thành giai đoạn:** Chạy chốt hạ 1 lần duy nhất trên tập Test cuối; hoàn thiện phân tích lỗi, trực quan hóa, giới hạn kết luận và bộ tài liệu tái lập.

*   **TV1:**
    *   Đóng băng toàn bộ siêu tham số và cấu hình pipeline; chạy **1 lần duy nhất trên tập Test cuối** và xuất bảng kết quả chính thức.
    *   Đóng gói mã nguồn sạch sẽ, viết tài liệu hướng dẫn tái lập thực nghiệm (`README.md`, cấu trúc thư mục, lệnh chạy).
*   **TV2:**
    *   Trực quan hóa 2–3 mạng lưới nhóm đáng ngờ điển hình (dùng *NetworkX / PyVis / Gephi*) thể hiện rõ sản phẩm chung, thời gian và tương đồng nội dung.
    *   Viết phần báo cáo **Khai phá nhóm & FP-Growth** (lưu ý chỉ gọi các cụm tìm được ở dữ liệu thật là *"nhóm đáng ngờ / cần xem xét"*, nêu rõ khả năng trùng với khách cùng sở thích).
*   **TV3:**
    *   Thực hiện **Error Analysis phần nội dung** trên kết quả Test cuối (review bị lọc nhưng không copy từ ngữ, câu quá ngắn, hoặc review hợp lệ dùng mẫu câu giống nhau).
    *   Viết phần báo cáo **Khai phá văn bản & LSH** (kèm bảng đánh đổi Recall/Runtime).
*   **TV4:**
    *   Hoàn thiện biểu đồ **Ablation Study** và đường cong **Scalability (Runtime/Peak RAM)** trên cấu hình chốt cuối (bao gồm cả việc báo cáo các cấu hình từng thất bại vì quá RAM/thời gian nếu có).
    *   Viết phần báo cáo **Hành vi thời gian & Khả năng mở rộng** (ghi rõ công thức GTW, GD, GETF và các biến thể nếu có chỉnh sửa).
*   **TV5:**
    *   Chạy chấm điểm cuối cùng trên bộ **Synthetic Test** (dùng seed/kịch bản tách biệt hoàn toàn với phần đã dùng để chọn ngưỡng ở Giai đoạn 3).
    *   Tổng hợp bảng kết quả hai tầng (Review-level B0 $\rightarrow$ B1 $\rightarrow$ B2 $\rightarrow$ B3 và Phục hồi nhóm bơm), viết phần **Giao thức đánh giá & Giới hạn kết luận** (nhấn mạnh nhãn Yelp là nhãn bộ lọc gần đúng), biên tập slide và kịch bản demo.

---

### Bảng tóm tắt ma trận công việc

| Giai đoạn | TV1 (Dữ liệu & Pipeline) | TV2 (FP-Growth & Đồ thị) | TV3 (Văn bản & LSH) | TV4 (Thời gian & Ablation) | TV5 (Benchmark & Synthetic) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GĐ 1: Nền tảng** | Check raw data, schema, snapshot, mock data | EDA bậc, định nghĩa giao dịch (giữ `<3` rev) | EDA text, lưu 2 bản, TF-IDF cho B1 | EDA spike, feature B1, chốt công thức GTW/GD | Chia split thời gian, Metric, chạy B0 & B1 |
| **GĐ 2: Khai phá & Ghép** | Check schema, chống rò rỉ, ghép pipeline | Feature cặp (B2), FP-Growth (B3) + giới hạn | MinHash/LSH + rescore chính xác, GCS | Burst quá khứ, GTW/GD/GETF, xử lý nhóm rỗng | Bộ sinh Synthetic khó + đối chứng, chạy thử B2/B3 |
| **GĐ 3: Kiểm chứng (Val)** | Đóng gói config, lát cắt tăng quy mô (YelpNYC) | Đo Coverage, check nhóm ít/nhiều lịch sử | Audit LSH Recall vs Exact trên tập mẫu | Ablation (đơn lẻ & rút bớt), đo Runtime/RAM | Đánh giá Synthetic (Jaccard 1-1, Candidate vs Top-K) |
| **GĐ 4: Chốt & Báo cáo** | Chạy Test 1 lần, đóng gói repo & README | Trực quan hóa mạng lưới, viết mục Nhóm | Error Analysis ngôn ngữ, viết mục Văn bản | Vẽ biểu đồ Ablation & Scalability, viết mục Thời gian | Synthetic Test cuối, tổng hợp 2 tầng, Slide/Demo |