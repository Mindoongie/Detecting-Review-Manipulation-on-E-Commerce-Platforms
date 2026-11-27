# TV2 · GĐ1 — Quyết định về user ít hoạt động, định nghĩa giao dịch và sản phẩm hot

- Code: `src/rmd/tv2/degree_eda.py` và `src/rmd/tv2/transactions.py`
- Tham số: `configs/default.yaml`

Cách chạy (`df` là bảng review có các cột `review_id`, `user_id`, `product_id`, `timestamp`; cột `is_filtered` là tùy chọn):

```python
import yaml
from rmd.tv2.degree_eda import run_degree_eda
from rmd.tv2.transactions import TxConfig, build_transactions

cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
eda = run_degree_eda(df, cfg["eda"]["low_activity_threshold"])  # bậc, user ít hoạt động, sản phẩm hot
tx = build_transactions(df, TxConfig.from_dict(cfg["mining"]))  # giao dịch cho FP-Growth
eda["user_summary"], eda["low_activity"], eda["by_user_degree"], eda["top_products"], tx.stats
```

> Tham số chỉ được chỉnh dựa trên **kích thước dữ liệu và tài nguyên chạy**, không dựa trên nhãn. Các số liệu theo nhãn chỉ dùng để mô tả dữ liệu.

---

## 1. Thống kê phân phối bậc User – Product và tỷ lệ user ít hoạt động

Hàm `run_degree_eda(df, low_threshold=3)` trả về các kết quả sau.

| Khóa | Nội dung |
| :--- | :--- |
| `user_summary`, `product_summary` | Các chỉ số mean, median, p75, p90, p99, max. `share_deg1` là tỷ lệ đỉnh có bậc 1. `top1pct_review_share` là tỷ lệ review thuộc 1% đỉnh có bậc cao nhất |
| `low_activity` | Số user và tỷ lệ user có ít hơn 3 review, số review và tỷ lệ review của họ. Nếu có nhãn thì thêm tỷ lệ bị lọc của nhóm này so với nhóm còn lại, và tỷ lệ review bị lọc thuộc về họ |
| `by_user_degree` | Số user, số review và tỷ lệ bị lọc theo các nhóm bậc 1, 2, 3–4, 5–9, 10–19, 20–49, ≥50 |
| `top_products` | 10 sản phẩm nhiều review nhất. Cột `pairs_if_one_tx` là số cặp user sinh ra nếu coi cả sản phẩm là một giao dịch |
| `user_deg`, `prod_deg` | Bậc của từng user và từng sản phẩm. Hàm `ccdf()` tính phân phối bù từ hai chuỗi này |

## 2. Không loại user có ít hơn 3 review khỏi tập đánh giá chính

**Quy tắc:**

- Mọi review đều được chấm điểm và đều nằm trong tập đánh giá, bất kể user có bao nhiêu review.
- Ngưỡng hoạt động **chỉ** dùng để quyết định ai được *tham gia khai phá nhóm*. Một user được làm item của FP-Growth khi xuất hiện trong **≥ `min_support` giao dịch có từ 2 user trở lên**.
- Điều kiện trên được áp dụng lặp lại cho tới khi ổn định: bỏ user không đạt, rồi bỏ giao dịch chỉ còn dưới 2 user, rồi kiểm tra lại.
- Review của user không đủ điều kiện vẫn được chấm. Đặc trưng nhóm của review đó bằng 0 và gắn cờ "không có bằng chứng nhóm".

**Lý do:**

1. Trên dữ liệu Yelp, user ít hoạt động thường chiếm phần lớn số user và là nơi tập trung review bị lọc. Nếu loại họ khỏi tập đánh giá, ta sẽ bỏ đúng phần khó nhất và làm điểm số đẹp ảo. Chỉ số `low_activity` dùng để định lượng điều này trên YelpChi.
2. Ngưỡng khai phá không làm mất thông tin. Một nhóm có support ≥ `s` thì mọi thành viên phải xuất hiện trong ≥ `s` giao dịch nhiều-user. Vì vậy loại các user không đạt điều kiện này không làm mất tập phổ biến nào có ≥ 2 user, và cũng không làm thay đổi support của các tập còn lại.
3. Với `min_support = 3`, quy tắc này trùng với ngưỡng "≥ 3 review" trong kế hoạch, nhưng chính xác hơn vì tính theo giao dịch chứ không theo số review.

## 3. Giao dịch = (sản phẩm × cửa sổ 28 ngày)

**Định nghĩa:**

- Một giao dịch là một cặp `(product_id, cửa sổ thời gian cố định 28 ngày)`.
- Item là các `user_id` đã review sản phẩm đó trong cửa sổ đó. Một user review nhiều lần trong cùng giao dịch chỉ tính 1 lần.
- Cửa sổ được tính từ một mốc gốc `origin` cố định. Mặc định `origin` là ngày nhỏ nhất của dữ liệu. **Khi chạy thật nên ghi cứng `origin` vào config** để các cửa sổ không thay đổi giữa các lần chạy và giữa các snapshot.
- Phương án đối chứng: `transaction_mode: product`, tức giao dịch là toàn bộ lịch sử của sản phẩm. Phương án này được giữ lại để so sánh và cho Ablation ở GĐ3.

**Lý do chọn cửa sổ thời gian:**

1. **Bám đúng giả thuyết cấu kết.** Spam theo nhóm thường đăng dồn trong một đợt. Ngưỡng 28 ngày khớp với ngưỡng cửa sổ thời gian nhóm GTW trong Mukherjee et al. (2012), nên định nghĩa giao dịch của TV2 nhất quán với chỉ số của TV4.
2. **Giảm ghép nhầm.** Hai "khách quen" review cùng một nhà hàng cách nhau vài năm sẽ không bị tính là cùng giao dịch.
3. **Giảm bùng nổ tổ hợp.** Mỗi sản phẩm bị chia thành nhiều giao dịch nhỏ, nên tổng số cặp `Σ n(n−1)/2` giảm mạnh. Chỉ số `pairs_raw` và `pairs_after_hot` dùng để so sánh hai phương án.

**Đánh đổi đã biết:**

- **Coverage thấp hơn.** Review rải rác theo thời gian khó gom thành nhóm hơn. Chỉ số `coverage_ub_reviews` là cận trên của Coverage và cần được so sánh giữa hai phương án.
- **Cửa sổ cố định có thể cắt đôi một đợt tấn công** nằm vắt qua ranh giới hai cửa sổ, làm support của nhóm bị chia đôi. Cách kiểm tra:
  - So sánh với cửa sổ 14 ngày và với phương án `product`.
  - Nếu tỷ lệ mất mát là đáng kể, cân nhắc chuyển sang cửa sổ trượt chồng lấn (stride bằng nửa cửa sổ) và khử trùng support.
- **Tương tác với snapshot.** `build_transactions` nhận dữ liệu **đã được cắt theo snapshot** và không tự cắt. Tại mỗi snapshot, cửa sổ chứa mốc cắt chỉ gồm phần dữ liệu đã quan sát được. Đây là hành vi đúng, không gây rò rỉ.

## 4. Xử lý sản phẩm quá hot: chia nhỏ cửa sổ, sau đó mới bỏ

**Quy tắc:** trần kích thước giao dịch là `max_tx_size = 200` user. Giao dịch vượt trần được xử lý như sau:

1. `hot_strategy: subwindow` (mặc định): **chia đôi cửa sổ** của giao dịch đó, lặp lại cho đến khi ≤ trần. Với `transaction_mode: product`, lần chia đầu tiên tách theo `window_days`.
2. Nếu cửa sổ đã nhỏ hơn `min_subwindow_days = 1` ngày mà giao dịch vẫn vượt trần, giao dịch đó mới bị **bỏ khỏi khai phá**. Số review bị bỏ được ghi trong chỉ số `reviews_dropped_hot`.
3. Review bị bỏ khỏi khai phá vẫn được chấm điểm bình thường, chỉ là không có bằng chứng nhóm.

Phương án đối chứng `hot_strategy: drop` bỏ ngay giao dịch vượt trần.

**Lý do:**

- Một sản phẩm có `n` reviewer sinh ra `n(n−1)/2` cặp. Chỉ một sản phẩm có 1.000 reviewer đã sinh khoảng 500 nghìn cặp.
- Chia nhỏ theo thời gian giữ lại được tín hiệu "cùng đăng trong vài ngày", vốn chính là tín hiệu cấu kết. Bỏ hẳn sản phẩm hot sẽ để lọt các nhóm nhắm vào sản phẩm nổi tiếng.

**Cách hiệu chỉnh trần trên YelpChi:** chạy `build_transactions` với nhiều mức trần (∞, 500, 200, 100) và chọn trần nhỏ nhất thỏa cả hai điều kiện:

- `reviews_dropped_hot` gần bằng 0.
- `pairs_after_hot` đủ nhỏ để chạy trong giới hạn RAM và thời gian.

## Tham số chuyển sang GĐ2

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| `transaction_mode` | `product_window` | Đối chứng: `product` |
| `window_days` | 28 | Thử thêm 14 |
| `min_support` | 3 | Tuyệt đối (số giao dịch). `mlxtend` nhận dạng tỷ lệ, nên phải truyền `min_support / số giao dịch` |
| `max_tx_size` | 200 | Hiệu chỉnh lại trên YelpChi |
| `hot_strategy` | `subwindow` | Đối chứng: `drop` |

Đầu ra của `build_transactions()` dùng trực tiếp cho GĐ2:

- `memberships` (`tx_id`, `user_id`): đầu vào của FP-Growth, lấy qua `.itemsets()`.
- `transactions` (`tx_id`, `product_id`, `window_start`, `window_end`, `size`): dùng để tra ra sản phẩm và thời gian của mỗi nhóm.
- `review_tx` (`review_id`, `tx_id`): dùng để gộp đặc trưng nhóm về cấp review.

## Số liệu thật (điền khi có YelpChi)

- [ ] Tỷ lệ user có ít hơn 3 review, và tỷ lệ review cùng tỷ lệ review bị lọc của nhóm này: …
- [ ] So sánh kích thước giao dịch theo `product` và theo `product × 28 ngày` (p99, max, `pairs_raw`): …
- [ ] Trần `max_tx_size` đã chọn và `reviews_dropped_hot` tương ứng: …
- [ ] Cận trên Coverage theo cấu hình chốt: …
