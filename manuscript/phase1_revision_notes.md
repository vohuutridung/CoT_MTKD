# Ghi chú cho bản viết lại mục 3.2 và 3.3

Bản tiếng Anh trong `phase1_sections_3_2_3_3.tex` dùng để thay trực tiếp hai mục tương ứng của draft. File không có preamble và dùng các macro, package đã có trong bản bạn gửi. `phase1_method_preview.tex` là bản độc lập để đọc và chỉnh trong trình soạn LaTeX; bố cục hai cột của bản xem không phải template ACL chính thức.

Bản chỉnh văn phong tập trung vào động cơ và định nghĩa phương pháp. Đã lược bỏ đoạn thực thi trên long trajectories, chi tiết số học, trạng thái optimizer, RNG và các giá trị hyperparameter cụ thể khỏi phần Method. Các ghi chú triển khai bên dưới chỉ dùng để đối chiếu; chúng không thuộc nội dung hai mục thay thế.

Đã đối chiếu với nhánh `output-space`, trên nền commit `f59e73b2890867f3c80323f4d4ddf0d507b9a324` và thay đổi local Kneedle ngày 04/10/2026. Các giá trị trong ghi chú này là cấu hình triển khai hiện tại, không phải bằng chứng về chất lượng hay hiệu năng của một lần huấn luyện.

## Những điểm được sửa trong hai mục

| Thành phần | Mô tả đúng với Phase 1 hiện tại | Nguồn trong code |
|---|---|---|
| Support | Loại token quan sát trước khi chọn; áp dụng local Kneedle trên top-512 non-target logits của từng expert, chuẩn hóa cả rank và logits trong cùng window; hợp các tập ứng viên. `search_k=512`, `k_min=8`, không có upper clipping; khi có đủ candidate mỗi expert giữ từ 8 đến 512 token, union của ba expert có thể chứa đến 1.536 token. Với `K<8`, giữ toàn bộ `K` candidate. | [kneedle.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/stage1/kneedle.py:6), [chunked_head.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/models/chunked_head.py:52) |
| SFT | Loss trên toàn bộ assistant response, gồm answer và token cấu trúc; dùng LoRA dropout 0.05. | [prepare.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/data/prepare.py:71), [config](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/configs/stage1/qwen25_7b_m3.yaml:13) |
| DPP | Trung bình Gram trong một bước rồi tính log-determinant; trung bình tiếp theo bước và theo ví dụ có reasoning. Chỉ dùng reasoning content. | [dpp.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/stage1/dpp.py:102) |
| RBF | Khoảng cách giữa các cập nhật hiệu dụng `(alpha/r) BA`, chuẩn hóa theo số phần tử từng module. | [rbf.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/stage1/rbf.py:20) |
| GAC | Trộn gradient bằng kernel chuẩn hóa theo expert nhận; giới hạn lực đẩy; ramp 10–30%; clip toàn bộ hướng cập nhật rồi AdamW. | [gac_gradient.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/stage1/gac_gradient.py:20), [trainer.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/stage1/trainer.py:1030) |
| Thực thi | One-pass giữ các graph có checkpointing; hai VJP riêng trong ramp, một VJP kết hợp ở full phase; two-pass có tái tạo RNG để giữ đúng LoRA dropout. | [trainer.py](/Users/vohuutridung/Desktop/FML/Code/CoT_MTKD/src/cot_mtkd/stage1/trainer.py:357) |

Tên “Kneedle-style” phân biệt quy tắc elbow có giới hạn trong code với thuật toán Kneedle đầy đủ, vốn còn có xử lý cực đại cục bộ và ngưỡng phát hiện. Xem [bài gốc của Satopää và cộng sự](https://www.cs.williams.edu/~jeannie/papers/kneedle-simplex11.pdf). Trích dẫn [DPP](https://arxiv.org/abs/1207.6083) và [SVGD](https://arxiv.org/abs/1608.04471) được giữ để chỉ nguồn ý tưởng; bản viết không khẳng định GAC thực hiện suy luận posterior theo SVGD.

Không giữ các số liệu `k=4.2`, `92%` mass hoặc `8%` target ngoài region từ draft vì chưa có artifact đo lường được xác minh trong lượt này. Việc loại target khỏi DPP không bảo đảm xác suất target không thay đổi sau cập nhật tham số. Tính bất biến khi đổi cách phân tích LoRA được phát biểu cho khoảng cách giữa các `BA`, không cho toàn bộ AdamW/GAC trong tọa độ factor.

## Các phần cần đồng bộ ở lượt sửa tiếp theo

- Mục 3.1 còn mô tả mask dead-end, formatting và answer sai. Các mask này không mô tả đường SFT hiện tại.
- Mục 3.4 còn dùng `eq:d2` của khoảng cách subspace cũ. Bản thay thế dùng label mới `eq:stage1-distance`; vì vậy tham chiếu cũ cần được xử lý khi sửa Phase 2. Mục 3.4 cũng đang dùng `eq:kneedle` như region chung có thể chứa target, trong khi công thức 3.2 mới chỉ định nghĩa support non-target của Phase 1.
- Abstract, Introduction và Related Work còn mô tả step dropout, tokenwise DPP và Grassmann repulsion.
- Thuật toán Stage 1 và bảng hyperparameter trong phụ lục còn theo phiên bản cũ. Cấu hình hiện tại dùng LR `5e-5`, DPP weight `0.2`, RBF weight `1.0`, LoRA dropout `0.05`, interaction ramp `10–30%`, jitter ban đầu `1e-4`, `search_k=512`, `k_min=8`; window là upper bound duy nhất.

Các phần trên nằm ngoài phạm vi thay thế 3.2–3.3 và chưa được viết lại. Bản xem độc lập được biên dịch và kiểm tra bố cục; đây không phải xác nhận rằng toàn bộ draft ARR đã được biên dịch hoặc đồng bộ nội dung.
