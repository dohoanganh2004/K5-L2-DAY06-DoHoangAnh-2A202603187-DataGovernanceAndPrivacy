# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T02 · Raw QC Triage (ảnh mờ, lóa, thiếu sáng) |
| Primary metric (tên đúng như evaluator in ra) | `Defect Recall@B` (kèm `Precision@B`) |
| Tăng tốt hay giảm tốt? | higher (càng cao càng tốt) |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | image (một ảnh) |
| Công thức + numerator / denominator | Recall@B = (số ảnh lỗi thật nằm trong top-B theo `score`) / (tổng số ảnh lỗi thật trong split eval). Precision@B = (số ảnh lỗi thật trong top-B) / B. Hoà điểm: giữ thứ tự gốc của file (sort ổn định). |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Ghép theo `image_id` (khớp chính xác). Chỉ các ảnh thuộc split eval được tính. |
| Candidate pool cố định | 40 ảnh split `eval` (10 cảnh gốc × 4 ảnh) của `fixtures/t02_raw_qc/`. Không thêm/bớt ảnh sau khi chốt. |
| Review budget B / K | B = 20 |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Mọi ngưỡng/trọng số chỉ được chọn bằng cách quan sát split **dev** (56 ảnh, 28 ảnh lỗi theo gold set). Giá trị chốt nằm trong `configs/lab.yaml` → `params` (kiểm bằng `python -m tools.calibrate_t02`, chỉ đọc nhãn dev): `blur_lap_rel=-2.5`, `dark_bright_rel=-1.0`, `glare_sat_rel=0.005`, `glare_bright_rel=0.2`, `comp_block_rel=1.35`, `min_group=3`; baseline `lap_threshold=745.0` (F1 dev tốt nhất). Không chỉnh lại sau khi thấy số trên eval. **Công khai:** trước khi có gold set của học viên đã có một lần chạy prototype đo trên eval bằng nhãn do trợ lý AI gán; các ngưỡng trên không đổi sau lần đó (riêng `noref_lap`/`noref_mean` đặt lại theo trung vị ảnh sạch dev, chỉ dùng cho ablation). Commit chứa ngưỡng: `5b7db6f` (2026-11-15 10:42 +0700), được tạo sau khi đã có số đo; giá trị ngưỡng không đổi giữa lúc chốt trên dev và lúc commit. |
| Eval split (không dùng để tune) | `eval` (40 ảnh, 20 ảnh lỗi theo gold set). Baseline và PoC dùng cùng pool, cùng B, cùng seed. |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | Miss = ảnh lỗi thật nằm ngoài top-B (làm giảm Recall). False alarm = ảnh sạch nằm trong top-B (báo theo slice `light`, đã có trong evaluator). |
| Slices sẽ báo cáo (≥2) | `light` (day / dusk / night), loại lỗi (`defect_type`: blur / dark / glare / compression) và `camera_id` (cam_A / cam_B, tính thêm bằng script riêng) |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | N = 40 (20 ảnh lỗi) nên báo kèm khoảng tin cậy Wilson 95% cho Recall@20 và Precision@20, và nêu rõ khác biệt dưới ~1–2 ảnh không có ý nghĩa thống kê. |
| Ngày giờ chốt + người chốt | 2026-11-14 · Do Hoang Anh (chốt ngưỡng trên dev); commit ghi lại: 2026-11-15 10:42 +0700 (`5b7db6f`) |

## Gold set (nhãn tự gán, vì repo học viên không có `gt/`)

- File: `reports/my_gold_labels.csv` (96 ảnh: `image_id, defect_type, is_defect, split, light, camera_id`); công cụ gán: `tools/label_tool.html` (không hiển thị điểm/đáp án).
- Nhãn: `none / blur / dark / glare / compression` (đúng tập nhãn của đề); `is_defect = 0` nếu `none`, ngược lại 1.
- Số lượng: 96 ảnh, 48 ảnh lỗi (compression 18, blur 12, dark 10, glare 8). Dev: 28/56 lỗi. Eval: 20/40 lỗi.
- **Quy tắc gán nhãn:** một ảnh là "lỗi" khi nó tệ hơn mức bình thường của chính camera đó. Ảnh `cam_B` mềm đồng đều (quang học mềm hơn, đặc biệt ban đêm) là bình thường (`none`), không phải lỗi.
- **Giới hạn trung thực:** một người gán, không có người gán thứ hai để đo độ đồng thuận. Ảnh eval đã được nhìn bằng mắt khi gán nhãn, nhưng **không** dùng để chọn ngưỡng/trọng số. Điểm chính thức do giảng viên chấm trên `gt/` và hidden eval, có thể khác số tự đo ở đây.

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.
