# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Mọi con số trong báo cáo lấy từ file do script sinh ra: `reports/metrics_baseline.json`, `reports/metrics_poc.json`,
> `reports/table1_baseline_vs_poc.md`, `reports/analysis_t02.{json,md}` (`tools/analyze_t02.py`), `reports/calibration_t02.json`
> (`tools/calibrate_t02.py`), `out/*/run_info.json`.
## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G-2A202603187 |
| Challenge | T02 · Raw QC Triage: ảnh mờ, lóa và thiếu sáng |
| Thành viên (5–6) + vai trò | 1. Do Hoang Anh (làm cá nhân, đảm nhiệm toàn bộ vai trò: Lead/Integrator, Data, Core algorithm, QA/metric, Review + report) |
| Git URL | https://github.com/dohoanganh2004/K5-L2-DAY06-DoHoangAnh-2A202603187-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | |
| Tag `final-24h` (commit) | `d1d67bb` |
| Deadline (giờ VN) | 12:00 ngày 15/11/2026 |

## 01 · Problem statement

- Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai? Đội huấn luyện/đánh giá mô hình nhận dữ liệu (ảnh mờ, lóa, tối lọt qua làm mô hình học nhiễu) và đội thu thập dữ liệu (ảnh tốt bị báo nhầm là lỗi bắt phải chụp lại; ảnh ban đêm hữu ích bị loại).
- Điểm nghẽn cụ thể: người review chỉ có ngân sách xem **B = 20 ảnh** trên 96 ảnh; cần xếp hạng sao cho 20 ảnh đó chứa nhiều ảnh lỗi thật nhất.
- Đầu vào → đầu ra: `fixtures/t02_raw_qc/` (`images.csv` + ảnh JPEG 320×180) → `out/ranked_review.csv` (`image_id, score`, cao = nghi lỗi hơn), kèm `review_queue.csv` top-20, `contact_sheet.html`, `audit_log.jsonl`, `decision.json`.
- Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval): lô là `KEEP` (=GO) nếu không còn ảnh nào bị đánh dấu `RECAPTURE` sau review; ngược lại `RECAPTURE` (=REWORK), `reason_code = DEFECTS_FLAGGED`. Số ảnh bị gắn cờ nằm ngoài top-20 (chưa có người xem) được ghi rõ trong lý do.
- Giả định của nhóm (cần DPO/legal/domain owner xác nhận): dữ liệu synthetic, không PII thật; định nghĩa "lỗi" theo camera (ảnh cam_B mềm đồng đều là bình thường) cần domain owner xác nhận; chưa có đánh giá pháp lý.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 ảnh (14 cảnh gốc), 28 ảnh lỗi | `fixtures/t02_raw_qc/` (giảng viên cấp) | CC0 (theo README) | Có |
| eval / heldout | 40 ảnh (10 cảnh gốc), 20 ảnh lỗi | `fixtures/t02_raw_qc/` | CC0 (theo README) | Có |

- Schema & class mapping: `images.csv` = `image_id, path, camera_id, light, split`. Nhãn gold: `none / blur / dark / glare / compression` → `is_defect = 0/1`.
- Seed: 42 · Cách chia split: theo **cảnh gốc** (4 ảnh của một cảnh luôn cùng split).
- `manifest.csv` verified (sha256): có — `python -m tools.verify_manifests`; `tools.preflight` ngày 2026-11-14: 0 FAIL. Lưu ý Windows: cần `git config --local core.autocrlf false`, nếu không CRLF làm lệch hash.
- Dữ liệu thật dùng (nếu có) và cách xây GT cho nó: không dùng.
- **Gold set tự gán** (repo học viên không có `gt/`): `reports/my_gold_labels.csv`, 96 ảnh, 48 ảnh lỗi (compression 18, blur 12, dark 10, glare 8). Gán bằng `tools/label_tool.html` (không hiển thị điểm/đáp án), một người gán, có hai lượt: lượt đầu gán nhầm 6 ảnh `_0` của cảnh cam_B (mềm/tối do camera) là lỗi; sau khi phát hiện, nhãn được sửa theo quy tắc "lỗi = tệ hơn mức bình thường của chính camera đó".

## 03 · Research & design

- Tài liệu tham khảo (xem mục 11) và **vì sao** chọn kỹ thuật:
  - ACDC ghép mỗi ảnh điều kiện xấu với ảnh **cùng cảnh** ở điều kiện bình thường → gợi ý dùng *ảnh cùng cảnh làm tham chiếu* thay vì ngưỡng tuyệt đối. Đây là ý tưởng cốt lõi của PoC.
  - CVAT QA dùng tập nhỏ có nhãn chuẩn (Ground Truth / honeypot) để ước lượng chất lượng → tương ứng gold set tự gán và việc báo kèm khoảng tin cậy khi N nhỏ.
  - ISO/IEC TR 5259-6 (khung trực quan hoá chất lượng dữ liệu) → định hướng contact sheet/slice thay vì chỉ một con số.
- Thuật toán PoC (`src/qc_features.py`, gọi từ `src/pipeline.py::process`): với mỗi ảnh đo 4 bằng chứng **so với ảnh cùng cảnh** (tham chiếu = trung vị pixel của các ảnh cùng cảnh, các ảnh đã căn chỉnh):
  - blur: `log(Laplacian variance / trung vị cảnh)` thấp;
  - dark: `log(độ sáng / trung vị cảnh)` thấp;
  - glare: tỉ lệ pixel > 245 tăng so với cảnh, hoặc độ sáng toàn cục tăng;
  - compression: tỉ số độ khối JPEG 8×8 so với cảnh cao.
  Mỗi bằng chứng chia cho ngưỡng (1.0 = chạm ngưỡng); `score` = bằng chứng mạnh nhất; `score ≥ 1.0` → `RECAPTURE`. Chỉ lấy hướng làm hỏng (ví dụ ảnh nét hơn bình thường **không** phải lỗi, vì ảnh `_1` của cảnh cam_B nét hơn nhưng bình thường).
- Sơ đồ data flow: `images.csv` + ảnh → load (kiểm manifest + file) → process (4 bằng chứng → `score`, `pred_action`) → rank (top-20 → `review_queue.csv` + `contact_sheet.html`) → review (`review_queue_reviewed.csv`, override, `audit_log.jsonl`) → decide (`decision.json`) → `ranked_review.csv`.
- Thành phần offline: không dùng model pretrained; chỉ numpy/scipy/Pillow, không gọi mạng.
- Trade-offs đã cân nhắc: (1) dùng cấu trúc "4 ảnh cùng cảnh" cho độ chính xác cao nhưng là **giả định về dữ liệu** (không có ở ảnh thật rời rạc) → có đường lui yếu hơn (fallback tuyệt đối, trọng số 0.5) và được đo bằng ablation ở mục 06; (2) ngưỡng đơn giản, dễ giải thích thay vì mô hình học máy, vì chỉ có 56 ảnh dev.

## 04 · Baseline vs PoC

- Baseline (`src/baseline_laplacian.py`): một ngưỡng Laplacian toàn ảnh duy nhất, `lap < 745 → RECAPTURE`, `score = −lap`; 745 chọn trên dev (F1 dev 0.842, `tools/calibrate_t02.py`). Commit: `4b2239a` (code), tham số trong `5b7db6f`.
- PoC (`src/qc_features.py` + `src/pipeline.py`, tham số trong `configs/lab.yaml`): như mục 03. Commit: `5b7db6f` (`qc_features.py`, ngưỡng) và `4b2239a` (`pipeline.py`); trạng thái code lúc đo: `129a46a`.
- Cùng budget (B=20) / cùng pool eval (40 ảnh) / cùng seed 42: **xác nhận có**. Baseline dùng chung toàn bộ hạ tầng load/rank/review/decide, chỉ khác `process()`.
- Mốc tham chiếu: random baseline của repo cho Recall@20 = 0.50 (10/20), Wilson 95% CI (0.299, 0.701).

**Bảng 1 · Baseline vs PoC & hard slices** (từ `reports/table1_baseline_vs_poc.md` và `reports/analysis_t02.md`; gold tự gán, split eval N=40):

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Defect Recall@20 (lỗi thật trong top-20 / lỗi thật) | 0.85 (17/20); CI (0.64, 0.948) | 1.00 (20/20); CI (0.839, 1.0) | +0.15; higher is better. Khoảng tin cậy chồng nhau nhiều → N=40 chưa đủ để khẳng định chắc chắn |
| Precision@20 | 0.85 | 1.00 | +0.15 (B cố định nên bằng Recall khi có đúng 20 ảnh lỗi) |
| Hard-slice: false alarm trong top-20, `light=night` | 0.25 (2/8) | 0.00 (0/8) | −0.25; N=8 ảnh sạch |
| Hard-slice: false alarm, `camera_id=cam_B` | 0.50 (3/6) | 0.00 (0/6) | −0.50; N=6 ảnh sạch |
| Review@B / miss / FP | miss 3, FP 3 | miss 0, FP 0 | B=20 |
| Defect-type accuracy (trên 20 ảnh lỗi) | 0.20 | 1.00 | baseline chỉ có nhãn "blur" nên chỉ mang tính tham khảo |
| Runtime / 100 items | 0.4376 s | 0.6986 s | CPU Intel64 Family 6 Model 141, Windows 11, Python 3.13.13 |

Ảnh PoC bắt thêm so với baseline: `img_01_3`, `img_13_2`, `img_05_3`; baseline không bắt được ảnh nào mà PoC bỏ sót.

## 05 · Evaluation protocol

- Công thức primary metric, đơn vị đánh giá, TP/FP/FN, denominator: Recall@20 = (ảnh lỗi thật trong top-20) / (tổng ảnh lỗi thật eval = 20); đơn vị = ảnh; xem `reports/metric_contract.md`.
- Threshold / budget B / K và lúc chốt: B = 20. Mọi ngưỡng chọn bằng cách quan sát **dev** (`python -m tools.calibrate_t02`, chỉ đọc nhãn dev): cả 4 detector có khoảng trống giữa ảnh lỗi yếu nhất và ảnh sạch mạnh nhất (blur 1.52 vs 0.06; dark 1.52 vs 0.02; glare 1.06 vs 0.08; compression 1.32 vs 0.33 — ngưỡng = 1.0 nằm trong khoảng trống), dev: TP 28, FP 0, FN 0, TN 28. Giá trị đóng băng trong `configs/lab.yaml` → `params`. Commit chứa ngưỡng: `5b7db6f` (2026-11-15 10:42 +0700); commit này được tạo **sau** khi đã có số đo, nhưng ngưỡng không đổi từ lúc chốt trên dev đến lúc commit (xem phần công khai quy trình bên dưới).
- **Công khai về quy trình:** trước khi có gold set của bạn, một bản prototype đã được chạy và đo trên eval bằng nhãn do trợ lý AI gán; các ngưỡng blur/dark/glare/compression được chọn từ dev và **không đổi sau lần chạy đó**. Riêng hai hằng số của fallback không-ảnh-cùng-cảnh (`noref_lap`, `noref_mean`) được đặt lại theo trung vị ảnh sạch của dev; chúng chỉ ảnh hưởng ablation, không ảnh hưởng số chính. Ảnh eval đã được nhìn bằng mắt khi gán nhãn nhưng không dùng để chọn ngưỡng. Điểm chính thức do giảng viên chấm trên `gt/` và hidden eval, có thể thấp hơn số tự đo.
- Độ bất định: N = 40 (20 ảnh lỗi); báo Wilson 95% CI (xem Bảng 1). PoC 20/20 có CI dưới 0.839, nên không thể kết luận "hoàn hảo" từ N này.
- Lệnh evaluate chính xác (gold tự gán):
  `python -m src.evaluate --track t02 --gt reports/my_gold_labels.csv --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json`
  (tương tự `out/baseline/...` → `metrics_baseline.json`). Các `metrics_*.json` hiện ghi git sha `129a46a` (trạng thái code đã commit, chạy lại sau khi commit).

## 06 · Result slices

| Slice (eval) | N | Baseline (recall / false alarm) | PoC (recall / false alarm) | Ghi chú |
|---|---|---|---|---|
| light=day | 20 | 0.90 / 0.00 | 1.00 / 0.00 | |
| light=dusk | 4 | 1.00 / 0.50 | 1.00 / 0.00 | N rất nhỏ |
| light=night | 16 | 0.75 / 0.25 | 1.00 / 0.00 | |
| camera_id=cam_A | 28 | 0.786 / 0.00 | 1.00 / 0.00 | tính thêm bằng script |
| camera_id=cam_B | 12 | 1.00 / 0.50 | 1.00 / 0.00 | tính thêm bằng script |
| defect=compression | 8 | 0.75 / — | 1.00 / — | |
| defect=glare | 4 | 0.75 / — | 1.00 / — | |

**Ablation (ngưỡng giữ nguyên, không tune lại; `reports/analysis_t02.md`):**

| Biến thể PoC | Recall@20 (CI) | False alarm trong top-20 |
|---|---|---|
| Đầy đủ (có ảnh cùng cảnh) | 1.00 (0.839–1.0) | 0 |
| **Không dùng ảnh cùng cảnh** (chỉ fallback tuyệt đối) | **0.65** (0.433–0.819) | **7** (night 0.875, cam_B 0.667) |
| Bỏ detector blur | 0.80 | 4 |
| Bỏ detector glare | 0.90 | 2 |
| Bỏ detector compression | 0.85 | 3 |
| Bỏ detector dark | 1.00 | 0 |

Ảnh dark vẫn được bắt khi bỏ detector dark vì bóng tối làm giảm Laplacian nên detector blur kích hoạt; detector dark do đó dư thừa trên eval nhưng giữ lại để gọi đúng loại lỗi. **Toàn bộ lợi thế của PoC đến từ ảnh cùng cảnh**: bỏ nó thì PoC (0.65) thua cả baseline Laplacian (0.85).

- **Failure case 1 — camera cam_B mềm hơn (failure mode 3, ảnh `demo/failure_cases_baseline.png`):** baseline có 3 false alarm trong top-20, cả 3 đều là ảnh sạch cam_B: `img_06_0` (night, Laplacian 276), `img_15_0` (dusk, 313), `img_03_0` (night, 354). Root cause: median Laplacian của ảnh sạch là 1113.2 (cam_A, n=32) so với 618.4 (cam_B, n=16), nên một ngưỡng chung không tách được; cam_B mềm là đặc tính camera. PoC so với ảnh cùng cảnh (cùng camera) nên 0/6 false alarm cam_B.
- **Failure case 2 — lỗi mà Laplacian không thấy (failure mode 2, cùng ảnh):** baseline bỏ sót `img_01_3` và `img_13_2` (night, compression, Laplacian 1094 và 1207) và `img_05_3` (day, glare **cục bộ** ở vùng xe/biển số, Laplacian 1324). Root cause: các ô vuông nén tạo thêm cạnh làm Laplacian *cao*, và vùng chói cục bộ không làm giảm độ nét toàn ảnh. PoC bắt cả 3 nhờ detector compression và glare.
- **Failure case 3 — ảnh ban đêm (failure mode 1), khi mất ảnh cùng cảnh:** PoC không-ảnh-cùng-cảnh báo nhầm 7/8 ảnh sạch ban đêm (`img_06_0`, `img_06_1`, `img_01_0`, `img_01_1`, `img_03_0`, `img_03_1`, `img_13_1`) và bỏ sót 7/20 ảnh lỗi: glare ban đêm cam_B (`img_06_3`, `img_03_2`), glare ban ngày (`img_05_3`) và 4 ảnh compression ban ngày (`img_16_3`, `img_23_2`, `img_05_2`, `img_19_2`). Root cause: thiếu tham chiếu, ngưỡng tuyệt đối coi ảnh tối/mềm theo bản chất là lỗi, đúng bẫy ảnh đêm của đề.
- Latency / compute (máy, CPU, ms/item): PoC 0.6707 s / 96 ảnh = 7.0 ms/ảnh (0.699 s/100 ảnh); baseline 0.438 s/100 ảnh (thời gian dao động giữa các lần chạy do máy); CPU Intel64 Family 6 Model 141 Stepping 1, Windows 11 (`out/*/run_info.json`).

## 07 · Human & decision

- Review queue (file, B, cách ưu tiên): `out/poc/review_queue.csv`, B = 20, top-20 theo `score` giảm dần (hoà điểm giữ thứ tự gốc); `out/poc/contact_sheet.html` hiển thị ảnh + score + lý do + hành động đề xuất + ô chọn quyết định + nút Export CSV.
- Cách override + audit log (`out/poc/audit_log.jsonl`): reviewer nhập tên và quyết định (`approve/reject/edit/escalate/skip`); `edit`/`reject` kèm `override_value` (`KEEP`|`RECAPTURE`) thay `pred_action`, giá trị máy gốc giữ ở `pred_action_machine`; mỗi thay đổi ghi một dòng JSONL (ai, khi nào, trước/sau, lý do). Thiếu `reviewer`, `override_value` sai → chạy thất bại rõ ràng (có test); `image_id` lạ có kiểm tra trong code nhưng chưa có test.
- Decision policy + `reason_code` + owner: `KEEP`/`NO_DEFECT_FLAGGED` hoặc `RECAPTURE`/`DEFECTS_FLAGGED`; owner = `Do Hoang Anh (individual owner / reviewer)`. Sau review (`out/poc/decision.json`): RECAPTURE (=REWORK), 48/96 ảnh bị gắn cờ, `n_reviewed = 20`, trong đó 28 ảnh gắn cờ nằm ngoài top-20 nên vẫn là quyết định chỉ do máy. `decision_log.csv` (append-only) có 2 dòng: lần chạy đầu chỉ do máy ("no human review file was applied") và lần chạy sau review.
- Khi nào HOLD/REWORK thay vì tự động release: còn bất kỳ ảnh nào bị đánh dấu RECAPTURE sau review; ảnh gắn cờ ngoài top-B chưa có người xem nên quyết định đó được ghi là "chỉ do máy".
- **Bằng chứng review (2026-11-15):** reviewer Đỗ Hoàng Anh đã duyệt 20/20 ảnh trong `contact_sheet.html` và xuất `review_queue_reviewed.csv`; kết quả: **20 `approve`, 0 `reject/edit`, 0 override**, không có ghi chú. Nghĩa là reviewer đồng ý với đề xuất của máy ở cả 20 ảnh, nên `pred_action` không bị đổi (đối chiếu gold: cả 20 ảnh đều là ảnh lỗi thật, khớp Precision@20 = 1.00). Bằng chứng nằm ở `reports/review_queue.csv` (có cột reviewer/decision), `reports/audit_log.jsonl` (20 dòng `review:approve`) và `reports/decision_log.csv`. Lưu ý: cột `reviewed_at` trong CSV là thời điểm bấm Export (giống nhau cho cả 20 dòng), không phải thời điểm xem từng ảnh. Vì không có ca nào bị đảo, luồng override (`reject/edit` kèm `override_value`) mới được kiểm bằng test tự động, chưa được kiểm bằng một quyết định thật của reviewer.

## 08 · Reproduce & tests

```bash
# Windows (PowerShell):
py -3.13 -m venv .venv ; .\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.main --input fixtures/t02_raw_qc --config configs/lab.yaml --output out/ --offline --seed 42
.\.venv\Scripts\python -m src.evaluate --track t02 --gt reports/my_gold_labels.csv --pred out/ranked_review.csv --out reports/metrics_poc.json
.\.venv\Scripts\python -m pytest -q
# số liệu trong báo cáo:
python -m tools.calibrate_t02 ; python -m tools.analyze_t02
```

- Output của `pytest -q` (2026-11-14): `65 passed, 39 skipped` (39 test bị skip cần `gt/` của giảng viên). Có 2 nhóm test của nhóm: `tests/test_t02_pipeline.py` (load/contract/review/override/decision, gồm các case lỗi) và `tests/test_t02_algorithm.py` (detector trên ảnh dựng sẵn, fallback không có ảnh cùng cảnh, tính tất định, hồi quy trên dev).
- Máy / CPU / OS / Python: Intel64 Family 6 Model 141, Windows 11, Python 3.13.13 · git sha: `129a46a` (trạng thái code lúc chạy `src.main`/`src.evaluate` lần cuối; các commit tiếp theo chỉ đổi báo cáo).

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: dữ liệu synthetic CC0, không người/biển số thật; không dùng `data/real/`.
- Điều chưa biết / false alarms / nếu scale 10×: (1) lợi thế phụ thuộc vào cấu trúc "4 ảnh căn chỉnh cùng cảnh" (`img_<cảnh>_<biến thể>`); với ảnh thật không có ảnh cùng cảnh, ablation cho thấy Recall rớt còn 0.65 và false alarm ban đêm 0.875, nên cần tham chiếu khác (ví dụ chuẩn hoá theo camera/giờ chụp); (2) nếu hidden eval đặt tên hoặc nhóm ảnh khác, PoC lùi về fallback; (3) một người gán nhãn, không đo được độ đồng thuận; lượt gán đầu đã sai 6 ảnh cam_B nên nhãn "mờ nhẹ" có thể mơ hồ; (4) N eval nhỏ (40) và toàn bộ là dữ liệu synthetic cùng một bộ sinh, nên chưa chứng minh được trên ảnh thật; (5) các ngưỡng khớp hoàn hảo với dev (28/28) nên cần thận trọng với overfit.
- Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động; giả định cần DPO/legal xác nhận. Bài này chỉ là tài liệu học tập, không phải tư vấn pháp lý.
- Kế hoạch cải tiến: kiểm 10 ảnh out-of-domain (ví dụ `data/real/` nếu giảng viên cấp, không đưa lên cloud); thay tham chiếu cùng cảnh bằng chuẩn hoá theo `camera_id`/`light`; người gán thứ hai để đo độ đồng thuận; thêm test cho `image_id` lạ trong file review.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Do Hoang Anh | dohoanganh2004 / dohoanganh92004@gmail.com | Toàn bộ (làm cá nhân) | `a738bcf`, `5b7db6f`, `4b2239a`, `129a46a` (và các commit sau đó) | Có |

Ghi chú minh bạch: phần lớn code (pipeline, thuật toán, test, script phân tích, bản nháp báo cáo) do trợ lý AI (Claude) viết theo yêu cầu của học viên; công cụ gán nhãn `tools/label_tool.html` cũng do trợ lý viết, còn **nhãn gold do học viên tự gán**. Học viên chịu trách nhiệm kiểm tra và giải thích lại trước khi nộp.

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| https://arxiv.org/abs/2104.13395 (ACDC: Adverse Conditions Dataset) | 2026-11-14 (qua tra cứu, chưa đọc toàn văn) | Bộ dữ liệu lái xe điều kiện xấu, mỗi ảnh xấu có ảnh cùng cảnh ở điều kiện bình thường làm cặp tham chiếu → ý tưởng dùng ảnh cùng cảnh làm tham chiếu. |
| https://docs.cvat.ai/docs/qa-analytics/auto-qa/ (CVAT Automated QA) | 2026-11-14 (qua tra cứu, chưa đọc toàn văn) | QA tự động so các job với một tập nhỏ Ground Truth (honeypot) để ước lượng chất lượng → tương ứng gold set nhỏ tự gán và việc báo độ bất định. |
| https://webstore.iec.ch/en/publication/113479 (ISO/IEC TR 5259-6) | 2026-11-14 (chỉ xem mô tả; toàn văn trả phí, chưa đọc) | Khung trực quan hoá kết quả đo chất lượng dữ liệu cho phân tích/ML → định hướng trình bày theo slice và contact sheet. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [ ] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics, ≥1 failure case (có), reviewer evidence (có: `reports/review_queue.csv`, `audit_log.jsonl`), decision log (có).
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N (có).
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên (chưa commit).
