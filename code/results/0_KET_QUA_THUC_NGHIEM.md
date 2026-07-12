# Tổng hợp kết quả thực nghiệm — CPV301 Phát hiện buồn ngủ

> **Trạng thái:** Phần ảnh tĩnh (DDD, yawn_eye) đã chốt. Phần video **TẠM THỜI** — mới chỉ có video của 1 thành viên (Bình), sẽ chạy lại khi có đủ video nhóm.

---

## 1. Cấu hình ngưỡng

| Tham số | Giá trị đang dùng | Trạng thái |
|---|---|---|
| `EAR_THRESHOLD` | **0.21** | ⚠️ **TẠM** — chọn theo bộ B, chờ thêm video xác nhận |
| `EAR_CONSEC_FRAMES` | **30** | ⚠️ **TẠM** — chọn theo bộ B, chờ thêm video xác nhận |
| `MAR_THRESHOLD` | 0.6 | Chốt |
| `MAR_CONSEC_FRAMES` | 15 | Chốt |
| `TILT_THRESHOLD` | 15° | Chốt |
| `TILT_CONSEC_FRAMES` | 20 | Chốt |

> ⚠️ **CẦN LÀM TRƯỚC KHI NỘP:** Hiện `evaluate.py` đang để `EAR_CONSEC_FRAMES = 30` nhưng `drowsiness_detection.py` vẫn là `20`. **Phải đồng bộ hai file** trước khi nộp, nếu không bảng kết quả sẽ không khớp với hệ thống demo.

---

## 2. Kết quả trên dataset ảnh tĩnh (ĐÃ CHỐT)

### 2.1. DDD — Đánh giá toàn hệ thống (FULL: 41.793 ảnh)

**Kết quả chính thức (dùng cho báo cáo):**

| Chỉ số | Giá trị |
|---|---|
| TP / TN / FP / FN | 10.438 / 17.878 / 1.559 / 11.901 |
| Accuracy | 0.6778 |
| Precision | 0.8701 |
| Recall | 0.4673 |
| F1-score | 0.6080 |
| Ảnh không detect được mặt | 17 / 41.793 (0.04%) |
| Thời gian chạy | 634.0s (~10,5 phút, CPU i5 gen 10) |

**So sánh mẫu 500 vs full** (kiểm chứng tính đại diện của mẫu):

| Chỉ số | limit 500 (1.000 ảnh) | Full (41.793 ảnh) | Chênh |
|---|---|---|---|
| Accuracy | 0.6730 | 0.6778 | +0.005 |
| Precision | 0.7952 | **0.8701** | **+0.075** |
| Recall | 0.4660 | 0.4673 | +0.001 |
| F1 | 0.5876 | 0.6080 | +0.020 |
| no_face_rate | 0.00% | 0.04% | +0.04% |

**Nhận xét:**

1. **Recall gần như không đổi khi tăng mẫu gấp 42 lần** (0.4660 → 0.4673, chênh 0.1%). Đây là bằng chứng mạnh cho thấy recall thấp **không phải do mẫu nhỏ hay ngẫu nhiên**, mà là **đặc tính ổn định** của phương pháp geometric trên dữ liệu ảnh tĩnh.

2. **Nguyên nhân recall thấp không phải lỗi thuật toán**, mà là hạn chế của việc đánh giá hệ thống **thời gian** bằng **ảnh tĩnh**: một ảnh gán nhãn "drowsy" không đảm bảo tại đúng khoảnh khắc đó mắt đang nhắm hay đang ngáp — người buồn ngủ vẫn có thể đang mở mắt tại khung hình được chụp. Với ảnh tĩnh không có khái niệm "N frame liên tiếp", nên chỉ so ngưỡng trên một khung hình duy nhất.

3. **`no_face_rate = 0.04%` loại trừ một lời giải thích cạnh tranh:** kết quả thấp KHÔNG phải do MediaPipe không tìm thấy mặt. Gần như toàn bộ 41.793 ảnh đều detect được landmark.

4. **Precision tăng 0.075 giữa mẫu 500 và full** — đây là khác biệt đáng kể duy nhất. Tỷ lệ FP ở mẫu 500 là 60/500 = 12%, ở full chỉ 1.559/19.445 = 8%. Cho thấy **500 ảnh đầu tiên không đại diện ngẫu nhiên** cho toàn dataset (có thể cùng subject / cùng điều kiện chụp). **Bài học:** kết quả `--limit` chỉ nên dùng để test tốc độ, không dùng để kết luận.

### 2.2. yawn_eye — Đánh giá riêng chỉ số MAR

Khảo sát 4 ngưỡng MAR (1446 ảnh, no_face = 2 ảnh / 0.14%):

| MAR_THRESHOLD | TP | TN | FP | FN | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 0.60 (đang dùng) | 473 | 724 | **0** | 249 | 1.0000 | 0.6551 | 0.7916 | 0.8278 |
| 0.55 | 522 | 724 | **0** | 200 | 1.0000 | 0.7230 | 0.8392 | 0.8617 |
| 0.50 | 574 | 724 | **0** | 148 | 1.0000 | 0.7950 | 0.8858 | 0.8976 |
| 0.45 | 622 | 724 | **0** | 100 | 1.0000 | 0.8615 | 0.9256 | 0.9308 |

**Nhận xét:** FP = 0 ở **cả 4 ngưỡng** — không có ảnh `no_yawn` nào có MAR ≥ 0.45. Trên ảnh tĩnh, hạ ngưỡng cho recall tăng "miễn phí" (+149 TP, 0 FP).

**Nhưng KHÔNG hạ ngưỡng.** Lý do ở mục 3.2.

---

## 3. Kết quả trên video (TẠM THỜI — N = 1 người)

**Dữ liệu:** `Binh_alert.mp4` (1351 frame) + `Binh_drowsy.mp4` (1421 frame), 16.6 fps.
Ground truth: 5 `eye_closed`, 3 `yawn`, 3 `head_tilt`.
`no_face_rate`: 0.0% (alert) / 1.48% (drowsy).

### 3.1. Khảo sát ngưỡng EAR — event-level (`eye_closed`)

| Bộ | EAR / CONSEC | hit | miss | false_alarm | Recall | Precision |
|---|---|---|---|---|---|---|
| Baseline | 0.21 / 20 | 5/5 | 0 | 10 | 1.0000 | 0.3333 |
| A | 0.18 / 20 | 5/5 | 0 | 7 | 1.0000 | 0.4167 |
| **B (chọn)** | **0.21 / 30** | **5/5** | 0 | **6** | 1.0000 | **0.4545** |
| C | 0.19 / 25 | 5/5 | 0 | 6 | 1.0000 | 0.4545 |

### 3.2. Cùng khảo sát — frame-level (`eye_closed`)

| Bộ | TP | TN | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Baseline | 298 | 2120 | 199 | 155 | 0.5996 | 0.6578 | 0.6274 |
| A | 213 | 2187 | 132 | 240 | 0.6174 | 0.4702 | 0.5338 |
| **B (chọn)** | **268** | 2165 | 154 | 185 | 0.6351 | **0.5916** | **0.6126** |
| C | 219 | 2188 | 131 | 234 | 0.6257 | 0.4834 | 0.5455 |

**Lý do chọn bộ B:** B và C cùng đạt `false_alarm = 6` ở event-level, nhưng B giữ được **268 frame TP** so với 219 của C. Nghĩa là B vẫn nhận diện được phần lớn **thời lượng** của mỗi lần nhắm mắt thật, còn C (do hạ EAR xuống 0.19) đã bắt đầu "cắt cụt" sự kiện thật. Cả 4 bộ đều giữ `hit = 5/5` — không bộ nào làm mất sự kiện thật.

### 3.3. Kết quả `yawn` và `head_tilt` trên video

Với **MAR = 0.6** (ngưỡng đang dùng):

| Sự kiện | true_events | hit | miss | false_alarm | Recall | Precision |
|---|---|---|---|---|---|---|
| `yawn` | 3 | 3 | 0 | **0** | 1.0000 | **1.0000** |
| `head_tilt` | 3 | 0 | 3 | 0 | 0.0000 | 0.0000 |

Frame-level `yawn` (MAR 0.6): TP=121, FP=0, FN=114 — precision 1.0000, recall 0.5149.

**So sánh MAR 0.6 vs 0.45 trên video:** event-level **giống hệt nhau** (3/3 hit, 0 false alarm). Frame-level chỉ nhích nhẹ (TP 121 → 128, ~0.4 giây ở 16.6fps).

---

## 4. Ba phát hiện chính từ thực nghiệm

### 4.1. Ngưỡng tối ưu trên ảnh tĩnh ≠ ngưỡng tối ưu trên hệ thống thật

Trên `yawn_eye`, hạ MAR từ 0.6 → 0.45 làm recall tăng mạnh (0.655 → 0.862). Nhưng trên **video** — tức hệ thống thật, có thêm ràng buộc `MAR_CONSEC_FRAMES` — cả hai ngưỡng cho kết quả event-level **y hệt**: 3/3 hit, 0 false alarm. Ngưỡng 0.6 đã đạt trần.

**Nguyên nhân:** `MAR_CONSEC_FRAMES` yêu cầu miệng mở liên tục ~0.9s mới tính là ngáp. Một cú ngáp thật có MAR vọt lên cao vượt xa 0.6, nên được bắt ở cả hai ngưỡng. Hạ ngưỡng chỉ giúp bắt thêm phần **đầu/đuôi** của cú ngáp (lúc miệng đang mở dần / đóng dần) — làm đẹp frame-level nhưng không đổi event-level.

**Quyết định:** Giữ MAR = 0.6. Hạ tiếp chỉ tăng rủi ro false positive để đổi lấy một chỉ số đã hoàn hảo.

### 4.2. False alarm của `eye_closed` đến từ THỜI LƯỢNG, không phải NGƯỠNG

Giả thuyết ban đầu: `EAR_THRESHOLD = 0.21` quá lỏng. **Dữ liệu bác bỏ giả thuyết này:**

- Bộ A (siết riêng EAR → 0.18): false_alarm chỉ giảm 10 → 7, nhưng **phá hủy frame TP** (298 → 213, mất 28%).
- Bộ B (siết riêng CONSEC → 30): false_alarm giảm 10 → 6 (**tốt hơn A**), chỉ mất 10% frame TP.

**Kết luận đúng:** Có những đợt mắt khép/nheo tự nhiên đủ sâu (EAR < 0.21) và kéo dài hơn 1.2s nhưng chưa tới 1.8s — đó chính là nguồn báo động giả. Hạ ngưỡng EAR không giải quyết được (vì chúng thật sự nhắm sâu), chỉ làm mất frame của sự kiện thật.

### 4.3. `head_tilt` miss 3/3 — nguyên nhân là dữ liệu quay, không phải công thức

Giả thuyết ban đầu: hệ thống thiếu đo **pitch** (gục đầu ra trước). **Dữ liệu bác bỏ:**

- Động tác nghiêng đầu trong video quá nhẹ (trung bình **3.8°**, chỉ 8/195 frame vượt ngưỡng 15°) và quá ngắn.
- Chỉ số `nose_ratio` cho thấy đầu **gần như không gục ra trước** (0.366 ≈ baseline 0.358) — tức pitch không phải yếu tố bị bỏ sót ở đây.

**Kết luận:** Hệ thống bỏ lỡ vì **động tác diễn quá nhẹ**, không phải vì công thức thiếu pitch.

---

## 5. Hạn chế (BẮT BUỘC ghi trong báo cáo)

1. **Dữ liệu video chỉ có 1 người (Bình), 5 sự kiện `eye_closed` / 3 `yawn` / 3 `head_tilt`.** Với N nhỏ như vậy, `EAR_CONSEC_FRAMES = 30` có nguy cơ **overfit** vào nhịp chớp mắt riêng của một người. Đây là lý do bộ B **chưa phải kết quả cuối cùng** — cần chạy lại 4 bộ EAR khi có đủ video nhóm.

2. **Precision `eye_closed` chỉ 0.4545** ngay cả ở cấu hình tốt nhất — hệ thống vẫn báo nhầm nhiều hơn báo đúng. Đây là hướng lệch **nguy hiểm cho ứng dụng thật**: nhiều cảnh báo giả sẽ khiến tài xế bỏ qua cảnh báo thật.

3. **Đánh đổi độ trễ:** Siết `EAR_CONSEC_FRAMES` lên 30 (~1.8s ở 16.6fps) khiến hệ thống **phản ứng chậm hơn**. Trong tình huống lái xe thật, 1.8s nhắm mắt ở 60 km/h tương đương đi được **30 mét** trong vô thức. Giảm false alarm phải trả giá bằng thời gian phản ứng.

4. **`FP = 0` trên `yawn_eye` không chứng minh ngưỡng an toàn.** Ảnh `no_yawn` trong dataset gần như chắc chắn là mặt miệng đóng bình thường. Trong thực tế, tài xế **nói chuyện, cười, hát** — miệng mở vừa phải. Những trạng thái đó **không tồn tại trong yawn_eye** (selection bias), nên không thể kết luận hệ thống sẽ không báo nhầm khi tài xế đang nói.

5. **`head_tilt` chỉ đo roll (nghiêng trái–phải), chưa đo pitch (gục ra trước)** — mà gục đầu ra trước lại là dấu hiệu buồn ngủ điển hình nhất. Có thể bổ sung bằng ước lượng 3D head pose của MediaPipe nếu có thêm thời gian.

6. **Ảnh tĩnh (DDD) không phản ánh đúng hệ thống thời gian.** Recall 0.4673 trên toàn bộ 41.793 ảnh chủ yếu do bản chất đánh giá, không phải do thuật toán yếu. Đã loại trừ hai lời giải thích cạnh tranh: (a) không phải do mẫu nhỏ — recall gần như không đổi khi tăng mẫu từ 1.000 lên 41.793 ảnh; (b) không phải do không detect được mặt — `no_face_rate` chỉ 0.04%.

7. **Dữ liệu video là "diễn", không phải buồn ngủ thật.** Chưa kiểm chứng trên tài xế buồn ngủ thực tế trong xe. Đây là hệ thống demo học thuật, chưa sẵn sàng ứng dụng thực tế.

---

## 6. Việc cần làm khi nhận đủ video nhóm

- [ ] Chạy lại **4 bộ EAR** (baseline / A / B / C) trên toàn bộ video nhóm
- [ ] Kiểm tra bộ B có còn là lựa chọn tốt nhất không, hay số liệu đổi khi N tăng
- [ ] Kiểm tra lại `MAR = 0.6` với video có người **nói chuyện nhiều** (kiểm chứng hạn chế số 4)
- [ ] Xem `head_tilt` có bắt được không khi người khác diễn động tác **mạnh và rõ hơn** (kiểm chứng phát hiện 4.3)
- [ ] **Đồng bộ ngưỡng cuối cùng vào CẢ HAI file** `evaluate.py` và `drowsiness_detection.py`
- [ ] Cập nhật lại toàn bộ số liệu trong báo cáo + slide
