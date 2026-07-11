# Phát hiện buồn ngủ khi lái xe — CPV301

Phát hiện dấu hiệu buồn ngủ của tài xế qua webcam bằng 3 đặc trưng hình học:
- **EAR** (Eye Aspect Ratio): mắt nhắm lâu → ngủ gật
- **MAR** (Mouth Aspect Ratio): miệng mở lâu → ngáp
- **Head Tilt**: đầu nghiêng quá mức (roll — nghiêng trái/phải)

Không dùng deep learning tự train. MediaPipe Face Mesh chỉ là công cụ tìm landmark có sẵn; toàn bộ logic phán đoán là hình học thuần (explainable).

> **Lưu ý về head tilt:** Hiện tại chỉ đo **roll** (nghiêng trái–phải), chưa đo **pitch** (gục đầu ra trước). Đây là hạn chế đã biết, ghi rõ trong báo cáo.

---

## ⚠️ Hai lỗi thường gặp — đọc trước khi cài

1. **Phiên bản Python:** MediaPipe (bản dùng ở đây, `0.10.21`) **chưa hỗ trợ Python 3.12+**. Máy cài Python 3.14 sẽ không có bản build phù hợp. Bắt buộc dùng **Python 3.11** (đã test với 3.11.9).

2. **Đường dẫn KHÔNG được chứa dấu tiếng Việt.** Nếu thư mục project nằm trong đường dẫn có dấu (ví dụ `Kì 4`), MediaPipe báo lỗi `FileNotFoundError: binarypb`. Đây là bug đã biết trong lõi C++ của MediaPipe. Đổi tên thành không dấu (ví dụ `Ki4`) để khắc phục.

---

## Cài đặt

Yêu cầu **Python 3.11** (xem lý do ở phần cảnh báo trên).

Tạo môi trường ảo riêng bằng Python 3.11, rồi cài thư viện (nhớ add Python.exe to PATH):

```powershell
# Windows (PowerShell) — trỏ đúng tới Python 3.11
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1

pip install -r requirements.txt
```

```bash
# macOS / Linux
python3.11 -m venv venv311
source venv311/bin/activate

pip install -r requirements.txt
```

Khi môi trường được kích hoạt, đầu dòng lệnh phải hiển thị `(venv311)`. **Luôn kích hoạt `venv311` trước khi chạy bất kỳ file Python nào.**

Phiên bản MediaPipe sử dụng: `mediapipe==0.10.21`.

---

## Chạy phát hiện realtime

```bash
python drowsiness_detection.py
```

Nhấn **Q** để thoát. Webcam mặc định là camera số 0.

Màn hình hiển thị realtime các chỉ số EAR / MAR / Tilt và bật cảnh báo (chữ đỏ) khi phát hiện dấu hiệu buồn ngủ.

---

## Chạy đánh giá (`evaluate.py`)

Script đánh giá tổng hợp trên các dataset ảnh tĩnh và video, tự chuẩn hóa `consec_frames` theo fps thực tế của video. Kết quả xuất ra thư mục `results/` (CSV) và in ra màn hình.

Kích hoạt `venv311` trước, sau đó:

```powershell
# Test nhanh trên DDD (~3–5 phút) trước khi chạy full
python evaluate.py --ddd ..\dataset\driver_drowsiness_dataset --limit 500

# Đánh giá riêng chỉ số MAR trên yawn_eye (vài phút)
python evaluate.py --yawn ..\dataset\yawn_eye_dataset_new

# Chạy full DDD sau khi đã chắc --limit 500 chạy ổn (~20–40 phút)
python evaluate.py --ddd ..\dataset\driver_drowsiness_dataset
```

Điều chỉnh đường dẫn `--ddd` / `--yawn` cho khớp với vị trí đặt dataset trên máy bạn.

---

## Tải dataset

Dataset **không được commit lên Git** (đã loại trong `.gitignore`), nên cần tải thủ công từ Kaggle và đặt vào thư mục `dataset/`:

- **DDD (Driver Drowsiness Dataset)** — 41.793 ảnh (22.348 drowsy + 19.445 non_drowsy), dùng đánh giá toàn hệ thống:
  https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

- **yawn_eye_dataset_new** — dùng phần `yawn/no_yawn` để đánh giá riêng chỉ số MAR:
  https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new

Cấu trúc thư mục `dataset/` gợi ý:

```
dataset/
├── driver_drowsiness_dataset/
└── yawn_eye_dataset_new/
```

---

## Tinh chỉnh ngưỡng

Mỗi khuôn mặt/camera khác nhau, nên chỉnh các hằng số ở đầu file cho đúng:
- `EAR_THRESHOLD` (mặc định 0.21): nếu báo ngủ gật quá nhạy thì giảm; nếu không bắt được thì tăng. Quan sát giá trị EAR in trên màn hình khi mắt mở/nhắm để chọn ngưỡng giữa hai mức.
- `MAR_THRESHOLD` (0.6): tương tự với ngáp.
- `TILT_THRESHOLD` (15 độ): góc nghiêng đầu cho phép.
- Các `*_CONSEC_FRAMES`: tăng lên nếu muốn giảm cảnh báo giả (yêu cầu dấu hiệu kéo dài hơn).

---

## Cấu trúc project

- `drowsiness_detection.py` — chương trình chính (phát hiện realtime từ webcam)
- `evaluate.py` — script đánh giá tổng hợp (DDD + yawn_eye + video tự quay)
- `requirements.txt` — thư viện cần cài
- `.gitignore` — loại `venv311/`, `dataset/`, `results/`, cache Python khỏi repo
- `dataset/` — chứa DDD + yawn_eye (tải thủ công, bị gitignore)
- `results/` — kết quả đánh giá dạng CSV
