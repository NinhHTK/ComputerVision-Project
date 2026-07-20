# Driver Drowsiness Detection — CPV301

Đồ án Computer Vision phát hiện ba dấu hiệu liên quan đến buồn ngủ của tài xế
qua webcam hoặc video:

- **Eye closure:** Eye Aspect Ratio (EAR) thấp trong một khoảng thời gian.
- **Yawn:** Mouth Aspect Ratio (MAR) cao trong một khoảng thời gian.
- **Head tilt:** góc nối hai khóe mắt vượt ngưỡng nghiêng trái/phải.

MediaPipe Face Mesh được dùng để trích xuất landmark. Phần quyết định cảnh báo
dựa trên các đặc trưng hình học và quy tắc thời gian, không huấn luyện mô hình
deep learning mới.

## 1. Môi trường đã kiểm thử

- Windows 10/11
- Python `3.11.9`
- OpenCV `4.11.0`
- MediaPipe `0.10.21`
- NumPy `1.26.4`

Project không kèm môi trường Python. Tất cả dependency trực tiếp được khai báo
trong `code/requirements.txt`.

> Nên đặt project trong đường dẫn không có dấu tiếng Việt. Nếu MediaPipe báo
> `FileNotFoundError` liên quan đến `binarypb`, hãy chuyển project sang đường
> dẫn không dấu, chẳng hạn `D:\CPV301\ComputerVision-Project`.

## 2. Cài đặt trên Windows

Mở PowerShell tại thư mục gốc của project:

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\code\requirements.txt
```

Kiểm tra dependency:

```powershell
python -m pip check
python -c "import cv2, mediapipe, numpy; print(cv2.__version__, mediapipe.__version__, numpy.__version__)"
```

Kết quả `pip check` mong đợi là `No broken requirements found.`

Nếu PowerShell chặn script kích hoạt môi trường, có thể gọi trực tiếp:

```powershell
.\venv311\Scripts\python.exe -m pip install -r .\code\requirements.txt
```

## 3. Chạy ứng dụng realtime

Từ thư mục gốc của project:

```powershell
python .\code\drowsiness_detection.py
```

Ứng dụng sử dụng webcam số `0`. Nhấn **Q** để thoát.

Các ngưỡng mặc định:

| Dấu hiệu | Ngưỡng hình học | Thời gian liên tục |
|---|---:|---:|
| Eye closure | EAR < 0.21 | 1.00 giây |
| Yawn | MAR > 0.60 | 0.50 giây |
| Head tilt | góc > 15° | 0.67 giây |

Ngưỡng được định nghĩa tập trung trong `code/config.py`. Demo realtime và video
evaluation cùng sử dụng `code/temporal_logic.py`, vì vậy thời gian cảnh báo không
phụ thuộc trực tiếp vào FPS của camera/video.

## 4. Dữ liệu đi kèm bài nộp

`dataset/samples/` chứa bộ mẫu có thể chạy ngay:

- DDD: 50 ảnh `drowsy` và 50 ảnh `non_drowsy`.
- yawn_eye: 50 ảnh cho mỗi lớp `yawn`/`no_yawn` trên mỗi split
  `train`/`test` (tổng 200 ảnh).
- `sample_manifest.csv`: đường dẫn tương đối và SHA-256 của 300 ảnh mẫu.

`dataset/video_submission/` chứa sáu video đã nén của ba thành viên và sáu file
nhãn cùng tên. CSV nhãn có cấu trúc:

```csv
start,end,event
3.0,6.0,eye_closed
```

`event` chỉ nhận một trong ba giá trị: `eye_closed`, `yawn`, `head_tilt`.

### Dataset đầy đủ

Dataset đầy đủ không nằm trong bài nộp vì giới hạn dung lượng. Có thể tải tại:

- [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- [Yawn Eye Dataset New](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new)

Sau khi tải và giải nén, đặt theo cấu trúc:

```text
dataset/
├── driver_drowsiness_dataset/
│   ├── drowsy/
│   └── non_drowsy/
└── yawn_eye_dataset_new/
    ├── train/
    │   ├── yawn/
    │   └── no_yawn/
    └── test/
        ├── yawn/
        └── no_yawn/
```

## 5. Chạy đánh giá

Mọi lệnh dưới đây được chạy từ thư mục gốc của project. `--output-dir` quyết
định nơi lưu CSV và giúp tránh ghi đè kết quả đã báo cáo.

### Smoke test trên 300 ảnh mẫu

```powershell
python .\code\evaluate.py --ddd .\dataset\samples\DDD --output-dir .\run_results\ddd_sample
python .\code\evaluate.py --yawn .\dataset\samples\yawn_eye --output-dir .\run_results\yawn_sample
```

Kết quả kiểm tra tham chiếu:

| Nguồn | Processed | No face | Accuracy | F1 |
|---|---:|---:|---:|---:|
| DDD sample | 100 | 0 | 0.7300 | 0.6494 |
| yawn_eye sample | 199 | 1 | 0.8442 | 0.8166 |

### Video tự quay

```powershell
python .\code\evaluate.py --video .\dataset\video_submission --output-dir .\run_results\video
```

Video evaluation xuất kết quả frame-level, event-level, theo từng thành viên,
theo từng file và một CSV lưu cấu hình chạy.

### Dataset đầy đủ

```powershell
# Kiểm tra nhanh: tối đa 500 ảnh cho mỗi lớp DDD
python .\code\evaluate.py --ddd .\dataset\driver_drowsiness_dataset --limit 500 --output-dir .\run_results\ddd_quick

# Đánh giá toàn bộ hai dataset ảnh tĩnh
python .\code\evaluate.py --ddd .\dataset\driver_drowsiness_dataset --output-dir .\run_results\ddd_full
python .\code\evaluate.py --yawn .\dataset\yawn_eye_dataset_new --output-dir .\run_results\yawn_full

# Chạy mọi nguồn dữ liệu đang có trong dataset/
python .\code\evaluate.py --all .\dataset --output-dir .\run_results\all
```

Ảnh không phát hiện được khuôn mặt được báo cáo riêng và không được đưa vào
confusion matrix. Với ảnh tĩnh, mỗi ảnh được so sánh trực tiếp với ngưỡng; quy
tắc duy trì theo thời gian chỉ áp dụng cho realtime và video.

## 6. Kết quả có sẵn

`code/results/` được chia thành:

- `full/static/`: kết quả trên dataset ảnh tĩnh đầy đủ.
- `sample_smoke_test/`: kết quả kiểm tra pipeline bằng 300 ảnh mẫu.
- `experiments/`: quick test và các thử nghiệm lựa chọn ngưỡng.
- `final_video/`: kết quả video cuối sau khi sửa logic thời gian, gồm kết quả
  tổng hợp, theo thành viên, theo file và cấu hình chạy.

Xem mô tả chi tiết tại `code/results/README.md`.

## 7. Cấu trúc project

```text
ComputerVision-Project-main/
├── README.md
├── .gitignore
├── code/
│   ├── config.py
│   ├── temporal_logic.py
│   ├── drowsiness_detection.py
│   ├── evaluate.py
│   ├── prepare_submission_samples.py
│   ├── requirements.txt
│   └── results/
└── dataset/
    ├── samples/
    └── video_submission/
```

Các môi trường `.venv/`, `venv311/`, full dataset, video gốc và cache Python
không được đưa vào gói nộp.

## 8. Hạn chế đã biết

- Head tilt hiện chỉ ước lượng **roll** từ đường nối hai khóe mắt; chuyển động
  cúi đầu ra trước (**pitch**) có thể không được phát hiện.
- Ba thành viên quay video với FPS, độ phân giải, góc quay và quy trình ghi nhãn
  khác nhau. Vì vậy kết quả theo từng thành viên là phân tích chính; kết quả gộp
  cần được diễn giải thận trọng.
- Hệ thống dùng ngưỡng hình học cố định nên khả năng khái quát sang khuôn mặt,
  camera và điều kiện chiếu sáng mới còn hạn chế.

