# Phát hiện buồn ngủ khi lái xe — CPV301

Phát hiện dấu hiệu buồn ngủ của tài xế qua webcam bằng 3 đặc trưng hình học:
- **EAR** (Eye Aspect Ratio): mắt nhắm lâu → ngủ gật
- **MAR** (Mouth Aspect Ratio): miệng mở lâu → ngáp
- **Head Tilt**: đầu nghiêng/gục quá mức

Không dùng deep learning tự train. MediaPipe Face Mesh chỉ là công cụ tìm landmark có sẵn; toàn bộ logic phán đoán là hình học thuần.

## Cài đặt

Yêu cầu Python 3.8–3.11 (MediaPipe chưa hỗ trợ tốt các bản mới hơn ở một số máy).

```bash
pip install -r requirements.txt
```

## Chạy

```bash
python drowsiness_detection.py
```

Nhấn **Q** để thoát. Webcam mặc định là camera số 0.

## Tinh chỉnh ngưỡng

Mỗi khuôn mặt/camera khác nhau, nên chỉnh các hằng số ở đầu file cho đúng:
- `EAR_THRESHOLD` (mặc định 0.21): nếu báo ngủ gật quá nhạy thì giảm; nếu không bắt được thì tăng. Quan sát giá trị EAR in trên màn hình khi mắt mở/nhắm để chọn ngưỡng giữa hai mức.
- `MAR_THRESHOLD` (0.6): tương tự với ngáp.
- `TILT_THRESHOLD` (15 độ): góc nghiêng đầu cho phép.
- Các `*_CONSEC_FRAMES`: tăng lên nếu muốn giảm cảnh báo giả (yêu cầu dấu hiệu kéo dài hơn).

## Cấu trúc

- `drowsiness_detection.py` — chương trình chính
- `requirements.txt` — thư viện cần cài
