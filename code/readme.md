# Ghi chú mã nguồn

- `config.py` là nguồn cấu hình duy nhất cho EAR, MAR, head tilt và thời gian
  duy trì của từng dấu hiệu.
- `temporal_logic.py` cung cấp `DurationTracker`, được dùng chung bởi demo
  realtime và video evaluation.
- `drowsiness_detection.py` chạy webcam realtime.
- `evaluate.py` đánh giá DDD, yawn_eye và video có nhãn.
- `prepare_submission_samples.py` tạo bộ ảnh mẫu xác định với seed và manifest
  SHA-256.

Không định nghĩa lại ngưỡng trong `drowsiness_detection.py` hoặc `evaluate.py`.
Nếu thay đổi `config.py`, cần chạy lại các thực nghiệm bị ảnh hưởng và lưu cấu
hình mới cùng kết quả.

Các lệnh cài đặt và chạy chuẩn được ghi tại `../README.md`.

