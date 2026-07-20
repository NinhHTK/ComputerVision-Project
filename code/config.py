"""Cấu hình dùng chung cho demo realtime và script đánh giá."""

# Ngưỡng hình học
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
TILT_THRESHOLD = 15.0

# Thời gian dấu hiệu phải xuất hiện liên tục trước khi cảnh báo.
# Đây là nguồn sự thật duy nhất; không định nghĩa lại bằng số frame.
EAR_CONSEC_SECONDS = 1.0          # tương đương 30 frame tại 30 FPS
MAR_CONSEC_SECONDS = 0.5          # tương đương 15 frame tại 30 FPS
TILT_CONSEC_SECONDS = 2.0 / 3.0   # tương đương 20 frame tại 30 FPS

# Chỉ dùng để giải thích/đối chiếu cấu hình cũ trong báo cáo.
REFERENCE_FPS = 30.0
