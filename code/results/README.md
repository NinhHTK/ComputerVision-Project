# Results Directory

## `full/static/`

Kết quả trên toàn bộ dataset ảnh tĩnh dùng trong báo cáo:

- `ddd_full.csv`: đánh giá toàn hệ thống trên DDD.
- `yawn_full_mar060.csv`: đánh giá riêng MAR trên yawn_eye với ngưỡng 0.60.

## `sample_smoke_test/`

Kết quả trên 300 ảnh mẫu đi kèm bài nộp. Các file này chỉ dùng xác minh pipeline,
không thay thế kết quả full dataset.

## `experiments/`

- `ddd_quick_test.csv`: quick test với tối đa 500 ảnh mỗi lớp DDD.
- `ear_threshold_sweep/`: thử nghiệm EAR và thời gian trên video của Bình dùng
  trong giai đoạn lựa chọn cấu hình.
- `mar_threshold_sweep/static/`: thử nghiệm ngưỡng MAR trên full yawn_eye.

### Cấu hình `ear_threshold_sweep/`

Các tên `Baseline`, `A`, `B`, `C` là những cấu hình nội bộ dùng để khảo sát
độ nhạy của ngưỡng EAR và thời gian mắt phải nhắm liên tục. Đây không phải là
các phương pháp baseline lấy từ related work.

| Cấu hình | EAR threshold | Số frame tham chiếu tại 30 FPS | Thời gian tương đương | Mục đích |
|---|---:|---:|---:|---|
| Baseline | 0.21 | 20 | 0.67 giây | Cấu hình ban đầu |
| A | 0.18 | 20 | 0.67 giây | Siết riêng ngưỡng EAR |
| B | 0.21 | 30 | 1.00 giây | Tăng riêng thời gian duy trì; cấu hình được chọn |
| C | 0.19 | 25 | 0.83 giây | Thay đổi đồng thời EAR và thời gian duy trì |

Các CSV `video_*_baseline.csv`, `video_*_earA.csv`, `video_*_earB.csv` và
`video_*_earC.csv` được tạo trên cặp video của Bình trong giai đoạn lựa chọn
cấu hình. Cột “số frame tham chiếu” chỉ dùng để mô tả thiết kế cũ tại 30 FPS;
giá trị thời gian tương đương mới là ý nghĩa cần dùng khi so sánh video có FPS
khác nhau.

Kết quả cuối trong `final_video/` đã được chạy lại bằng `DurationTracker` theo
giây với cấu hình B: `EAR_THRESHOLD=0.21` và `EAR_CONSEC_SECONDS=1.0`. Vì vậy,
không suy diễn trực tiếp các CSV sweep một người thành kết quả tổng quát cho cả
ba thành viên.

### Cấu hình `mar_threshold_sweep/static/`

Các file trong thư mục này tương ứng với bốn giá trị `MAR_THRESHOLD`: 0.45,
0.50, 0.55 và 0.60. Cấu hình cuối sử dụng `MAR_THRESHOLD=0.60`; kết quả cuối
trên full yawn_eye nằm tại `full/static/yawn_full_mar060.csv`.

## `final_video/`

Kết quả video cuối sau khi chuyển logic cảnh báo sang thời gian:

- `video_frame_level.csv`: frame-level tổng hợp.
- `video_event_level.csv`: event-level tổng hợp.
- `video_frame_level_per_subject.csv`: frame-level theo thành viên.
- `video_event_level_per_subject.csv`: event-level theo thành viên.
- `video_per_file.csv`: FPS, số frame và tỷ lệ mất mặt của từng video.
- `video_run_config.csv`: ngưỡng, thời gian duy trì và event overlap của lần chạy.

Kết quả theo thành viên là phân tích chính vì giao thức quay của ba thành viên
không đồng nhất. Chỉ số tổng hợp được dùng như thông tin bổ sung.

## Kết quả lưu trữ cục bộ

Kết quả trước khi sửa timing và kết quả có input video trùng lặp được giữ cục bộ
để audit nhưng không nằm trong kết quả cuối hoặc gói submission.
