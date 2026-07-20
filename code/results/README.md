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

