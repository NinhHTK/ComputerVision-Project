"""
===================================================================
ĐỒ ÁN CPV301 - COMPUTER VISION
Đề tài: Phát hiện buồn ngủ khi lái xe (Driver Drowsiness Detection)
Phương pháp: Geometric features (EAR + MAR + Head Tilt) trên MediaPipe Face Mesh
-------------------------------------------------------------------
Ý tưởng:
  - EAR (Eye Aspect Ratio): mắt nhắm lâu  -> ngủ gật
  - MAR (Mouth Aspect Ratio): miệng mở lâu -> ngáp
  - Head Tilt: đầu nghiêng/gục quá mức     -> mất tập trung
Không dùng deep learning tự train. MediaPipe chỉ đóng vai trò
công cụ tìm landmark có sẵn; toàn bộ logic phán đoán là hình học.
===================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time

from config import (
    EAR_CONSEC_SECONDS,
    EAR_THRESHOLD,
    MAR_CONSEC_SECONDS,
    MAR_THRESHOLD,
    TILT_CONSEC_SECONDS,
    TILT_THRESHOLD,
)
from temporal_logic import DurationTracker

# ------------------------------------------------------------------
# 2. CHỈ SỐ LANDMARK CỦA MEDIAPIPE FACE MESH (468 điểm)
#    Thứ tự 6 điểm mỗi mắt theo đúng công thức EAR:
#    p1 (góc trong) - p2,p3 (mí trên) - p4 (góc ngoài) - p5,p6 (mí dưới)
# ------------------------------------------------------------------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Miệng: 2 điểm ngang (mép) + 2 điểm dọc (môi trên - môi dưới)
MOUTH = [78, 308, 13, 14]   # [mép trái, mép phải, môi trên giữa, môi dưới giữa]

# Điểm để tính độ nghiêng đầu: mắt trái ngoài (33) và mắt phải ngoài (263)
LEFT_EYE_CORNER  = 33
RIGHT_EYE_CORNER = 263


# ------------------------------------------------------------------
# 3. CÁC HÀM TÍNH TOÁN HÌNH HỌC
# ------------------------------------------------------------------
def euclid(p1, p2):
    """Khoảng cách Euclid giữa 2 điểm (x, y)."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(eye_pts):
    """
    Tính EAR theo công thức Soukupova & Cech (2016):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    eye_pts: list 6 điểm (x, y) theo đúng thứ tự p1..p6
    """
    A = euclid(eye_pts[1], eye_pts[5])   # ||p2 - p6||
    B = euclid(eye_pts[2], eye_pts[4])   # ||p3 - p5||
    C = euclid(eye_pts[0], eye_pts[3])   # ||p1 - p4||
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth_pts):
    """
    Tính MAR = khoảng cách dọc (môi trên - môi dưới)
             / khoảng cách ngang (mép - mép).
    mouth_pts: [mép trái, mép phải, môi trên, môi dưới]
    """
    vertical   = euclid(mouth_pts[2], mouth_pts[3])   # môi trên <-> môi dưới
    horizontal = euclid(mouth_pts[0], mouth_pts[1])   # mép <-> mép
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def head_tilt_angle(left_corner, right_corner):
    """
    Tính góc nghiêng của đầu dựa trên đường nối 2 khóe mắt ngoài.
    Trả về góc (độ) so với phương ngang. 0 = đầu thẳng.
    """
    dy = right_corner[1] - left_corner[1]
    dx = right_corner[0] - left_corner[0]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)


# ------------------------------------------------------------------
# 4. CHƯƠNG TRÌNH CHÍNH
# ------------------------------------------------------------------
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,       # tăng độ chính xác vùng mắt/miệng
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)        # 0 = webcam mặc định
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại camera.")
        return

    # Cùng một logic thời gian được dùng trong evaluate.py.
    eye_tracker = DurationTracker(EAR_CONSEC_SECONDS)
    mouth_tracker = DurationTracker(MAR_CONSEC_SECONDS)
    tilt_tracker = DurationTracker(TILT_CONSEC_SECONDS)
    yawn_count = 0

    prev_time = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.monotonic()
        frame_delta = max(0.0, now - prev_time)
        prev_time = now

        frame = cv2.flip(frame, 1)   # lật ngang cho giống gương
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        alert_msgs = []              # danh sách cảnh báo hiển thị

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Đổi landmark chuẩn hóa (0..1) sang tọa độ pixel
            def px(idx):
                return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

            # ---- EAR ----
            left_eye  = [px(i) for i in LEFT_EYE]
            right_eye = [px(i) for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # ---- MAR ----
            mouth_pts = [px(i) for i in MOUTH]
            mar = mouth_aspect_ratio(mouth_pts)

            # ---- Head tilt ----
            tilt = head_tilt_angle(px(LEFT_EYE_CORNER), px(RIGHT_EYE_CORNER))

            # ---- Vẽ landmark mắt & miệng để trực quan ----
            for p in left_eye + right_eye + mouth_pts:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            # ---- Logic thời gian (không phụ thuộc FPS) ----
            if eye_tracker.update(ear < EAR_THRESHOLD, frame_delta):
                alert_msgs.append("CANH BAO: NGU GAT!")

            if mouth_tracker.update(mar > MAR_THRESHOLD, frame_delta):
                alert_msgs.append("CANH BAO: DANG NGAP!")
            if mouth_tracker.just_activated:
                yawn_count += 1

            if tilt_tracker.update(tilt > TILT_THRESHOLD, frame_delta):
                alert_msgs.append("CANH BAO: NGHIENG DAU!")

            # ---- Hiển thị chỉ số ----
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Tilt: {tilt:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"So lan ngap: {yawn_count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Mất mặt làm gián đoạn chuỗi thời gian của mọi dấu hiệu.
            eye_tracker.reset()
            mouth_tracker.reset()
            tilt_tracker.reset()
            cv2.putText(frame, "Khong phat hien khuon mat", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---- Hiển thị cảnh báo (chữ đỏ, lớn) ----
        y0 = 160
        for i, msg in enumerate(alert_msgs):
            cv2.putText(frame, msg, (10, y0 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # ---- FPS ----
        fps = 1.0 / frame_delta if frame_delta > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Drowsiness Detection - CPV301 (nhan Q de thoat)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
