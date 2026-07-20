"""
===================================================================
CPV301 COMPUTER VISION PROJECT
Topic: Driver Drowsiness Detection
Method: Geometric features (EAR + MAR + Head Tilt) using MediaPipe Face Mesh
-------------------------------------------------------------------
Concept:
  - EAR (Eye Aspect Ratio): prolonged eye closure -> drowsiness alert
  - MAR (Mouth Aspect Ratio): prolonged mouth opening -> yawn alert
  - Head Tilt: excessive left/right roll -> attention alert
No new deep-learning model is trained. MediaPipe supplies facial landmarks;
all alert decisions use geometric and temporal rules.
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
# 2. MEDIAPIPE FACE MESH LANDMARK INDICES (468 landmarks)
#    Six eye points ordered for the EAR formula:
#    p1 (inner corner), p2/p3 (upper lid), p4 (outer corner),
#    p5/p6 (lower lid)
# ------------------------------------------------------------------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth: two horizontal corners and two vertical lip points
MOUTH = [78, 308, 13, 14]   # [left corner, right corner, upper lip, lower lip]

# Outer eye corners used to estimate head roll
LEFT_EYE_CORNER  = 33
RIGHT_EYE_CORNER = 263


# ------------------------------------------------------------------
# 3. GEOMETRIC FEATURE FUNCTIONS
# ------------------------------------------------------------------
def euclid(p1, p2):
    """Return the Euclidean distance between two (x, y) points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(eye_pts):
    """
    Compute EAR using the Soukupova and Cech (2016) formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    eye_pts: six (x, y) points ordered as p1..p6
    """
    A = euclid(eye_pts[1], eye_pts[5])   # ||p2 - p6||
    B = euclid(eye_pts[2], eye_pts[4])   # ||p3 - p5||
    C = euclid(eye_pts[0], eye_pts[3])   # ||p1 - p4||
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth_pts):
    """
    Compute MAR as vertical lip distance divided by mouth-corner distance.
    mouth_pts: [left corner, right corner, upper lip, lower lip]
    """
    vertical   = euclid(mouth_pts[2], mouth_pts[3])   # upper lip <-> lower lip
    horizontal = euclid(mouth_pts[0], mouth_pts[1])   # left corner <-> right corner
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def head_tilt_angle(left_corner, right_corner):
    """
    Estimate head roll from the line between the two outer eye corners.
    Return the absolute angle in degrees relative to horizontal; 0 is upright.
    """
    dy = right_corner[1] - left_corner[1]
    dx = right_corner[0] - left_corner[0]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)


# ------------------------------------------------------------------
# 4. MAIN APPLICATION
# ------------------------------------------------------------------
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,       # improve eye and mouth landmark accuracy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)        # 0 = default webcam
    if not cap.isOpened():
        print("Unable to open the webcam. Check the camera connection.")
        return

    # The same temporal logic is used in evaluate.py.
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

        frame = cv2.flip(frame, 1)   # horizontal mirror view
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        alert_msgs = []              # alerts displayed on the frame

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Convert normalized landmarks (0..1) to pixel coordinates
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

            # ---- Draw eye and mouth landmarks ----
            for p in left_eye + right_eye + mouth_pts:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            # ---- FPS-independent temporal logic ----
            if eye_tracker.update(ear < EAR_THRESHOLD, frame_delta):
                alert_msgs.append("WARNING: EYES CLOSED!")

            if mouth_tracker.update(mar > MAR_THRESHOLD, frame_delta):
                alert_msgs.append("WARNING: YAWNING!")
            if mouth_tracker.just_activated:
                yawn_count += 1

            if tilt_tracker.update(tilt > TILT_THRESHOLD, frame_delta):
                alert_msgs.append("WARNING: HEAD TILT!")

            # ---- Display metrics ----
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Tilt: {tilt:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Yawn count: {yawn_count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Losing the face interrupts every continuous-duration sequence.
            eye_tracker.reset()
            mouth_tracker.reset()
            tilt_tracker.reset()
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---- Display large red alerts ----
        y0 = 160
        for i, msg in enumerate(alert_msgs):
            cv2.putText(frame, msg, (10, y0 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # ---- FPS ----
        fps = 1.0 / frame_delta if frame_delta > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Drowsiness Detection - CPV301 (press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
