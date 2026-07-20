"""
===================================================================
CPV301 - SCRIPT ĐÁNH GIÁ TỔNG HỢP (Driver Drowsiness Detection)
-------------------------------------------------------------------
Đánh giá hệ thống geometric (EAR + MAR + Head Tilt / MediaPipe Face Mesh)
trên 3 nguồn dữ liệu:

  1. DDD (ảnh tĩnh, drowsy/non_drowsy)  -> đánh giá TOÀN HỆ THỐNG
  2. yawn_eye (ảnh tĩnh, yawn/no_yawn)  -> đánh giá RIÊNG chỉ số MAR
  3. Video tự quay + nhãn CSV           -> đánh giá theo THỜI GIAN
        (a) frame-level   (b) event-level

LƯU Ý PHƯƠNG PHÁP (ghi rõ trong báo cáo):
  - Với ẢNH TĨNH: KHÔNG có khái niệm "N frame liên tiếp".
    Ta chỉ so sánh giá trị EAR/MAR/tilt trên từng ảnh với ngưỡng.
  - Với VIDEO: dùng cùng ngưỡng theo GIÂY và cùng DurationTracker với code
    realtime để kết quả không phụ thuộc FPS của nguồn video.
  - Ảnh mà MediaPipe KHÔNG tìm được mặt sẽ bị LOẠI khỏi phần tính toán,
    và tỷ lệ loại này được BÁO CÁO (không giấu, không bịa).

Cách chạy (demo):
    py evaluate.py --ddd ../dataset/driver_drowsiness_dataset --limit 500
    
Cách chạy:
    py evaluate.py --ddd    ../dataset/driver_drowsiness_dataset
    py evaluate.py --yawn   ../dataset/yawn_eye_dataset_new
    py evaluate.py --video  ../dataset/video_submission
    py evaluate.py --all    ../dataset          # chạy tất cả nguồn có sẵn

Kết quả: in ra màn hình + lưu CSV trong thư mục được chọn bằng --output-dir.
===================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import glob
import argparse
import math
import time
from collections import defaultdict

from config import (
    EAR_CONSEC_SECONDS,
    EAR_THRESHOLD,
    MAR_CONSEC_SECONDS,
    MAR_THRESHOLD,
    TILT_CONSEC_SECONDS,
    TILT_THRESHOLD,
)
from temporal_logic import DurationTracker


RESULTS_DIR = "results"

# ==================================================================
# 2. CHỈ SỐ LANDMARK (giống code gốc)
# ==================================================================
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]           # [mép trái, mép phải, môi trên, môi dưới]
LEFT_EYE_CORNER  = 33
RIGHT_EYE_CORNER = 263

# Các đuôi file ảnh chấp nhận
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG")


# ==================================================================
# 3. HÀM HÌNH HỌC (giống hệt code gốc để đảm bảo nhất quán)
# ==================================================================
def euclid(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(eye_pts):
    A = euclid(eye_pts[1], eye_pts[5])
    B = euclid(eye_pts[2], eye_pts[4])
    C = euclid(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth_pts):
    vertical   = euclid(mouth_pts[2], mouth_pts[3])
    horizontal = euclid(mouth_pts[0], mouth_pts[1])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def head_tilt_angle(left_corner, right_corner):
    dy = right_corner[1] - left_corner[1]
    dx = right_corner[0] - left_corner[0]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)


# ==================================================================
# 4. HÀM LÕI: từ 1 ảnh/frame -> (ear, mar, tilt) hoặc None nếu ko thấy mặt
# ==================================================================
def extract_metrics(image_bgr, face_mesh):
    """
    Nhận ảnh BGR (OpenCV), trả về dict {ear, mar, tilt} hoặc None nếu
    MediaPipe không tìm được khuôn mặt.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    def px(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    left_eye  = [px(i) for i in LEFT_EYE]
    right_eye = [px(i) for i in RIGHT_EYE]
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

    mouth_pts = [px(i) for i in MOUTH]
    mar = mouth_aspect_ratio(mouth_pts)

    tilt = head_tilt_angle(px(LEFT_EYE_CORNER), px(RIGHT_EYE_CORNER))

    return {"ear": ear, "mar": mar, "tilt": tilt}


def make_face_mesh(static=True):
    """
    static_image_mode=True cho ảnh tĩnh (mỗi ảnh xử lý độc lập),
    =False cho video (MediaPipe tận dụng tracking giữa các frame -> giống demo).
    """
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ==================================================================
# 5. HÀM TÍNH CÁC CHỈ SỐ PHÂN LOẠI TỪ CONFUSION MATRIX
# ==================================================================
def compute_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
    }


def print_metrics(title, m, extra=None):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Confusion:  TP={m['TP']}  TN={m['TN']}  "
          f"FP={m['FP']}  FN={m['FN']}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1-score : {m['f1']:.4f}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_metrics_csv(filename, rows, fieldnames):
    """rows: list các dict. Ghi vào RESULTS_DIR/<filename>."""
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  -> Đã lưu: {path}")


# ==================================================================
# 6. ĐÁNH GIÁ DDD (ảnh tĩnh, drowsy / non_drowsy) - TOÀN HỆ THỐNG
# ==================================================================
def evaluate_ddd(ddd_dir, limit=None):
    """
    ddd_dir chứa 2 thư mục con: drowsy/ và non_drowsy/
    Quy ước dự đoán: 1 ảnh = "drowsy" nếu BẤT KỲ dấu hiệu nào kích hoạt:
        EAR < ngưỡng  HOẶC  MAR > ngưỡng  HOẶC  tilt > ngưỡng
    (Ảnh tĩnh nên không dùng đếm frame liên tiếp.)

    limit: nếu đặt số (vd 2000) thì chỉ lấy tối đa limit ảnh mỗi lớp
           -> để chạy thử nhanh. None = chạy toàn bộ.
    """
    print("\n########## ĐÁNH GIÁ DDD (toàn hệ thống) ##########")
    face_mesh = make_face_mesh(static=True)

    # nhãn thật: drowsy = positive (1), non_drowsy = negative (0)
    classes = {"drowsy": 1, "non_drowsy": 0}

    tp = tn = fp = fn = 0
    no_face = 0            # số ảnh không detect được mặt
    processed = 0
    t0 = time.time()

    for cls_name, y_true in classes.items():
        cls_dir = os.path.join(ddd_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  [CẢNH BÁO] Không thấy thư mục: {cls_dir} -> bỏ qua")
            continue

        files = sorted(f for f in glob.glob(os.path.join(cls_dir, "*"))
                       if f.endswith(IMG_EXTS))
        if limit:
            files = files[:limit]
        print(f"  Đang xử lý lớp '{cls_name}': {len(files)} ảnh...")

        for i, fpath in enumerate(files):
            img = cv2.imread(fpath)
            if img is None:
                continue
            metrics = extract_metrics(img, face_mesh)
            if metrics is None:
                no_face += 1
                continue

            processed += 1
            # Dự đoán drowsy nếu bất kỳ dấu hiệu nào vượt ngưỡng
            pred_drowsy = (
                metrics["ear"] < EAR_THRESHOLD or
                metrics["mar"] > MAR_THRESHOLD or
                metrics["tilt"] > TILT_THRESHOLD
            )
            y_pred = 1 if pred_drowsy else 0

            if   y_true == 1 and y_pred == 1: tp += 1
            elif y_true == 0 and y_pred == 0: tn += 1
            elif y_true == 0 and y_pred == 1: fp += 1
            elif y_true == 1 and y_pred == 0: fn += 1

            if (i + 1) % 2000 == 0:
                print(f"    ...{i+1} ảnh ({cls_name})")

    face_mesh.close()
    elapsed = time.time() - t0

    total_seen = processed + no_face
    no_face_rate = (no_face / total_seen * 100) if total_seen else 0.0

    m = compute_metrics(tp, tn, fp, fn)
    print_metrics("KẾT QUẢ DDD (toàn hệ thống)", m, extra={
        "Ảnh xử lý được (detect ra mặt)": processed,
        "Ảnh KHÔNG detect được mặt": f"{no_face} ({no_face_rate:.2f}%)",
        "Thời gian": f"{elapsed:.1f}s",
    })

    # lưu CSV
    row = dict(m)
    row.update({
        "source": "DDD",
        "processed": processed,
        "no_face": no_face,
        "no_face_rate_percent": round(no_face_rate, 2),
        "seconds": round(elapsed, 1),
    })
    save_metrics_csv("ddd_result.csv", [row],
                     fieldnames=["source", "TP", "TN", "FP", "FN",
                                 "accuracy", "precision", "recall", "f1",
                                 "processed", "no_face",
                                 "no_face_rate_percent", "seconds"])
    return m


# ==================================================================
# 7. ĐÁNH GIÁ yawn_eye (ảnh tĩnh, yawn / no_yawn) - RIÊNG MAR
# ==================================================================
def evaluate_yawn(yawn_dir, splits=("train", "test"), limit=None):
    """
    Cấu trúc: yawn_dir/<split>/yawn/*.jpg và .../no_yawn/*.jpg
    CHỈ đánh giá chỉ số MAR:
        pred "yawn" nếu MAR > ngưỡng, ngược lại "no_yawn".
    Gộp train + test lại để đánh giá (đây là đánh giá thuần geometric,
    KHÔNG train gì, nên không cần tách train/test theo nghĩa học máy).
    """
    print("\n########## ĐÁNH GIÁ yawn_eye (riêng MAR) ##########")
    face_mesh = make_face_mesh(static=True)

    classes = {"yawn": 1, "no_yawn": 0}
    tp = tn = fp = fn = 0
    no_face = 0
    processed = 0
    t0 = time.time()

    for split in splits:
        for cls_name, y_true in classes.items():
            cls_dir = os.path.join(yawn_dir, split, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"  [CẢNH BÁO] Không thấy: {cls_dir} -> bỏ qua")
                continue

            files = sorted(f for f in glob.glob(os.path.join(cls_dir, "*"))
                           if f.endswith(IMG_EXTS))
            if limit:
                files = files[:limit]
            print(f"  '{split}/{cls_name}': {len(files)} ảnh...")

            for fpath in files:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                metrics = extract_metrics(img, face_mesh)
                if metrics is None:
                    no_face += 1
                    continue

                processed += 1
                y_pred = 1 if metrics["mar"] > MAR_THRESHOLD else 0

                if   y_true == 1 and y_pred == 1: tp += 1
                elif y_true == 0 and y_pred == 0: tn += 1
                elif y_true == 0 and y_pred == 1: fp += 1
                elif y_true == 1 and y_pred == 0: fn += 1

    face_mesh.close()
    elapsed = time.time() - t0

    total_seen = processed + no_face
    no_face_rate = (no_face / total_seen * 100) if total_seen else 0.0

    m = compute_metrics(tp, tn, fp, fn)
    print_metrics("KẾT QUẢ yawn_eye (riêng MAR)", m, extra={
        "Ảnh xử lý được": processed,
        "Ảnh KHÔNG detect được mặt": f"{no_face} ({no_face_rate:.2f}%)",
        "Thời gian": f"{elapsed:.1f}s",
    })

    row = dict(m)
    row.update({
        "source": "yawn_eye(MAR)",
        "processed": processed,
        "no_face": no_face,
        "no_face_rate_percent": round(no_face_rate, 2),
        "seconds": round(elapsed, 1),
    })
    save_metrics_csv("yawn_result.csv", [row],
                     fieldnames=["source", "TP", "TN", "FP", "FN",
                                 "accuracy", "precision", "recall", "f1",
                                 "processed", "no_face",
                                 "no_face_rate_percent", "seconds"])
    return m


# ==================================================================
# 8. ĐÁNH GIÁ VIDEO (theo thời gian)
# ==================================================================
# Ánh xạ tên event trong CSV <-> loại dấu hiệu
EVENT_TYPES = ["eye_closed", "yawn", "head_tilt"]


def load_ground_truth(csv_path):
    """
    Đọc file nhãn CSV (start,end,event) -> list các dict.
    Kiểm tra định dạng cơ bản: start<end, event hợp lệ.
    """
    events = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # dòng 2 trở đi (dòng 1 header)
            try:
                start = float(row["start"])
                end = float(row["end"])
                ev = row["event"].strip()
            except (KeyError, ValueError):
                print(f"    [CSV LỖI] {csv_path} dòng {i}: sai định dạng -> bỏ qua")
                continue
            if ev not in EVENT_TYPES:
                print(f"    [CSV LỖI] {csv_path} dòng {i}: event '{ev}' "
                      f"không hợp lệ (chỉ dùng {EVENT_TYPES}) -> bỏ qua")
                continue
            if start >= end:
                print(f"    [CSV LỖI] {csv_path} dòng {i}: start>=end -> bỏ qua")
                continue
            events.append({"start": start, "end": end, "event": ev})
    return events


def find_csv_for_video(video_path):
    """Tìm file CSV cùng tên với video (An_drowsy.mp4 -> An_drowsy.csv)."""
    base = os.path.splitext(video_path)[0]
    csv_path = base + ".csv"
    return csv_path if os.path.isfile(csv_path) else None


def subject_from_video_name(video_name):
    """Binh_alert.mp4 / Binh_drowsy.mp4 -> Binh."""
    stem = os.path.splitext(os.path.basename(video_name))[0]
    for suffix in ("_alert", "_drowsy"):
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def run_system_on_video(video_path):
    """
    Chạy hệ thống với cùng DurationTracker và ngưỡng theo giây như demo.
    Mỗi frame đóng góp 1/fps giây vào chuỗi liên tục. Trả về:
      - fps của video
      - per_frame: list, mỗi phần tử là dict báo mỗi loại dấu hiệu
        có ĐANG cảnh báo ở frame đó không (sau khi qua bộ đếm liên tiếp)
        vd: {"eye_closed": True/False, "yawn": ..., "head_tilt": ...}
      - no_face_frames: số frame không detect được mặt
      - total_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [LỖI] Không mở được video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback nếu không đọc được

    # Các con số frame dưới đây chỉ để hiển thị. Nguồn sự thật là thời gian.
    ear_frames = max(1, math.ceil(EAR_CONSEC_SECONDS * fps - 1e-9))
    mar_frames = max(1, math.ceil(MAR_CONSEC_SECONDS * fps - 1e-9))
    tilt_frames = max(1, math.ceil(TILT_CONSEC_SECONDS * fps - 1e-9))
    print(f"      Ngưỡng theo fps={fps:.1f}: "
          f"eye>={ear_frames}f ({EAR_CONSEC_SECONDS:.2f}s) | "
          f"yawn>={mar_frames}f ({MAR_CONSEC_SECONDS:.2f}s) | "
          f"tilt>={tilt_frames}f ({TILT_CONSEC_SECONDS:.2f}s)")

    # dùng static=False để tận dụng tracking giữa frame (giống demo realtime)
    face_mesh = make_face_mesh(static=False)

    frame_duration = 1.0 / fps
    eye_tracker = DurationTracker(EAR_CONSEC_SECONDS)
    mouth_tracker = DurationTracker(MAR_CONSEC_SECONDS)
    tilt_tracker = DurationTracker(TILT_CONSEC_SECONDS)

    per_frame = []
    no_face_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        metrics = extract_metrics(frame, face_mesh)

        active = {"eye_closed": False, "yawn": False, "head_tilt": False}

        if metrics is None:
            no_face_frames += 1
            # Mất mặt làm gián đoạn chuỗi thời gian của mọi dấu hiệu.
            eye_tracker.reset()
            mouth_tracker.reset()
            tilt_tracker.reset()
            per_frame.append(active)
            continue

        active["eye_closed"] = eye_tracker.update(
            metrics["ear"] < EAR_THRESHOLD, frame_duration)
        active["yawn"] = mouth_tracker.update(
            metrics["mar"] > MAR_THRESHOLD, frame_duration)
        active["head_tilt"] = tilt_tracker.update(
            metrics["tilt"] > TILT_THRESHOLD, frame_duration)

        per_frame.append(active)

    cap.release()
    face_mesh.close()

    return {
        "fps": fps,
        "per_frame": per_frame,
        "no_face_frames": no_face_frames,
        "total_frames": total_frames,
    }


def build_frame_labels(events, total_frames, fps):
    """
    Từ danh sách sự kiện (giây) -> nhãn theo TỪNG FRAME.
    Trả về list độ dài total_frames, mỗi phần tử là set các event đang diễn ra.
    """
    labels = [set() for _ in range(total_frames)]
    for ev in events:
        f_start = int(round(ev["start"] * fps))
        f_end   = int(round(ev["end"] * fps))
        f_start = max(0, f_start)
        f_end   = min(total_frames - 1, f_end)
        for fi in range(f_start, f_end + 1):
            labels[fi].add(ev["event"])
    return labels


def evaluate_video_frame_level(per_frame, frame_labels):
    """
    Đánh giá FRAME-LEVEL cho từng loại dấu hiệu.
    Trả về dict: {event_type: metrics_dict}
    """
    results = {}
    for ev_type in EVENT_TYPES:
        tp = tn = fp = fn = 0
        for pred, truth in zip(per_frame, frame_labels):
            y_pred = 1 if pred[ev_type] else 0
            y_true = 1 if ev_type in truth else 0
            if   y_true == 1 and y_pred == 1: tp += 1
            elif y_true == 0 and y_pred == 0: tn += 1
            elif y_true == 0 and y_pred == 1: fp += 1
            elif y_true == 1 and y_pred == 0: fn += 1
        results[ev_type] = compute_metrics(tp, tn, fp, fn)
    return results


def evaluate_video_event_level(per_frame, events, fps, overlap_ratio=0.3):
    """
    Đánh giá EVENT-LEVEL.
    - Mỗi sự kiện thật (1 dòng CSV) coi là "BẮT ĐƯỢC" (hit) nếu trong khoảng
      thời gian của nó, hệ thống báo cảnh báo tương ứng ở >= overlap_ratio
      tỷ lệ số frame (mặc định 30%). Ngược lại là MISS (false negative).
    - False alarm (báo động giả): đếm số "cụm" frame hệ thống báo cảnh báo
      mà KHÔNG trùng với bất kỳ sự kiện thật nào cùng loại.

    Trả về dict: {event_type: {"hit":, "miss":, "false_alarm":,
                                "recall":, ...}}
    """
    total_frames = len(per_frame)
    results = {}

    for ev_type in EVENT_TYPES:
        # frame nào hệ thống đang báo dấu hiệu này?
        pred_frames = [1 if per_frame[i][ev_type] else 0
                       for i in range(total_frames)]

        # --- HIT / MISS trên từng sự kiện thật ---
        hit = 0
        miss = 0
        true_events = [e for e in events if e["event"] == ev_type]
        # đánh dấu các frame đã được "giải thích" bởi 1 sự kiện thật
        covered = [False] * total_frames

        for ev in true_events:
            f_start = max(0, int(round(ev["start"] * fps)))
            f_end   = min(total_frames - 1, int(round(ev["end"] * fps)))
            span = f_end - f_start + 1
            if span <= 0:
                continue
            fired = sum(pred_frames[f_start:f_end + 1])
            if fired >= overlap_ratio * span:
                hit += 1
            else:
                miss += 1
            for fi in range(f_start, f_end + 1):
                covered[fi] = True

        # --- FALSE ALARM: các cụm frame báo động nằm ngoài mọi sự kiện thật ---
        false_alarm = 0
        in_cluster = False
        cluster_outside = False
        for fi in range(total_frames):
            if pred_frames[fi] == 1:
                if not in_cluster:
                    in_cluster = True
                    cluster_outside = not covered[fi]
                else:
                    if not covered[fi]:
                        cluster_outside = True
            else:
                if in_cluster and cluster_outside:
                    false_alarm += 1
                in_cluster = False
                cluster_outside = False
        if in_cluster and cluster_outside:
            false_alarm += 1

        n_true = hit + miss
        recall = hit / n_true if n_true else 0.0
        # precision kiểu event: hit / (hit + false_alarm)
        precision = hit / (hit + false_alarm) if (hit + false_alarm) else 0.0

        results[ev_type] = {
            "true_events": n_true,
            "hit": hit,
            "miss": miss,
            "false_alarm": false_alarm,
            "recall": recall,
            "precision": precision,
        }
    return results


def evaluate_video(video_dir, overlap_ratio=0.3):
    """
    Duyệt tất cả video (.mp4) trong video_dir có file .csv đi kèm,
    chạy hệ thống, rồi đánh giá cả frame-level và event-level.
    Tổng hợp kết quả toàn bộ video và lưu CSV.
    """
    print("\n########## ĐÁNH GIÁ VIDEO (theo thời gian) ##########")

    video_files = set()
    for ext in ("*.mp4", "*.MP4", "*.avi", "*.mov"):
        video_files.update(glob.glob(os.path.join(video_dir, ext)))
    video_files = sorted(video_files)

    if not video_files:
        print(f"  [THÔNG BÁO] Chưa có video nào trong {video_dir}.")
        print("  -> Khi nào quay xong, đặt .mp4 + .csv cùng tên vào đây rồi chạy lại.")
        return

    # cộng dồn frame-level trên tất cả video
    fl_accum = {ev: {"TP":0,"TN":0,"FP":0,"FN":0} for ev in EVENT_TYPES}
    # cộng dồn event-level
    el_accum = {ev: {"true_events":0,"hit":0,"miss":0,"false_alarm":0}
                for ev in EVENT_TYPES}

    # Tổng hợp riêng theo thành viên để tránh video dài chi phối mà không rõ.
    subject_fl = defaultdict(
        lambda: {ev: {"TP":0,"TN":0,"FP":0,"FN":0}
                 for ev in EVENT_TYPES})
    subject_el = defaultdict(
        lambda: {ev: {"true_events":0,"hit":0,"miss":0,"false_alarm":0}
                 for ev in EVENT_TYPES})

    per_video_rows = []   # để lưu chi tiết từng video

    for vpath in sorted(video_files):
        vname = os.path.basename(vpath)
        subject = subject_from_video_name(vname)
        csv_path = find_csv_for_video(vpath)
        if csv_path is None:
            print(f"  [BỎ QUA] {vname}: không có file .csv nhãn cùng tên.")
            continue

        print(f"\n  >>> Video: {vname}")
        events = load_ground_truth(csv_path)
        print(f"      Nhãn: {len(events)} sự kiện từ {os.path.basename(csv_path)}")

        run = run_system_on_video(vpath)
        if run is None:
            continue

        fps = run["fps"]
        per_frame = run["per_frame"]
        total_frames = run["total_frames"]
        nf = run["no_face_frames"]
        nf_rate = (nf / total_frames * 100) if total_frames else 0.0
        print(f"      {total_frames} frame @ {fps:.1f} fps | "
              f"mất mặt: {nf} frame ({nf_rate:.1f}%)")

        frame_labels = build_frame_labels(events, total_frames, fps)

        # --- frame-level ---
        fl = evaluate_video_frame_level(per_frame, frame_labels)
        for ev in EVENT_TYPES:
            for k in ("TP","TN","FP","FN"):
                fl_accum[ev][k] += fl[ev][k]
                subject_fl[subject][ev][k] += fl[ev][k]

        # --- event-level ---
        el = evaluate_video_event_level(per_frame, events, fps, overlap_ratio)
        for ev in EVENT_TYPES:
            for k in ("true_events","hit","miss","false_alarm"):
                el_accum[ev][k] += el[ev][k]
                subject_el[subject][ev][k] += el[ev][k]

        # in nhanh kết quả video này
        for ev in EVENT_TYPES:
            print(f"        [{ev:10s}] frame: acc={fl[ev]['accuracy']:.3f} "
                  f"P={fl[ev]['precision']:.3f} R={fl[ev]['recall']:.3f}  |  "
                  f"event: hit={el[ev]['hit']}/{el[ev]['true_events']} "
                  f"false_alarm={el[ev]['false_alarm']}")

        per_video_rows.append({
            "subject": subject, "video": vname, "total_frames": total_frames,
            "fps": round(fps,1), "no_face_frames": nf,
            "no_face_rate_percent": round(nf_rate,2),
        })

    # ============ TỔNG HỢP FRAME-LEVEL ============
    print("\n" + "#"*55)
    print("  TỔNG HỢP FRAME-LEVEL (cộng dồn mọi video)")
    print("#"*55)
    fl_rows = []
    for ev in EVENT_TYPES:
        a = fl_accum[ev]
        m = compute_metrics(a["TP"], a["TN"], a["FP"], a["FN"])
        print_metrics(f"[frame-level] {ev}", m)
        row = dict(m); row["event_type"] = ev; row["level"] = "frame"
        fl_rows.append(row)

    # ============ TỔNG HỢP EVENT-LEVEL ============
    print("\n" + "#"*55)
    print("  TỔNG HỢP EVENT-LEVEL (cộng dồn mọi video)")
    print("#"*55)
    el_rows = []
    for ev in EVENT_TYPES:
        a = el_accum[ev]
        n_true = a["true_events"]
        recall = a["hit"]/n_true if n_true else 0.0
        precision = (a["hit"]/(a["hit"]+a["false_alarm"])
                     if (a["hit"]+a["false_alarm"]) else 0.0)
        print(f"\n  [event-level] {ev}")
        print(f"    Sự kiện thật : {n_true}")
        print(f"    Bắt được(hit): {a['hit']}")
        print(f"    Bỏ lỡ (miss) : {a['miss']}")
        print(f"    Báo động giả : {a['false_alarm']}")
        print(f"    Recall       : {recall:.4f}")
        print(f"    Precision    : {precision:.4f}")
        el_rows.append({
            "event_type": ev, "level": "event",
            "true_events": n_true, "hit": a["hit"], "miss": a["miss"],
            "false_alarm": a["false_alarm"],
            "recall": round(recall,4), "precision": round(precision,4),
        })

    # ============ LƯU CSV ============
    save_metrics_csv("video_frame_level.csv", fl_rows,
                     fieldnames=["level","event_type","TP","TN","FP","FN",
                                 "accuracy","precision","recall","f1"])
    save_metrics_csv("video_event_level.csv", el_rows,
                     fieldnames=["level","event_type","true_events","hit",
                                 "miss","false_alarm","recall","precision"])
    if per_video_rows:
        save_metrics_csv("video_per_file.csv", per_video_rows,
                         fieldnames=["subject","video","total_frames","fps",
                                     "no_face_frames","no_face_rate_percent"])

    # ============ KẾT QUẢ RIÊNG THEO THÀNH VIÊN ============
    subject_fl_rows = []
    subject_el_rows = []
    for subject in sorted(subject_fl):
        for ev in EVENT_TYPES:
            a = subject_fl[subject][ev]
            m = compute_metrics(a["TP"], a["TN"], a["FP"], a["FN"])
            subject_fl_rows.append({
                "subject": subject,
                "level": "frame",
                "event_type": ev,
                **m,
                "positive_frames": a["TP"] + a["FN"],
                "negative_frames": a["TN"] + a["FP"],
            })

            e = subject_el[subject][ev]
            n_true = e["true_events"]
            n_pred = e["hit"] + e["false_alarm"]
            recall = e["hit"] / n_true if n_true else ""
            precision = e["hit"] / n_pred if n_pred else ""
            subject_el_rows.append({
                "subject": subject,
                "level": "event",
                "event_type": ev,
                "true_events": n_true,
                "hit": e["hit"],
                "miss": e["miss"],
                "false_alarm": e["false_alarm"],
                "recall": recall,
                "precision": precision,
            })

    if subject_fl_rows:
        save_metrics_csv(
            "video_frame_level_per_subject.csv",
            subject_fl_rows,
            fieldnames=["subject","level","event_type","TP","TN","FP","FN",
                        "accuracy","precision","recall","f1",
                        "positive_frames","negative_frames"],
        )
        save_metrics_csv(
            "video_event_level_per_subject.csv",
            subject_el_rows,
            fieldnames=["subject","level","event_type","true_events","hit",
                        "miss","false_alarm","recall","precision"],
        )

    save_metrics_csv(
        "video_run_config.csv",
        [
            {"parameter": "EAR_THRESHOLD", "value": EAR_THRESHOLD, "unit": "ratio"},
            {"parameter": "EAR_CONSEC_SECONDS", "value": EAR_CONSEC_SECONDS, "unit": "seconds"},
            {"parameter": "MAR_THRESHOLD", "value": MAR_THRESHOLD, "unit": "ratio"},
            {"parameter": "MAR_CONSEC_SECONDS", "value": MAR_CONSEC_SECONDS, "unit": "seconds"},
            {"parameter": "TILT_THRESHOLD", "value": TILT_THRESHOLD, "unit": "degrees"},
            {"parameter": "TILT_CONSEC_SECONDS", "value": TILT_CONSEC_SECONDS, "unit": "seconds"},
            {"parameter": "EVENT_OVERLAP_RATIO", "value": overlap_ratio, "unit": "ratio"},
        ],
        fieldnames=["parameter", "value", "unit"],
    )


# ==================================================================
# 9. MAIN / CLI
# ==================================================================
def main():
    global RESULTS_DIR
    parser = argparse.ArgumentParser(
        description="Script đánh giá tổng hợp CPV301 - Drowsiness Detection")
    parser.add_argument("--ddd", metavar="DIR",
                        help="Đường dẫn thư mục DDD (chứa drowsy/ và non_drowsy/)")
    parser.add_argument("--yawn", metavar="DIR",
                        help="Đường dẫn thư mục yawn_eye_dataset_new")
    parser.add_argument("--video", metavar="DIR",
                        help="Đường dẫn thư mục chứa video + csv nhãn")
    parser.add_argument("--all", metavar="DATASET_DIR",
                        help="Đường dẫn thư mục 'dataset' -> chạy cả 3 nguồn")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số ảnh mỗi lớp (để chạy thử nhanh)")
    parser.add_argument("--overlap", type=float, default=0.3,
                        help="Tỷ lệ chồng lấn tối thiểu để tính 'bắt được' "
                             "sự kiện (event-level, mặc định 0.3)")
    parser.add_argument("--output-dir", default="results", metavar="DIR",
                        help="Thư mục lưu CSV (mặc định: ./results)")
    args = parser.parse_args()
    RESULTS_DIR = os.path.abspath(args.output_dir)

    if args.all:
        ddd_dir  = os.path.join(args.all, "driver_drowsiness_dataset")
        yawn_dir = os.path.join(args.all, "yawn_eye_dataset_new")
        video_submission = os.path.join(args.all, "video_submission")
        video_dir = (video_submission if os.path.isdir(video_submission)
                     else os.path.join(args.all, "video"))
        if os.path.isdir(ddd_dir):
            evaluate_ddd(ddd_dir, limit=args.limit)
        if os.path.isdir(yawn_dir):
            evaluate_yawn(yawn_dir, limit=args.limit)
        if os.path.isdir(video_dir):
            evaluate_video(video_dir, overlap_ratio=args.overlap)
        return

    ran = False
    if args.ddd:
        evaluate_ddd(args.ddd, limit=args.limit); ran = True
    if args.yawn:
        evaluate_yawn(args.yawn, limit=args.limit); ran = True
    if args.video:
        evaluate_video(args.video, overlap_ratio=args.overlap); ran = True

    if not ran:
        parser.print_help()
        print("\nVí dụ chạy nhanh (thử 500 ảnh mỗi lớp):")
        print("  python evaluate.py --ddd ../dataset/driver_drowsiness_dataset --limit 500")


if __name__ == "__main__":
    main()
