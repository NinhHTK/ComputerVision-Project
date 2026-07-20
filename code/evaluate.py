"""
===================================================================
CPV301 COMPREHENSIVE EVALUATION SCRIPT (Driver Drowsiness Detection)
-------------------------------------------------------------------
Evaluate the geometric system (EAR + MAR + Head Tilt / MediaPipe Face Mesh)
on three data sources:

  1. DDD static images (drowsy/non_drowsy) -> complete-system evaluation
  2. yawn_eye static images (yawn/no_yawn) -> MAR-only evaluation
  3. Self-recorded video + CSV labels      -> temporal evaluation
        (a) frame-level   (b) event-level

METHODOLOGY NOTES (state these explicitly in the report):
  - STATIC IMAGES have no concept of N consecutive frames. Each image's
    EAR/MAR/tilt values are compared directly with the thresholds.
  - VIDEO uses the same second-based thresholds and DurationTracker as the
    realtime application, so results are independent of source FPS.
  - Images for which MediaPipe cannot detect a face are excluded from metric
    calculation, and the exclusion rate is reported transparently.

Quick example:
    py evaluate.py --ddd ../dataset/driver_drowsiness_dataset --limit 500

Usage:
    py evaluate.py --ddd    ../dataset/driver_drowsiness_dataset
    py evaluate.py --yawn   ../dataset/yawn_eye_dataset_new
    py evaluate.py --video  ../dataset/video_submission
    py evaluate.py --all    ../dataset          # run every available source

Results are printed to the terminal and saved under the selected --output-dir.
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
# 2. LANDMARK INDICES (shared with the realtime application)
# ==================================================================
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]           # [left corner, right corner, upper lip, lower lip]
LEFT_EYE_CORNER  = 33
RIGHT_EYE_CORNER = 263

# Accepted image extensions
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG")


# ==================================================================
# 3. GEOMETRIC FUNCTIONS (identical to the realtime implementation)
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
# 4. CORE EXTRACTION: image/frame -> (ear, mar, tilt), or None if no face
# ==================================================================
def extract_metrics(image_bgr, face_mesh):
    """
    Accept an OpenCV BGR image and return {ear, mar, tilt}, or None when
    MediaPipe cannot detect a face.
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
    static_image_mode=True processes every static image independently.
    False enables inter-frame tracking for video, matching the realtime demo.
    """
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ==================================================================
# 5. CLASSIFICATION METRICS FROM A CONFUSION MATRIX
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
    """Write a list of row dictionaries to RESULTS_DIR/<filename>."""
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  -> Saved: {path}")


# ==================================================================
# 6. DDD STATIC-IMAGE EVALUATION (complete system)
# ==================================================================
def evaluate_ddd(ddd_dir, limit=None):
    """
    ddd_dir contains drowsy/ and non_drowsy/ subdirectories.
    An image is predicted as drowsy when ANY sign crosses its threshold:
        EAR below threshold OR MAR above threshold OR tilt above threshold.
    Static images do not use continuous-frame logic.

    limit: maximum images per class for a quick run. None processes all images.
    """
    print("\n########## DDD EVALUATION (complete system) ##########")
    face_mesh = make_face_mesh(static=True)

    # Ground truth: drowsy = positive (1), non_drowsy = negative (0)
    classes = {"drowsy": 1, "non_drowsy": 0}

    tp = tn = fp = fn = 0
    no_face = 0            # images for which no face was detected
    processed = 0
    t0 = time.time()

    for cls_name, y_true in classes.items():
        cls_dir = os.path.join(ddd_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  [WARNING] Directory not found: {cls_dir}; skipping")
            continue

        files = sorted(f for f in glob.glob(os.path.join(cls_dir, "*"))
                       if f.endswith(IMG_EXTS))
        if limit:
            files = files[:limit]
        print(f"  Processing class '{cls_name}': {len(files)} images...")

        for i, fpath in enumerate(files):
            img = cv2.imread(fpath)
            if img is None:
                continue
            metrics = extract_metrics(img, face_mesh)
            if metrics is None:
                no_face += 1
                continue

            processed += 1
            # Predict drowsy if any sign crosses its threshold
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
                print(f"    ...{i+1} images ({cls_name})")

    face_mesh.close()
    elapsed = time.time() - t0

    total_seen = processed + no_face
    no_face_rate = (no_face / total_seen * 100) if total_seen else 0.0

    m = compute_metrics(tp, tn, fp, fn)
    print_metrics("DDD RESULTS (complete system)", m, extra={
        "Images with a detected face": processed,
        "Images with no detected face": f"{no_face} ({no_face_rate:.2f}%)",
        "Elapsed time": f"{elapsed:.1f}s",
    })

    # Save CSV
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
# 7. yawn_eye STATIC-IMAGE EVALUATION (MAR only)
# ==================================================================
def evaluate_yawn(yawn_dir, splits=("train", "test"), limit=None):
    """
    Structure: yawn_dir/<split>/yawn/*.jpg and .../no_yawn/*.jpg.
    Evaluate MAR only: predict yawn if MAR is above the threshold.
    Train and test folders are combined because this is a geometric evaluation;
    no model is trained, so no learned-model split is required.
    """
    print("\n########## yawn_eye EVALUATION (MAR only) ##########")
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
                print(f"  [WARNING] Directory not found: {cls_dir}; skipping")
                continue

            files = sorted(f for f in glob.glob(os.path.join(cls_dir, "*"))
                           if f.endswith(IMG_EXTS))
            if limit:
                files = files[:limit]
            print(f"  '{split}/{cls_name}': {len(files)} images...")

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
    print_metrics("yawn_eye RESULTS (MAR only)", m, extra={
        "Processed images": processed,
        "Images with no detected face": f"{no_face} ({no_face_rate:.2f}%)",
        "Elapsed time": f"{elapsed:.1f}s",
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
# 8. TEMPORAL VIDEO EVALUATION
# ==================================================================
# CSV event names used by the detector
EVENT_TYPES = ["eye_closed", "yawn", "head_tilt"]


def load_ground_truth(csv_path):
    """
    Read a (start,end,event) annotation CSV into a list of dictionaries.
    Validate start < end and the allowed event names.
    """
    events = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # data starts at line 2
            try:
                start = float(row["start"])
                end = float(row["end"])
                ev = row["event"].strip()
            except (KeyError, ValueError):
                print(f"    [CSV ERROR] {csv_path}, line {i}: invalid format; skipping")
                continue
            if ev not in EVENT_TYPES:
                print(f"    [CSV ERROR] {csv_path}, line {i}: event '{ev}' "
                      f"is invalid (allowed: {EVENT_TYPES}); skipping")
                continue
            if start >= end:
                print(f"    [CSV ERROR] {csv_path}, line {i}: start>=end; skipping")
                continue
            events.append({"start": start, "end": end, "event": ev})
    return events


def find_csv_for_video(video_path):
    """Find a same-stem CSV file (An_drowsy.mp4 -> An_drowsy.csv)."""
    base = os.path.splitext(video_path)[0]
    csv_path = base + ".csv"
    return csv_path if os.path.isfile(csv_path) else None


def subject_from_video_name(video_name):
    """Map Binh_alert.mp4 or Binh_drowsy.mp4 to subject Binh."""
    stem = os.path.splitext(os.path.basename(video_name))[0]
    for suffix in ("_alert", "_drowsy"):
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def run_system_on_video(video_path):
    """
    Run the same DurationTracker and second-based thresholds as the demo.
    Each frame contributes 1/fps seconds to a continuous sequence. Return:
      - source video FPS
      - per_frame: dictionaries indicating whether each sign is currently
        active after its duration threshold
      - no_face_frames: number of frames with no detected face
      - total_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Unable to open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback when FPS metadata cannot be read

    # Frame counts below are for display only; duration is the source of truth.
    ear_frames = max(1, math.ceil(EAR_CONSEC_SECONDS * fps - 1e-9))
    mar_frames = max(1, math.ceil(MAR_CONSEC_SECONDS * fps - 1e-9))
    tilt_frames = max(1, math.ceil(TILT_CONSEC_SECONDS * fps - 1e-9))
    print(f"      Thresholds at fps={fps:.1f}: "
          f"eye>={ear_frames}f ({EAR_CONSEC_SECONDS:.2f}s) | "
          f"yawn>={mar_frames}f ({MAR_CONSEC_SECONDS:.2f}s) | "
          f"tilt>={tilt_frames}f ({TILT_CONSEC_SECONDS:.2f}s)")

    # static=False enables inter-frame tracking, matching the realtime demo
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
            # Losing the face interrupts every continuous-duration sequence.
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
    Convert second-based events into per-frame labels.
    Return one set of active ground-truth events for each frame.
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
    Evaluate each sign at frame level.
    Return {event_type: metrics_dict}.
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
    Evaluate each sign at event level.
    - A ground-truth CSV event is a hit when the corresponding alert is active
      for at least overlap_ratio of its frames (30% by default); otherwise it
      is a miss.
    - A false alarm is a contiguous predicted cluster that does not overlap any
      ground-truth event of the same type.

    Return {event_type: {"hit", "miss", "false_alarm", "recall", ...}}.
    """
    total_frames = len(per_frame)
    results = {}

    for ev_type in EVENT_TYPES:
        # Frames in which this sign is active
        pred_frames = [1 if per_frame[i][ev_type] else 0
                       for i in range(total_frames)]

        # --- HIT / MISS for each ground-truth event ---
        hit = 0
        miss = 0
        true_events = [e for e in events if e["event"] == ev_type]
        # Mark frames covered by a ground-truth event
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

        # --- FALSE ALARM: predicted clusters outside ground-truth events ---
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
        # Event precision: hit / (hit + false_alarm)
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
    Process every supported video with a matching CSV annotation file.
    Evaluate frame-level and event-level metrics, aggregate the results, and
    save CSV outputs.
    """
    print("\n########## TEMPORAL VIDEO EVALUATION ##########")

    video_files = set()
    for ext in ("*.mp4", "*.MP4", "*.avi", "*.mov"):
        video_files.update(glob.glob(os.path.join(video_dir, ext)))
    video_files = sorted(video_files)

    if not video_files:
        print(f"  [INFO] No video files found in {video_dir}.")
        print("  -> Add same-stem video and CSV annotation files, then run again.")
        return

    # Aggregate frame-level confusion counts across all videos
    fl_accum = {ev: {"TP":0,"TN":0,"FP":0,"FN":0} for ev in EVENT_TYPES}
    # Aggregate event-level counts
    el_accum = {ev: {"true_events":0,"hit":0,"miss":0,"false_alarm":0}
                for ev in EVENT_TYPES}

    # Aggregate by subject so long videos do not dominate without visibility.
    subject_fl = defaultdict(
        lambda: {ev: {"TP":0,"TN":0,"FP":0,"FN":0}
                 for ev in EVENT_TYPES})
    subject_el = defaultdict(
        lambda: {ev: {"true_events":0,"hit":0,"miss":0,"false_alarm":0}
                 for ev in EVENT_TYPES})

    per_video_rows = []   # processing metadata for each video

    for vpath in sorted(video_files):
        vname = os.path.basename(vpath)
        subject = subject_from_video_name(vname)
        csv_path = find_csv_for_video(vpath)
        if csv_path is None:
            print(f"  [SKIP] {vname}: no matching annotation CSV file.")
            continue

        print(f"\n  >>> Video: {vname}")
        events = load_ground_truth(csv_path)
        print(f"      Labels: {len(events)} events from {os.path.basename(csv_path)}")

        run = run_system_on_video(vpath)
        if run is None:
            continue

        fps = run["fps"]
        per_frame = run["per_frame"]
        total_frames = run["total_frames"]
        nf = run["no_face_frames"]
        nf_rate = (nf / total_frames * 100) if total_frames else 0.0
        print(f"      {total_frames} frame @ {fps:.1f} fps | "
              f"no face: {nf} frames ({nf_rate:.1f}%)")

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

        # Print a concise result for this video
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

    # ============ AGGREGATE FRAME-LEVEL RESULTS ============
    print("\n" + "#"*55)
    print("  AGGREGATE FRAME-LEVEL RESULTS (all videos)")
    print("#"*55)
    fl_rows = []
    for ev in EVENT_TYPES:
        a = fl_accum[ev]
        m = compute_metrics(a["TP"], a["TN"], a["FP"], a["FN"])
        print_metrics(f"[frame-level] {ev}", m)
        row = dict(m); row["event_type"] = ev; row["level"] = "frame"
        fl_rows.append(row)

    # ============ AGGREGATE EVENT-LEVEL RESULTS ============
    print("\n" + "#"*55)
    print("  AGGREGATE EVENT-LEVEL RESULTS (all videos)")
    print("#"*55)
    el_rows = []
    for ev in EVENT_TYPES:
        a = el_accum[ev]
        n_true = a["true_events"]
        recall = a["hit"]/n_true if n_true else 0.0
        precision = (a["hit"]/(a["hit"]+a["false_alarm"])
                     if (a["hit"]+a["false_alarm"]) else 0.0)
        print(f"\n  [event-level] {ev}")
        print(f"    Ground-truth events: {n_true}")
        print(f"    Hits               : {a['hit']}")
        print(f"    Misses             : {a['miss']}")
        print(f"    False alarms       : {a['false_alarm']}")
        print(f"    Recall       : {recall:.4f}")
        print(f"    Precision    : {precision:.4f}")
        el_rows.append({
            "event_type": ev, "level": "event",
            "true_events": n_true, "hit": a["hit"], "miss": a["miss"],
            "false_alarm": a["false_alarm"],
            "recall": round(recall,4), "precision": round(precision,4),
        })

    # ============ SAVE CSV OUTPUTS ============
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

    # ============ PER-SUBJECT RESULTS ============
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
        description="CPV301 comprehensive Driver Drowsiness Detection evaluation")
    parser.add_argument("--ddd", metavar="DIR",
                        help="DDD directory containing drowsy/ and non_drowsy/")
    parser.add_argument("--yawn", metavar="DIR",
                        help="Path to yawn_eye_dataset_new")
    parser.add_argument("--video", metavar="DIR",
                        help="Directory containing videos and annotation CSVs")
    parser.add_argument("--all", metavar="DATASET_DIR",
                        help="Path to dataset/; run every available source")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum images per class for a quick run")
    parser.add_argument("--overlap", type=float, default=0.3,
                        help="Minimum overlap ratio for an event-level hit "
                             "(default: 0.3)")
    parser.add_argument("--output-dir", default="results", metavar="DIR",
                        help="CSV output directory (default: ./results)")
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
        print("\nQuick example (up to 500 images per class):")
        print("  python evaluate.py --ddd ../dataset/driver_drowsiness_dataset --limit 500")


if __name__ == "__main__":
    main()
