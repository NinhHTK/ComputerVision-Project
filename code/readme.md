# Source Code Notes

- `config.py` is the single source of truth for EAR, MAR, head-tilt thresholds,
  and the required continuous duration of each sign.
- `temporal_logic.py` provides `DurationTracker`, shared by the realtime demo
  and video evaluation.
- `drowsiness_detection.py` runs realtime webcam detection.
- `evaluate.py` evaluates DDD, yawn_eye, and annotated videos.
- `prepare_submission_samples.py` creates a deterministic sample set with a
  fixed seed and a SHA-256 manifest.

Do not redefine thresholds inside `drowsiness_detection.py` or `evaluate.py`.
After changing `config.py`, rerun every affected experiment and save the new
configuration with its results.

Standard installation and execution commands are documented in `../README.md`.

