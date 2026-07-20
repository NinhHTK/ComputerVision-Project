# Driver Drowsiness Detection — CPV301

This Computer Vision project detects three signs associated with driver
drowsiness from a webcam or recorded video:

- **Eye closure:** the Eye Aspect Ratio (EAR) remains below a threshold.
- **Yawn:** the Mouth Aspect Ratio (MAR) remains above a threshold.
- **Head tilt:** the angle between the outer eye corners exceeds a roll limit.

MediaPipe Face Mesh extracts facial landmarks. Alert decisions are based on
explainable geometric features and temporal rules; no new deep-learning model
is trained.

## 1. Tested environment

- Windows 10/11
- Python `3.11.9`
- OpenCV `4.11.0`
- MediaPipe `0.10.21`
- NumPy `1.26.4`

The project does not bundle a Python environment. All direct dependencies are
declared in `code/requirements.txt`.

> Prefer a project path containing only ASCII characters. If MediaPipe raises a
> `FileNotFoundError` related to `binarypb`, move the project to a path such as
> `D:\CPV301\ComputerVision-Project`.

## 2. Installation on Windows

Open PowerShell in the project root:

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\code\requirements.txt
```

Verify the environment:

```powershell
python -m pip check
python -c "import cv2, mediapipe, numpy; print(cv2.__version__, mediapipe.__version__, numpy.__version__)"
```

The expected `pip check` output is `No broken requirements found.`

If PowerShell blocks environment activation, call the interpreter directly:

```powershell
.\venv311\Scripts\python.exe -m pip install -r .\code\requirements.txt
```

## 3. Run the realtime application

From the project root:

```powershell
python .\code\drowsiness_detection.py
```

The application uses camera index `0`. Press **Q** to exit.

Default configuration:

| Sign | Geometric threshold | Required continuous duration |
|---|---:|---:|
| Eye closure | EAR < 0.21 | 1.00 second |
| Yawn | MAR > 0.60 | 0.50 second |
| Head tilt | angle > 15° | 0.67 second |

Thresholds are defined centrally in `code/config.py`. The realtime demo and
video evaluation share `code/temporal_logic.py`, so alert duration is not tied
to the input FPS.

## 4. Bundled data

`dataset/samples/` contains a runnable sample set:

- DDD: 50 `drowsy` and 50 `non_drowsy` images.
- yawn_eye: 50 images for each `yawn`/`no_yawn` class in each `train`/`test`
  split (200 images total).
- `sample_manifest.csv`: relative paths and SHA-256 hashes for all 300 samples.

`dataset/video_submission/` contains six compressed videos from three subjects
and six matching annotation files. Annotation CSV files use this format:

```csv
start,end,event
3.0,6.0,eye_closed
```

`event` must be one of `eye_closed`, `yawn`, or `head_tilt`.

### Full datasets

The full datasets are excluded from the submission because of the upload-size
limit. Download them from:

- [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- [Yawn Eye Dataset New](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new)

Extract them into this structure:

```text
dataset/
├── driver_drowsiness_dataset/
│   ├── drowsy/
│   └── non_drowsy/
└── yawn_eye_dataset_new/
    ├── train/
    │   ├── yawn/
    │   └── no_yawn/
    └── test/
        ├── yawn/
        └── no_yawn/
```

## 5. Run evaluation

Run all commands below from the project root. `--output-dir` selects where CSV
outputs are written and prevents the reported results from being overwritten.

### Smoke test on the 300 bundled images

```powershell
python .\code\evaluate.py --ddd .\dataset\samples\DDD --output-dir .\run_results\ddd_sample
python .\code\evaluate.py --yawn .\dataset\samples\yawn_eye --output-dir .\run_results\yawn_sample
```

Reference smoke-test results:

| Source | Processed | No face | Accuracy | F1 |
|---|---:|---:|---:|---:|
| DDD sample | 100 | 0 | 0.7300 | 0.6494 |
| yawn_eye sample | 199 | 1 | 0.8442 | 0.8166 |

### Self-recorded video

```powershell
python .\code\evaluate.py --video .\dataset\video_submission --output-dir .\run_results\video
```

Video evaluation exports aggregate and per-subject frame-level/event-level
metrics, per-file processing metadata, and the run configuration.

### Full datasets

```powershell
# Quick check: at most 500 DDD images per class
python .\code\evaluate.py --ddd .\dataset\driver_drowsiness_dataset --limit 500 --output-dir .\run_results\ddd_quick

# Evaluate both full static-image datasets
python .\code\evaluate.py --ddd .\dataset\driver_drowsiness_dataset --output-dir .\run_results\ddd_full
python .\code\evaluate.py --yawn .\dataset\yawn_eye_dataset_new --output-dir .\run_results\yawn_full

# Run every available source under dataset/
python .\code\evaluate.py --all .\dataset --output-dir .\run_results\all
```

Images for which MediaPipe cannot detect a face are reported separately and
excluded from the confusion matrix. Static images are compared directly with
the geometric thresholds; continuous-duration rules apply only to realtime and
video evaluation.

## 6. Bundled results

`code/results/` is organized as follows:

- `full/static/`: results on the complete external image datasets.
- `sample_smoke_test/`: pipeline checks using the 300 bundled images.
- `experiments/`: quick checks and threshold-selection experiments.
- `final_video/`: final time-corrected video results, including aggregate,
  per-subject, per-file, and run-configuration outputs.

See `code/results/README.md` for details.

## 7. Project structure

```text
ComputerVision-Project-main/
├── README.md
├── .gitignore
├── code/
│   ├── config.py
│   ├── temporal_logic.py
│   ├── drowsiness_detection.py
│   ├── evaluate.py
│   ├── prepare_submission_samples.py
│   ├── requirements.txt
│   └── results/
└── dataset/
    ├── samples/
    └── video_submission/
```

Python environments (`.venv/`, `venv311/`), full datasets, original videos,
and Python caches are excluded from the submission.

## 8. Known limitations

- Head tilt estimates only **roll** from the outer eye corners. Forward head
  movement (**pitch**) may not be detected.
- The three subjects used different FPS values, resolutions, camera angles, and
  recording/annotation procedures. Per-subject results are therefore primary;
  pooled results must be interpreted cautiously.
- Fixed geometric thresholds may not generalize to new faces, cameras, or
  lighting conditions.

