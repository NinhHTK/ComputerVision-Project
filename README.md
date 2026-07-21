# Driver Drowsiness Detection — CPV301

This Computer Vision project detects three signs associated with driver
drowsiness from a webcam or recorded video:

- **Eye closure:** the Eye Aspect Ratio (EAR) remains below a threshold.
- **Yawn:** the Mouth Aspect Ratio (MAR) remains above a threshold.
- **Head tilt:** the angle between the outer eye corners exceeds a roll limit.

MediaPipe Face Mesh extracts facial landmarks. Alert decisions are based on
explainable geometric features and temporal rules; no new deep-learning model
is trained.

## Important: Where to find the reported results

The submission **already includes the CSV results used in the report**. A
reviewer does not need to download the full multi-gigabyte datasets merely to
inspect the recorded experimental results.

| Evidence | Files to inspect | How it should be used |
|---|---|---|
| Final full-dataset image results | `code/results/full/static/ddd_full.csv` and `code/results/full/static/yawn_full_mar060.csv` | Primary static-image results |
| Final video results by subject | `code/results/final_video/video_frame_level_per_subject.csv` and `video_event_level_per_subject.csv` | Primary video results because recording protocols differ by subject |
| Final pooled video results (`N = 3`) | `code/results/final_video/video_frame_level.csv` and `video_event_level.csv` | Supplementary summary; not a controlled cross-subject generalization result |
| Video input and run details | `code/results/final_video/video_per_file.csv` and `video_run_config.csv` | FPS, frame count, no-face rate, thresholds, and timing configuration |
| EAR/MAR configuration experiments | `code/results/experiments/` | Parameter-sensitivity and configuration-selection evidence |
| Submission smoke test | `code/results/sample_smoke_test/` | Reproducibility check on the 300 bundled sample images; not the final scientific result |
| Generated visualizations | `figures/` | Report-ready figures generated directly from the CSV files |

Start with [`code/results/README.md`](code/results/README.md) for metric
definitions, the exact meaning of every result group, configuration A/B/C
mapping, and interpretation cautions. Suggested figure captions and source-CSV
mapping are provided in [`figures/README.md`](figures/README.md).

## 1. Tested environment

- Windows 10/11
- Python `3.11.9`
- OpenCV `4.11.0`
- MediaPipe `0.10.21`
- NumPy `1.26.4`
- Matplotlib `3.11.0`

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
python -c "import cv2, mediapipe, numpy, matplotlib; print(cv2.__version__, mediapipe.__version__, numpy.__version__, matplotlib.__version__)"
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

### Generate report figures

Generate all visualizations from the bundled result CSV files with one command:

```powershell
python .\code\visualize_results.py
```

The seven PNG files are written to the project-root `figures/` directory. The output includes
static confusion matrices, threshold-sensitivity plots, aggregate video
metrics, per-subject event metrics, and video data-quality information.

## 7. Project structure

```text
ComputerVision-Project-main/
├── README.md
├── .gitignore
├── figures/
│   ├── README.md
│   └── fig01_...png through fig07_...png
├── code/
│   ├── config.py
│   ├── temporal_logic.py
│   ├── drowsiness_detection.py
│   ├── evaluate.py
│   ├── prepare_submission_samples.py
│   ├── visualize_results.py
│   ├── requirements.txt
│   └── results/
│       ├── full/static/
│       ├── final_video/
│       ├── experiments/
│       └── sample_smoke_test/
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
