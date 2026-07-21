# Result Files and Interpretation Guide

This directory contains the numerical evidence used in the report. The CSV
files are preserved so that every table and figure can be traced to a recorded
experiment. Report-ready PNG files are generated separately in the project-root
`figures/` directory by `code/visualize_results.py`.

## Which results should be reported?

Use the result groups in this order:

1. **Primary static-image results:** `full/static/`.
2. **Primary video results:** the per-subject files in `final_video/`.
3. **Supplementary video results:** the aggregate files in `final_video/`.
4. **Parameter-selection evidence:** `experiments/`.
5. **Reproducibility check only:** `sample_smoke_test/`.

The three subjects did not follow one uniform recording protocol. Their videos
differ in FPS, resolution, camera placement, behavior, and annotation style.
For this reason, the per-subject video results are primary. The pooled `N = 3`
results are useful as a summary but must not be presented as evidence from a
controlled three-subject study.

## Metric definitions

### Static-image and frame-level files

- `TP`: positive samples correctly detected.
- `TN`: negative samples correctly rejected.
- `FP`: negative samples incorrectly detected as positive.
- `FN`: positive samples missed by the system.
- `accuracy`: `(TP + TN) / (TP + TN + FP + FN)`.
- `precision`: `TP / (TP + FP)`; how reliable a positive prediction is.
- `recall`: `TP / (TP + FN)`; how many true positive cases are detected.
- `f1`: harmonic mean of precision and recall.
- `processed`: samples included in the confusion matrix.
- `no_face`: samples excluded because MediaPipe did not detect a face.
- `no_face_rate_percent`: percentage of inputs excluded for that reason.
- `seconds`: total evaluation time.

### Event-level video files

- `true_events`: annotated ground-truth events.
- `hit`: annotated events overlapped by at least one predicted event.
- `miss`: annotated events with no matching prediction.
- `false_alarm`: predicted events with no matching annotation.
- `recall`: `hit / true_events`.
- `precision`: `hit / (hit + false_alarm)`.

An empty metric means that its denominator is zero. For example, event
precision is undefined when the system produces no event. Generated figures
display such cells as `N/A`; they must not be interpreted as zero.

## `full/static/` — final external-dataset evaluation

- `ddd_full.csv`: final evaluation on the complete processed DDD dataset.
- `yawn_full_mar060.csv`: final MAR-only evaluation on yawn_eye with
  `MAR_THRESHOLD = 0.60`.

The two files evaluate different positive signs and should not be merged into
one overall score. In the report, use their confusion matrices and metrics as
separate experiments.

Key interpretation:

- DDD has high precision but considerably lower recall. The system is
  conservative: a positive drowsy prediction is often correct, but many
  drowsy images are missed.
- The yawn_eye experiment has no false positives at the selected threshold,
  but it still misses some yawning images. This is a precision/recall trade-off,
  not evidence of perfect yawn detection.
- Images with no detected face are reported separately and are excluded from
  the confusion-matrix denominator.

## `final_video/` — final time-corrected video evaluation

These files were produced after replacing frame-count timing with
seconds-based temporal logic:

- `video_frame_level_per_subject.csv`: frame metrics for each subject and sign;
  use this for detailed error analysis.
- `video_event_level_per_subject.csv`: event precision and recall for each
  subject and sign; this is the clearest primary video result.
- `video_frame_level.csv`: pooled frame metrics across all three subjects;
  supplementary only.
- `video_event_level.csv`: pooled event metrics across all three subjects;
  supplementary only.
- `video_per_file.csv`: subject, video name, frame count, FPS, and no-face rate;
  use this to document input heterogeneity and face-detection coverage.
- `video_run_config.csv`: exact thresholds, continuous durations, and event
  overlap rule used for the final run.

The aggregate event files answer “how did the system perform after pooling all
annotations and predictions?” They do not answer “how well does the system
generalize to an average new person?” A larger, standardized subject sample
would be required for the latter claim.

## `experiments/` — parameter selection

### `ear_threshold_sweep/`

`Baseline`, `A`, `B`, and `C` are internal configurations. They are not
baseline methods copied from related work.

| Configuration | EAR threshold | Reference frames at 30 FPS | Equivalent duration | Purpose |
|---|---:|---:|---:|---|
| Baseline | 0.21 | 20 | 0.67 seconds | Initial configuration |
| A | 0.18 | 20 | 0.67 seconds | Change only the EAR threshold |
| B | 0.21 | 30 | 1.00 second | Increase only duration; selected |
| C | 0.19 | 25 | 0.83 seconds | Change both EAR and duration |

The `video_*_baseline.csv`, `video_*_earA.csv`, `video_*_earB.csv`, and
`video_*_earC.csv` files come from Binh's video pair. Configuration B was
selected as a practical trade-off: it reduced eye-closure false alarms from 10
to 6 while preserving event recall in this configuration recording. Its
frame-level F1 was not the largest, so the report should describe the selection
as a false-alarm/duration trade-off rather than an absolute optimum.

“Reference frames” document the earlier 30-FPS implementation. Final video
results use `EAR_THRESHOLD = 0.21` and `EAR_CONSEC_SECONDS = 1.0` through the
seconds-based `DurationTracker`. Because the sweep used one subject, it is a
sensitivity experiment and must not be generalized to all three subjects.

### `mar_threshold_sweep/static/`

The files evaluate `MAR_THRESHOLD` values 0.45, 0.50, 0.55, and 0.60 on the
full yawn_eye dataset. Increasing the threshold reduces recall while precision
remains high. The final threshold 0.60 favors fewer false positives; its final
result is copied to `full/static/yawn_full_mar060.csv`.

### `ddd_quick_test.csv`

This is a development-time check with at most 500 DDD images per class. It is
not the final DDD result and should not replace `full/static/ddd_full.csv`.

## `sample_smoke_test/` — submission reproducibility check

These files were generated from the 300 sample images included in the
submission. They demonstrate that a reviewer can install the dependencies and
run the pipeline without downloading the multi-gigabyte datasets. They are not
used as the main scientific results because the sample subset is small.

## Generate all figures

From the project root, run:

```powershell
py -3.11 code\visualize_results.py
```

The command reads the CSV files without changing them and writes seven PNG
files to the project-root `figures/` directory. It works regardless of the
terminal's current directory. Optional overrides:

```powershell
py -3.11 code\visualize_results.py --output-dir .\my_figures --dpi 300
```

Do not edit generated PNG files by hand. Update the source CSV or visualization
script and regenerate them so the numerical evidence remains traceable.

## Locally archived results

Pre-timing-fix results and results generated from accidentally duplicated video
inputs may be retained locally for auditing, but they are excluded from the
final results and submission package.
