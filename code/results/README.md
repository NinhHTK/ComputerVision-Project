# Results Directory

## `full/static/`

Results on the complete external image datasets used in the report:

- `ddd_full.csv`: complete DDD system evaluation.
- `yawn_full_mar060.csv`: MAR-only yawn_eye evaluation at threshold 0.60.

## `sample_smoke_test/`

Results on the 300 images bundled with the submission. These files verify the
pipeline and do not replace the full-dataset results.

## `experiments/`

- `ddd_quick_test.csv`: quick test using at most 500 DDD images per class.
- `ear_threshold_sweep/`: EAR threshold/duration experiments on Binh's videos
  during configuration selection.
- `mar_threshold_sweep/static/`: MAR threshold experiments on the full
  yawn_eye dataset.

### `ear_threshold_sweep/` configuration

`Baseline`, `A`, `B`, and `C` are internal configurations used to study
sensitivity to the EAR threshold and continuous eye-closure duration. They are
not baseline methods taken from related work.

| Configuration | EAR threshold | Reference frames at 30 FPS | Equivalent duration | Purpose |
|---|---:|---:|---:|---|
| Baseline | 0.21 | 20 | 0.67 seconds | Initial configuration |
| A | 0.18 | 20 | 0.67 seconds | Tighten only the EAR threshold |
| B | 0.21 | 30 | 1.00 second | Increase only duration; selected configuration |
| C | 0.19 | 25 | 0.83 seconds | Change both EAR and duration |

The `video_*_baseline.csv`, `video_*_earA.csv`, `video_*_earB.csv`, and
`video_*_earC.csv` files were generated from Binh's video pair during
configuration selection. “Reference frames” describe the legacy design at
30 FPS; equivalent duration is the correct interpretation across videos with
different FPS values.

Final results in `final_video/` were rerun with the seconds-based
`DurationTracker` and configuration B: `EAR_THRESHOLD=0.21` and
`EAR_CONSEC_SECONDS=1.0`. The one-subject sweep must not be interpreted as a
general result for all three subjects.

### `mar_threshold_sweep/static/` configuration

The files correspond to `MAR_THRESHOLD` values 0.45, 0.50, 0.55, and 0.60. The
final configuration uses `MAR_THRESHOLD=0.60`; its complete yawn_eye result is
stored in `full/static/yawn_full_mar060.csv`.

## `final_video/`

Final video results after converting temporal logic to seconds:

- `video_frame_level.csv`: aggregate frame-level metrics.
- `video_event_level.csv`: aggregate event-level metrics.
- `video_frame_level_per_subject.csv`: frame-level metrics by subject.
- `video_event_level_per_subject.csv`: event-level metrics by subject.
- `video_per_file.csv`: FPS, frame count, and no-face rate for each video.
- `video_run_config.csv`: thresholds, continuous durations, and event overlap.

Per-subject results are primary because the three recording protocols were not
uniform. Aggregate metrics are supplementary.

## Locally archived results

Pre-timing-fix results and results generated from accidentally duplicated video
inputs are retained locally for audit purposes but excluded from the final
results and submission package.

