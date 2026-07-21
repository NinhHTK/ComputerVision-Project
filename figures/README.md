# Generated Figures and Suggested Captions

Run `py -3.11 code\visualize_results.py` from the project root to regenerate
all PNG files in this directory.

The captions below state what each figure can support without overstating the
evidence. Copy and adapt them in the report.

| File | Suggested caption | Source CSV files |
|---|---|---|
| `fig01_static_confusion_matrices.png` | **Row-normalized confusion matrices on the full processed DDD and yawn_eye datasets.** Counts are shown with percentages normalized within each actual class; images without a detected face are excluded and reported separately. | `code/results/full/static/*.csv` |
| `fig02_static_metrics.png` | **Static-image accuracy, precision, recall, and F1.** DDD drowsiness and yawn_eye yawning are separate tasks, so the bars summarize two experiments rather than a single combined benchmark. | `code/results/full/static/*.csv` |
| `fig03_mar_threshold_sensitivity.png` | **Sensitivity of yawn_eye performance to the MAR threshold.** Raising the threshold from 0.45 to 0.60 reduces recall while maintaining precision; 0.60 is the selected conservative configuration. | `code/results/experiments/mar_threshold_sweep/static/*.csv` |
| `fig04_ear_configuration_comparison.png` | **Eye-closure configuration comparison on one subject.** Configuration B was selected as a false-alarm/duration trade-off. This one-subject sensitivity experiment is not a general three-subject result. | `code/results/experiments/ear_threshold_sweep/*.csv` |
| `fig05_video_aggregate_metrics.png` | **Supplementary pooled frame- and event-level video metrics for three subjects.** These values summarize pooled observations from non-uniform recording protocols and should be interpreted with the per-subject results. | `code/results/final_video/video_frame_level.csv`; `video_event_level.csv` |
| `fig06_video_event_metrics_per_subject.png` | **Primary per-subject event precision and recall.** Differences across subjects reflect both system behavior and non-uniform recording/annotation protocols. `N/A` denotes an undefined or unsupported case, not a zero score. | `code/results/final_video/video_event_level_per_subject.csv` |
| `fig07_video_data_quality.png` | **Frame count, FPS, and no-face rate for the six submitted videos.** The figure documents variation in input volume and face-detection coverage. | `code/results/final_video/video_per_file.csv` |

Recommended report order: use Figures 1, 3, 4, and 6 in the main experiment
section. Figures 2, 5, and 7 may be used when space allows or moved to an
appendix. Figure 6 should remain the primary video comparison because the three
recording protocols were not standardized.

`N/A` in the per-subject heatmap means that a metric is undefined or the case
is unsupported; it must not be interpreted as a score of zero.
