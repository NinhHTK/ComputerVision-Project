# Results Directory

## full/static

Full external-dataset results reported in the paper.

- `ddd_full.csv`: complete DDD evaluation.
- `yawn_full_mar060.csv`: complete yawn_eye evaluation with MAR=0.60.

## sample_smoke_test

Results produced from the 300 bundled sample images. These results are
only for pipeline verification and are not the full-dataset results.

## experiments/ddd_quick_test.csv

A quick test using at most 500 images per DDD class. This test is not
used as the primary reported result.

## experiments/ear_threshold_sweep

Single-subject EAR threshold and temporal-duration experiments conducted
using Bình's videos.

## experiments/mar_threshold_sweep/static

MAR threshold experiments on the complete yawn_eye dataset.

## final_video

Reserved for final per-subject video results generated after correcting
the timing logic.

## Archived results

Results generated before the timing correction or from accidentally
duplicated video inputs are retained locally for audit purposes but are
not included as final results.