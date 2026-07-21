"""Generate the report figures from the CSV files in ``code/results``.

Run from the project root:

    py -3.11 code\visualize_results.py

By default, every PNG is written to ``figures`` in the project root. Paths are
resolved relative to this file, so the command also works from another current
working directory.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"

BLUE = "#2E74B5"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#7A5195"
GRAY = "#6B7280"
GRID = "#D1D5DB"


def read_rows(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    """Read a CSV file and validate the columns used by a figure."""
    if not path.is_file():
        raise FileNotFoundError(f"Required result file was not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = set(required_columns) - columns
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing columns: {missing_text}")
        return list(reader)


def number(value: str | None) -> float:
    """Convert a CSV value to float; empty and invalid values become NaN."""
    if value is None or not value.strip():
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def integer(value: str | None) -> int:
    value_as_float = number(value)
    return 0 if math.isnan(value_as_float) else int(value_as_float)


def event_label(value: str) -> str:
    return {
        "eye_closed": "Eye closure",
        "yawn": "Yawn",
        "head_tilt": "Head tilt",
    }.get(value, value.replace("_", " ").title())


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def add_percent_labels(ax: plt.Axes, bars, digits: int = 1) -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            f"{height * 100:.{digits}f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def confusion_matrix_figure(results_dir: Path, output_dir: Path, dpi: int) -> None:
    sources = [
        (
            results_dir / "full" / "static" / "ddd_full.csv",
            "DDD",
            ("Non-drowsy", "Drowsy"),
        ),
        (
            results_dir / "full" / "static" / "yawn_full_mar060.csv",
            "yawn_eye (MAR = 0.60)",
            ("No yawn", "Yawn"),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    last_image = None
    for ax, (path, title, labels) in zip(axes, sources):
        row = read_rows(path, ["TP", "TN", "FP", "FN"])[0]
        matrix = np.array(
            [
                [integer(row["TN"]), integer(row["FP"])],
                [integer(row["FN"]), integer(row["TP"])],
            ],
            dtype=int,
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        last_image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for y in range(2):
            for x in range(2):
                value = normalized[y, x]
                color = "white" if value >= 0.58 else "#111827"
                ax.text(
                    x,
                    y,
                    f"{matrix[y, x]:,}\n({value * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.grid(False)

    fig.colorbar(last_image, ax=axes, label="Row-normalized proportion", shrink=0.82)
    save_figure(fig, output_dir / "fig01_static_confusion_matrices.png", dpi)


def static_metrics_figure(results_dir: Path, output_dir: Path, dpi: int) -> None:
    sources = [
        (results_dir / "full" / "static" / "ddd_full.csv", "DDD"),
        (
            results_dir / "full" / "static" / "yawn_full_mar060.csv",
            "yawn_eye",
        ),
    ]
    metrics = ["accuracy", "precision", "recall", "f1"]
    display_names = ["Accuracy", "Precision", "Recall", "F1"]
    values = []
    for path, _ in sources:
        row = read_rows(path, metrics)[0]
        values.append([number(row[key]) for key in metrics])

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(display_names))
    width = 0.34
    bars_ddd = ax.bar(x - width / 2, values[0], width, label=sources[0][1], color=BLUE)
    bars_yawn = ax.bar(x + width / 2, values[1], width, label=sources[1][1], color=ORANGE)
    add_percent_labels(ax, bars_ddd)
    add_percent_labels(ax, bars_yawn)
    ax.set_ylabel("Score")
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=2, loc="upper center")
    save_figure(fig, output_dir / "fig02_static_metrics.png", dpi)


def mar_sensitivity_figure(results_dir: Path, output_dir: Path, dpi: int) -> None:
    sweep_dir = results_dir / "experiments" / "mar_threshold_sweep" / "static"
    files = [
        (0.45, sweep_dir / "yawn_result_mar045.csv"),
        (0.50, sweep_dir / "yawn_result_mar05.csv"),
        (0.55, sweep_dir / "yawn_result_mar055.csv"),
        (0.60, sweep_dir / "yawn_result_mar06.csv"),
    ]
    metric_specs = [
        ("accuracy", "Accuracy", BLUE, "o"),
        ("recall", "Recall", ORANGE, "s"),
        ("f1", "F1", GREEN, "^"),
        ("precision", "Precision", GRAY, "D"),
    ]
    values: dict[str, list[float]] = {key: [] for key, *_ in metric_specs}
    for _, path in files:
        row = read_rows(path, values.keys())[0]
        for key in values:
            values[key].append(number(row[key]))

    thresholds = [threshold for threshold, _ in files]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for key, label, color, marker in metric_specs:
        ax.plot(
            thresholds,
            values[key],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )
    ax.axvline(0.60, color=RED, linestyle="--", linewidth=1.4, label="Selected: 0.60")
    ax.set_xlabel("MAR threshold")
    ax.set_ylabel("Score")
    ax.set_xticks(thresholds)
    ax.set_ylim(0.60, 1.035)
    ax.legend(ncol=3, loc="lower left")
    save_figure(fig, output_dir / "fig03_mar_threshold_sensitivity.png", dpi)


def ear_configuration_figure(results_dir: Path, output_dir: Path, dpi: int) -> None:
    sweep_dir = results_dir / "experiments" / "ear_threshold_sweep"
    configs = [
        ("Baseline", "baseline"),
        ("A", "earA"),
        ("B*", "earB"),
        ("C", "earC"),
    ]
    frame_metrics = {"precision": [], "recall": [], "f1": []}
    event_precision = []
    false_alarms = []

    for _, suffix in configs:
        frame_path = sweep_dir / f"video_frame_level_{suffix}.csv"
        event_path = sweep_dir / f"video_event_level_{suffix}.csv"
        frame_rows = read_rows(frame_path, ["event_type", *frame_metrics.keys()])
        event_rows = read_rows(event_path, ["event_type", "precision", "false_alarm"])
        frame = next(row for row in frame_rows if row["event_type"] == "eye_closed")
        event = next(row for row in event_rows if row["event_type"] == "eye_closed")
        for key in frame_metrics:
            frame_metrics[key].append(number(frame[key]))
        event_precision.append(number(event["precision"]))
        false_alarms.append(integer(event["false_alarm"]))

    labels = [label for label, _ in configs]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    width = 0.24
    for offset, (key, label, color) in zip(
        (-width, 0, width),
        [
            ("precision", "Precision", BLUE),
            ("recall", "Recall", ORANGE),
            ("f1", "F1", GREEN),
        ],
    ):
        axes[0].bar(x + offset, frame_metrics[key], width, label=label, color=color)
    axes[0].set_title("Frame-level eye-closure metrics")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(ncol=3, loc="upper center")

    colors = [BLUE, BLUE, GREEN, BLUE]
    bars = axes[1].bar(x, false_alarms, color=colors, alpha=0.88, label="False alarms")
    axes[1].set_title("Event-level eye-closure errors")
    axes[1].set_ylabel("False-alarm count")
    axes[1].set_xticks(x, labels)
    axes[1].bar_label(bars, padding=3)
    precision_axis = axes[1].twinx()
    precision_axis.plot(
        x,
        event_precision,
        color=PURPLE,
        marker="o",
        linewidth=2,
        label="Event precision",
    )
    precision_axis.set_ylabel("Event precision")
    precision_axis.set_ylim(0, 1.05)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = precision_axis.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    save_figure(fig, output_dir / "fig04_ear_configuration_comparison.png", dpi)


def grouped_metric_bars(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metrics: list[tuple[str, str, str]],
    title: str,
) -> None:
    event_order = ["eye_closed", "yawn", "head_tilt"]
    by_event = {row["event_type"]: row for row in rows}
    available_events = [event for event in event_order if event in by_event]
    x = np.arange(len(available_events))
    width = 0.75 / len(metrics)
    start = -width * (len(metrics) - 1) / 2
    for index, (key, label, color) in enumerate(metrics):
        values = [number(by_event[event][key]) for event in available_events]
        ax.bar(x + start + index * width, values, width, label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xticks(x, [event_label(event) for event in available_events])
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=len(metrics), loc="upper center")


def aggregate_video_figure(results_dir: Path, output_dir: Path, dpi: int) -> None:
    frame_rows = read_rows(
        results_dir / "final_video" / "video_frame_level.csv",
        ["event_type", "precision", "recall", "f1"],
    )
    event_rows = read_rows(
        results_dir / "final_video" / "video_event_level.csv",
        ["event_type", "precision", "recall"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    grouped_metric_bars(
        axes[0],
        frame_rows,
        [
            ("precision", "Precision", BLUE),
            ("recall", "Recall", ORANGE),
            ("f1", "F1", GREEN),
        ],
        "Frame-level metrics",
    )
    grouped_metric_bars(
        axes[1],
        event_rows,
        [("precision", "Precision", BLUE), ("recall", "Recall", ORANGE)],
        "Event-level metrics",
    )
    save_figure(fig, output_dir / "fig05_video_aggregate_metrics.png", dpi)


def subject_event_heatmap_figure(results_dir: Path, output_dir: Path, dpi: int) -> None:
    rows = read_rows(
        results_dir / "final_video" / "video_event_level_per_subject.csv",
        ["subject", "event_type", "recall", "precision"],
    )
    subjects = list(dict.fromkeys(row["subject"] for row in rows))
    event_order = ["eye_closed", "yawn", "head_tilt"]
    events = [event for event in event_order if any(row["event_type"] == event for row in rows)]
    lookup = {(row["subject"], row["event_type"]): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    image = None
    for ax, metric, title in zip(
        axes,
        ["recall", "precision"],
        ["Event recall", "Event precision"],
    ):
        matrix = np.full((len(subjects), len(events)), np.nan)
        for y, subject in enumerate(subjects):
            for x, event in enumerate(events):
                row = lookup.get((subject, event))
                if row is not None:
                    matrix[y, x] = number(row[metric])

        masked = np.ma.masked_invalid(matrix)
        color_map = plt.get_cmap("YlGnBu").copy()
        color_map.set_bad("#E5E7EB")
        image = ax.imshow(masked, cmap=color_map, vmin=0, vmax=1)
        for y in range(len(subjects)):
            for x in range(len(events)):
                value = matrix[y, x]
                text = "N/A" if math.isnan(value) else f"{value * 100:.1f}%"
                color = "white" if not math.isnan(value) and value >= 0.62 else "#111827"
                ax.text(x, y, text, ha="center", va="center", color=color, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(events)), [event_label(event) for event in events])
        ax.set_yticks(np.arange(len(subjects)), subjects)
        ax.set_xlabel("Event type")
        ax.set_ylabel("Subject")
        ax.grid(False)

    fig.colorbar(image, ax=axes, label="Score", shrink=0.82)
    save_figure(fig, output_dir / "fig06_video_event_metrics_per_subject.png", dpi)


def video_quality_figure(results_dir: Path, output_dir: Path, dpi: int) -> None:
    rows = read_rows(
        results_dir / "final_video" / "video_per_file.csv",
        ["subject", "video", "total_frames", "fps", "no_face_rate_percent"],
    )
    labels = [f"{row['subject']}\n{Path(row['video']).stem}" for row in rows]
    frames = [integer(row["total_frames"]) for row in rows]
    no_face_rates = [number(row["no_face_rate_percent"]) for row in rows]
    fps_values = [number(row["fps"]) for row in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bars = ax.bar(x, frames, color=BLUE, alpha=0.85, label="Total frames")
    ax.set_ylabel("Total frames")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    for bar, fps in zip(bars, fps_values):
        ax.annotate(
            f"{fps:.1f} FPS",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1F2937",
        )

    rate_axis = ax.twinx()
    rate_axis.plot(
        x,
        no_face_rates,
        color=RED,
        marker="o",
        linewidth=2,
        label="No-face rate",
    )
    rate_axis.set_ylabel("No-face rate (%)")
    rate_axis.set_ylim(bottom=0)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = rate_axis.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    save_figure(fig, output_dir / "fig07_video_data_quality.png", dpi)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all report figures from the project's result CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Result CSV directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Figure output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    generators = [
        confusion_matrix_figure,
        static_metrics_figure,
        mar_sensitivity_figure,
        ear_configuration_figure,
        aggregate_video_figure,
        subject_event_heatmap_figure,
        video_quality_figure,
    ]
    for generator in generators:
        generator(results_dir, output_dir, args.dpi)

    print(f"Generated {len(generators)} figures in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
