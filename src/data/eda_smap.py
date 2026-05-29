import argparse
import ast
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class EdaSummary:
    channel: str
    split: str
    num_samples: int
    num_features: int
    has_timestamps: bool
    is_sampling_regular: str
    missing_count: int
    missing_ratio: float
    value_min: float
    value_max: float
    value_mean: float
    value_std: float
    labels_present: bool
    label_sequences: list
    label_count: int
    label_ratio: float


def _normalize_series(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data


def _parse_anomaly_sequences(label_str: str) -> list:
    if not isinstance(label_str, str) or label_str.strip() == "":
        return []
    try:
        return ast.literal_eval(label_str)
    except (ValueError, SyntaxError):
        return []


def _build_label_vector(length: int, sequences: list) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int32)
    for seq in sequences:
        if not isinstance(seq, (list, tuple)) or len(seq) != 2:
            continue
        start, end = seq
        if start is None or end is None:
            continue
        start = max(0, int(start))
        end = min(length - 1, int(end))
        if start <= end:
            labels[start : end + 1] = 1
    return labels


def _safe_float(value: float) -> float:
    if np.isnan(value):
        return float("nan")
    return float(value)


def run_eda(channel: str, split: str, data_root: str, report_dir: str, plot_limit: int | None) -> EdaSummary:
    data_path = os.path.join(data_root, split, f"{channel}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    raw_data = np.load(data_path)
    data = _normalize_series(raw_data)
    num_samples = data.shape[0]
    num_features = 1 if data.ndim == 1 else data.shape[1]

    missing_count = int(np.isnan(data).sum())
    missing_ratio = missing_count / float(num_samples * num_features)

    value_min = _safe_float(np.nanmin(data))
    value_max = _safe_float(np.nanmax(data))
    value_mean = _safe_float(np.nanmean(data))
    value_std = _safe_float(np.nanstd(data))

    labels_present = False
    label_sequences = []
    label_count = 0
    label_ratio = 0.0

    labels_path = os.path.join(os.path.dirname(data_root), "labeled_anomalies.csv")
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        row = labels_df[labels_df["chan_id"] == channel]
        if not row.empty:
            labels_present = True
            label_sequences = _parse_anomaly_sequences(row.iloc[0]["anomaly_sequences"])
            labels = _build_label_vector(num_samples, label_sequences)
            label_count = int(labels.sum())
            label_ratio = label_count / float(num_samples)

    # Plot signal with anomaly spans.
    os.makedirs(report_dir, exist_ok=True)
    plot_path = os.path.join(report_dir, f"eda_{channel}_{split}.png")

    plot_data = data[:, 0] if data.ndim > 1 else data
    if plot_limit is not None and plot_limit > 0:
        plot_data = plot_data[:plot_limit]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_data, linewidth=0.8)
    ax.set_title(f"SMAP {channel} ({split})")
    ax.set_xlabel("time index")
    ax.set_ylabel("value")

    if labels_present and label_sequences:
        for start, end in label_sequences:
            if plot_limit is not None and start > plot_limit:
                continue
            end_clip = end if plot_limit is None else min(end, plot_limit)
            ax.axvspan(start, end_clip, color="#ffb347", alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return EdaSummary(
        channel=channel,
        split=split,
        num_samples=num_samples,
        num_features=num_features,
        has_timestamps=False,
        is_sampling_regular="Unknown (no timestamps)",
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        value_min=value_min,
        value_max=value_max,
        value_mean=value_mean,
        value_std=value_std,
        labels_present=labels_present,
        label_sequences=label_sequences,
        label_count=label_count,
        label_ratio=label_ratio,
    )


def write_report(summary: EdaSummary, report_dir: str) -> str:
    report_path = os.path.join(report_dir, f"eda_{summary.channel}_{summary.split}.md")
    plot_name = f"eda_{summary.channel}_{summary.split}.png"

    lines = [
        "# SMAP EDA Report",
        "",
        f"Channel: {summary.channel}",
        f"Split: {summary.split}",
        "",
        "## Answers (light EDA)",
        f"- Number of channels: {summary.num_features}",
        f"- Univariate or multivariate: {'Univariate' if summary.num_features == 1 else 'Multivariate'}",
        f"- Sampling regular: {summary.is_sampling_regular}",
        f"- Missing values: {summary.missing_count} ({summary.missing_ratio:.6f})",
        f"- Value scale (min/max/mean/std): {summary.value_min:.6f} / {summary.value_max:.6f} / {summary.value_mean:.6f} / {summary.value_std:.6f}",
        f"- Label format: {'anomaly_sequences list of [start,end]' if summary.labels_present else 'not found'}",
        f"- Label coverage: {summary.label_count} points ({summary.label_ratio:.6f})",
        f"- Sequence length: {summary.num_samples}",
        "",
        "## Plot",
        f"![signal plot]({plot_name})",
        "",
        "## Data Contract (to freeze after EDA)",
        "WINDOW_SIZE = TBD",
        "N_FEATURES = TBD",
        "PREDICTION_DIM = TBD",
        "ROLLING_WINDOW = TBD",
        "",
        "## Notes",
        "- This report is a light EDA focused on the 7 questions only.",
        "- If you switch to multivariate, update N_FEATURES and re-run.",
    ]

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Light EDA for SMAP dataset")
    parser.add_argument("--channel", default="P-1", help="Channel id, e.g. P-1")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument(
        "--data-root",
        default=os.path.join("data", "raw", "NASA_SMAP", "data", "data"),
        help="Root folder containing train/ and test/",
    )
    parser.add_argument(
        "--report-dir",
        default=os.path.join("report"),
        help="Directory to save report and plots",
    )
    parser.add_argument(
        "--plot-limit",
        type=int,
        default=None,
        help="Optional limit for plot length",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_eda(
        channel=args.channel,
        split=args.split,
        data_root=args.data_root,
        report_dir=args.report_dir,
        plot_limit=args.plot_limit,
    )
    report_path = write_report(summary, args.report_dir)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
