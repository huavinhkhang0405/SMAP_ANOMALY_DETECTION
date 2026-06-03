import os
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from src import config
from src.data.loader import load_channel_npy, train_test_split_series


def _ensure_2d(data: np.ndarray) -> np.ndarray:
	if data.ndim == 1:
		return data.reshape(-1, 1)
	return data


def create_sequences(
	data: np.ndarray,
	window_size: int,
	prediction_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
	data = _ensure_2d(data)
	num_samples = data.shape[0]
	num_features = data.shape[1]

	if prediction_dim > num_features:
		raise ValueError("prediction_dim cannot exceed num_features")

	sample_count = num_samples - window_size
	if sample_count <= 0:
		raise ValueError("window_size is too large for the series length")

	X = np.zeros((sample_count, window_size, num_features), dtype=np.float32)
	y = np.zeros((sample_count, prediction_dim), dtype=np.float32)

	for idx in range(sample_count):
		X[idx] = data[idx : idx + window_size]
		y[idx] = data[idx + window_size, :prediction_dim]

	return X, y


def verify_shapes(X: np.ndarray, y: np.ndarray) -> None:
	if X.ndim != 3:
		raise ValueError("X must be 3D: (samples, window, features)")
	if y.ndim != 2:
		raise ValueError("y must be 2D: (samples, prediction_dim)")
	if X.shape[0] != y.shape[0]:
		raise ValueError("X and y must have the same sample size")


def save_processed(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	output_dir: str,
) -> None:
	os.makedirs(output_dir, exist_ok=True)
	np.save(os.path.join(output_dir, "X_train.npy"), X_train)
	np.save(os.path.join(output_dir, "y_train.npy"), y_train)
	np.save(os.path.join(output_dir, "X_test.npy"), X_test)
	np.save(os.path.join(output_dir, "y_test.npy"), y_test)


def plot_baselines(
	series: np.ndarray,
	window_size: int,
	train_split: float,
	report_dir: str,
	channel: str,
) -> None:
	os.makedirs(report_dir, exist_ok=True)
	series = series.squeeze()
	if series.ndim > 1:
		series = series[:, 0]

	fig, ax = plt.subplots(figsize=(12, 4))
	ax.plot(series, linewidth=0.8, label="raw")
	if window_size > 1:
		rolling = np.convolve(series, np.ones(window_size) / window_size, mode="valid")
		ax.plot(np.arange(window_size - 1, window_size - 1 + len(rolling)), rolling, label="rolling mean")
	ax.set_title(f"Raw signal: {channel}")
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(report_dir, f"baseline_raw_{channel}.png"), dpi=150)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(6, 4))
	ax.hist(series, bins=50, color="#4c72b0", alpha=0.85)
	ax.set_title(f"Histogram: {channel}")
	fig.tight_layout()
	fig.savefig(os.path.join(report_dir, f"baseline_hist_{channel}.png"), dpi=150)
	plt.close(fig)

	if len(series) > window_size:
		sample = series[: window_size + 1]
		fig, ax = plt.subplots(figsize=(8, 3))
		ax.plot(np.arange(window_size), sample[:window_size], label="X window")
		ax.scatter([window_size], [sample[-1]], color="#dd8452", label="y target")
		ax.set_title(f"Window sample: {channel}")
		ax.legend()
		fig.tight_layout()
		fig.savefig(os.path.join(report_dir, f"baseline_window_{channel}.png"), dpi=150)
		plt.close(fig)

	split_idx = int(len(series) * train_split)
	fig, ax = plt.subplots(figsize=(12, 4))
	ax.plot(series, linewidth=0.8)
	ax.axvline(split_idx, color="#c44e52", linestyle="--", label="train/test split")
	ax.set_title(f"Train/Test split: {channel}")
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(report_dir, f"baseline_split_{channel}.png"), dpi=150)
	plt.close(fig)


def run_preprocessing(channel: str, split: str = "train") -> None:
	data_root = config.RAW_DATA_ROOT
	processed_dir = config.PROCESSED_DIR
	report_dir = config.REPORT_DIR

	series = load_channel_npy(data_root, split, channel)
	series = series.astype(np.float32)

	train_series, test_series = train_test_split_series(series, config.TRAIN_SPLIT)

	X_train, y_train = create_sequences(train_series, config.WINDOW_SIZE, config.PREDICTION_DIM)
	X_test, y_test = create_sequences(test_series, config.WINDOW_SIZE, config.PREDICTION_DIM)

	verify_shapes(X_train, y_train)
	verify_shapes(X_test, y_test)

	save_processed(X_train, y_train, X_test, y_test, processed_dir)
	plot_baselines(series, config.WINDOW_SIZE, config.TRAIN_SPLIT, report_dir, channel)


if __name__ == "__main__":
	run_preprocessing(channel=config.DEFAULT_CHANNEL)
