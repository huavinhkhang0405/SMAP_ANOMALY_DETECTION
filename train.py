import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from src import config
from src.model.lstm_forecast import train_model
from src.utils.residual_features import compute_residuals, compute_rolling_features


def load_processed() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))
	y_train = np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy"))
	X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
	y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
	return X_train, y_train, X_test, y_test


def split_train_val(
	X: np.ndarray, y: np.ndarray, val_split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if val_split <= 0 or val_split >= 1:
		raise ValueError("val_split must be between 0 and 1")
	split_idx = int(len(X) * (1 - val_split))
	return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def save_residual_artifacts(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	residuals: np.ndarray,
	residual_features: np.ndarray,
) -> None:
	os.makedirs(config.PROCESSED_DIR, exist_ok=True)
	np.save(config.Y_TRUE_PATH, y_true)
	np.save(config.Y_PRED_PATH, y_pred)
	np.save(config.RESIDUALS_PATH, residuals)
	np.save(config.RESIDUAL_FEATURES_PATH, residual_features)


def plot_residuals(residuals: np.ndarray, report_dir: str) -> None:
	os.makedirs(report_dir, exist_ok=True)
	fig, ax = plt.subplots(figsize=(12, 4))
	ax.plot(residuals, linewidth=0.8)
	ax.set_title("Residuals (abs error)")
	ax.set_xlabel("time index")
	ax.set_ylabel("residual")
	fig.tight_layout()
	fig.savefig(os.path.join(report_dir, "residuals.png"), dpi=150)
	plt.close(fig)


def main() -> None:
	X_train, y_train, X_test, y_test = load_processed()
	X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, config.VAL_SPLIT)

	model = train_model(X_tr, y_tr, X_val, y_val)

	y_pred = model.predict(X_test, batch_size=config.BATCH_SIZE)
	residuals = compute_residuals(y_test, y_pred)
	residual_features, _ = compute_rolling_features(residuals, config.ROLLING_WINDOW)

	save_residual_artifacts(y_test, y_pred, residuals, residual_features)
	plot_residuals(residuals, config.REPORT_DIR)


if __name__ == "__main__":
	main()
