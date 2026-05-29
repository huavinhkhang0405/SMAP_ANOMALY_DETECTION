import os
from typing import Tuple

import numpy as np


def load_channel_npy(data_root: str, split: str, channel: str) -> np.ndarray:
    data_path = os.path.join(data_root, split, f"{channel}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")
    return np.load(data_path)


def train_test_split_series(series: np.ndarray, train_split: float) -> Tuple[np.ndarray, np.ndarray]:
    if train_split <= 0 or train_split >= 1:
        raise ValueError("train_split must be between 0 and 1")
    split_idx = int(len(series) * train_split)
    return series[:split_idx], series[split_idx:]


def zscore_normalize(series: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(series))
    std = float(np.std(series))
    if std == 0:
        return series, mean, std
    return (series - mean) / std, mean, std
