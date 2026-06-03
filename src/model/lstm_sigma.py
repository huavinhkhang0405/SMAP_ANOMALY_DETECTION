"""
lstm_sigma.py
-------------
Baseline 1: Pure LSTM + 3-Sigma Threshold (Mô hình học sâu tĩnh).

Mục đích (theo paper Hundman et al. - NASA, 2018):
    Chứng minh rằng nếu chỉ dùng LSTM để dự báo rồi áp một ngưỡng toán học
    cố định (mean + 3σ) mà **không** dùng Isolation Forest và Temporal Smoothing,
    hệ thống sẽ sinh ra vô vàn cảnh báo giả (false positives).

Cách hoạt động:
    1. Load model LSTM đã huấn luyện sẵn (lstm.keras).
    2. Predict trên tập test → tính residuals = |y_true - y_pred|.
    3. Đặt threshold = mean(residuals) + 3 * std(residuals).
    4. Nhãn anomaly = 1 (True) nếu residual > threshold.

Không có:
    - Isolation Forest
    - Temporal Smoothing (majority vote, merge, filter)
    - PR-Curve calibration

Tham chiếu:
    Hundman, K., LaDDha, V., Baker, C., Colwell, I., & Soderstrom, T. (2018).
    Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.
    KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219845
"""

import numpy as np
from typing import Tuple


def compute_sigma_threshold(
    residuals: np.ndarray,
    n_sigma: float = 3.0,
) -> Tuple[float, float, float]:
    """
    Tính ngưỡng phát hiện bất thường theo quy tắc N-Sigma.

    Công thức: threshold = mean + n_sigma * std

    Args:
        residuals : array (N,) hoặc (N, 1) — absolute residuals từ LSTM.
        n_sigma   : hệ số sigma, mặc định 3.0 (quy tắc 3-sigma / 99.73%).

    Returns:
        threshold : float — ngưỡng phân tách normal / anomaly.
        mean      : float — giá trị trung bình của residuals.
        std       : float — độ lệch chuẩn của residuals.
    """
    r = np.array(residuals).squeeze()
    mean = float(np.mean(r))
    std  = float(np.std(r))
    threshold = mean + n_sigma * std
    return threshold, mean, std


def predict_anomalies_sigma(
    residuals: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phát hiện bất thường bằng ngưỡng cố định trên residuals.

    Nhãn:
        -1 = anomaly  (residual > threshold)
         1 = normal   (residual <= threshold)

    Args:
        residuals : array (N,) — absolute residuals từ LSTM.
        threshold : ngưỡng sigma đã tính qua compute_sigma_threshold().

    Returns:
        labels    : array (N,) gồm -1 (anomaly) và 1 (normal).
        residuals : array (N,) residuals gốc (để vẽ biểu đồ so sánh).
    """
    r = np.array(residuals).squeeze()
    labels = np.where(r > threshold, -1, 1).astype(np.int32)
    return labels, r


def run_lstm_sigma_detection(
    lstm_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    n_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Pipeline hoàn chỉnh: LSTM predict → residuals → 3-Sigma threshold → labels.

    Args:
        lstm_model : Keras model đã được load (lstm.keras).
        X_test     : array (N, window_size, n_features) — test sequences.
        y_test     : array (N, prediction_dim) — ground truth targets.
        batch_size : batch size cho LSTM predict.
        n_sigma    : hệ số sigma (mặc định 3.0).

    Returns:
        labels    : array (N,)  — nhãn dự đoán (-1 / 1).
        residuals : array (N,)  — absolute residuals.
        threshold : float       — ngưỡng 3-sigma đã tính.
        mean      : float       — mean của residuals.
        std       : float       — std của residuals.
    """
    y_pred = lstm_model.predict(X_test, batch_size=batch_size, verbose=0)

    y_true_flat = y_test.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    residuals = np.abs(y_true_flat - y_pred_flat)

    threshold, mean, std = compute_sigma_threshold(residuals, n_sigma=n_sigma)

    labels, residuals = predict_anomalies_sigma(residuals, threshold)

    return labels, residuals, threshold, mean, std
