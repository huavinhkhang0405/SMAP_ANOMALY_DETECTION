from typing import Tuple, Any
import numpy as np

from src import config
from src.data.preprocessing import create_sequences
from src.utils.residual_features import compute_rolling_features

def run_inference(
    series: np.ndarray,
    lstm_model: Any = None,
    if_model: Any = None,
    window_size: int | None = None,
    rolling_window: int | None = None,
    threshold_scale: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if rolling_window is None:
        rolling_window = config.ROLLING_WINDOW

    # 1. Xử lý đồng bộ số chiều (Zero-Padding)
    data_2d = np.asarray(series, dtype=np.float32)
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(-1, 1)

    # Nếu file nạp vào có ít hơn 25 kênh (ví dụ file demo chỉ có 1 cột)
    # Ta tự động đệm thêm các cột số 0 để không bị crash mô hình LSTM
    if data_2d.shape[1] < 25:
        padded = np.zeros((data_2d.shape[0], 25), dtype=np.float32)
        padded[:, :data_2d.shape[1]] = data_2d
        data_2d = padded

    # Tạo cửa sổ chuỗi thời gian
    X, y_true = create_sequences(data_2d, window_size, config.PREDICTION_DIM)

    # ---------------------------------------------------------
    # TẦNG 1: DỰ BÁO LSTM
    # ---------------------------------------------------------
    if lstm_model is not None:
        y_pred = lstm_model.predict(X, verbose=0)
    else:
        y_pred = X[:, -1, :] # Naive fallback

    # Lấy giá trị của kênh đầu tiên (Cột 0 - Kênh viễn trắc chính) để tính sai số
    y_true_main = y_true[:, 0] if y_true.ndim > 1 else y_true
    y_pred_main = y_pred[:, 0] if y_pred.ndim > 1 else y_pred

    # Tính sai số tuyệt đối
    residuals = np.abs(y_true_main.reshape(-1) - y_pred_main.reshape(-1))
    features, _ = compute_rolling_features(residuals, rolling_window)

    # ---------------------------------------------------------
    # TẦNG 2: PHÂN LẬP VỚI ISOLATION FOREST
    # ---------------------------------------------------------
    if if_model is not None:
        labels = if_model.predict(features)
    else:
        mean = float(np.mean(residuals))
        std = float(np.std(residuals))
        threshold = mean + threshold_scale * std
        labels = np.where(residuals > threshold, -1, 1)

    return labels.astype(np.int32), residuals.astype(np.float32), features