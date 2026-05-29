from typing import Tuple
import os
import numpy as np
import joblib     
                        
from tensorflow.keras.models import load_model 

from src import config
from src.data.preprocessing import create_sequences
from src.utils.residual_features import compute_rolling_features

def run_inference(
    series: np.ndarray,
    window_size: int | None = None,
    rolling_window: int | None = None,
    threshold_scale: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Luồng xử lý tích hợp (Inference Pipeline) phục vụ cho giao diện Streamlit.
    Quy ước nhãn đầu ra: 1 là Bình thường (Normal), -1 là Bất thường (Anomaly)
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if rolling_window is None:
        rolling_window = config.ROLLING_WINDOW

    # 1. Tiền xử lý dữ liệu (Phần của Khang)
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    X, y_true = create_sequences(series, window_size, config.PREDICTION_DIM)

    # 2. KHỐI DỰ BÁO CHUỖI THỜI GIAN (LSTM)
    if os.path.exists(config.MODEL_PATH):
        # ✅ KHI CÓ MODEL THẬT (Kiệt - Sprint 4)
        # model = load_model(config.MODEL_PATH)
        # y_pred = model.predict(X).reshape(-1)
        pass
    else:
        # ⚠️ BẢN CHẠY THỬ (Khang Baseline)
        y_pred = X[:, -1, 0].reshape(-1)

    # 3. Tính toán sai số trượt (Phần kết hợp)
    residuals = np.abs(y_true.reshape(-1) - y_pred)
    features, _ = compute_rolling_features(residuals, rolling_window)

    # 4. KHỐI PHÁT HIỆN BẤT THƯỜNG (Isolation Forest)
    if os.path.exists(config.IF_MODEL_PATH):
        # ✅ KHI CÓ MODEL THẬT (Kiệt - Sprint 4)
        # if_model = joblib.load(config.IF_MODEL_PATH)
        # labels = if_model.predict(features) 
        pass
    else:
        # ⚠️ BẢN CHẠY THỬ (Khang Heuristic)
        mean = float(np.mean(residuals))
        std = float(np.std(residuals))
        threshold = mean + threshold_scale * std
        labels = np.where(residuals > threshold, -1, 1)

    return labels.astype(np.int32), residuals.astype(np.float32), features