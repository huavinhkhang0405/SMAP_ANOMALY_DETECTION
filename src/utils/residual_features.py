from typing import Tuple, List
import numpy as np

def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return np.abs(y_true - y_pred)

def compute_rolling_features(residuals: np.ndarray, window: int) -> Tuple[np.ndarray, List[str]]:
    if window <= 0:
        raise ValueError("window must be > 0")

    residuals = residuals.reshape(-1)
    n = residuals.shape[0]
    features = np.zeros((n, 5), dtype=np.float32)

    for idx in range(n):
        start = max(0, idx - window + 1)
        chunk = residuals[start : idx + 1]
        
        features[idx, 0] = residuals[idx]                  
        features[idx, 1] = float(np.mean(chunk))           
        features[idx, 2] = float(np.std(chunk))            
        features[idx, 3] = float(np.max(chunk))            
        if idx == 0:
            features[idx, 4] = 0.0
        else:
            features[idx, 4] = residuals[idx] - residuals[idx - 1]  

    feature_names = [
        "residual",
        "rolling_mean",
        "rolling_std",
        "rolling_max",
        "residual_delta",
    ]

    return features, feature_names