import numpy as np
from typing import Tuple, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_raw_features(
    raw_series: np.ndarray,
    window_size: int = 100,
) -> np.ndarray:
    s = np.asarray(raw_series, dtype=np.float32)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if s.ndim != 2:
        raise ValueError("raw_series must be 1D or 2D array")

    n, d = s.shape
    features = np.zeros((n, d * 5), dtype=np.float32)

    for idx in range(n):
        start = max(0, idx - window_size + 1)
        chunk = s[start : idx + 1]

        features[idx, 0:d] = s[idx]
        features[idx, d:2 * d] = np.mean(chunk, axis=0)
        features[idx, 2 * d:3 * d] = np.std(chunk, axis=0)
        features[idx, 3 * d:4 * d] = np.max(chunk, axis=0)
        if idx > 0:
            features[idx, 4 * d:5 * d] = s[idx] - s[idx - 1]
        else:
            features[idx, 4 * d:5 * d] = 0.0

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def train_standard_isolation_forest(
    features: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.089,
    max_samples: str | int = "auto",
    random_state: int = 42,
) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(features)
    return pipeline


def predict_anomalies_raw_if(
    model: Pipeline,
    features: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = model.decision_function(features)

    if threshold is None:
        labels = model.predict(features)
    else:
        labels = np.where(scores < threshold, -1, 1).astype(np.int32)

    return labels, scores


def run_standard_if_detection(
    train_series: np.ndarray,
    test_series: np.ndarray,
    window_size: int = 100,
    n_estimators: int = 200,
    contamination: float = 0.089,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    train_features = build_raw_features(train_series, window_size=window_size)
    test_features  = build_raw_features(test_series,  window_size=window_size)

    model = train_standard_isolation_forest(
        features=train_features,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    labels, scores = predict_anomalies_raw_if(model, test_features)

    return labels, scores, model
