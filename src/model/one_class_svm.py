import numpy as np
from typing import Tuple, Optional

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_one_class_svm(
    features: np.ndarray,
    kernel: str = "rbf",
    nu: float = 0.089,
    gamma: str = "scale",
) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
        )),
    ])
    pipeline.fit(features)
    return pipeline


def predict_anomalies_ocsvm(
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


def run_ocsvm_detection(
    lstm_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    compute_residuals_fn,
    compute_features_fn,
    window_size: int,
    batch_size: int = 32,
    nu: float = 0.089,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    y_pred_train = lstm_model.predict(X_train, batch_size=batch_size, verbose=0)
    residuals_train = compute_residuals_fn(y_train, y_pred_train)
    train_features, _ = compute_features_fn(residuals_train, window_size)

    model = train_one_class_svm(
        features=train_features,
        kernel=kernel,
        nu=nu,
        gamma=gamma,
    )

    y_pred_test = lstm_model.predict(X_test, batch_size=batch_size, verbose=0)
    residuals_test = compute_residuals_fn(y_test, y_pred_test)
    test_features, _ = compute_features_fn(residuals_test, window_size)

    labels, scores = predict_anomalies_ocsvm(model, test_features)

    return labels, scores, model
