import os
from typing import Tuple
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

def train_isolation_forest(
    features: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.02,
    random_state: int = 42,
) -> IsolationForest:
    """
    Fits an Isolation Forest model on the provided residual features.
    
    Args:
        features: 2D array of residual features shape (n_samples, n_features)
        n_estimators: number of trees in forest
        contamination: proportion of outliers in the data
        random_state: seed for reproducible random selection
        
    Returns:
        Fitted IsolationForest model.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(features)
    return model

def save_model(model: IsolationForest, path: str) -> None:
    """Saves the fitted model to the specified path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> IsolationForest:
    """Loads the model from the specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def predict_anomalies(model: IsolationForest, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts anomaly labels and returns decision scores.
    
    Args:
        model: Fitted IsolationForest model
        features: 2D array of features shape (n_samples, n_features)
        
    Returns:
        labels: 1D array of shape (n_samples,), where -1 represents anomaly and 1 represents normal
        scores: 1D array of anomaly scores, where lower (more negative) values mean more anomalous
    """
    labels = model.predict(features)  # returns -1 for anomaly, 1 for normal
    scores = model.decision_function(features)  # signed distance, smaller is more anomalous
    return labels, scores
