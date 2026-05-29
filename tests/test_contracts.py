import numpy as np
import pytest

from src import config
from src.data.preprocessing import create_sequences
from src.utils.residual_features import compute_rolling_features
from src.pipeline.inference_pipeline import run_inference


def test_data_shape_contract():
    data = np.arange(105, dtype=np.float32)
    X, y = create_sequences(data, config.WINDOW_SIZE, config.PREDICTION_DIM)
    assert X.ndim == 3
    assert y.ndim == 2
    assert X.shape == (5, config.WINDOW_SIZE, 1)
    assert y.shape == (5, config.PREDICTION_DIM)


def test_config_integrity():
    required_attrs = [
        "WINDOW_SIZE",
        "N_FEATURES",
        "PREDICTION_DIM",
        "ROLLING_WINDOW",
        "IF_CONTAMINATION",
        "MODEL_PATH",
        "IF_MODEL_PATH",
    ]
    for attr in required_attrs:
        assert hasattr(config, attr), f"Missing config: {attr}"

def test_rolling_features_shape():
    residuals = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    features, names = compute_rolling_features(residuals, window=3)
    assert features.shape == (10, 5)
    assert len(names) == 5

@pytest.mark.skip(reason="Chờ Kiệt tích hợp Full Pipeline ở Day 4")
def test_pipeline_sanity_check():
    data = np.random.rand(105).astype(np.float32)
    labels, residuals, features = run_inference(data)
    assert labels.shape[0] == 5
    assert residuals.shape[0] == 5
    assert features.shape[0] == 5
    assert set(np.unique(labels)).issubset({-1, 1})
