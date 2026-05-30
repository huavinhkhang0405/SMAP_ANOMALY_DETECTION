import os
import tempfile
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from src.model.isolation_forest import train_isolation_forest, save_model, load_model, predict_anomalies
from run_sprint3 import load_anomaly_labels


def test_train_isolation_forest():
    # Create dummy residual features (shape: N, 5)
    np.random.seed(42)
    features = np.random.rand(100, 5).astype(np.float32)
    
    model = train_isolation_forest(features, n_estimators=10, contamination=0.05, random_state=42)
    assert isinstance(model, IsolationForest)
    
    # Check predictions
    labels, scores = predict_anomalies(model, features)
    assert labels.shape == (100,)
    assert scores.shape == (100,)
    assert set(np.unique(labels)).issubset({-1, 1})


def test_model_save_load():
    np.random.seed(42)
    features = np.random.rand(50, 5).astype(np.float32)
    model = train_isolation_forest(features, n_estimators=5, contamination=0.05, random_state=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "if_model.pkl")
        save_model(model, model_path)
        assert os.path.exists(model_path)
        
        loaded = load_model(model_path)
        assert isinstance(loaded, IsolationForest)
        
        # Predictions match original model
        orig_labels, orig_scores = predict_anomalies(model, features)
        load_labels, load_scores = predict_anomalies(loaded, features)
        np.testing.assert_array_equal(orig_labels, load_labels)
        np.testing.assert_array_almost_equal(orig_scores, load_scores)


def test_load_anomaly_labels():
    # For channel P-1, total samples is 8505 in test split.
    # Let's test that load_anomaly_labels loads and labels correctly.
    num_samples = 8505
    labels = load_anomaly_labels("P-1", num_samples)
    
    assert labels.shape == (num_samples,)
    assert set(np.unique(labels)).issubset({-1, 1})
    
    # Verify specific anomalies are labeled as -1.
    # From CSV: [[2149, 2349], [4536, 4844], [3539, 3779]]
    # These indices should be -1 (anomalies)
    assert labels[2149] == -1
    assert labels[2200] == -1
    assert labels[2349] == -1
    
    assert labels[3539] == -1
    assert labels[3600] == -1
    assert labels[3779] == -1
    
    assert labels[4536] == -1
    assert labels[4600] == -1
    assert labels[4844] == -1
    
    # Other normal regions should be 1 (normal)
    assert labels[0] == 1
    assert labels[2000] == 1
    assert labels[2400] == 1
    assert labels[3500] == 1
    assert labels[4000] == 1
    assert labels[5000] == 1
    assert labels[8500] == 1
