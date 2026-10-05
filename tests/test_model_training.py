"""Unit and regression tests for model training, metrics, and leakage guard."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.training.leak_guard import LeakageGuard, LeakageGuardError
from ml.evaluation.evaluator import ModelEvaluator
from ml.preprocessing.pipeline_preprocessor import TabularDataPreprocessor, ALL_FEATURE_COLUMNS


def test_leakage_guard_detection():
    """Verify LeakageGuard raises LeakageGuardError on forbidden and target columns."""
    guard = LeakageGuard()

    # Clean input
    clean_df = pd.DataFrame({
        "hour": [8, 9],
        "deviceid": ["262", "263"],
        "segment": [1, 2]
    })
    assert guard.validate_features(clean_df) is True

    # Leaked target
    leak_target_df = pd.DataFrame({
        "hour": [8, 9],
        "eta_to_next_stop_sec": [120.0, 150.0]
    })
    with pytest.raises(LeakageGuardError):
        guard.validate_features(leak_target_df)

    # Future column substring
    leak_future_df = pd.DataFrame({
        "hour": [8, 9],
        "future_dwell_time": [10.0, 20.0]
    })
    with pytest.raises(LeakageGuardError):
        guard.validate_features(leak_future_df)


def test_model_evaluator_metrics():
    """Verify ModelEvaluator produces mathematically correct error statistics in seconds."""
    evaluator = ModelEvaluator()
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0, 520.0])  # errors: 10, 10, 10, 10, 20

    metrics = evaluator.calculate_metrics(y_true, y_pred)
    assert metrics["samples"] == 5
    assert metrics["mae_sec"] == 12.0
    assert metrics["median_ae_sec"] == 10.0
    assert metrics["r2"] > 0.95
    assert metrics["sanity_checks"]["non_positive_predictions_count"] == 0


def test_tabular_preprocessor_shape():
    """Verify preprocessor handles categorical casting and numerical column extraction."""
    preprocessor = TabularDataPreprocessor()
    
    # Create mock feature dataframe
    mock_data = {col: [1.0] * 5 for col in ALL_FEATURE_COLUMNS}
    mock_data["deviceid"] = ["262", "262", "263", "263", "262"]
    mock_data["direction"] = [1, 1, 2, 2, 1]
    mock_data["segment"] = [1, 2, 3, 4, 5]
    df = pd.DataFrame(mock_data)

    preprocessor.fit(df)
    tree_features = preprocessor.transform_for_trees(df)
    assert tree_features.shape == (5, len(ALL_FEATURE_COLUMNS))
    assert tree_features["deviceid"].dtype.name == "category"
