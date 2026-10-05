"""Tests for production ETA inference service."""
import pytest
import pandas as pd
from pathlib import Path

from ml.inference.predictor import ETAPredictor, predict_eta
from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS
from config.settings import PROCESSED_DATA_DIR
from ml.training.leak_guard import LeakageGuardError


def test_eta_predictor_health():
    """Verify ETAPredictor initializes and reports HEALTHY status."""
    predictor = ETAPredictor()
    health = predictor.health_check()
    assert health["status"] == "HEALTHY"
    assert "model_lightgbm_v1" in health["model_id"]
    assert health["model_version"].startswith("v1.")


def test_predict_eta_single_dict():
    """Verify predict_eta accepts a single dictionary input and returns valid ETA."""
    val_path = PROCESSED_DATA_DIR / "kandy_eta_validation.parquet"
    df_val = pd.read_parquet(val_path)
    sample = df_val.iloc[0][ALL_FEATURE_COLUMNS].to_dict()

    response = predict_eta(sample)
    assert response["status"] == "SUCCESS"
    assert response["records_processed"] == 1
    assert isinstance(response["predicted_eta_sec"], float)
    assert 3.0 <= response["predicted_eta_sec"] <= 3600.0
    assert response["latency_ms"] >= 0.0


def test_predict_eta_batch_dataframe():
    """Verify ETAPredictor processes batch DataFrame with low latency."""
    predictor = ETAPredictor()
    val_path = PROCESSED_DATA_DIR / "kandy_eta_validation.parquet"
    df_val = pd.read_parquet(val_path).iloc[:20][ALL_FEATURE_COLUMNS]

    response = predictor.predict(df_val)
    assert response["status"] == "SUCCESS"
    assert response["records_processed"] == 20
    assert len(response["predicted_eta_sec"]) == 20
    assert all(3.0 <= eta <= 3600.0 for eta in response["predicted_eta_sec"])


def test_inference_leakage_guard_blocks_target():
    """Verify inference pipeline strictly blocks requests containing target columns."""
    predictor = ETAPredictor()
    val_path = PROCESSED_DATA_DIR / "kandy_eta_validation.parquet"
    df_val = pd.read_parquet(val_path).iloc[:5]  # Contains eta_to_next_stop_sec

    with pytest.raises(LeakageGuardError):
        predictor.predict(df_val)
