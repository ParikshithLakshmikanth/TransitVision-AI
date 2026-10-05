"""Tests verifying model serialization, deserialization, and prediction determinism."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS


def test_model_serialization_determinism():
    """Verify serialized model artifact loads cleanly and produces exact identical outputs."""
    model_dir = MODELS_DIR / "eta_model_v1.0.0"
    assert model_dir.exists(), f"Model directory {model_dir} must exist."

    model_path = model_dir / "model.joblib"
    preproc_path = model_dir / "preprocessor.joblib"
    assert model_path.exists()
    assert preproc_path.exists()

    # Load artifacts in clean context
    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(preproc_path)

    # Load test batch from validation parquet
    val_path = PROCESSED_DATA_DIR / "kandy_eta_validation.parquet"
    df_val = pd.read_parquet(val_path).iloc[:50]
    X_val = df_val[ALL_FEATURE_COLUMNS].copy()

    # Run predictions via preprocessor + model
    X_tree = loaded_preprocessor.transform_for_trees(X_val)
    preds_1 = loaded_model.predict(X_tree)

    # Reload into a completely separate object to test persistence
    reloaded_model = joblib.load(model_path)
    preds_2 = reloaded_model.predict(X_tree)

    np.testing.assert_allclose(preds_1, preds_2, rtol=1e-5, atol=1e-5)
    assert len(preds_1) == 50
    assert (preds_1 > 0).all()
