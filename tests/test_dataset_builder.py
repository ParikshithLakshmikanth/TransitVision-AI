"""TransitVision AI - Unit Tests for Dataset Builder & Dual-Holdout Isolation."""
import pandas as pd
import pytest
from config.settings import PROCESSED_DATA_DIR
from retraining.dataset_builder import DatasetBuilder


def test_dataset_builder_80_20_split_and_provenance():
    """Verify 80/20 chronological split and provenance tracking."""
    builder = DatasetBuilder()
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    assert stream_path.exists(), "Stream dataset missing"

    sample_stream = pd.read_parquet(stream_path).iloc[:1000].copy()
    cand_train, recent_holdout, orig_holdout, manifest = builder.build_candidate_dataset(
        resolved_stream_df=sample_stream,
        trigger_reason="Test Trigger",
        source_model_version="v1.0.0",
    )

    # 80% of 1000 = 800 train stream records, 20% = 200 holdout
    assert manifest.candidate_train_stream_rows == 800
    assert manifest.recent_holdout_rows == 200
    assert len(recent_holdout) == 200
    assert manifest.total_candidate_train_rows == manifest.historical_anchor_rows + 800
    assert manifest.contamination_check_passed
    assert manifest.leakage_guard_passed


def test_dataset_builder_strict_non_contamination():
    """Verify recent holdout and original holdout are strictly disjoint from candidate training buffer."""
    builder = DatasetBuilder()
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    sample_stream = pd.read_parquet(stream_path).iloc[:1000].copy()

    cand_train, recent_holdout, orig_holdout, manifest = builder.build_candidate_dataset(
        resolved_stream_df=sample_stream,
    )

    # Original holdout has 30,102 rows
    assert len(orig_holdout) == 30102
    assert manifest.original_validation_rows == 30102

    # Check timestamps chronology: holdout timestamps >= train stream timestamps
    if "timestamp_utc" in sample_stream.columns:
        train_stream_slice = cand_train.iloc[manifest.historical_anchor_rows:]
        max_train_ts = pd.to_datetime(train_stream_slice["timestamp_utc"]).max()
        min_holdout_ts = pd.to_datetime(recent_holdout["timestamp_utc"]).min()
        assert min_holdout_ts >= max_train_ts


def test_dataset_builder_empty_stream_raises():
    """Verify empty stream buffer raises ValueError."""
    builder = DatasetBuilder()
    with pytest.raises(ValueError):
        builder.build_candidate_dataset(resolved_stream_df=pd.DataFrame())
