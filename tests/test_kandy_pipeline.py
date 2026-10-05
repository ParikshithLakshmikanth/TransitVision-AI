"""Comprehensive unit and integration tests for Kandy Real GPS Transit Data Pipeline."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.preprocessing.kandy_cleaner import KandyDataCleaner
from ml.features.kandy_target_builder import KandyETATargetBuilder
from ml.features.kandy_feature_pipeline import KandyFeatureEngineeringPipeline
from ml.preprocessing.kandy_splitter import KandyDatasetSplitter
from ml.preprocessing.kandy_topology import KandyTopologyBuilder


def test_kandy_cleaner_filtering():
    """Verify KandyDataCleaner removes malformed timestamps and negative run times."""
    cleaner = KandyDataCleaner(min_run_time_sec=3.0, max_run_time_sec=3600.0)
    
    mock_df = pd.DataFrame({
        "trip_id": [1.0, 1.0, 1.0, 2.0],
        "deviceid": [262.0, 262.0, 262.0, 262.0],
        "direction": [1.0, 1.0, 1.0, 1.0],
        "segment": [1.0, 2.0, 3.0, 1.0],
        "date": ["2021-10-01", "2021-10-01", "invalid_date", "2021-10-01"],
        "start_time": ["06:30:00", "06:35:00", "06:40:00", "07:00:00"],
        "end_time": ["06:32:00", "06:38:00", "06:42:00", "07:02:00"],
        "run_time_in_seconds": [120.0, -10.0, 120.0, 150.0],  # Middle one has negative run time
        "length": [0.62, 1.28, 0.80, 0.62]
    })

    cleaned_df, report = cleaner.clean_running_times(mock_df)
    assert len(cleaned_df) == 2  # Only row 0 and row 3 are valid
    assert report["total_records_removed"] == 2
    assert (cleaned_df["run_time_in_seconds"] >= 3.0).all()


def test_kandy_target_construction_and_no_leakage():
    """Verify ETA target calculation and strict feature leakage prevention."""
    target_builder = KandyETATargetBuilder(min_eta_sec=3.0, max_eta_sec=3600.0)
    feature_pipeline = KandyFeatureEngineeringPipeline()

    mock_df = pd.DataFrame({
        "trip_id": ["1", "1", "1"],
        "deviceid": ["262", "262", "262"],
        "direction": [1, 1, 1],
        "segment": [1, 2, 3],
        "length": [0.62, 1.28, 0.80],
        "timestamp_utc": pd.to_datetime(["2021-10-01 06:30:00", "2021-10-01 06:35:00", "2021-10-01 06:40:00"], utc=True),
        "run_time_in_seconds": [120.0, 180.0, 150.0]
    })

    labeled_df, report = target_builder.construct_target(mock_df)
    assert len(labeled_df) == 3
    assert labeled_df.iloc[0]["eta_to_next_stop_sec"] == 120.0
    assert labeled_df.iloc[1]["eta_to_next_stop_sec"] == 180.0

    # Transform features
    featured_df = feature_pipeline.transform(labeled_df)
    assert "eta_to_next_stop_sec" in featured_df.columns

    # Causal validation: previous_segment_run_time of first segment must be historical mean, second must be 120.0
    assert featured_df.iloc[1]["previous_segment_run_time"] == 120.0

    # Leakage test
    with pytest.raises(ValueError):
        feature_pipeline.feature_columns.append("eta_to_next_stop_sec")
        feature_pipeline.verify_no_leakage(featured_df)
    feature_pipeline.feature_columns.remove("eta_to_next_stop_sec")


def test_kandy_temporal_splitter_chronology():
    """Verify Kandy dataset temporal partitioning enforces strict time separation."""
    splitter = KandyDatasetSplitter(train_ratio=0.70, val_ratio=0.15, save_files=False)

    dates = pd.date_range("2021-10-01", periods=100, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "trip_id": [f"T_{i}" for i in range(100)],
        "deviceid": ["262"] * 100,
        "feature_val": np.random.randn(100)
    })

    train_df, val_df, stream_df, split_info = splitter.split_dataset(df)
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(stream_df) == 15

    # Strictly chronological
    assert train_df["timestamp_utc"].max() < val_df["timestamp_utc"].min()
    assert val_df["timestamp_utc"].max() < stream_df["timestamp_utc"].min()


def test_kandy_topology():
    """Verify route topology loading and stop coordinates."""
    builder = KandyTopologyBuilder()
    topology_df = builder.build_topology()
    
    assert len(topology_df) == 31
    assert "latitude" in topology_df.columns
    assert "longitude" in topology_df.columns
    assert "direction_id" in topology_df.columns
    assert (topology_df["latitude"] > 7.0).all()
    assert (topology_df["longitude"] > 80.0).all()
