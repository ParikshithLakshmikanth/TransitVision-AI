"""Unit and integration tests for Phase 2 data pipeline components."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from ml.preprocessing.cleaner import TransitDataCleaner
from ml.features.target_builder import ETATargetBuilder
from ml.features.feature_pipeline import FeatureEngineeringPipeline
from ml.preprocessing.temporal_splitter import TemporalDatasetSplitter
from ml.preprocessing.gtfs_topology import haversine_distance_km
from config.settings import CONFIG


def test_haversine_distance():
    """Verify haversine distance calculation is accurate on known coordinates."""
    # O'Connell Bridge to Dublin Airport (~9.5 km)
    dist = haversine_distance_km(53.3475, -6.2592, 53.4264, -6.2499)
    assert 8.5 <= dist <= 10.5


def test_cleaner_filters_anomalies():
    """Verify TransitDataCleaner strips out-of-bounds coordinates and impossible speeds."""
    cleaner = TransitDataCleaner(CONFIG)
    raw_df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:01:00", "2023-01-01 10:02:00"], utc=True),
        "route_id": ["15", "15", "15"],
        "trip_id": ["T1", "T1", "T1"],
        "stop_id": ["S1", "S2", "S3"],
        "latitude": [53.3498, 12.0, 53.3510],  # Middle one is outside Dublin
        "longitude": [-6.2603, -6.2600, -6.2590],
        "current_speed_kmh": [25.0, 30.0, 150.0] # Last one exceeds 100 km/h
    })

    cleaned_df, report = cleaner.clean_records(raw_df)
    assert len(cleaned_df) == 1
    assert report["total_records_removed"] == 2
    assert report["initial_records"] == 3


def test_target_builder_and_no_leakage():
    """Verify target calculation eta_to_next_stop_sec is accurate and forbids negative/leaked values."""
    target_builder = ETATargetBuilder(CONFIG)
    feature_pipeline = FeatureEngineeringPipeline()

    sample_df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:01:30", "2023-01-01 10:03:00"], utc=True),
        "trip_id": ["T1", "T1", "T1"],
        "route_id": ["15", "15", "15"],
        "stop_id": ["S1", "S2", "S3"],
        "latitude": [53.3498, 53.3505, 53.3512],
        "longitude": [-6.2603, -6.2600, -6.2595],
        "current_speed_kmh": [20.0, 25.0, 22.0]
    })

    labeled_df, report = target_builder.construct_eta_target(sample_df)

    # First record ETA to S2 is 90 seconds (10:00:00 to 10:01:30)
    assert len(labeled_df) == 2  # Last record has no downstream stop, so rejected
    assert labeled_df.iloc[0]["eta_to_next_stop_sec"] == 90.0
    assert labeled_df.iloc[1]["eta_to_next_stop_sec"] == 90.0
    assert (labeled_df["eta_to_next_stop_sec"] > 0).all()

    # Feature transformation
    featured_df = feature_pipeline.transform(labeled_df)
    assert "eta_to_next_stop_sec" in featured_df.columns

    # Verify no leakage check catches illegal target columns in features
    with pytest.raises(ValueError):
        feature_pipeline.feature_columns.append("eta_to_next_stop_sec")
        feature_pipeline.verify_no_leakage(featured_df)
    feature_pipeline.feature_columns.remove("eta_to_next_stop_sec")


def test_temporal_dataset_splitter_chronology():
    """Verify temporal splitter preserves strict time ordering without overlap between splits."""
    splitter = TemporalDatasetSplitter(train_ratio=0.70, val_ratio=0.15)
    
    dates = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "feature_a": np.random.randn(100)
    })

    train_df, val_df, stream_df, split_info = splitter.split_dataset(df)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(stream_df) == 15

    # Strict temporal ordering assertions
    assert train_df["timestamp_utc"].max() < val_df["timestamp_utc"].min()
    assert val_df["timestamp_utc"].max() < stream_df["timestamp_utc"].min()
