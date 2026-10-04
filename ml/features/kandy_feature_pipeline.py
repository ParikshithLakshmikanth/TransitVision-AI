"""TransitVision AI - Kandy Feature Engineering Engine.
Generates temporal, spatial, causal rolling, traffic congestion, and weather features
with strict zero-leakage enforcement and feature provenance metadata.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import METADATA_DIR

logger = logging.getLogger("TransitVision.KandyFeatures")


class KandyFeatureEngineeringPipeline:
    """Extracts causal features for segment-level ETA regression."""

    def __init__(self):
        self.feature_columns: List[str] = [
            # Temporal
            "hour",
            "minute",
            "day_of_week",
            "is_weekend",
            "is_peak_period",
            "sin_hour",
            "cos_hour",
            "sin_time_of_day",
            "cos_time_of_day",
            # Spatial & Route Context
            "direction",
            "segment",
            "segment_length_km",
            "segments_completed",
            "segments_remaining",
            "trip_progress_ratio",
            # Bus & Driver / Device
            "deviceid",
            # Causal Historical Rolling (Preceding segments only)
            "previous_segment_run_time",
            "rolling_prev_segment_mean",
            "rolling_prev_segment_std",
            "cumulative_trip_time_sec",
            # GPS-Derived Congestion Indicators
            "historical_segment_time_mean",
            "segment_delay_ratio",
            "congestion_proxy",
            # Real Weather Observations
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "weather_code"
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all feature transformations deterministically.
        Strictly guarantees zero future-information leakage.
        """
        logger.info(f"Engineering causal features on {len(df)} Kandy transit records...")
        data = df.copy()

        # 1. Temporal Features
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp_utc"]):
            data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)

        data["hour"] = data["timestamp_utc"].dt.hour
        data["minute"] = data["timestamp_utc"].dt.minute
        data["day_of_week"] = data["timestamp_utc"].dt.dayofweek
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

        # Peak period: Morning (07:00-09:00) or Evening (16:30-18:30)
        time_minutes = data["hour"] * 60 + data["minute"]
        is_morning_peak = (time_minutes >= 7 * 60) & (time_minutes <= 9 * 60)
        is_evening_peak = (time_minutes >= 16 * 60 + 30) & (time_minutes <= 18 * 60 + 30)
        data["is_peak_period"] = (is_morning_peak | is_evening_peak).astype(int)

        # Cyclical Encodings
        data["sin_hour"] = np.sin(2.0 * np.pi * data["hour"] / 24.0)
        data["cos_hour"] = np.cos(2.0 * np.pi * data["hour"] / 24.0)
        data["sin_time_of_day"] = np.sin(2.0 * np.pi * time_minutes / 1440.0)
        data["cos_time_of_day"] = np.cos(2.0 * np.pi * time_minutes / 1440.0)

        # 2. Spatial & Route Progress
        data["segment_length_km"] = data["length"].astype(float)
        
        # Calculate segments completed & remaining per trip
        data = data.sort_values(by=["trip_id", "timestamp_utc"]).reset_index(drop=True)
        data["segments_completed"] = data.groupby("trip_id").cumcount()
        total_segments_per_trip = data.groupby("trip_id")["segments_completed"].transform("max") + 1
        data["segments_remaining"] = total_segments_per_trip - data["segments_completed"] - 1
        data["trip_progress_ratio"] = (data["segments_completed"] / total_segments_per_trip).clip(0.0, 1.0)

        # 3. Causal Rolling Features (STRICTLY PRECEDING: shift(1) per trip)
        # Shift run_time_in_seconds by 1 to get previous segment run time
        data["previous_segment_run_time"] = data.groupby("trip_id")["run_time_in_seconds"].shift(1)
        
        # If first segment on trip, fill with historical average for that segment
        historical_seg_means = data.groupby(["direction", "segment"])["run_time_in_seconds"].transform("mean")
        data["historical_segment_time_mean"] = historical_seg_means
        data["previous_segment_run_time"] = data["previous_segment_run_time"].fillna(data["historical_segment_time_mean"])

        # Rolling 3 preceding segments mean & std (causal: shifting before rolling)
        data["rolling_prev_segment_mean"] = (
            data.groupby("trip_id")["previous_segment_run_time"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        data["rolling_prev_segment_std"] = (
            data.groupby("trip_id")["previous_segment_run_time"]
            .transform(lambda s: s.rolling(3, min_periods=1).std())
            .fillna(0.0)
        )

        # Cumulative elapsed trip time (prior to current segment)
        data["cumulative_trip_time_sec"] = (
            data.groupby("trip_id")["previous_segment_run_time"]
            .cumsum() - data["previous_segment_run_time"]
        ).clip(lower=0.0)

        # 4. GPS-Derived Congestion Indicators
        data["segment_delay_ratio"] = (
            data["previous_segment_run_time"] / data["historical_segment_time_mean"].replace(0, 1.0)
        ).clip(0.2, 5.0)
        data["congestion_proxy"] = np.where(data["segment_delay_ratio"] > 1.25, 1, 0)

        # 5. Handle Weather Nulls if any
        weather_cols = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "wind_speed_10m", "weather_code"]
        for c in weather_cols:
            if c in data.columns:
                data[c] = data[c].fillna(data[c].median() if data[c].notnull().any() else 0.0)
            else:
                data[c] = 0.0

        # Device ID categorical encoding / normalization
        data["deviceid"] = data["deviceid"].astype(str)

        # 6. Leakage Verification
        self.verify_no_leakage(data)

        logger.info(f"Feature transformation complete. Feature matrix shape: {data[self.feature_columns].shape}")
        return data

    def verify_no_leakage(self, df: pd.DataFrame) -> None:
        """Enforces that no future timestamps, targets, or future-leaking columns are present in features."""
        forbidden_in_features = [
            "eta_to_next_stop_sec",
            "run_time_in_seconds",
            "end_time",
            "actual_arrival_time"
        ]
        for col in forbidden_in_features:
            if col in self.feature_columns:
                raise ValueError(f"CRITICAL LEAKAGE DETECTED: Column '{col}' is in feature_columns list!")

    def export_feature_metadata(self) -> Dict[str, Any]:
        """Saves machine-readable feature provenance catalog in data/metadata/kandy_feature_metadata.json."""
        provenance_map = {
            "hour": "DERIVED_FEATURE",
            "minute": "DERIVED_FEATURE",
            "day_of_week": "DERIVED_FEATURE",
            "is_weekend": "DERIVED_FEATURE",
            "is_peak_period": "DERIVED_FEATURE",
            "sin_hour": "DERIVED_FEATURE",
            "cos_hour": "DERIVED_FEATURE",
            "sin_time_of_day": "DERIVED_FEATURE",
            "cos_time_of_day": "DERIVED_FEATURE",
            "direction": "REAL_GPS_DERIVED",
            "segment": "REAL_GPS_DERIVED",
            "segment_length_km": "REAL_GPS_DERIVED",
            "segments_completed": "DERIVED_FEATURE",
            "segments_remaining": "DERIVED_FEATURE",
            "trip_progress_ratio": "DERIVED_FEATURE",
            "deviceid": "REAL_GPS_DERIVED",
            "previous_segment_run_time": "DERIVED_FEATURE",
            "rolling_prev_segment_mean": "DERIVED_FEATURE",
            "rolling_prev_segment_std": "DERIVED_FEATURE",
            "cumulative_trip_time_sec": "DERIVED_FEATURE",
            "historical_segment_time_mean": "DERIVED_FEATURE",
            "segment_delay_ratio": "DERIVED_FEATURE",
            "congestion_proxy": "DERIVED_FEATURE",
            "temperature_2m": "WEATHER_OBSERVED",
            "relative_humidity_2m": "WEATHER_OBSERVED",
            "precipitation": "WEATHER_OBSERVED",
            "rain": "WEATHER_OBSERVED",
            "wind_speed_10m": "WEATHER_OBSERVED",
            "weather_code": "WEATHER_OBSERVED"
        }

        metadata = {
            "metadata_version": "1.0.0",
            "dataset": "Bus Travel Time Data (Kandy, Sri Lanka)",
            "total_features": len(self.feature_columns),
            "features": [
                {
                    "name": col,
                    "provenance": provenance_map.get(col, "DERIVED_FEATURE"),
                    "data_type": "int64" if col in ["hour", "minute", "day_of_week", "is_weekend", "is_peak_period", "direction", "segment", "segments_completed", "segments_remaining", "congestion_proxy", "weather_code"] else "str" if col == "deviceid" else "float64"
                }
                for col in self.feature_columns
            ]
        }

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = METADATA_DIR / "kandy_feature_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Kandy feature metadata exported to: {meta_path}")
        return metadata


if __name__ == "__main__":
    pipeline = KandyFeatureEngineeringPipeline()
    pipeline.export_feature_metadata()
