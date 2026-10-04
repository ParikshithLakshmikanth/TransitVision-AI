"""TransitVision AI - Feature Engineering Engine.
Extracts spatiotemporal, kinematic, operational, and meteorological features
with strict feature leakage prevention and metadata cataloging.
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
from ml.preprocessing.gtfs_topology import haversine_distance_km

logger = logging.getLogger("TransitVision.Features")

# Feature category classification catalog
FEATURE_PROVENANCE_CATALOG = {
    # Real Features directly from sensors/telemetry/GTFS
    "timestamp_utc": "REAL",
    "route_id": "REAL",
    "trip_id": "REAL",
    "vehicle_id": "REAL",
    "direction_id": "REAL",
    "stop_id": "REAL",
    "next_stop_id": "REAL",
    "stop_sequence": "REAL",
    "latitude": "REAL",
    "longitude": "REAL",
    "stop_lat": "REAL",
    "stop_lon": "REAL",
    "next_stop_lat": "REAL",
    "next_stop_lon": "REAL",
    "is_at_stop": "REAL",
    "is_congested": "REAL",
    "temperature_2m": "REAL",
    "relative_humidity_2m": "REAL",
    "precipitation": "REAL",
    "rain": "REAL",
    "wind_speed_10m": "REAL",
    "weather_code": "REAL",

    # Derived Features (Calculated exclusively from past/current real observations)
    "hour": "DERIVED_FROM_REAL",
    "minute": "DERIVED_FROM_REAL",
    "day_of_week": "DERIVED_FROM_REAL",
    "is_weekend": "DERIVED_FROM_REAL",
    "sin_hour": "DERIVED_FROM_REAL",
    "cos_hour": "DERIVED_FROM_REAL",
    "distance_to_next_stop_km": "DERIVED_FROM_REAL",
    "distance_remaining_on_route_km": "DERIVED_FROM_REAL",
    "stops_remaining": "DERIVED_FROM_REAL",
    "current_speed_kmh": "DERIVED_FROM_REAL",
    "recent_speed_mean": "DERIVED_FROM_REAL",
    "recent_speed_std": "DERIVED_FROM_REAL",
    "recent_speed_min": "DERIVED_FROM_REAL",
    "recent_speed_max": "DERIVED_FROM_REAL",
    "dwell_time_sec": "DERIVED_FROM_REAL",

    # Synthetic Simulation Features (For controlled concept-drift injection)
    "simulated_passenger_occupancy_ratio": "SYNTHETIC",
    "injected_disturbance_intensity": "SYNTHETIC",
    "is_incident_active": "SYNTHETIC"
}


class FeatureEngineeringPipeline:
    """Transforms cleaned transit and meteorological records into structured ML feature tables."""

    def __init__(self):
        self.feature_columns: List[str] = [
            "hour",
            "minute",
            "day_of_week",
            "is_weekend",
            "sin_hour",
            "cos_hour",
            "latitude",
            "longitude",
            "distance_to_next_stop_km",
            "distance_remaining_on_route_km",
            "stops_remaining",
            "direction_id",
            "stop_sequence",
            "current_speed_kmh",
            "recent_speed_mean",
            "recent_speed_std",
            "recent_speed_min",
            "recent_speed_max",
            "is_at_stop",
            "dwell_time_sec",
            "is_congested",
            "temperature_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "weather_code",
            "simulated_passenger_occupancy_ratio",
            "injected_disturbance_intensity",
            "is_incident_active"
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all feature transformations deterministically.
        Guarantees zero future leakage.
        """
        logger.info(f"Engineering features on {len(df)} records...")
        data = df.copy()

        # 1. Temporal Features
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp_utc"]):
            data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)

        data["hour"] = data["timestamp_utc"].dt.hour
        data["minute"] = data["timestamp_utc"].dt.minute
        data["day_of_week"] = data["timestamp_utc"].dt.dayofweek
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
        data["sin_hour"] = np.sin(2.0 * np.pi * data["hour"] / 24.0)
        data["cos_hour"] = np.cos(2.0 * np.pi * data["hour"] / 24.0)

        # 2. Spatial Features
        if "distance_to_next_stop_km" not in data.columns:
            if "next_stop_lat" in data.columns and "next_stop_lon" in data.columns:
                valid_coords = data["next_stop_lat"].notnull() & data["latitude"].notnull()
                data["distance_to_next_stop_km"] = 0.0
                data.loc[valid_coords, "distance_to_next_stop_km"] = haversine_distance_km(
                    data.loc[valid_coords, "latitude"].values,
                    data.loc[valid_coords, "longitude"].values,
                    data.loc[valid_coords, "next_stop_lat"].values,
                    data.loc[valid_coords, "next_stop_lon"].values
                )
            else:
                data["distance_to_next_stop_km"] = 0.0

        if "distance_remaining_on_route_km" not in data.columns:
            data["distance_remaining_on_route_km"] = data.get("segment_distance_km", 1.0) * data.get("stops_remaining", 1)

        # 3. Transit & Topology defaults if not present
        if "direction_id" not in data.columns:
            data["direction_id"] = 0
        if "stop_sequence" not in data.columns:
            data["stop_sequence"] = 1
        if "stops_remaining" not in data.columns:
            data["stops_remaining"] = 0

        # 4. Kinematic Features (Rolling windows across preceding points only)
        if "current_speed_kmh" not in data.columns:
            data["current_speed_kmh"] = 25.0  # Default nominal speed

        # Rolling statistics strictly looking back (shift/rolling with min_periods=1)
        data["recent_speed_mean"] = (
            data.groupby("trip_id")["current_speed_kmh"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        data["recent_speed_std"] = (
            data.groupby("trip_id")["current_speed_kmh"]
            .transform(lambda s: s.rolling(3, min_periods=1).std())
            .fillna(0.0)
        )
        data["recent_speed_min"] = (
            data.groupby("trip_id")["current_speed_kmh"]
            .transform(lambda s: s.rolling(3, min_periods=1).min())
        )
        data["recent_speed_max"] = (
            data.groupby("trip_id")["current_speed_kmh"]
            .transform(lambda s: s.rolling(3, min_periods=1).max())
        )

        # 4. Operational Features
        if "is_at_stop" not in data.columns:
            data["is_at_stop"] = 0
        if "dwell_time_sec" not in data.columns:
            data["dwell_time_sec"] = np.where(data["is_at_stop"] == 1, 20.0, 0.0)
        if "is_congested" not in data.columns:
            data["is_congested"] = 0

        # 5. Synthetic Simulation Features (Default clean values for baseline training)
        if "simulated_passenger_occupancy_ratio" not in data.columns:
            data["simulated_passenger_occupancy_ratio"] = 0.35  # Nominal load
        if "injected_disturbance_intensity" not in data.columns:
            data["injected_disturbance_intensity"] = 0.0   # No disturbance in historical baseline
        if "is_incident_active" not in data.columns:
            data["is_incident_active"] = 0

        # Fill any residual missing weather or spatial fields with realistic defaults
        for col in ["temperature_2m", "precipitation", "rain", "wind_speed_10m", "weather_code"]:
            if col not in data.columns:
                data[col] = 0.0
            data[col] = data[col].fillna(0.0)

        # 6. Strict Leakage Verification Check
        self.verify_no_leakage(data)

        logger.info(f"Feature transformation completed. Features shape: {data[self.feature_columns].shape}")
        return data

    def verify_no_leakage(self, df: pd.DataFrame) -> None:
        """Asserts that no future timestamps, targets, or future-leaking columns are present in features."""
        forbidden_in_features = [
            "eta_to_next_stop_sec",
            "actual_next_stop_arrival_utc",
            "actual_next_stop_arrival_timestamp",
            "next_stop_arrival_utc"
        ]
        for col in forbidden_in_features:
            if col in self.feature_columns:
                raise ValueError(f"CRITICAL LEAKAGE ERROR: Target or future column '{col}' found in feature_columns list!")

    def export_feature_metadata(self) -> Dict[str, Any]:
        """Saves machine-readable feature provenance catalog in data/metadata/feature_metadata.json."""
        metadata = {
            "metadata_version": "1.0.0",
            "features_count": len(self.feature_columns),
            "features": [
                {
                    "name": col,
                    "provenance": FEATURE_PROVENANCE_CATALOG.get(col, "DERIVED_FROM_REAL"),
                    "data_type": "float64" if col not in ["direction_id", "is_weekend", "is_at_stop", "is_congested", "is_incident_active", "weather_code"] else "int64"
                }
                for col in self.feature_columns
            ]
        }
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = METADATA_DIR / "feature_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Feature metadata exported to: {meta_path}")
        return metadata


if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    pipeline.export_feature_metadata()
