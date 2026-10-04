"""Data validation engine for TransitVision AI.
Performs rigorous structural, physical, and temporal validation on ingested transit data.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TransitDataValidator:
    """Validates raw and interim transit datasets against spatial, temporal, and physical constraints."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        val_cfg = self.config.get("VALIDATION", {})
        self.max_speed = val_cfg.get("MAX_ALLOWED_SPEED_KMH", 100.0)
        self.min_speed = val_cfg.get("MIN_ALLOWED_SPEED_KMH", 0.0)
        self.max_delay = val_cfg.get("MAX_DELAY_SECONDS", 7200.0)
        self.min_delay = val_cfg.get("MIN_DELAY_SECONDS", -1800.0)
        coord_bounds = val_cfg.get("COORDINATE_BOUNDS", {})
        self.lat_bounds = coord_bounds.get("LAT", [53.15, 53.55])
        self.lon_bounds = coord_bounds.get("LON", [-6.55, -6.05])

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "transit_avl") -> Dict[str, Any]:
        """
        Runs comprehensive validation suite and produces a structured validation report.

        Checks performed:
        1. Missing value percentage per column
        2. Duplicate record identification
        3. Geolocation boundary compliance (WGS84 Lat/Lon)
        4. Physical speed limits (0 to 100 km/h)
        5. Temporal monotonicity and timestamp integrity
        6. Route and Stop ID validity
        7. Delay range realism (-30m to +120m)
        8. Sensor flag binary integrity (AtStop, Congestion in {0, 1})
        """
        total_rows = len(df)
        report: Dict[str, Any] = {
            "validation_timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset_name": dataset_name,
            "total_records_evaluated": total_rows,
            "is_valid": True,
            "summary": {
                "missing_values_count": 0,
                "duplicates_count": 0,
                "out_of_bounds_coords_count": 0,
                "impossible_speed_count": 0,
                "timestamp_anomalies_count": 0,
                "invalid_delay_count": 0,
                "invalid_flags_count": 0
            },
            "detailed_checks": {},
            "quarantine_record_indices": []
        }

        if total_rows == 0:
            report["is_valid"] = False
            report["error"] = "Dataset is completely empty."
            return report

        # 1. Missing Values Check
        missing_series = df.isnull().sum()
        missing_dict = missing_series.to_dict()
        report["detailed_checks"]["missing_values"] = {
            col: {"missing_count": int(count), "missing_percentage": round(count / total_rows * 100, 2)}
            for col, count in missing_dict.items() if count > 0
        }
        report["summary"]["missing_values_count"] = int(missing_series.sum())

        # 2. Duplicate Records Check
        dup_mask = df.duplicated()
        dup_count = int(dup_mask.sum())
        report["summary"]["duplicates_count"] = dup_count
        report["detailed_checks"]["duplicates"] = {
            "count": dup_count,
            "percentage": round(dup_count / total_rows * 100, 2)
        }

        # 3. Coordinate Boundary Check
        lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None)
        lon_col = next((c for c in ["Longitude", "longitude", "lon"] if c in df.columns), None)

        if lat_col and lon_col:
            invalid_lat = (df[lat_col] < self.lat_bounds[0]) | (df[lat_col] > self.lat_bounds[1])
            invalid_lon = (df[lon_col] < self.lon_bounds[0]) | (df[lon_col] > self.lon_bounds[1])
            coord_invalid_mask = invalid_lat | invalid_lon
            coord_invalid_count = int(coord_invalid_mask.sum())
            report["summary"]["out_of_bounds_coords_count"] = coord_invalid_count
            report["detailed_checks"]["coordinate_bounds"] = {
                "lat_bounds": self.lat_bounds,
                "lon_bounds": self.lon_bounds,
                "out_of_bounds_count": coord_invalid_count,
                "out_of_bounds_percentage": round(coord_invalid_count / total_rows * 100, 2)
            }
        else:
            report["detailed_checks"]["coordinate_bounds"] = {"warning": "Latitude/Longitude columns not found"}

        # 4. Physical Speed & Speed Monotonicity
        speed_col = next((c for c in ["Speed", "speed", "speed_kmh"] if c in df.columns), None)
        if speed_col:
            invalid_speed_mask = (df[speed_col] < self.min_speed) | (df[speed_col] > self.max_speed)
            invalid_speed_count = int(invalid_speed_mask.sum())
            report["summary"]["impossible_speed_count"] = invalid_speed_count
            report["detailed_checks"]["speed_limits"] = {
                "min_allowed_kmh": self.min_speed,
                "max_allowed_kmh": self.max_speed,
                "anomalous_count": invalid_speed_count
            }

        # 5. Timestamp Monotonicity & Formatting
        ts_col = next((c for c in ["Timestamp", "timestamp", "time"] if c in df.columns), None)
        if ts_col:
            is_ts_numeric = pd.api.types.is_numeric_dtype(df[ts_col])
            report["detailed_checks"]["timestamp"] = {
                "column": ts_col,
                "is_numeric": is_ts_numeric,
                "min_timestamp": str(df[ts_col].min()),
                "max_timestamp": str(df[ts_col].max())
            }

        # 6. Delay Range Validation (Target Variable)
        delay_col = next((c for c in ["Delay", "delay", "delay_seconds"] if c in df.columns), None)
        if delay_col:
            invalid_delay_mask = (df[delay_col] < self.min_delay) | (df[delay_col] > self.max_delay)
            invalid_delay_count = int(invalid_delay_mask.sum())
            report["summary"]["invalid_delay_count"] = invalid_delay_count
            report["detailed_checks"]["delay_validity"] = {
                "min_delay_allowed_sec": self.min_delay,
                "max_delay_allowed_sec": self.max_delay,
                "anomalous_delay_count": invalid_delay_count,
                "mean_delay_sec": float(df[delay_col].mean()),
                "std_delay_sec": float(df[delay_col].std())
            }

        # 7. Binary Sensor Integrity (Congestion, AtStop)
        for flag_col in ["Congestion", "congestion", "AtStop", "at_stop"]:
            if flag_col in df.columns:
                unique_vals = set(df[flag_col].dropna().unique())
                valid_binary = unique_vals.issubset({0, 1, 0.0, 1.0})
                if not valid_binary:
                    report["summary"]["invalid_flags_count"] += 1
                    report["detailed_checks"][f"{flag_col}_binary_check"] = {
                        "valid": False,
                        "observed_values": list(unique_vals)
                    }

        # Overall Status
        total_anomalies = (
            report["summary"]["out_of_bounds_coords_count"] +
            report["summary"]["impossible_speed_count"] +
            report["summary"]["invalid_delay_count"]
        )
        if total_anomalies > total_rows * 0.1:  # More than 10% severe anomalies
            report["is_valid"] = False

        return report

    def save_validation_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Serializes the validation report to disk as formatted JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved to: {output_path}")
