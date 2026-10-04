"""TransitVision AI - Data Cleaning & Normalization Engine.
Performs deterministic filtering, sanitization, coordinate boundary enforcement,
and produces an audit report in data/metadata/cleaning_report.json.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import INTERIM_DATA_DIR, METADATA_DIR, CONFIG

logger = logging.getLogger("TransitVision.Cleaner")


class TransitDataCleaner:
    """Cleans raw transit data records deterministically and records filtering audit trails."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or CONFIG
        val_cfg = self.config.get("VALIDATION", {})
        coord_bounds = val_cfg.get("COORDINATE_BOUNDS", {})
        self.lat_bounds = coord_bounds.get("LAT", [53.05, 53.65])
        self.lon_bounds = coord_bounds.get("LON", [-6.65, -6.00])
        self.max_speed = val_cfg.get("MAX_ALLOWED_SPEED_KMH", 100.0)
        self.min_speed = val_cfg.get("MIN_ALLOWED_SPEED_KMH", 0.0)

    def clean_records(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies sequential cleaning filters, tracking records removed at every stage.

        Filters:
        1. Remove exact duplicate rows.
        2. Remove missing or malformed timestamps.
        3. Remove missing or out-of-bounds GPS coordinates.
        4. Remove impossible speeds (< 0 or > 100 km/h).
        5. Remove missing route_id or trip_id.
        6. Remove invalid/missing stop IDs.
        7. Ensure chronological order per trip/vehicle.
        """
        initial_count = len(df)
        audit_trail = []
        current_df = df.copy()

        # Step 1: Remove exact duplicates
        count_before = len(current_df)
        current_df = current_df.drop_duplicates()
        removed_dups = count_before - len(current_df)
        audit_trail.append({
            "stage": "duplicate_removal",
            "records_before": count_before,
            "records_removed": removed_dups,
            "records_remaining": len(current_df),
            "reason": "Exact duplicate records across all fields"
        })

        # Step 2: Timestamp validity
        count_before = len(current_df)
        if "timestamp_utc" in current_df.columns:
            valid_ts = current_df["timestamp_utc"].notnull()
            current_df = current_df[valid_ts]
        removed_ts = count_before - len(current_df)
        audit_trail.append({
            "stage": "timestamp_sanitization",
            "records_before": count_before,
            "records_removed": removed_ts,
            "records_remaining": len(current_df),
            "reason": "Missing or null UTC timestamps"
        })

        # Step 3: Coordinate bounds (WGS84 Dublin bounding box)
        count_before = len(current_df)
        if "latitude" in current_df.columns and "longitude" in current_df.columns:
            valid_lat = (current_df["latitude"] >= self.lat_bounds[0]) & (current_df["latitude"] <= self.lat_bounds[1])
            valid_lon = (current_df["longitude"] >= self.lon_bounds[0]) & (current_df["longitude"] <= self.lon_bounds[1])
            current_df = current_df[valid_lat & valid_lon]
        removed_coords = count_before - len(current_df)
        audit_trail.append({
            "stage": "coordinate_bounding_box",
            "records_before": count_before,
            "records_removed": removed_coords,
            "records_remaining": len(current_df),
            "reason": f"Latitude outside {self.lat_bounds} or Longitude outside {self.lon_bounds}"
        })

        # Step 4: Speed limits
        count_before = len(current_df)
        if "current_speed_kmh" in current_df.columns:
            valid_speed = (current_df["current_speed_kmh"] >= self.min_speed) & (current_df["current_speed_kmh"] <= self.max_speed)
            current_df = current_df[valid_speed]
        removed_speed = count_before - len(current_df)
        audit_trail.append({
            "stage": "speed_limit_enforcement",
            "records_before": count_before,
            "records_removed": removed_speed,
            "records_remaining": len(current_df),
            "reason": f"Speed outside physical range [{self.min_speed}, {self.max_speed}] km/h"
        })

        # Step 5: Route and Trip ID validity
        count_before = len(current_df)
        valid_ids = current_df["route_id"].notnull() & current_df["trip_id"].notnull()
        current_df = current_df[valid_ids]
        removed_ids = count_before - len(current_df)
        audit_trail.append({
            "stage": "identifier_integrity",
            "records_before": count_before,
            "records_removed": removed_ids,
            "records_remaining": len(current_df),
            "reason": "Missing route_id or trip_id identifier"
        })

        # Step 6: Stop validity
        count_before = len(current_df)
        if "stop_id" in current_df.columns:
            valid_stop = current_df["stop_id"].notnull()
            current_df = current_df[valid_stop]
        removed_stops = count_before - len(current_df)
        audit_trail.append({
            "stage": "stop_identification",
            "records_before": count_before,
            "records_removed": removed_stops,
            "records_remaining": len(current_df),
            "reason": "Missing current stop identifier"
        })

        # Step 7: Chronological sorting
        sort_cols = [c for c in ["trip_id", "timestamp_utc", "stop_sequence"] if c in current_df.columns]
        if sort_cols:
            current_df = current_df.sort_values(by=sort_cols).reset_index(drop=True)

        total_removed = initial_count - len(current_df)
        retention_rate = round(len(current_df) / initial_count * 100, 2) if initial_count > 0 else 0.0

        cleaning_report = {
            "cleaning_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "initial_records": initial_count,
            "final_clean_records": len(current_df),
            "total_records_removed": total_removed,
            "overall_retention_rate_pct": retention_rate,
            "audit_trail": audit_trail
        }

        # Save cleaning report
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        report_path = METADATA_DIR / "cleaning_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cleaning_report, f, indent=2)

        logger.info(f"Cleaning complete. Retained {len(current_df)} / {initial_count} records ({retention_rate}%). Report: {report_path}")
        return current_df, cleaning_report
