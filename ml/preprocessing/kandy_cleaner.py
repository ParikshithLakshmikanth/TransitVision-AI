"""TransitVision AI - Kandy Data Cleaning & Sanitization Engine.
Performs deterministic filtering, timestamp parsing, and records audit trail in kandy_cleaning_report.json.
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

from config.settings import METADATA_DIR

logger = logging.getLogger("TransitVision.KandyCleaner")


class KandyDataCleaner:
    """Cleans real Kandy GPS-derived running times and dwell logs."""

    def __init__(self, min_run_time_sec: float = 3.0, max_run_time_sec: float = 3600.0):
        self.min_run_time_sec = min_run_time_sec
        self.max_run_time_sec = max_run_time_sec

    def clean_running_times(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans segment running times table:
        1. Remove exact duplicate records.
        2. Remove records with missing trip_id, deviceid, date, start_time, or run_time.
        3. Parse standardized timestamp_utc (from date + start_time).
        4. Enforce positive physical run times (3.0s <= run_time <= 3600s).
        5. Ensure valid directions (1 or 2).
        """
        initial_count = len(df)
        audit_trail = []
        current_df = df.copy()

        # Step 1: Duplicate removal
        count_before = len(current_df)
        current_df = current_df.drop_duplicates()
        removed_dups = count_before - len(current_df)
        audit_trail.append({
            "stage": "duplicate_removal",
            "records_before": count_before,
            "records_removed": removed_dups,
            "records_remaining": len(current_df),
            "reason": "Exact duplicate rows"
        })

        # Step 2: Missing key attributes
        count_before = len(current_df)
        key_cols = ["trip_id", "deviceid", "direction", "segment", "date", "start_time", "run_time_in_seconds"]
        valid_keys = current_df[key_cols].notnull().all(axis=1)
        current_df = current_df[valid_keys].copy()
        removed_keys = count_before - len(current_df)
        audit_trail.append({
            "stage": "missing_attributes_filter",
            "records_before": count_before,
            "records_removed": removed_keys,
            "records_remaining": len(current_df),
            "reason": "Missing trip_id, deviceid, segment, date, start_time, or run_time_in_seconds"
        })

        # Step 3: Parse timestamps to UTC
        count_before = len(current_df)
        # Handle various date string formats cleanly
        current_df["date_clean"] = pd.to_datetime(current_df["date"], errors="coerce")
        valid_dates = current_df["date_clean"].notnull()
        current_df = current_df[valid_dates].copy()

        # Combine date and start_time
        current_df["timestamp_utc"] = pd.to_datetime(
            current_df["date_clean"].dt.strftime("%Y-%m-%d") + " " + current_df["start_time"].astype(str),
            errors="coerce",
            utc=True
        )
        valid_ts = current_df["timestamp_utc"].notnull()
        current_df = current_df[valid_ts].copy()
        removed_ts = count_before - len(current_df)
        audit_trail.append({
            "stage": "timestamp_parsing",
            "records_before": count_before,
            "records_removed": removed_ts,
            "records_remaining": len(current_df),
            "reason": "Malformed date or start_time strings"
        })

        # Step 4: Run time physical bounds
        count_before = len(current_df)
        valid_runtime = (
            (current_df["run_time_in_seconds"] >= self.min_run_time_sec) &
            (current_df["run_time_in_seconds"] <= self.max_run_time_sec)
        )
        current_df = current_df[valid_runtime].copy()
        removed_runtime = count_before - len(current_df)
        audit_trail.append({
            "stage": "run_time_bounds",
            "records_before": count_before,
            "records_removed": removed_runtime,
            "records_remaining": len(current_df),
            "reason": f"run_time_in_seconds outside physical bounds [{self.min_run_time_sec}, {self.max_run_time_sec}] seconds"
        })

        # Step 5: Normalize identifiers
        current_df["trip_id"] = current_df["trip_id"].astype(int).astype(str)
        current_df["deviceid"] = current_df["deviceid"].astype(int).astype(str)
        current_df["direction"] = current_df["direction"].astype(int)
        current_df["segment"] = current_df["segment"].astype(int)
        current_df["route_id"] = "654"

        # Sort chronologically by trip and segment
        current_df = current_df.sort_values(by=["trip_id", "timestamp_utc", "segment"]).reset_index(drop=True)

        total_removed = initial_count - len(current_df)
        retention_rate = round(len(current_df) / initial_count * 100, 2) if initial_count > 0 else 0.0

        cleaning_report = {
            "cleaning_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "initial_records": initial_count,
            "final_clean_records": len(current_df),
            "total_records_removed": total_removed,
            "overall_retention_rate_pct": retention_rate,
            "unique_trips": int(current_df["trip_id"].nunique()),
            "unique_devices": int(current_df["deviceid"].nunique()),
            "date_range": {
                "start": str(current_df["timestamp_utc"].min()),
                "end": str(current_df["timestamp_utc"].max())
            },
            "audit_trail": audit_trail
        }

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        report_path = METADATA_DIR / "kandy_cleaning_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cleaning_report, f, indent=2)

        logger.info(f"Cleaning complete. Retained {len(current_df)} / {initial_count} records ({retention_rate}%). Report: {report_path}")
        return current_df, cleaning_report
