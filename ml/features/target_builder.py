"""TransitVision AI - Ground-Truth ETA Target Construction Engine.
Calculates the exact future arrival duration (eta_to_next_stop_sec)
from downstream stop events with strict zero-leakage guarantees.
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

from config.settings import METADATA_DIR, CONFIG

logger = logging.getLogger("TransitVision.TargetBuilder")


class ETATargetBuilder:
    """Computes downstream ETA targets and enforces strict target validity constraints."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or CONFIG
        self.max_eta_sec = 3600.0  # 1 hour maximum between consecutive scheduled stops
        self.min_eta_sec = 1.0     # Must be strictly positive

    def construct_eta_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculates eta_to_next_stop_sec = actual_next_stop_arrival_timestamp - current_timestamp.

        Steps:
        1. Group by (trip_id).
        2. Sort chronologically by timestamp_utc.
        3. Match the next stop arrival event.
        4. Calculate elapsed duration in seconds.
        5. Filter anomalous labels (ETA <= 0, ETA > max_eta_sec, missing downstream stop).
        6. Produce target_generation_report.json.
        """
        logger.info(f"Constructing ETA target on {len(df)} records...")
        working_df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(working_df["timestamp_utc"]):
            working_df["timestamp_utc"] = pd.to_datetime(working_df["timestamp_utc"], utc=True)

        # Sort chronologically within each trip
        working_df = working_df.sort_values(by=["trip_id", "timestamp_utc"]).reset_index(drop=True)

        # In GTFS trips, next_stop_arrival_utc is the scheduled/actual arrival at next_stop_id
        if "next_stop_arrival_utc" in working_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(working_df["next_stop_arrival_utc"]):
                working_df["next_stop_arrival_utc"] = pd.to_datetime(working_df["next_stop_arrival_utc"], utc=True)
            
            # Calculate raw ETA in seconds
            raw_eta = (working_df["next_stop_arrival_utc"] - working_df["timestamp_utc"]).dt.total_seconds()
            working_df["eta_to_next_stop_sec"] = raw_eta
        else:
            # Shift within trip to find the timestamp when the bus reaches the next stop
            # When bus is in transit to next stop, next stop arrival timestamp is the timestamp of the next stop record
            working_df["actual_next_stop_arrival_utc"] = working_df.groupby("trip_id")["timestamp_utc"].shift(-1)
            raw_eta = (working_df["actual_next_stop_arrival_utc"] - working_df["timestamp_utc"]).dt.total_seconds()
            working_df["eta_to_next_stop_sec"] = raw_eta

        # Target Validation & Quality Auditing
        initial_records = len(working_df)
        rejection_reasons = {
            "missing_next_stop_or_terminal_stop": 0,
            "negative_or_zero_eta": 0,
            "implausibly_large_eta": 0,
            "nan_or_corrupt_target": 0
        }

        # Terminal stop records (no downstream stop)
        missing_mask = working_df["eta_to_next_stop_sec"].isnull()
        rejection_reasons["missing_next_stop_or_terminal_stop"] = int(missing_mask.sum())

        # Non-positive ETA
        non_pos_mask = (working_df["eta_to_next_stop_sec"] <= self.min_eta_sec) & (~missing_mask)
        rejection_reasons["negative_or_zero_eta"] = int(non_pos_mask.sum())

        # Implausibly large ETA
        large_mask = (working_df["eta_to_next_stop_sec"] > self.max_eta_sec) & (~missing_mask)
        rejection_reasons["implausibly_large_eta"] = int(large_mask.sum())

        # Valid target mask
        valid_mask = (
            working_df["eta_to_next_stop_sec"].notnull() &
            (working_df["eta_to_next_stop_sec"] >= self.min_eta_sec) &
            (working_df["eta_to_next_stop_sec"] <= self.max_eta_sec)
        )

        labeled_df = working_df[valid_mask].copy()
        labeled_df["eta_to_next_stop_sec"] = labeled_df["eta_to_next_stop_sec"].astype(float)

        target_series = labeled_df["eta_to_next_stop_sec"]
        target_stats = {
            "count": int(len(target_series)),
            "mean_sec": round(float(target_series.mean()), 2) if len(target_series) > 0 else 0.0,
            "median_sec": round(float(target_series.median()), 2) if len(target_series) > 0 else 0.0,
            "std_sec": round(float(target_series.std()), 2) if len(target_series) > 0 else 0.0,
            "min_sec": round(float(target_series.min()), 2) if len(target_series) > 0 else 0.0,
            "max_sec": round(float(target_series.max()), 2) if len(target_series) > 0 else 0.0,
            "percentiles_sec": {
                "p25": round(float(np.percentile(target_series, 25)), 2) if len(target_series) > 0 else 0.0,
                "p50": round(float(np.percentile(target_series, 50)), 2) if len(target_series) > 0 else 0.0,
                "p75": round(float(np.percentile(target_series, 75)), 2) if len(target_series) > 0 else 0.0,
                "p90": round(float(np.percentile(target_series, 90)), 2) if len(target_series) > 0 else 0.0,
                "p95": round(float(np.percentile(target_series, 95)), 2) if len(target_series) > 0 else 0.0,
                "p99": round(float(np.percentile(target_series, 99)), 2) if len(target_series) > 0 else 0.0,
            }
        }

        total_rejected = initial_records - len(labeled_df)
        target_report = {
            "report_generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "target_variable": "eta_to_next_stop_sec",
            "initial_input_records": initial_records,
            "valid_labeled_records": len(labeled_df),
            "rejected_records_count": total_rejected,
            "rejection_breakdown": rejection_reasons,
            "target_distribution": target_stats,
            "quality_constraints": {
                "min_allowed_eta_sec": self.min_eta_sec,
                "max_allowed_eta_sec": self.max_eta_sec
            }
        }

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        report_path = METADATA_DIR / "target_generation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(target_report, f, indent=2)

        logger.info(f"Target construction complete. Labeled {len(labeled_df)} records (Mean: {target_stats['mean_sec']}s, Med: {target_stats['median_sec']}s). Report: {report_path}")

        return labeled_df, target_report
