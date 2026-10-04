"""TransitVision AI - Kandy Real Observed ETA Target Engine.
Constructs the genuine observed segment running time target (eta_to_next_stop_sec)
from GPS-derived segment observations.
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

logger = logging.getLogger("TransitVision.KandyTargetBuilder")


class KandyETATargetBuilder:
    """Constructs and validates the real observed segment-level ETA target."""

    def __init__(self, min_eta_sec: float = 3.0, max_eta_sec: float = 3600.0):
        self.min_eta_sec = min_eta_sec
        self.max_eta_sec = max_eta_sec

    def construct_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extracts eta_to_next_stop_sec = run_time_in_seconds.
        Enforces strict target validity constraints and generates real_eta_target_report.json.
        """
        logger.info(f"Constructing real ETA target on {len(df)} Kandy segment records...")
        working_df = df.copy()

        # Primary ML Target: Real GPS-derived segment travel time
        working_df["eta_to_next_stop_sec"] = working_df["run_time_in_seconds"].astype(float)

        initial_count = len(working_df)
        rejection_reasons = {
            "non_positive_or_zero_eta": 0,
            "excessive_eta_outlier": 0,
            "missing_or_nan_eta": 0
        }

        # Check NaN
        nan_mask = working_df["eta_to_next_stop_sec"].isnull()
        rejection_reasons["missing_or_nan_eta"] = int(nan_mask.sum())

        # Check non-positive
        non_pos_mask = (working_df["eta_to_next_stop_sec"] < self.min_eta_sec) & (~nan_mask)
        rejection_reasons["non_positive_or_zero_eta"] = int(non_pos_mask.sum())

        # Check large outliers
        large_mask = (working_df["eta_to_next_stop_sec"] > self.max_eta_sec) & (~nan_mask)
        rejection_reasons["excessive_eta_outlier"] = int(large_mask.sum())

        # Valid mask
        valid_mask = (
            working_df["eta_to_next_stop_sec"].notnull() &
            (working_df["eta_to_next_stop_sec"] >= self.min_eta_sec) &
            (working_df["eta_to_next_stop_sec"] <= self.max_eta_sec)
        )

        labeled_df = working_df[valid_mask].copy()
        target_series = labeled_df["eta_to_next_stop_sec"]

        target_stats = {
            "count": int(len(target_series)),
            "mean_sec": round(float(target_series.mean()), 2),
            "median_sec": round(float(target_series.median()), 2),
            "std_sec": round(float(target_series.std()), 2),
            "min_sec": round(float(target_series.min()), 2),
            "max_sec": round(float(target_series.max()), 2),
            "percentiles_sec": {
                "p25": round(float(np.percentile(target_series, 25)), 2),
                "p50": round(float(np.percentile(target_series, 50)), 2),
                "p75": round(float(np.percentile(target_series, 75)), 2),
                "p90": round(float(np.percentile(target_series, 90)), 2),
                "p95": round(float(np.percentile(target_series, 95)), 2),
                "p99": round(float(np.percentile(target_series, 99)), 2)
            }
        }

        total_rejected = initial_count - len(labeled_df)
        report = {
            "report_generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "target_variable": "eta_to_next_stop_sec",
            "prediction_granularity": "segment_level_eta_prediction",
            "total_observed_records": initial_count,
            "valid_eta_labels": len(labeled_df),
            "rejected_records_count": total_rejected,
            "rejection_breakdown": rejection_reasons,
            "target_distribution": target_stats,
            "quality_constraints": {
                "min_allowed_eta_sec": self.min_eta_sec,
                "max_allowed_eta_sec": self.max_eta_sec
            }
        }

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        report_path = METADATA_DIR / "real_eta_target_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Target construction complete. Valid ETA labels: {len(labeled_df)} (Mean: {target_stats['mean_sec']}s, Med: {target_stats['median_sec']}s). Report: {report_path}")
        return labeled_df, report
