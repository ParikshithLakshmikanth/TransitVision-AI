"""TransitVision AI - Temporal Partitioning Engine.
Splits real transit datasets strictly chronologically into Training, Validation, and Streaming Replay sets.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import PROCESSED_DATA_DIR, METADATA_DIR

logger = logging.getLogger("TransitVision.TemporalSplitter")


class TemporalDatasetSplitter:
    """Performs chronological dataset partitioning without temporal leakage."""

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.stream_ratio = 1.0 - train_ratio - val_ratio

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits DataFrame chronologically based on timestamp_utc.

        Returns:
            (train_df, val_df, stream_df, split_metadata)
        """
        logger.info(f"Partitioning {len(df)} records chronologically...")
        df_sorted = df.sort_values(by="timestamp_utc").reset_index(drop=True)

        total_rows = len(df_sorted)
        train_idx = int(total_rows * self.train_ratio)
        val_idx = int(total_rows * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_idx].copy()
        val_df = df_sorted.iloc[train_idx:val_idx].copy()
        stream_df = df_sorted.iloc[val_idx:].copy()

        split_info = {
            "split_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "total_records": total_rows,
            "train": {
                "records": len(train_df),
                "percentage": round(len(train_df) / total_rows * 100, 2),
                "start_time_utc": str(train_df["timestamp_utc"].min()),
                "end_time_utc": str(train_df["timestamp_utc"].max()),
                "file": "data/processed/eta_training.parquet"
            },
            "validation": {
                "records": len(val_df),
                "percentage": round(len(val_df) / total_rows * 100, 2),
                "start_time_utc": str(val_df["timestamp_utc"].min()),
                "end_time_utc": str(val_df["timestamp_utc"].max()),
                "file": "data/processed/eta_validation.parquet"
            },
            "stream_replay": {
                "records": len(stream_df),
                "percentage": round(len(stream_df) / total_rows * 100, 2),
                "start_time_utc": str(stream_df["timestamp_utc"].min()),
                "end_time_utc": str(stream_df["timestamp_utc"].max()),
                "file": "data/processed/eta_stream.parquet"
            }
        }

        # Save to Parquet
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(PROCESSED_DATA_DIR / "eta_training.parquet", index=False)
        val_df.to_parquet(PROCESSED_DATA_DIR / "eta_validation.parquet", index=False)
        stream_df.to_parquet(PROCESSED_DATA_DIR / "eta_stream.parquet", index=False)

        # Save metadata
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        split_meta_path = METADATA_DIR / "dataset_split.json"
        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)

        logger.info(f"Dataset split complete. Train: {len(train_df)}, Val: {len(val_df)}, Stream: {len(stream_df)}. Metadata: {split_meta_path}")

        return train_df, val_df, stream_df, split_info
