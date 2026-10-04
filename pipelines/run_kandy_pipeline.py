"""TransitVision AI - Kandy End-to-End Real Data Processing Pipeline.
Integrates genuine Kandy GPS-derived travel time tables, Open-Meteo historical weather,
extracts causal features, builds real observed ETA labels, and performs temporal partitioning.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR
from ml.preprocessing.kandy_topology import KandyTopologyBuilder
from ml.preprocessing.kandy_cleaner import KandyDataCleaner
from ml.preprocessing.kandy_weather_integrator import KandyWeatherIntegrator
from ml.features.kandy_target_builder import KandyETATargetBuilder
from ml.features.kandy_feature_pipeline import KandyFeatureEngineeringPipeline
from ml.preprocessing.kandy_splitter import KandyDatasetSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TransitVision.KandyPipeline")


class KandyPreprocessingPipeline:
    """Orchestrates end-to-end processing of real Kandy bus GPS datasets."""

    def __init__(self):
        self.topology_builder = KandyTopologyBuilder()
        self.cleaner = KandyDataCleaner()
        self.weather_integrator = KandyWeatherIntegrator()
        self.target_builder = KandyETATargetBuilder()
        self.feature_pipeline = KandyFeatureEngineeringPipeline()
        self.splitter = KandyDatasetSplitter()

    def calculate_baseline_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates natural statistical distributions for concept-drift monitoring."""
        baseline = {
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "total_records": len(df),
            "eta_to_next_stop_sec": {
                "mean": round(float(df["eta_to_next_stop_sec"].mean()), 2),
                "std": round(float(df["eta_to_next_stop_sec"].std()), 2),
                "p25": round(float(np.percentile(df["eta_to_next_stop_sec"], 25)), 2),
                "p50": round(float(np.percentile(df["eta_to_next_stop_sec"], 50)), 2),
                "p75": round(float(np.percentile(df["eta_to_next_stop_sec"], 75)), 2),
                "p90": round(float(np.percentile(df["eta_to_next_stop_sec"], 90)), 2),
                "p99": round(float(np.percentile(df["eta_to_next_stop_sec"], 99)), 2)
            },
            "segment_length_km": {
                "mean": round(float(df["segment_length_km"].mean()), 4),
                "std": round(float(df["segment_length_km"].std()), 4)
            },
            "previous_segment_run_time": {
                "mean": round(float(df["previous_segment_run_time"].mean()), 2),
                "std": round(float(df["previous_segment_run_time"].std()), 2)
            },
            "temperature_2m": {
                "mean": round(float(df["temperature_2m"].mean()), 2),
                "std": round(float(df["temperature_2m"].std()), 2)
            },
            "precipitation": {
                "mean": round(float(df["precipitation"].mean()), 4),
                "max": round(float(df["precipitation"].max()), 2)
            }
        }
        out_path = METADATA_DIR / "kandy_baseline_distributions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"Kandy baseline distributions saved to: {out_path}")
        return baseline

    def update_column_lineage(self) -> None:
        """Updates data/metadata/column_lineage.json with verified real Kandy GPS classifications."""
        lineage = {
            "audit_version": "2.0.0",
            "audit_date": pd.Timestamp.now(tz="UTC").isoformat(),
            "provenance_status": "VERIFIED",
            "dataset_name": "Bus Travel Time Data (Kandy, Sri Lanka)",
            "primary_source_url": "https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data",
            "weather_source_url": "https://archive-api.open-meteo.com/v1/archive",
            "columns": {
                "trip_id": {
                    "source_file": "data/raw/kandy_bus/bus_running_times_654.csv",
                    "source_column": "trip_id",
                    "transformation": "Parsed string identifier",
                    "data_classification": "REAL_GPS_DERIVED",
                    "is_observed": True
                },
                "deviceid": {
                    "source_file": "data/raw/kandy_bus/bus_running_times_654.csv",
                    "source_column": "deviceid",
                    "transformation": "On-board GPS hardware device identifier",
                    "data_classification": "REAL_GPS_DERIVED",
                    "is_observed": True
                },
                "direction": {
                    "source_file": "data/raw/kandy_bus/bus_running_times_654.csv",
                    "source_column": "direction",
                    "transformation": "Route direction (1: Kandy->Digana, 2: Digana->Kandy)",
                    "data_classification": "REAL_GPS_DERIVED",
                    "is_observed": True
                },
                "segment": {
                    "source_file": "data/raw/kandy_bus/bus_running_times_654.csv",
                    "source_column": "segment",
                    "transformation": "Roadway segment index between consecutive stops",
                    "data_classification": "REAL_GPS_DERIVED",
                    "is_observed": True
                },
                "segment_length_km": {
                    "source_file": "data/raw/kandy_bus/bus_running_times_654.csv",
                    "source_column": "length",
                    "transformation": "Physical roadway distance of segment in kilometers",
                    "data_classification": "REAL_GPS_DERIVED",
                    "is_observed": True
                },
                "timestamp_utc": {
                    "source_file": "data/raw/kandy_bus/bus_running_times_654.csv",
                    "source_column": "date + start_time",
                    "transformation": "Parsed ISO UTC timestamp from GPS departure time",
                    "data_classification": "REAL_GPS_DERIVED",
                    "is_observed": True
                },
                "eta_to_next_stop_sec": {
                    "source_file": "data/raw/kandy_bus/bus_running_times_654.csv",
                    "source_column": "run_time_in_seconds",
                    "transformation": "Actual measured GPS travel duration to next stop (Target)",
                    "data_classification": "REAL_PROCESSED_FROM_GPS",
                    "is_observed": True
                },
                "hour": {
                    "source_file": "derived",
                    "source_column": "timestamp_utc.hour",
                    "transformation": "Hour of day (0-23)",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "minute": {
                    "source_file": "derived",
                    "source_column": "timestamp_utc.minute",
                    "transformation": "Minute of hour (0-59)",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "day_of_week": {
                    "source_file": "derived",
                    "source_column": "timestamp_utc.dayofweek",
                    "transformation": "Day of week (0-6)",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "is_weekend": {
                    "source_file": "derived",
                    "source_column": "day_of_week >= 5",
                    "transformation": "Weekend binary indicator",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "is_peak_period": {
                    "source_file": "derived",
                    "source_column": "hour, minute",
                    "transformation": "Rush hour indicator (07:00-09:00, 16:30-18:30)",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "previous_segment_run_time": {
                    "source_file": "derived",
                    "source_column": "run_time_in_seconds.shift(1)",
                    "transformation": "Causal shift of previous segment run time within trip",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "rolling_prev_segment_mean": {
                    "source_file": "derived",
                    "source_column": "previous_segment_run_time.rolling(3).mean()",
                    "transformation": "Causal rolling 3-segment backward mean",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "cumulative_trip_time_sec": {
                    "source_file": "derived",
                    "source_column": "previous_segment_run_time.cumsum()",
                    "transformation": "Cumulative elapsed travel time on trip before current segment",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "historical_segment_time_mean": {
                    "source_file": "derived",
                    "source_column": "run_time_in_seconds",
                    "transformation": "Historical average travel time for segment & direction",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "segment_delay_ratio": {
                    "source_file": "derived",
                    "source_column": "previous_segment_run_time / historical_segment_time_mean",
                    "transformation": "Congestion delay ratio indicator",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "congestion_proxy": {
                    "source_file": "derived",
                    "source_column": "segment_delay_ratio > 1.25",
                    "transformation": "Binary congestion indicator",
                    "data_classification": "DERIVED_FEATURE",
                    "is_observed": False
                },
                "temperature_2m": {
                    "source_file": "data/raw/kandy_weather/historical_weather_kandy.csv",
                    "source_column": "temperature_2m",
                    "transformation": "Open-Meteo historical hourly air temperature for Kandy",
                    "data_classification": "WEATHER_OBSERVED",
                    "is_observed": True
                },
                "relative_humidity_2m": {
                    "source_file": "data/raw/kandy_weather/historical_weather_kandy.csv",
                    "source_column": "relative_humidity_2m",
                    "transformation": "Open-Meteo historical hourly relative humidity",
                    "data_classification": "WEATHER_OBSERVED",
                    "is_observed": True
                },
                "precipitation": {
                    "source_file": "data/raw/kandy_weather/historical_weather_kandy.csv",
                    "source_column": "precipitation",
                    "transformation": "Open-Meteo historical hourly precipitation (mm)",
                    "data_classification": "WEATHER_OBSERVED",
                    "is_observed": True
                },
                "wind_speed_10m": {
                    "source_file": "data/raw/kandy_weather/historical_weather_kandy.csv",
                    "source_column": "wind_speed_10m",
                    "transformation": "Open-Meteo historical hourly wind speed (km/h)",
                    "data_classification": "WEATHER_OBSERVED",
                    "is_observed": True
                },
                "weather_code": {
                    "source_file": "data/raw/kandy_weather/historical_weather_kandy.csv",
                    "source_column": "weather_code",
                    "transformation": "Open-Meteo WMO weather code",
                    "data_classification": "WEATHER_OBSERVED",
                    "is_observed": True
                }
            }
        }
        lineage_path = METADATA_DIR / "column_lineage.json"
        with open(lineage_path, "w", encoding="utf-8") as f:
            json.dump(lineage, f, indent=2)
        logger.info(f"Updated column lineage saved to: {lineage_path}")

    def run_pipeline(self) -> Dict[str, Any]:
        """Runs the entire real Kandy GPS data preparation pipeline."""
        logger.info("=== STARTING KANDY REAL DATA PIPELINE ===")

        # Step 1: Topology
        logger.info("[Step 1/6] Building Kandy Route Topology...")
        topology_df = self.topology_builder.build_topology()

        # Step 2: Load raw running times & clean
        logger.info("[Step 2/6] Loading raw running times and cleaning...")
        raw_running_path = RAW_DATA_DIR / "kandy_bus" / "bus_running_times_654.csv"
        raw_df = pd.read_csv(raw_running_path)
        cleaned_df, clean_report = self.cleaner.clean_running_times(raw_df)

        # Step 3: Weather Fusion
        logger.info("[Step 3/6] Fusing Open-Meteo Historical Weather for Kandy...")
        weather_df = self.weather_integrator.enrich_with_weather(cleaned_df)

        # Step 4: ETA Target Construction
        logger.info("[Step 4/6] Constructing Real Observed ETA Target (run_time_in_seconds)...")
        labeled_df, target_report = self.target_builder.construct_target(weather_df)

        # Step 5: Causal Feature Engineering & Zero-Leakage Check
        logger.info("[Step 5/6] Engineering Causal Features and Verifying Zero Leakage...")
        featured_df = self.feature_pipeline.transform(labeled_df)
        self.feature_pipeline.export_feature_metadata()

        # Step 6: Chronological Partitioning
        logger.info("[Step 6/6] Chronologically Splitting into Train, Validation, and Stream sets...")
        train_df, val_df, stream_df, split_info = self.splitter.split_dataset(featured_df)

        # Calculate Baseline Distributions
        baseline_stats = self.calculate_baseline_distributions(featured_df)

        # Update Lineage
        self.update_column_lineage()

        # Pipeline Summary Report
        summary = {
            "pipeline_executed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "status": "SUCCESS",
            "provenance_status": "VERIFIED",
            "dataset_name": "Bus Travel Time Data (Kandy, Sri Lanka)",
            "total_raw_records": len(raw_df),
            "clean_records": len(cleaned_df),
            "labeled_valid_records": len(labeled_df),
            "train_records": len(train_df),
            "validation_records": len(val_df),
            "stream_records": len(stream_df),
            "unique_trips": int(featured_df["trip_id"].nunique()),
            "unique_devices": int(featured_df["deviceid"].nunique()),
            "temporal_range": {
                "start": str(featured_df["timestamp_utc"].min()),
                "end": str(featured_df["timestamp_utc"].max())
            },
            "target_statistics": target_report["target_distribution"],
            "features_count": len(self.feature_pipeline.feature_columns),
            "weather_match_rate_pct": round(featured_df["temperature_2m"].notnull().sum() / len(featured_df) * 100, 2)
        }

        dash_path = METADATA_DIR / "kandy_data_quality_dashboard.json"
        with open(dash_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("=== KANDY REAL DATA PIPELINE COMPLETED SUCCESSFULLY ===")
        return summary


if __name__ == "__main__":
    pipeline = KandyPreprocessingPipeline()
    res = pipeline.run_pipeline()
    print("Pipeline Execution Summary:\n", json.dumps(res, indent=2))
