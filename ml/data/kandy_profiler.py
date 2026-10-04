"""TransitVision AI - Kandy Real Dataset Profiler.
Analyzes downloaded Kandy bus travel time tables and generates kandy_raw_profile.json.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_DATA_DIR, METADATA_DIR

logger = logging.getLogger("TransitVision.KandyProfiler")


def sanitize_dict_for_json(d: dict) -> dict:
    """Recursively converts numpy types to standard Python primitives."""
    sanitized = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, np.int64, np.int32)):
            sanitized[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            sanitized[k] = float(v)
        elif isinstance(v, dict):
            sanitized[k] = sanitize_dict_for_json(v)
        elif isinstance(v, list):
            sanitized[k] = [int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, (np.floating, np.float64)) else x for x in v]
        else:
            sanitized[k] = v
    return sanitized


def profile_kandy_raw_datasets() -> Dict[str, Any]:
    """Inspects the raw downloaded Kandy CSV tables and produces kandy_raw_profile.json."""
    kandy_dir = RAW_DATA_DIR / "kandy_bus"
    weather_file = RAW_DATA_DIR / "kandy_weather" / "historical_weather_kandy.csv"

    profile: Dict[str, Any] = {
        "profile_generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_name": "Bus Travel Time Data (Kandy, Sri Lanka)",
        "source_url": "https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data",
        "tables": {}
    }

    # 1. Running Times (Segment Running Time & Travel Duration)
    df_running = pd.read_csv(kandy_dir / "bus_running_times_654.csv")
    profile["tables"]["bus_running_times_654"] = {
        "filename": "bus_running_times_654.csv",
        "row_count": len(df_running),
        "columns": list(df_running.columns),
        "dtypes": {col: str(dtype) for col, dtype in df_running.dtypes.items()},
        "missing_values": {k: int(v) for k, v in df_running.isnull().sum().items()},
        "unique_trips": int(df_running["trip_id"].dropna().nunique()),
        "unique_devices": int(df_running["deviceid"].dropna().nunique()),
        "unique_segments": [int(s) for s in sorted(df_running["segment"].dropna().unique())],
        "directions": [int(d) for d in sorted(df_running["direction"].dropna().unique())],
        "date_range": {
            "min_date": str(df_running["date"].min()),
            "max_date": str(df_running["date"].max())
        },
        "running_time_seconds_stats": {
            "mean": round(float(df_running["run_time_in_seconds"].mean()), 2),
            "median": round(float(df_running["run_time_in_seconds"].median()), 2),
            "min": round(float(df_running["run_time_in_seconds"].min()), 2),
            "max": round(float(df_running["run_time_in_seconds"].max()), 2),
            "std": round(float(df_running["run_time_in_seconds"].std()), 2)
        },
        "segment_length_km_stats": {
            "mean": round(float(df_running["length"].mean()), 4),
            "min": round(float(df_running["length"].min()), 4),
            "max": round(float(df_running["length"].max()), 4)
        }
    }

    # 2. Dwell Times (Bus Stop Dwell)
    df_dwell = pd.read_csv(kandy_dir / "bus_dwell_times_654.csv")
    profile["tables"]["bus_dwell_times_654"] = {
        "filename": "bus_dwell_times_654.csv",
        "row_count": len(df_dwell),
        "columns": list(df_dwell.columns),
        "missing_values": {k: int(v) for k, v in df_dwell.isnull().sum().items()},
        "unique_trips": int(df_dwell["trip_id"].dropna().nunique()),
        "unique_devices": int(df_dwell["deviceid"].dropna().nunique()),
        "unique_bus_stops": int(df_dwell["bus_stop"].dropna().nunique()),
        "dwell_time_seconds_stats": {
            "mean": round(float(df_dwell["dwell_time_in_seconds"].mean()), 2),
            "median": round(float(df_dwell["dwell_time_in_seconds"].median()), 2),
            "min": round(float(df_dwell["dwell_time_in_seconds"].min()), 2),
            "max": round(float(df_dwell["dwell_time_in_seconds"].max()), 2)
        }
    }

    # 3. Stops and Terminals (Topology & GPS Coordinates)
    df_stops = pd.read_csv(kandy_dir / "bus_stops_and_terminals_654.csv")
    profile["tables"]["bus_stops_and_terminals_654"] = {
        "filename": "bus_stops_and_terminals_654.csv",
        "row_count": len(df_stops),
        "columns": list(df_stops.columns),
        "unique_stops": int(df_stops["stop_id"].nunique()),
        "routes": [int(r) if isinstance(r, (np.integer, int)) else str(r) for r in df_stops["route_id"].unique()],
        "directions": [str(d) for d in df_stops["direction"].unique()],
        "geographic_bounds": {
            "min_latitude": float(df_stops["latitude"].min()),
            "max_latitude": float(df_stops["latitude"].max()),
            "min_longitude": float(df_stops["longitude"].min()),
            "max_longitude": float(df_stops["longitude"].max())
        }
    }

    # 4. Trips
    df_trips = pd.read_csv(kandy_dir / "bus_trips_654.csv")
    profile["tables"]["bus_trips_654"] = {
        "filename": "bus_trips_654.csv",
        "row_count": len(df_trips),
        "columns": list(df_trips.columns),
        "unique_trips": int(df_trips["trip_id"].nunique()),
        "unique_devices": int(df_trips["deviceid"].nunique()),
        "duration_in_mins_stats": {
            "mean": round(float(df_trips["duration_in_mins"].mean()), 2),
            "median": round(float(df_trips["duration_in_mins"].median()), 2),
            "min": round(float(df_trips["duration_in_mins"].min()), 2),
            "max": round(float(df_trips["duration_in_mins"].max()), 2)
        }
    }

    # 5. Weather
    df_weather = pd.read_csv(weather_file)
    profile["tables"]["historical_weather_kandy"] = {
        "filename": "historical_weather_kandy.csv",
        "row_count": len(df_weather),
        "temporal_range": {
            "start": str(df_weather["time"].min()),
            "end": str(df_weather["time"].max())
        },
        "temperature_2m_stats": {
            "mean_c": round(float(df_weather["temperature_2m"].mean()), 2),
            "min_c": round(float(df_weather["temperature_2m"].min()), 2),
            "max_c": round(float(df_weather["temperature_2m"].max()), 2)
        },
        "total_precipitation_mm": round(float(df_weather["precipitation"].sum()), 2)
    }

    sanitized_profile = sanitize_dict_for_json(profile)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METADATA_DIR / "kandy_raw_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_profile, f, indent=2)

    logger.info(f"Kandy dataset profile written to: {out_path}")
    return sanitized_profile


if __name__ == "__main__":
    profile_kandy_raw_datasets()
