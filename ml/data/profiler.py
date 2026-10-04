"""TransitVision AI - Dataset Profiling Module.
Inspects raw GTFS tables and Open-Meteo weather datasets to generate raw_dataset_profile.json.
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

logger = logging.getLogger("TransitVision.Profiler")


def profile_raw_datasets() -> Dict[str, Any]:
    """Inspects the raw downloaded tables and writes raw_dataset_profile.json."""
    gtfs_dir = RAW_DATA_DIR / "gtfs_dublin"
    weather_file = RAW_DATA_DIR / "historical_weather_dublin.csv"

    profile: Dict[str, Any] = {
        "profile_generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "datasets": {}
    }

    # 1. Routes
    routes_df = pd.read_csv(gtfs_dir / "routes.txt")
    profile["datasets"]["routes"] = {
        "file": "routes.txt",
        "row_count": len(routes_df),
        "columns": list(routes_df.columns),
        "dtypes": {col: str(dtype) for col, dtype in routes_df.dtypes.items()},
        "unique_routes_count": int(routes_df["route_id"].nunique()),
        "sample_routes": routes_df["route_short_name"].head(10).tolist(),
        "missing_values": routes_df.isnull().sum().to_dict(),
        "duplicates": int(routes_df.duplicated().sum())
    }

    # 2. Stops
    stops_df = pd.read_csv(gtfs_dir / "stops.txt")
    profile["datasets"]["stops"] = {
        "file": "stops.txt",
        "row_count": len(stops_df),
        "columns": list(stops_df.columns),
        "unique_stops_count": int(stops_df["stop_id"].nunique()),
        "geographic_bounds": {
            "min_latitude": float(stops_df["stop_lat"].min()),
            "max_latitude": float(stops_df["stop_lat"].max()),
            "min_longitude": float(stops_df["stop_lon"].min()),
            "max_longitude": float(stops_df["stop_lon"].max())
        },
        "missing_values": stops_df.isnull().sum().to_dict(),
        "duplicates": int(stops_df.duplicated().sum())
    }

    # 3. Trips
    trips_df = pd.read_csv(gtfs_dir / "trips.txt")
    profile["datasets"]["trips"] = {
        "file": "trips.txt",
        "row_count": len(trips_df),
        "columns": list(trips_df.columns),
        "unique_trips_count": int(trips_df["trip_id"].nunique()),
        "unique_blocks_count": int(trips_df["block_id"].nunique()),
        "unique_routes_in_trips": int(trips_df["route_id"].nunique()),
        "unique_shapes_count": int(trips_df["shape_id"].nunique()),
        "missing_values": trips_df.isnull().sum().to_dict(),
        "duplicates": int(trips_df.duplicated().sum())
    }

    # 4. Stop Times (Chunked inspection due to 3M rows)
    stop_times_path = gtfs_dir / "stop_times.txt"
    st_dtypes = {"trip_id": str, "arrival_time": str, "departure_time": str, "stop_id": str, "stop_sequence": int}
    
    st_row_count = 0
    st_missing = {}
    st_unique_trips = set()
    st_unique_stops = set()
    
    for chunk in pd.read_csv(stop_times_path, chunksize=250000, dtype=st_dtypes, usecols=list(st_dtypes.keys())):
        st_row_count += len(chunk)
        st_unique_trips.update(chunk["trip_id"].unique())
        st_unique_stops.update(chunk["stop_id"].unique())
        for c, count in chunk.isnull().sum().items():
            st_missing[c] = st_missing.get(c, 0) + int(count)

    profile["datasets"]["stop_times"] = {
        "file": "stop_times.txt",
        "row_count": st_row_count,
        "unique_trips_referenced": len(st_unique_trips),
        "unique_stops_referenced": len(st_unique_stops),
        "missing_values": st_missing
    }

    # 5. Weather
    weather_df = pd.read_csv(weather_file)
    profile["datasets"]["weather"] = {
        "file": "historical_weather_dublin.csv",
        "row_count": len(weather_df),
        "columns": list(weather_df.columns),
        "temporal_range": {
            "start": str(weather_df["time"].min()),
            "end": str(weather_df["time"].max())
        },
        "missing_values": weather_df.isnull().sum().to_dict(),
        "duplicates": int(weather_df.duplicated().sum()),
        "metrics_summary": {
            "mean_temp_c": float(weather_df["temperature_2m"].mean()),
            "min_temp_c": float(weather_df["temperature_2m"].min()),
            "max_temp_c": float(weather_df["temperature_2m"].max()),
            "mean_wind_kmh": float(weather_df["wind_speed_10m"].mean()),
            "total_precip_mm": float(weather_df["precipitation"].sum())
        }
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    out_profile_path = METADATA_DIR / "raw_dataset_profile.json"
    with open(out_profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Raw dataset profile saved to: {out_profile_path}")
    return profile


if __name__ == "__main__":
    profile_raw_datasets()
