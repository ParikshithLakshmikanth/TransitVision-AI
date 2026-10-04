"""TransitVision AI - GTFS Transit Topology Module.
Extracts and integrates routes, trips, stops, shapes, and stop sequences into a canonical transit topology table.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_DATA_DIR, INTERIM_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TransitVision.GTFSTopology")


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class GTFSTopologyBuilder:
    """Parses raw GTFS tables and builds clean route-stop sequence matrices with spatial metrics."""

    def __init__(self, gtfs_dir: Optional[Path] = None):
        self.gtfs_dir = gtfs_dir or (RAW_DATA_DIR / "gtfs_dublin")
        self.stops_df: Optional[pd.DataFrame] = None
        self.routes_df: Optional[pd.DataFrame] = None
        self.trips_df: Optional[pd.DataFrame] = None
        self.stop_times_df: Optional[pd.DataFrame] = None

    def load_gtfs_tables(self, sample_routes: Optional[list] = None) -> None:
        """Loads and filters GTFS tables."""
        logger.info(f"Loading GTFS tables from {self.gtfs_dir}...")
        self.routes_df = pd.read_csv(
            self.gtfs_dir / "routes.txt",
            usecols=["route_id", "route_short_name", "route_long_name"]
        )
        self.stops_df = pd.read_csv(
            self.gtfs_dir / "stops.txt",
            usecols=["stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon"]
        )
        self.trips_df = pd.read_csv(
            self.gtfs_dir / "trips.txt",
            usecols=["route_id", "service_id", "trip_id", "direction_id", "block_id", "shape_id"]
        )
        
        # Load stop_times
        st_dtypes = {"trip_id": str, "arrival_time": str, "departure_time": str, "stop_id": str, "stop_sequence": int}
        self.stop_times_df = pd.read_csv(
            self.gtfs_dir / "stop_times.txt",
            dtype=st_dtypes,
            usecols=list(st_dtypes.keys())
        )
        logger.info(f"Loaded {len(self.routes_df)} routes, {len(self.stops_df)} stops, {len(self.trips_df)} trips, {len(self.stop_times_df)} stop times.")

    def build_integrated_stop_sequences(self, max_trips: Optional[int] = None) -> pd.DataFrame:
        """
        Merges stop_times with stops, trips, and routes to build an ordered sequence table.
        Computes inter-stop distances and cumulative distances along each trip pattern.
        """
        if self.routes_df is None:
            self.load_gtfs_tables()

        logger.info("Building integrated stop sequences...")
        trips_subset = self.trips_df
        if max_trips is not None:
            trips_subset = trips_subset.head(max_trips)

        # Merge trips with routes
        merged_trips = trips_subset.merge(self.routes_df, on="route_id", how="inner")

        # Merge with stop times
        merged = merged_trips.merge(self.stop_times_df, on="trip_id", how="inner")

        # Merge with stop coordinates
        merged = merged.merge(self.stops_df, on="stop_id", how="inner")

        # Sort by trip and stop sequence
        merged = merged.sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)

        # Calculate next stop coordinates and segment distances
        merged["next_stop_id"] = merged.groupby("trip_id")["stop_id"].shift(-1)
        merged["next_stop_lat"] = merged.groupby("trip_id")["stop_lat"].shift(-1)
        merged["next_stop_lon"] = merged.groupby("trip_id")["stop_lon"].shift(-1)

        # Compute inter-stop distance
        valid_coords = merged["next_stop_lat"].notnull()
        merged.loc[valid_coords, "segment_distance_km"] = haversine_distance_km(
            merged.loc[valid_coords, "stop_lat"].values,
            merged.loc[valid_coords, "stop_lon"].values,
            merged.loc[valid_coords, "next_stop_lat"].values,
            merged.loc[valid_coords, "next_stop_lon"].values
        )
        merged["segment_distance_km"] = merged["segment_distance_km"].fillna(0.0)

        # Compute total stops on trip and stops remaining
        total_stops_per_trip = merged.groupby("trip_id")["stop_sequence"].transform("max")
        merged["total_stops_on_trip"] = total_stops_per_trip
        merged["stops_remaining"] = total_stops_per_trip - merged["stop_sequence"]

        # Cumulative distance
        merged["cumulative_distance_km"] = merged.groupby("trip_id")["segment_distance_km"].cumsum()

        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = INTERIM_DATA_DIR / "gtfs_stop_sequences.parquet"
        merged.to_parquet(out_path, index=False)
        logger.info(f"Saved integrated GTFS stop sequence table ({len(merged)} records) to: {out_path}")

        return merged


if __name__ == "__main__":
    builder = GTFSTopologyBuilder()
    builder.load_gtfs_tables()
    builder.build_integrated_stop_sequences()
