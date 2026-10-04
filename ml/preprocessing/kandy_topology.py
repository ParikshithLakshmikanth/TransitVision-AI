"""TransitVision AI - Kandy Bus Route Topology Engine.
Extracts and builds the route and stop topology for Route 654 (Kandy <-> Digana).
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

logger = logging.getLogger("TransitVision.KandyTopology")


class KandyTopologyBuilder:
    """Builds structured route topology and stop coordinate tables for Kandy bus routes."""

    def __init__(self, kandy_dir: Optional[Path] = None):
        self.kandy_dir = kandy_dir or (RAW_DATA_DIR / "kandy_bus")

    def build_topology(self) -> pd.DataFrame:
        """
        Parses bus_stops_and_terminals_654.csv and constructs a canonical route topology table.
        Saves to data/processed/kandy_route_topology.parquet.
        """
        stops_file = self.kandy_dir / "bus_stops_and_terminals_654.csv"
        if not stops_file.exists():
            raise FileNotFoundError(f"Stops file not found at: {stops_file}")

        logger.info(f"Loading stop topology from: {stops_file}")
        stops_df = pd.read_csv(stops_file)

        # Normalize direction (1: Kandy-Digana, 2: Digana-Kandy)
        direction_map = {
            "Kandy-Digana": 1,
            "Digana-Kandy": 2
        }
        stops_df["direction_id"] = stops_df["direction"].map(direction_map).fillna(1).astype(int)
        stops_df["route_id"] = stops_df["route_id"].astype(str)

        # Build stop sequence per direction
        stops_df["stop_sequence"] = stops_df.groupby("direction_id").cumcount() + 1

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / "kandy_route_topology.parquet"
        stops_df.to_parquet(out_path, index=False)
        logger.info(f"Kandy route topology ({len(stops_df)} stops) saved to: {out_path}")

        return stops_df


if __name__ == "__main__":
    builder = KandyTopologyBuilder()
    builder.build_topology()
