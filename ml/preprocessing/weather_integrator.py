"""TransitVision AI - Meteorological Fusion Engine.
Merges historical hourly Open-Meteo weather records onto transit records by matching UTC timestamp hours.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_DATA_DIR

logger = logging.getLogger("TransitVision.WeatherIntegrator")


class WeatherIntegrator:
    """Performs temporal join between bus transit timestamps and Open-Meteo weather observations."""

    def __init__(self, weather_file: Optional[Path] = None):
        self.weather_file = weather_file or (RAW_DATA_DIR / "historical_weather_dublin.csv")
        self.weather_df: Optional[pd.DataFrame] = None
        self._load_weather()

    def _load_weather(self) -> None:
        if not self.weather_file.exists():
            logger.warning(f"Weather file not found at {self.weather_file}")
            return
        df = pd.read_csv(self.weather_file)
        df["weather_time_utc"] = pd.to_datetime(df["time"], utc=True)
        self.weather_df = df

    def enrich_transit_with_weather(self, transit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges weather observations to transit DataFrame using nearest-hour UTC alignment.
        Preserves missing values without fabricating arbitrary records.
        """
        if self.weather_df is None or len(self.weather_df) == 0:
            logger.warning("No weather data available. Leaving weather columns empty.")
            for col in ["temperature_2m", "precipitation", "rain", "wind_speed_10m", "weather_code"]:
                transit_df[col] = np.nan
            return transit_df

        logger.info(f"Enriching {len(transit_df)} transit records with weather observations...")
        
        # Ensure timestamp is datetime
        df = transit_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

        # Floor to nearest hour for deterministic historical matching
        df["weather_hour_key"] = df["timestamp_utc"].dt.floor("h")

        # Weather columns to join
        weather_cols = [
            "weather_time_utc",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "weather_code",
            "wind_speed_10m"
        ]
        weather_sub = self.weather_df[weather_cols].rename(columns={"weather_time_utc": "weather_hour_key"})

        # Left join on hourly key
        merged = df.merge(weather_sub, on="weather_hour_key", how="left")
        merged = merged.drop(columns=["weather_hour_key"])

        matched_count = merged["temperature_2m"].notnull().sum()
        match_rate = round(matched_count / len(merged) * 100, 2)
        logger.info(f"Weather enrichment complete: {matched_count} / {len(merged)} records matched ({match_rate}%).")

        return merged


if __name__ == "__main__":
    integrator = WeatherIntegrator()
    print("Weather rows loaded:", len(integrator.weather_df) if integrator.weather_df is not None else 0)
