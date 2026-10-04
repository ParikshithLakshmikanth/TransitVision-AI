"""TransitVision AI - Kandy Weather Fusion Module.
Joins Open-Meteo historical weather observations for Kandy onto transit segment records.
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

logger = logging.getLogger("TransitVision.KandyWeather")


class KandyWeatherIntegrator:
    """Fuses historical Kandy meteorological records onto transit segment records."""

    def __init__(self, weather_file: Optional[Path] = None):
        self.weather_file = weather_file or (RAW_DATA_DIR / "kandy_weather" / "historical_weather_kandy.csv")
        self.weather_df: Optional[pd.DataFrame] = None
        self._load_weather()

    def _load_weather(self) -> None:
        if not self.weather_file.exists():
            logger.warning(f"Kandy weather file not found at: {self.weather_file}")
            return
        df = pd.read_csv(self.weather_file)
        df["weather_time_utc"] = pd.to_datetime(df["time"], utc=True)
        self.weather_df = df
        logger.info(f"Loaded {len(df)} historical weather observations for Kandy.")

    def enrich_with_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merges hourly weather records onto transit records based on UTC hour."""
        if self.weather_df is None or len(self.weather_df) == 0:
            logger.warning("No weather data available. Assigning NaN to weather columns.")
            for col in ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "wind_speed_10m", "weather_code"]:
                df[col] = np.nan
            return df

        working_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(working_df["timestamp_utc"]):
            working_df["timestamp_utc"] = pd.to_datetime(working_df["timestamp_utc"], utc=True)

        working_df["weather_hour_key"] = working_df["timestamp_utc"].dt.floor("h")

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

        merged = working_df.merge(weather_sub, on="weather_hour_key", how="left")
        merged = merged.drop(columns=["weather_hour_key"])

        matched_count = merged["temperature_2m"].notnull().sum()
        match_rate = round(matched_count / len(merged) * 100, 2)
        logger.info(f"Weather enrichment complete: {matched_count} / {len(merged)} records matched ({match_rate}%).")

        return merged


if __name__ == "__main__":
    integrator = KandyWeatherIntegrator()
    print("Weather rows:", len(integrator.weather_df) if integrator.weather_df is not None else 0)
