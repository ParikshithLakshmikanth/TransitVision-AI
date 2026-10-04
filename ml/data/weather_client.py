"""Client for retrieving historical meteorological observations from Open-Meteo API."""
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class OpenMeteoWeatherClient:
    """Fetches real historical weather data from Open-Meteo archive API without requiring API keys."""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or self.BASE_URL

    def fetch_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        hourly_variables: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Fetch hourly historical weather observations for a given coordinate bounding point and date range.

        Args:
            latitude: Geographic latitude (e.g., 53.3498 for Dublin)
            longitude: Geographic longitude (e.g., -6.2603 for Dublin)
            start_date: YYYY-MM-DD string
            end_date: YYYY-MM-DD string
            hourly_variables: List of metrics (temperature_2m, precipitation, rain, weather_code, wind_speed_10m)

        Returns:
            pd.DataFrame with indexed timestamp and weather metrics.
        """
        if hourly_variables is None:
            hourly_variables = [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "weather_code",
                "surface_pressure",
                "wind_speed_10m"
            ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_variables),
            "timezone": "UTC"
        }

        logger.info(f"Querying Open-Meteo Archive API: {self.base_url} for ({latitude}, {longitude}) [{start_date} to {end_date}]")
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        if "hourly" not in payload:
            raise ValueError(f"Unexpected response structure from Open-Meteo: {payload}")

        df_weather = pd.DataFrame(payload["hourly"])
        df_weather["time"] = pd.to_datetime(df_weather["time"], utc=True)
        logger.info(f"Successfully retrieved {len(df_weather)} hourly weather records from Open-Meteo.")
        return df_weather
