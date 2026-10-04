"""TransitVision AI - Kandy Bus Dataset & Weather Acquisition.
Manages downloading the Kaggle Kandy Bus Travel Time dataset and matching Open-Meteo historical weather.
"""
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import requests

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_DATA_DIR, METADATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TransitVision.KandyDownload")


def compute_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_kandy_weather(weather_dir: Path) -> Path:
    """Fetches real historical weather for Kandy, Sri Lanka from Open-Meteo."""
    weather_dir.mkdir(parents=True, exist_ok=True)
    out_csv = weather_dir / "historical_weather_kandy.csv"

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 7.2906,
        "longitude": 80.6337,
        "start_date": "2021-10-01",
        "end_date": "2022-11-01",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,surface_pressure,wind_speed_10m",
        "timezone": "UTC"
    }

    logger.info("Fetching historical meteorological records for Kandy (7.2906 N, 80.6337 E) [2021-10-01 to 2022-11-01]...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    df_weather = pd.DataFrame(payload["hourly"])
    df_weather.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df_weather)} hourly weather records to: {out_csv}")
    return out_csv


def generate_kandy_manifest() -> dict:
    """Inspects raw downloaded Kandy bus files and weather records to build manifest."""
    kandy_raw_dir = RAW_DATA_DIR / "kandy_bus"
    kandy_weather_dir = RAW_DATA_DIR / "kandy_weather"

    # Download weather if not present
    weather_csv = kandy_weather_dir / "historical_weather_kandy.csv"
    if not weather_csv.exists():
        download_kandy_weather(kandy_weather_dir)

    manifest = {
        "manifest_version": "1.0.0",
        "dataset_name": "Bus Travel Time Data (Kandy, Sri Lanka)",
        "source_url": "https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": []
    }

    for f in sorted(list(kandy_raw_dir.glob("*.csv")) + list(kandy_weather_dir.glob("*.csv"))):
        sha = compute_sha256(f)
        size = f.stat().st_size
        manifest["files"].append({
            "filename": f.name,
            "relative_path": str(f.relative_to(ROOT_DIR)),
            "size_bytes": size,
            "sha256": sha
        })

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = METADATA_DIR / "kandy_download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2)

    logger.info(f"Kandy download manifest generated at: {manifest_path}")
    return manifest


if __name__ == "__main__":
    generate_kandy_manifest()
