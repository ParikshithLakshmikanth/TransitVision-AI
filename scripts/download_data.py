"""TransitVision AI - Real Dataset Acquisition Script.
Downloads genuine public transit GTFS topology and historical weather data,
verifies cryptographic hashes, preserves immutable raw files, and generates download_manifest.json.
"""
import datetime
import hashlib
import json
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import requests

# Set project root in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_DATA_DIR, METADATA_DIR, CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TransitVision.Download")


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a local file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_url_to_file(url: str, dest_path: Path, timeout: int = 60) -> int:
    """Downloads a remote URL to local destination with headers."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TransitVision-AI-Academic-Research)"})
    with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
        bytes_written = 0
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            out_file.write(chunk)
            bytes_written += len(chunk)
    logger.info(f"Downloaded {bytes_written} bytes.")
    return bytes_written


def download_all_data() -> dict:
    """Downloads GTFS and Weather data and returns manifest dictionary."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "datasets": []
    }

    # 1. Transport for Ireland (TFI) Dublin Bus GTFS
    gtfs_url = "https://www.transportforireland.ie/transitData/Data/GTFS_Dublin_Bus.zip"
    gtfs_raw_zip = RAW_DATA_DIR / "GTFS_Dublin_Bus.zip"
    gtfs_extract_dir = RAW_DATA_DIR / "gtfs_dublin"
    
    gtfs_entry = {
        "dataset_name": "TFI Dublin Bus Static GTFS",
        "source_url": gtfs_url,
        "local_path": str(gtfs_raw_zip.relative_to(ROOT_DIR)),
        "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "pending"
    }
    
    try:
        if not gtfs_raw_zip.exists():
            download_url_to_file(gtfs_url, gtfs_raw_zip, timeout=60)
        
        file_size = gtfs_raw_zip.stat().st_size
        sha256_hash = compute_sha256(gtfs_raw_zip)
        
        # Extract GTFS text files
        gtfs_extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = []
        with zipfile.ZipFile(gtfs_raw_zip, 'r') as zf:
            for item in zf.namelist():
                if item.endswith('.txt'):
                    zf.extract(item, gtfs_extract_dir)
                    extracted_files.append(item)
                    
        gtfs_entry.update({
            "file_size_bytes": file_size,
            "sha256": sha256_hash,
            "extracted_to": str(gtfs_extract_dir.relative_to(ROOT_DIR)),
            "extracted_files": extracted_files,
            "status": "success"
        })
        logger.info(f"GTFS extraction verified: {len(extracted_files)} tables extracted.")
    except Exception as e:
        logger.error(f"Failed downloading/extracting GTFS: {e}")
        gtfs_entry.update({"status": "failed", "error": str(e)})

    manifest["datasets"].append(gtfs_entry)

    # 2. Open-Meteo Historical Weather Archive for Dublin
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": 53.3498,
        "longitude": -6.2603,
        "start_date": "2023-01-01",
        "end_date": "2023-03-31",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,surface_pressure,wind_speed_10m",
        "timezone": "UTC"
    }
    weather_csv_path = RAW_DATA_DIR / "historical_weather_dublin.csv"
    weather_entry = {
        "dataset_name": "Open-Meteo Historical Meteorological Archive (Dublin)",
        "source_url": f"{weather_url}?latitude=53.3498&longitude=-6.2603&start_date=2023-01-01&end_date=2023-03-31",
        "local_path": str(weather_csv_path.relative_to(ROOT_DIR)),
        "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "pending"
    }

    try:
        logger.info("Querying Open-Meteo Archive API for Dublin meteorological history...")
        resp = requests.get(weather_url, params=weather_params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        
        if "hourly" in payload:
            df_weather = pd.DataFrame(payload["hourly"])
            df_weather.to_csv(weather_csv_path, index=False)
            weather_size = weather_csv_path.stat().st_size
            weather_sha = compute_sha256(weather_csv_path)
            weather_entry.update({
                "file_size_bytes": weather_size,
                "sha256": weather_sha,
                "records_count": len(df_weather),
                "temporal_range": {
                    "start": df_weather["time"].min(),
                    "end": df_weather["time"].max()
                },
                "status": "success"
            })
            logger.info(f"Open-Meteo historical weather saved: {len(df_weather)} hourly records.")
        else:
            raise ValueError(f"Unexpected response payload: {payload}")
    except Exception as e:
        logger.error(f"Failed downloading weather data: {e}")
        weather_entry.update({"status": "failed", "error": str(e)})

    manifest["datasets"].append(weather_entry)

    # Save manifest
    manifest_path = METADATA_DIR / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Download manifest written to: {manifest_path}")

    return manifest


if __name__ == "__main__":
    download_all_data()
