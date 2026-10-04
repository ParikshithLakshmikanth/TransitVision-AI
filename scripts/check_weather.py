import pandas as pd
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import PROCESSED_DATA_DIR

stream_df = pd.read_parquet(PROCESSED_DATA_DIR / "kandy_eta_stream.parquet")
train_df = pd.read_parquet(PROCESSED_DATA_DIR / "kandy_eta_training.parquet")

p_str_gt0 = float((stream_df["precipitation"] > 0).mean()) * 100
p_tr_gt0 = float((train_df["precipitation"] > 0).mean()) * 100

print("--- REAL STREAM WEATHER DISTRIBUTION (Oct 2022) ---")
print(f"Stream precipitation > 0: {p_str_gt0:.2f}% | max: {stream_df['precipitation'].max():.2f} mm/h | mean: {stream_df['precipitation'].mean():.2f} mm/h")
print("Stream weather codes:", stream_df["weather_code"].value_counts().to_dict())

print("\n--- REAL TRAINING WEATHER DISTRIBUTION (Oct 2021 - Jul 2022) ---")
print(f"Train precipitation > 0: {p_tr_gt0:.2f}% | max: {train_df['precipitation'].max():.2f} mm/h | mean: {train_df['precipitation'].mean():.2f} mm/h")
print("Train weather codes:", train_df["weather_code"].value_counts().head(5).to_dict())
