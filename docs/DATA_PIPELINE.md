# TransitVision AI — Data Pipeline Specification

**Provenance Status**: `VERIFIED` *(Phase 4 Real Kandy GPS Transit Dataset)*

This document specifies the data transformation flow, data lineage, and semantic distinctions across the real Kandy bus GPS pipeline.

---

## 1. Pipeline Flow & Provenance Architecture

```
data/raw/kandy_bus/ (Immutable Real GPS Records from Kaggle)
  ├── bus_running_times_654.csv (203,128 segment records)
  ├── bus_dwell_times_654.csv (191,602 dwell records)
  ├── bus_stops_and_terminals_654.csv (31 stop coordinates)
  └── bus_trips_654.csv (10,224 complete trip runs)
data/raw/kandy_weather/
  └── historical_weather_kandy.csv (9,528 hourly records from Open-Meteo)
       ↓
kandy_cleaner.py (Sanitizes timestamps, filters non-positive times)
       ↓
kandy_weather_integrator.py (UTC hourly weather fusion for Kandy)
       ↓
kandy_target_builder.py (eta_to_next_stop_sec from measured GPS run times)
       ↓
kandy_feature_pipeline.py (Causal rolling features, congestion proxies & weather)
       ↓
kandy_splitter.py (Chronological 70/15/15 partitioning)
       ↓
data/processed/
  ├── kandy_eta_training.parquet (140,475 records)
  ├── kandy_eta_validation.parquet (30,102 records)
  ├── kandy_eta_stream.parquet (30,102 records)
  └── kandy_route_topology.parquet (31 stops)
```

---

## 2. Ingestion & Quality Filters

1. **Raw Storage Immutability**: All original downloaded assets in `data/raw/kandy_bus/` are strictly read-only and audited via SHA-256 in `data/metadata/kandy_download_manifest.json`.
2. **Deterministic Data Cleaning (`ml/preprocessing/kandy_cleaner.py`)**:
   - Runtime bounds filter: $3.0\text{s} \le \text{Run Time} \le 3600.0\text{s}$.
   - Retains 200,679 valid records (98.79% retention rate).
   - Full audit trail recorded in `data/metadata/kandy_cleaning_report.json`.
3. **Temporal Weather Fusion (`ml/preprocessing/kandy_weather_integrator.py`)**:
   - Matches records via floored UTC hour (`weather_hour_key`).
   - Achieves 100.0% match rate across the 13-month timeline.

---

## 3. Chronological Dataset Partitioning

To avoid lookahead bias and accurately benchmark streaming concept drift:
- **Training Set (`data/processed/kandy_eta_training.parquet`)**: 140,475 rows (`2021-10-01` to `2022-07-27`).
- **Validation Set (`data/processed/kandy_eta_validation.parquet`)**: 30,102 rows (`2022-07-27` to `2022-09-15`).
- **Streaming Buffer (`data/processed/kandy_eta_stream.parquet`)**: 30,102 rows (`2022-09-15` to `2022-11-01`).
