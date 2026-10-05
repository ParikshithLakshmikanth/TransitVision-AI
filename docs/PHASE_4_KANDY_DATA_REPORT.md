# KANDY REAL DATA STATUS: VERIFIED

# TransitVision AI — Phase 4 Real Bus GPS Dataset Integration Report
**Dataset: Bus Travel Time Data (Kandy, Sri Lanka)**

**Date of Execution**: September 11, 2026  
**Auditor & ML Engineer**: Lead Software Architect & ML/MLOps Engineer  

---

## 1. Executive Summary & Verification Declaration

The primary real-world telemetry layer of TransitVision AI has been successfully transitioned to genuine GPS-derived bus travel time telemetry from the **Kandy Bus Travel Time Dataset** (University of Peradeniya / Kaggle).

* **Source URL**: [https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data](https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data)
* **Dataset Scale**: 203,128 raw segment observations covering ~13 months of transit operations.
* **Ground-Truth Target**: `eta_to_next_stop_sec` derived directly from measured physical GPS segment travel duration (`run_time_in_seconds`).
* **Environmental Data**: 9,528 hourly meteorological observations retrieved from the Open-Meteo Historical Archive API for Kandy ($7.2906^{\circ}\text{N}, 80.6337^{\circ}\text{E}$).
* **Feature Engineering**: 29 strictly causal features with verified zero future-information leakage.

---

## 2. Dataset Ingestion & Physical Specifications

| File Name | Size (Bytes) | SHA-256 Checksum | Total Rows | Content Description |
| :--- | :--- | :--- | :--- | :--- |
| `bus_running_times_654.csv` | 12,773,065 | `217a45d0b1a4...` | 203,128 | Real GPS segment travel times & distances |
| `bus_dwell_times_654.csv` | 9,448,335 | `009e5691555b...` | 191,602 | Real bus stop arrival, departure & dwell times |
| `bus_stops_and_terminals_654.csv` | 1,733 | `86016cf96a02...` | 31 | Real physical bus stops with WGS84 coordinates |
| `bus_trips_654.csv` | 651,609 | `7f19ed501ce2...` | 10,224 | Complete trip start/end times and total duration |
| `historical_weather_kandy.csv` | 445,805 | `7c447d5c7e10...` | 9,528 | Real hourly meteorological observations |

*All downloaded artifacts are preserved immutably under [`data/raw/kandy_bus/`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/raw/kandy_bus) and [`data/raw/kandy_weather/`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/raw/kandy_weather).*

---

## 3. Measured Dataset & Operational Metrics

* **Total Raw Trajectory Records**: **203,128**
* **Cleaned & Retained Records**: **200,679** ($98.79\%$ retention rate; 2,449 malformed date records quarantined)
* **Unique Physical Trips**: **14,126**
* **Unique On-board GPS Devices / Buses**: **26**
* **Routes Covered**: **Route 654 (Kandy $\leftrightarrow$ Digana)**
* **Directions**: Direction 1 (Kandy $\rightarrow$ Digana) & Direction 2 (Digana $\rightarrow$ Kandy)
* **Roadway Segments**: 30 distinct corridor segments (Segments 1–15 & 21–34)
* **Unique Physical Bus Stops**: **29** (Geographic Bounds: Lat $[7.279117^{\circ}\text{N}, 7.298960^{\circ}\text{N}]$, Lon $[80.634978^{\circ}\text{E}, 80.734720^{\circ}\text{E}]$)
* **Temporal Observation Range**: **2021-10-01 06:39:49 UTC** to **2022-11-01 18:57:10 UTC**

---

## 4. Primary Ground-Truth Target (`eta_to_next_stop_sec`) Distribution

The primary ML target represents the actual measured GPS segment transit time remaining to reach the downstream bus stop:
$$\text{eta\_to\_next\_stop\_sec} = \text{run\_time\_in\_seconds}$$

* **Valid Labeled Examples**: **200,679**
* **Mean Travel Duration**: **188.13 seconds** (~3.14 minutes per segment)
* **Median Duration ($P_{50}$)**: **162.00 seconds**
* **Standard Deviation**: **125.17 seconds**
* **Minimum / Maximum**: **3.00s / 1,199.00s** (Legitimate extreme peak congestion conditions preserved)
* **Percentiles**:
  * $P_{25}$: **105.00s**
  * $P_{50}$: **162.00s**
  * $P_{75}$: **233.00s**
  * $P_{90}$: **370.00s**
  * $P_{95}$: **441.00s**
  * $P_{99}$: **580.00s**

---

## 5. Chronological Dataset Partitioning

The dataset is partitioned strictly chronologically across the 13-month timeline:

| Split Partition | Record Count | Percentage | Date Range (UTC) | Storage File Path |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | **140,475** | $70.00\%$ | `2021-10-01 06:39:49` $\rightarrow$ `2022-07-27 10:14:02` | [`data/processed/kandy_eta_training.parquet`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/processed/kandy_eta_training.parquet) |
| **Validation Set** | **30,102** | $15.00\%$ | `2022-07-27 10:14:02` $\rightarrow$ `2022-09-15 08:32:15` | [`data/processed/kandy_eta_validation.parquet`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/processed/kandy_eta_validation.parquet) |
| **Streaming Replay Set** | **30,102** | $15.00\%$ | `2022-09-15 08:32:15` $\rightarrow$ `2022-11-01 18:57:10` | [`data/processed/kandy_eta_stream.parquet`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/processed/kandy_eta_stream.parquet) |

---

## 6. Weather Fusion & Feature Engineering

* **Meteorological Match Rate**: **100.0%** (All 200,679 records matched with Open-Meteo historical observations for Kandy).
* **Feature Count**: **29 causal features** cataloged in [`data/metadata/kandy_feature_metadata.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/kandy_feature_metadata.json).
* **Causal Feature Lineage**:
  * **Real GPS Observations**: `deviceid`, `direction`, `segment`, `segment_length_km`, `timestamp_utc`.
  * **Real Weather Observations**: `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `wind_speed_10m`, `weather_code`.
  * **Causal Rolling Features**: `previous_segment_run_time`, `rolling_prev_segment_mean`, `rolling_prev_segment_std`, `cumulative_trip_time_sec`.
  * **Traffic Congestion Proxies**: `historical_segment_time_mean`, `segment_delay_ratio`, `congestion_proxy`.
  * **Synthetic Variables**: **NONE** ($0$ synthetic records or artificial noise injected at this stage).
* **Leakage Prevention**: **VERIFIED PASSED** (Automated tests enforce that target and future segment durations are isolated strictly from $X$).

---

## 7. Automated Unit Test Verification

```text
tests/test_kandy_pipeline.py::test_kandy_cleaner_filtering PASSED        [ 10%]
tests/test_kandy_pipeline.py::test_kandy_target_construction_and_no_leakage PASSED [ 20%]
tests/test_kandy_pipeline.py::test_kandy_temporal_splitter_chronology PASSED [ 30%]
tests/test_kandy_pipeline.py::test_kandy_topology PASSED                 [ 40%]
tests/test_pipeline_phase2.py::test_haversine_distance PASSED            [ 50%]
tests/test_pipeline_phase2.py::test_cleaner_filters_anomalies PASSED     [ 60%]
tests/test_pipeline_phase2.py::test_target_builder_and_no_leakage PASSED [ 70%]
tests/test_pipeline_phase2.py::test_temporal_dataset_splitter_chronology PASSED [ 80%]
tests/test_validation.py::test_config_loading PASSED                     [ 90%]
tests/test_validation.py::test_validator_detects_clean_and_anomalous_data PASSED [100%]
============================= 10 passed in 0.45s ==============================
```

---

## 8. Phase 4 Deliverables & Metadata

1. [`docs/KAGGLE_DATA_ACCESS.md`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/docs/KAGGLE_DATA_ACCESS.md) — Kaggle authentication & API configuration guide.
2. [`docs/PHASE_4_KANDY_DATA_REPORT.md`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/docs/PHASE_4_KANDY_DATA_REPORT.md) — This verification report.
3. [`data/metadata/kandy_download_manifest.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/kandy_download_manifest.json) — Cryptographic audit manifest.
4. [`data/metadata/kandy_raw_profile.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/kandy_raw_profile.json) — Schema & statistical profiles.
5. [`data/metadata/kandy_cleaning_report.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/kandy_cleaning_report.json) — Filter audit trail.
6. [`data/metadata/real_eta_target_report.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/real_eta_target_report.json) — Target distribution report.
7. [`data/metadata/kandy_feature_metadata.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/kandy_feature_metadata.json) — 29-feature provenance catalog.
8. [`data/metadata/kandy_dataset_split.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/kandy_dataset_split.json) — Chronological split metadata.
9. [`data/metadata/kandy_baseline_distributions.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/kandy_baseline_distributions.json) — Concept drift reference distributions.
10. [`data/metadata/column_lineage.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/column_lineage.json) — Unified column provenance.
11. [`data/processed/kandy_eta_training.parquet`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/processed/kandy_eta_training.parquet) (140,475 records).
12. [`data/processed/kandy_eta_validation.parquet`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/processed/kandy_eta_validation.parquet) (30,102 records).
13. [`data/processed/kandy_eta_stream.parquet`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/processed/kandy_eta_stream.parquet) (30,102 records).

---

## 9. Stop Condition

In strict accordance with Phase 4 instructions:
* **No ML models have been trained yet.**
* **No frontend components have been built.**
* **No synthetic disruptions have been injected.**

The repository now contains a verified, real-world GPS-derived transit dataset ready for ML model training, evaluation, and drift monitoring in the next phase.
