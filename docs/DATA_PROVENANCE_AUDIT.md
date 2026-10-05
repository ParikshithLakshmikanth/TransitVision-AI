# PROVENANCE STATUS: REQUIRES CORRECTION

# TransitVision AI — Rigorous Data Provenance & Ground-Truth Audit Report

**Audit Date**: September 11, 2026  
**Audit Scope**: End-to-end lineage from `data/raw/` through `data/interim/` to `data/processed/` and all metadata catalogs.  
**Auditor**: Lead Software Architect & ML/MLOps Engineer  

---

## 1. Executive Summary & Critical Audit Finding

This audit was conducted prior to training machine learning models to verify the origin and legitimacy of all records in the processed dataset (`data/processed/eta_training.parquet`, `eta_validation.parquet`, `eta_stream.parquet`).

### Key Finding:
* The raw Dublin Bus GPS AVL download link (`https://opendata.dublincity.ie/TrafficOpenData/sir010113-310113.zip`) was unavailable (HTTP 400 Bad Request / deprecated legacy endpoint by Dublin City Council).
* While **Transport for Ireland (TFI) Static GTFS** (37.5 MB, 10 tables) and **Open-Meteo Historical Weather** (2,160 hourly records) were successfully acquired and verified, the **283,696 trajectory records** and **279,496 ETA labels** were **GENERATED FROM GTFS SCHEDULES EXPANDED ACROSS CALENDAR DATES**, rather than observed from physical on-board GPS/AVL hardware modems.
* Therefore, the primary ML target currently represents **SCHEDULE-DERIVED RUNNING TIME**, NOT genuine observed bus arrival timestamps.
* In accordance with academic integrity guidelines, **NO ML MODELS WILL BE TRAINED** until genuine observed transit telemetry is ingested.

---

## 2. Exact Source of Every Dataset

| Dataset / Table | Local Path | Origin Source & URL | Size / Row Count | Genuine Observed vs Synthesized / Inferred |
| :--- | :--- | :--- | :--- | :--- |
| **TFI Dublin GTFS Archive** | `data/raw/GTFS_Dublin_Bus.zip` | [Transport for Ireland](https://www.transportforireland.ie/transitData/Data/GTFS_Dublin_Bus.zip) | 37,531,684 bytes / 10 tables | **REAL** static timetable & topology |
| **GTFS Stops Table** | `data/raw/gtfs_dublin/stops.txt` | TFI GTFS Feed | 4,337 stops | **REAL** physical stop geolocations |
| **GTFS Stop Times Table** | `data/raw/gtfs_dublin/stop_times.txt` | TFI GTFS Feed | 3,035,625 rows | **REAL** scheduled timetable |
| **GTFS Trips Table** | `data/raw/gtfs_dublin/trips.txt` | TFI GTFS Feed | 56,054 trips | **REAL** scheduled trip definitions |
| **Open-Meteo Weather Archive** | `data/raw/historical_weather_dublin.csv` | [Open-Meteo API](https://archive-api.open-meteo.com/v1/archive) | 99,444 bytes / 2,160 rows | **REAL** historical weather observations |
| **Processed ETA Parquet** | `data/processed/eta_training.parquet` (and val/stream) | `pipelines/run_preprocessing_pipeline.py` | 279,496 total records | **GTFS SCHEDULE-DERIVED / INFERRED** |

---

## 3. Exact Source of the 283,696 Trajectory Records

The 283,696 records were produced by `generate_trajectory_stream_from_gtfs()` in [`pipelines/run_preprocessing_pipeline.py`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/pipelines/run_preprocessing_pipeline.py#L58-L118):
1. The script sampled 150 unique scheduled trips across 3 routes from `data/interim/gtfs_stop_sequences.parquet`.
2. It iterated across 28 synthetic operational calendar dates (`2023-01-01` to `2023-01-28`).
3. For each trip, it parsed scheduled arrival strings `arrival_time` (HH:MM:SS) and combined them with the calendar date to produce `timestamp_utc`.
4. It computed inter-stop scheduled durations:
   $$\text{segment\_time\_sec} = \text{next\_stop\_arrival\_utc} - \text{timestamp\_utc}$$
5. It inferred instantaneous speeds as:
   $$\text{current\_speed\_kmh} = \frac{\text{segment\_distance\_km}}{\text{segment\_time\_sec} / 3600.0}$$
6. It created pseudo-vehicle IDs `vehicle_id = "BUS_" + block_id.split("_")[-1]`.

---

## 4. Exact Source of the 279,496 ETA Labels

* **Target Variable**: `eta_to_next_stop_sec`
* **Calculation in Code**: [`ml/features/target_builder.py`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/ml/features/target_builder.py#L56-L65)
  $$\text{eta\_to\_next\_stop\_sec} = \text{next\_stop\_arrival\_utc} - \text{timestamp\_utc}$$
* **Ground Truth Source**: Timetable difference between consecutive stops from `stop_times.txt`.
* **Classification**: **SCHEDULE-DERIVED TARGET** (Scheduled inter-stop running time), **NOT** actual observed delay or sensor-measured arrival time.

---

## 5. Definitive Data Classification Catalog

| Column Name | Source File | Source Column | Classification | Observed? |
| :--- | :--- | :--- | :--- | :--- |
| `route_id`, `route_short_name` | `gtfs_dublin/routes.txt` | `route_id`, `route_short_name` | `REAL_GTFS_TOPOLOGY` | Yes |
| `trip_id`, `direction_id`, `block_id` | `gtfs_dublin/trips.txt` | `trip_id`, `direction_id`, `block_id` | `REAL_GTFS_TOPOLOGY` | Yes |
| `stop_id`, `stop_name` | `gtfs_dublin/stops.txt` | `stop_id`, `stop_name` | `REAL_GTFS_TOPOLOGY` | Yes |
| `latitude`, `longitude` | `gtfs_dublin/stops.txt` | `stop_lat`, `stop_lon` | `REAL_GTFS_TOPOLOGY` | Yes |
| `stop_sequence` | `gtfs_dublin/stop_times.txt` | `stop_sequence` | `REAL_GTFS_TIMETABLE` | Yes |
| `arrival_time`, `departure_time` | `gtfs_dublin/stop_times.txt` | `arrival_time`, `departure_time` | `REAL_GTFS_TIMETABLE` | Yes |
| `temperature_2m`, `precipitation`, `wind_speed_10m` | `historical_weather_dublin.csv` | `temperature_2m`, `precipitation`, `wind_speed_10m` | `REAL_OBSERVED_WEATHER` | Yes |
| `service_date`, `timestamp_utc` | `run_preprocessing_pipeline.py` | `date + arrival_time` | `GTFS_SCHEDULE_EXPANDED` | No |
| `vehicle_id` | `run_preprocessing_pipeline.py` | `block_id` prefix | `INFERRED_FROM_GTFS` | No |
| `segment_distance_km` | `gtfs_topology.py` | `stop_lat, stop_lon, next_stop_lat, next_stop_lon` | `DERIVED_FROM_REAL_GTFS` | Yes |
| `current_speed_kmh`, `recent_speed_mean` | `run_preprocessing_pipeline.py` | `segment_distance_km / segment_time_sec` | `INFERRED_FROM_GTFS_SCHEDULE` | No |
| `is_at_stop`, `is_congested` | `run_preprocessing_pipeline.py` | `thresholding` | `INFERRED_FROM_GTFS_SCHEDULE` | No |
| `eta_to_next_stop_sec` | `target_builder.py` | `next_stop_arrival_utc - timestamp_utc` | `SCHEDULE_DERIVED_TARGET` | No |
| `simulated_passenger_occupancy_ratio` | `feature_pipeline.py` | Constant ($0.35$) | `SYNTHETIC` | No |
| `injected_disturbance_intensity` | `feature_pipeline.py` | Constant ($0.0$) | `SYNTHETIC` | No |
| `is_incident_active` | `feature_pipeline.py` | Constant ($0$) | `SYNTHETIC` | No |

---

## 6. GPS Realism & Telemetry Analysis

Quantitative evaluation of the 279,496 processed records reveals characteristics inconsistent with genuine continuous GPS tracking probes:

* **Total Processed Records**: 279,496
* **Unique Coordinates**: **205** *(exactly equal to the 205 designated GTFS bus stops)*
* **Intermediate Breadcrumbs**: **0** *(no GPS traces between stops, no traffic light stops, no GPS coordinate jitter)*
* **Stationary Points**: $0.0\%$
* **Speed Range**: $5.0\text{ km/h}$ to $58.42\text{ km/h}$ (Mean: $17.67\text{ km/h}$, Std: $7.10\text{ km/h}$)
* **Temporal Interval ($\Delta t$)**: Mean $83.38\text{s}$, Median $67.0\text{s}$, Min $10.0\text{s}$, Max $819.0\text{s}$

**Verdict**: The records represent discrete scheduled stop arrival events derived from timetable arithmetic, rather than physical probe breadcrumbs from a moving vehicle.

---

## 7. GTFS Integration Analysis

* **GTFS Source**: `https://www.transportforireland.ie/transitData/Data/GTFS_Dublin_Bus.zip` (37,531,684 bytes).
* **Integrity**: 100% valid GTFS schema with 116 routes, 4,337 stops, 56,054 trips, and 3,035,625 stop times.
* **Semantics**: GTFS schedules specify expected arrival times, but do not record actual real-world transit delays, traffic congestion, or dwell variations.

---

## 8. Weather Provenance Analysis

* **Source**: Open-Meteo Historical Meteorological Archive REST API.
* **Requested Coordinates**: Latitude $53.3498^{\circ}\text{N}$, Longitude $-6.2603^{\circ}\text{W}$ (Dublin City Center).
* **Date Range**: `2023-01-01` to `2023-03-31` (2,160 hourly records).
* **Status**: **GENUINE HISTORICAL OBSERVATIONS**. Correctly aligned with UTC timestamps via nearest/backward hour join.

---

## 9. Temporal Leakage & Split Audit

* **Training Set**: 195,647 records (`2023-01-01 05:32:00` to `2023-01-20 20:41:20 UTC`)
* **Validation Set**: 41,924 records (`2023-01-20 20:41:20` to `2023-01-24 23:25:00 UTC`)
* **Streaming Set**: 41,925 records (`2023-01-24 23:25:00` to `2023-01-29 00:19:49 UTC`)
* **Temporal Overlap**: Zero timestamp overlap between partitions.
* **Feature Leakage**: Feature matrix $X$ contains no target or future arrival timestamps.

---

## 10. Required Corrections & Public Replacement Candidates

To satisfy the project requirement of training on genuine real-world GPS/AVL telemetry, we must ingest a verified public AVL trajectory dataset:

### Recommended Genuine Open Transit Datasets:
1. **Rio de Janeiro Bus GPS Fleet Probe Dataset** (Kaggle / Prefeitura do Rio):
   - **Source**: `https://www.kaggle.com/datasets/igorbalteiro/gps-data-from-rio-de-janeiro-buses`
   - **Properties**: Millions of real-time AVL probe records with physical GPS coordinates, instantaneous vehicle speeds, timestamps, vehicle IDs, and route lines.
   - **License**: Creative Commons / Open Public Data.
2. **Transjakarta Bus GPS Dataset** (Kaggle):
   - **Source**: `https://www.kaggle.com/datasets/aditirout/transjakarta-bus-gps-data`
   - **Properties**: Real GPS trajectories with vehicle codes, trip IDs, timestamps, and route corridors.
   - **License**: Open Access.
3. **Chicago Transit Authority (CTA) Bus Tracker API / Historical Archives**:
   - **Source**: `https://data.cityofchicago.org/`
   - **Properties**: Real-time and historical bus GPS locations, speeds, and actual arrival delays.
   - **License**: City of Chicago Open Data Terms (Public Domain).
4. **Transport for Ireland (TFI) GTFS-Realtime Stream / RTPI**:
   - **Source**: `https://data.transportforireland.ie/` / `https://developer.nationaltransport.ie/`
   - **Properties**: Live vehicle position feeds broadcasting real-time bus locations and delays across Dublin.

---

## 11. Conclusion & Next Steps

* **Status**: `PROVENANCE STATUS: REQUIRES CORRECTION`
* **Action Taken**:
  - Full audit completed and recorded in [`docs/DATA_PROVENANCE_AUDIT.md`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/docs/DATA_PROVENANCE_AUDIT.md).
  - Column-level lineage saved in [`data/metadata/column_lineage.json`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/data/metadata/column_lineage.json).
  - All documentation updated to eliminate misleading claims of "observed AVL delay".
  - Machine learning model training halted until a real-world AVL probe stream is connected.
