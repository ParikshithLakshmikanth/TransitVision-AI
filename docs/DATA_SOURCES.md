# TransitVision AI — External Data Sources & Provenance Catalog

**Provenance Status**: `VERIFIED` *(Phase 4 Real Kandy GPS Transit Dataset Integration)*

This document establishes the verified provenance, cryptographic hashes, schemas, and operational status of external datasets acquired for **TransitVision AI**.

---

## 1. Verified Download Manifest & Provenance Audit

| Dataset Name | Source Provider & URL | Local Path | SHA-256 Checksum | Size (Bytes) | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kandy Bus Travel Time Dataset (Primary)** | [Kaggle / University of Peradeniya](https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data) | `data/raw/kandy_bus/` | (4 CSV files audited) | **VERIFIED (Active Primary)** |
| **Kandy Historical Meteorological Archive** | [Open-Meteo Historical API](https://archive-api.open-meteo.com/v1/archive) | `data/raw/kandy_weather/historical_weather_kandy.csv` | `7c447d5c7e10...` | 445,805 | **VERIFIED (Active Primary)** |
| **TFI Dublin Bus Static GTFS** | [Transport for Ireland](https://www.transportforireland.ie/transitData/Data/GTFS_Dublin_Bus.zip) | `data/raw/GTFS_Dublin_Bus.zip` | `1e9d1bf762e8...` | 37,531,684 | **VERIFIED (Secondary / Topology Reference)** |
| **Legacy Dublin Bus AVL CSV** | `opendata.dublincity.ie/TrafficOpenData/` | N/A | N/A | 0 | **UNAVAILABLE (Deprecated)** |

---

## 2. Real Kandy Bus GPS Telemetry Schema & Structure

### Tables in `data/raw/kandy_bus/`:
1. **`bus_running_times_654.csv`** (203,128 rows):
   - `trip_id`: Unique trip run identifier (14,128 unique trips)
   - `deviceid`: On-board GPS hardware modem ID (26 unique buses)
   - `direction`: Direction 1 (Kandy $\rightarrow$ Digana) & Direction 2 (Digana $\rightarrow$ Kandy)
   - `segment`: Roadway segment index (Segments 1–15 & 21–34)
   - `date`: Service date (`2021-10-01` to `2022-11-01`)
   - `start_time`, `end_time`: Physical GPS segment departure and arrival times
   - `run_time_in_seconds`: **Real measured GPS segment travel duration (Target: `eta_to_next_stop_sec`)**
   - `length`: Segment distance in kilometers
2. **`bus_dwell_times_654.csv`** (191,602 rows):
   - `trip_id`, `deviceid`, `direction`, `bus_stop`, `date`, `arrival_time`, `departure_time`, `dwell_time_in_seconds`
3. **`bus_stops_and_terminals_654.csv`** (31 rows):
   - `stop_id`, `route_id`, `direction`, `address`, `latitude`, `longitude` (WGS84 Coordinates)
4. **`bus_trips_654.csv`** (10,224 rows):
   - `trip_id`, `deviceid`, `date`, `start_terminal`, `end_terminal`, `duration_in_mins`

---

## 3. Real Observational vs Derived vs Synthetic Variables

| Feature Name | Type | Classification | Source |
| :--- | :--- | :--- | :--- |
| `deviceid`, `direction`, `segment`, `segment_length_km` | Discrete/Float | **REAL_GPS_DERIVED** | Kandy On-board Bus GPS |
| `timestamp_utc` | `datetime64[ns, UTC]` | **REAL_GPS_DERIVED** | GPS Probe Timestamps |
| `eta_to_next_stop_sec` | `float64` | **REAL_PROCESSED_FROM_GPS** | Measured GPS segment running time |
| `temperature_2m`, `precipitation`, `wind_speed_10m` | `float64` | **WEATHER_OBSERVED** | Open-Meteo Historical Archive |
| `hour`, `minute`, `day_of_week`, `is_peak_period` | Numerical | **DERIVED_FEATURE** | Time encodings |
| `previous_segment_run_time`, `rolling_prev_segment_mean` | `float64` | **DERIVED_FEATURE** | Causal backward rolling statistics |
| `historical_segment_time_mean`, `segment_delay_ratio` | `float64` | **DERIVED_FEATURE** | GPS-derived congestion indicators |
| `simulated_passenger_occupancy_ratio` | `float64` | **SYNTHETIC** | Not present in Phase 4 (0 synthetic variables) |
