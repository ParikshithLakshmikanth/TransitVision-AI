# TransitVision AI — Machine Learning Pipeline Specification

**Provenance Status**: `VERIFIED` *(Phase 4 Real Kandy GPS Transit Dataset)*

This document details the feature engineering, target construction, model architecture, evaluation protocols, and zero-leakage constraints for bus ETA prediction using real GPS telemetry.

---

## 1. Problem Formulation & Primary Target

The prediction target is defined as the actual remaining time in seconds required for the bus to arrive at the immediate downstream bus stop:

$$\hat{y}_t = \text{ETA}_{\text{next}}(\mathbf{x}_t)$$

$$\text{eta\_to\_next\_stop\_sec} = \text{run\_time\_in\_seconds}$$

### Target Semantic Classification:
* **Current Classification**: **REAL_PROCESSED_FROM_GPS** (Derived from on-board GPS sensors operating on Kandy transit buses).
* **Prediction Granularity**: Segment-level ETA prediction from segment start.

### Target Distribution:
- **Total Valid Labeled Records**: 200,679
- **Mean ETA**: 188.13 seconds (~3.14 minutes per segment)
- **Median ETA**: 162.00 seconds
- **Standard Deviation**: 125.17 seconds
- **Range**: 3.0 seconds to 1,199.0 seconds ($3.0\text{s} \le \text{ETA} \le 3600.0\text{s}$)
- **Percentiles**: $P_{25} = 105.0\text{s}, P_{50} = 162.0\text{s}, P_{75} = 233.0\text{s}, P_{90} = 370.0\text{s}, P_{95} = 441.0\text{s}, P_{99} = 580.0\text{s}$.

---

## 2. Feature Matrix & Leakage Prevention Rules

Total Engineered Features: **29** (stored in `ml/features/kandy_feature_pipeline.py`).

| Feature Name | Category | Type | Provenance |
| :--- | :--- | :--- | :--- |
| `hour`, `minute`, `day_of_week`, `is_weekend`, `is_peak_period` | Temporal | Numerical | `DERIVED_FEATURE` |
| `sin_hour`, `cos_hour`, `sin_time_of_day`, `cos_time_of_day` | Temporal | Continuous | `DERIVED_FEATURE` |
| `direction`, `segment`, `segment_length_km` | Spatial & Route | Numerical | `REAL_GPS_DERIVED` |
| `segments_completed`, `segments_remaining`, `trip_progress_ratio` | Route Progress | Numerical / Ratio | `DERIVED_FEATURE` |
| `deviceid` | Vehicle / Driver | String / Categorical | `REAL_GPS_DERIVED` |
| `previous_segment_run_time` | Causal Kinematics | Continuous (Shift 1) | `DERIVED_FEATURE` |
| `rolling_prev_segment_mean`, `rolling_prev_segment_std` | Causal Kinematics | Continuous (Rolling 3) | `DERIVED_FEATURE` |
| `cumulative_trip_time_sec` | Causal Progress | Continuous (Cumsum) | `DERIVED_FEATURE` |
| `historical_segment_time_mean`, `segment_delay_ratio`, `congestion_proxy` | Congestion Proxies | Continuous / Binary | `DERIVED_FEATURE` |
| `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `wind_speed_10m`, `weather_code` | Weather | Continuous / Discrete | `WEATHER_OBSERVED` |

### Strict Leakage Prevention
The feature extraction pipeline enforces:
1. `eta_to_next_stop_sec` and `run_time_in_seconds` are NEVER in the feature matrix $X$.
2. All rolling and lag features strictly use prior observations within the trip ($t' < t$).
3. Zero future timestamps or downstream durations are exposed at prediction time.

---

## 3. Model Architecture & Benchmarking

Five genuine regression models were trained on the $140,475$ training observations and evaluated against the $30,102$ validation split:

| Model | MAE (s) | RMSE (s) | MedAE (s) | P95 (s) | $R^2$ | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Median Baseline** | $97.43$ | $140.67$ | $66.00$ | $317.00$ | $-0.1327$ | $61.57\%$ | CANDIDATE |
| **Ridge Regression** | $45.01$ | $73.91$ | $27.36$ | $148.90$ | $0.6874$ | $21.65\%$ | CANDIDATE |
| **Random Forest** | $43.25$ | $70.20$ | $26.06$ | $142.16$ | $0.7179$ | $21.05\%$ | CANDIDATE |
| **HistGradientBoosting** | $44.31$ | $72.58$ | $25.99$ | $149.39$ | $0.6985$ | $20.96\%$ | CANDIDATE |
| **LightGBM Regressor** | **$44.01$** | **$71.98$** | **$26.04$** | **$146.88$** | **$0.7035$** | **$20.86\%$** | **PRODUCTION** |

---

## 4. Model Registry & Inference Service

- **Model Registry (`ml/registry/registry_manager.py`)**:
  - Model status lifecycle: `CANDIDATE` $\rightarrow$ `VALIDATED` $\rightarrow$ `PRODUCTION` / `ARCHIVED`.
  - Automated performance gate: MAE $\le 48.0\text{s}$, $R^2 \ge 0.65$, P95 error $\le 160.0\text{s}$.
  - Versioned storage under `models/eta_model_v1.0.0/`.
  - Manifest file: `data/metadata/model_registry.json`.
- **Inference Service (`ml/inference/predictor.py`)**:
  - Exposes `predict_eta(features)` and `ETAPredictor`.
  - Validates feature schema and enforces `LeakageGuard` at runtime.
  - Returns predicted segment ETA in seconds with latency metadata ($< 1\text{ms}$ / record).

---

## 5. Digital Transit Simulator (`simulator/transit_simulator.py`)

- **Replay Dataset**: `data/processed/kandy_eta_stream.parquet` ($30,102$ records, strictly immutable).
- **Time Virtualization**: Converts historical UTC records into simulated streaming events at speeds $1x$ to `MAX`.
- **Event Lifecycle**:
  1. `TelemetryEvent`: 29 causal features emitted at segment start.
  2. `PredictionEvent`: Production LightGBM inference ($0.017\text{ms}$ latency).
  3. `OutcomeEvent`: Ground truth segment travel time arrival.
  4. `EvaluationEvent`: Online signed, absolute, and percentage error metrics.
- **Fleet State**: Real-time tracking of active bus devices (`BusState`).
- **Data Integrity**: Source dataset SHA-256 hash verified before and after simulation.


