# TransitVision AI — Phase 5 Model Training & Evaluation Report

## 1. Executive Summary

- **Phase**: 5 — Real ETA Regression Model Training, Evaluation & Model Registry
- **Status**: **MODEL TRAINING STATUS: VERIFIED**
- **Corridor**: Kandy $\leftrightarrow$ Digana (Route 654, Sri Lanka)
- **Primary Target**: `eta_to_next_stop_sec` (Real observed GPS-derived segment travel time in seconds)
- **Selected Production Algorithm**: `LightGBMRegressor` (Model ID: `model_lightgbm_v1`, Version: `v1.0.0`)
- **Key Validation Metrics**:
  - **MAE**: $44.01\text{ seconds}$ (vs. Baseline $97.43\text{ seconds}$, a $54.8\%$ reduction)
  - **Median Absolute Error**: $26.04\text{ seconds}$ (vs. Baseline $66.00\text{ seconds}$)
  - **RMSE**: $71.98\text{ seconds}$ (vs. Baseline $140.67\text{ seconds}$)
  - **$R^2$ Score**: $0.7035$
  - **MAPE**: $20.86\%$ (safe evaluation on targets $\ge 10\text{s}$)
  - **sMAPE**: $21.92\%$
  - **P90 Error**: $101.02\text{ seconds}$
  - **P95 Error**: $146.88\text{ seconds}$

---

## 2. Dataset Topology & Temporal Partitioning

The models were trained and validated exclusively on the verified Kandy Bus GPS telemetry dataset.

| Partition | Records | Share | Start UTC | End UTC | Unique Trips | Unique Devices | Storage File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training** | $140,475$ | $70.0\%$ | `2021-10-01 06:39:49` | `2022-07-02 16:13:58` | $9,888$ | $26$ | `data/processed/kandy_eta_training.parquet` |
| **Validation** | $30,102$ | $15.0\%$ | `2022-07-02 16:14:28` | `2022-09-02 13:55:09` | $2,119$ | $26$ | `data/processed/kandy_eta_validation.parquet` |
| **Streaming (Untouched)** | $30,102$ | $15.0\%$ | `2022-09-02 13:55:40` | `2022-11-01 18:57:10` | $2,119$ | $26$ | `data/processed/kandy_eta_stream.parquet` |
| **Total** | $200,679$ | $100.0\%$ | `2021-10-01` | `2022-11-01` | $14,126$ | $26$ | — |

### Target Distribution Summary (`eta_to_next_stop_sec`)
- **Mean**: $188.13\text{s}$ ($\pm 125.17\text{s}$)
- **Median ($P_{50}$)**: $162.00\text{s}$
- **$P_{25}$**: $105.00\text{s}$
- **$P_{75}$**: $233.00\text{s}$
- **$P_{90}$**: $370.00\text{s}$
- **$P_{95}$**: $441.00\text{s}$
- **Min / Max**: $3.00\text{s}$ / $1,199.00\text{s}$

---

## 3. Feature Set & Zero-Leakage Enforcement

A total of 29 causal, strictly backward-looking features were engineered and verified:

1. **Temporal (9)**: `hour`, `minute`, `day_of_week`, `is_weekend`, `is_peak_period`, `sin_hour`, `cos_hour`, `sin_time_of_day`, `cos_time_of_day`.
2. **Spatial & Progress (6)**: `direction`, `segment`, `segment_length_km`, `segments_completed`, `segments_remaining`, `trip_progress_ratio`.
3. **Vehicle / Hardware (1)**: `deviceid`.
4. **Causal Kinematics (4)**: `previous_segment_run_time`, `rolling_prev_segment_mean`, `rolling_prev_segment_std`, `cumulative_trip_time_sec`.
5. **Traffic & Congestion Proxies (3)**: `historical_segment_time_mean`, `segment_delay_ratio`, `congestion_proxy`.
6. **Weather Conditions (6)**: `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `wind_speed_10m`, `weather_code`.

### Leakage Guard Verification
The automated `LeakageGuard` passed with zero errors:
- No target variables (`eta_to_next_stop_sec`, `run_time_in_seconds`) present in feature matrices.
- No future-looking segment times, dwell times, or arrival timestamps present.
- All rolling aggregates use causal shift ($t-1$) excluding the current segment.

---

## 4. Multi-Model Benchmark Results

| Model | MAE (s) | RMSE (s) | MedAE (s) | P90 (s) | P95 (s) | $R^2$ | MAPE (%) | sMAPE (%) | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Median Baseline** | $97.43$ | $140.67$ | $66.00$ | $246.00$ | $317.00$ | $-0.1327$ | $61.57\%$ | $49.17\%$ | $0.01\text{s}$ |
| **Ridge Regression** | $45.01$ | $73.91$ | $27.36$ | $99.56$ | $148.90$ | $0.6874$ | $21.65\%$ | $23.30\%$ | $0.10\text{s}$ |
| **Random Forest** | $43.25$ | $70.20$ | $26.06$ | $97.96$ | $142.16$ | $0.7179$ | $21.05\%$ | $21.45\%$ | $13.50\text{s}$ |
| **HistGradientBoosting** | $44.31$ | $72.58$ | $25.99$ | $102.37$ | $149.39$ | $0.6985$ | $20.96\%$ | $22.03\%$ | $3.60\text{s}$ |
| **LightGBM Regressor** | **$44.01$** | **$71.98$** | **$26.04$** | **$101.02$** | **$146.88$** | **$0.7035$** | **$20.86\%$** | **$21.92\%$** | **$1.64\text{s}$** |

---

## 5. Stratified Error Analysis

Comprehensive error analysis was conducted on the $30,102$ validation records using the production LightGBM model:

### 5.1 By Direction
- **Outbound (Kandy $\rightarrow$ Digana)**: $15,478$ records | MAE: $41.45\text{s}$ | RMSE: $70.74\text{s}$ | MedAE: $23.83\text{s}$ | MAPE: $19.82\%$
- **Inbound (Digana $\rightarrow$ Kandy)**: $14,624$ records | MAE: $46.71\text{s}$ | RMSE: $73.26\text{s}$ | MedAE: $28.48\text{s}$ | MAPE: $22.13\%$

### 5.2 By Operational Period
- **Peak Hours (07:00–09:00, 16:30–18:30)**: $9,561$ records | MAE: $41.75\text{s}$ | RMSE: $68.46\text{s}$ | MedAE: $24.97\text{s}$
- **Off-Peak Hours**: $20,541$ records | MAE: $45.05\text{s}$ | RMSE: $73.56\text{s}$ | MedAE: $26.48\text{s}$

### 5.3 By Weather & Precipitation
- **No Rain ($0\text{ mm}$)**: $26,892$ records | MAE: $43.91\text{s}$ | RMSE: $71.85\text{s}$
- **Light Rain ($0-1\text{ mm}$)**: $2,410$ records | MAE: $44.52\text{s}$ | RMSE: $72.68\text{s}$
- **Moderate / Heavy Rain ($>1\text{ mm}$)**: $800$ records | MAE: $45.89\text{s}$ | RMSE: $74.20\text{s}$

### 5.4 By Travel-Time Duration Bucket
- **Short Segments ($< 60\text{s}$)**: MAE: $18.42\text{s}$ | MAPE: $34.12\%$
- **Medium Segments ($60-180\text{s}$)**: MAE: $27.50\text{s}$ | MAPE: $22.10\%$
- **Standard Segments ($180-360\text{s}$)**: MAE: $51.20\text{s}$ | MAPE: $19.45\%$
- **Long / Bottleneck Segments ($> 360\text{s}$)**: MAE: $98.15\text{s}$ | MAPE: $18.20\%$

---

## 6. Feature Importance (Top 10 Drivers)

Normalized gain contribution from LightGBM:
1. `segment` ($34.68\%$): Route corridor topology and geometric distance between stops.
2. `historical_segment_time_mean` ($32.31\%$): Historical baseline running time for the segment.
3. `segment_length_km` ($22.48\%$): Physical road distance in kilometers.
4. `sin_time_of_day` ($1.57\%$): Diurnal traffic cyclicality.
5. `cos_time_of_day` ($1.24\%$): Time-of-day progression.
6. `cos_hour` ($0.94\%$): Hourly scheduling window.
7. `previous_segment_run_time` ($0.75\%$): Upstream traffic delay propagation.
8. `rolling_prev_segment_mean` ($0.67\%$): Multi-segment moving average pace.
9. `relative_humidity_2m` ($0.63\%$): Microclimatic visibility / precipitation factor.
10. `deviceid` ($0.60\%$): Driver / vehicle acceleration profile.

---

## 7. Visualizations Generated

All 7 evaluation plots are generated and stored in `reports/figures/`:
1. `actual_vs_predicted_scatter.png`: Scatter comparison against the ideal $y = x$ diagonal.
2. `residual_distribution.png`: Centered residual distribution with mean residual $-0.26\text{s}$.
3. `absolute_error_distribution.png`: Cumulative error distribution with $P_{50} = 26.0\text{s}$, $P_{90} = 101.0\text{s}$, $P_{95} = 146.9\text{s}$.
4. `error_by_segment.png`: MAE distribution across all 29 individual route segments.
5. `error_by_hour.png`: Diurnal MAE profile highlighting evening traffic variance.
6. `feature_importance.png`: Relative feature gain contributions.
7. `actual_vs_predicted_time_series.png`: Chronological validation window actual vs predicted tracking.

---

## 8. Test Suite Summary

- **Total Unit & Integration Tests**: 18
- **Passed**: 18 ($100\%$)
- **Failed**: 0
- **Execution Command**: `python -m pytest tests/ -v`
- **Execution Time**: $2.13\text{s}$

---

## 9. Academic Requirement & Provenance Declaration

> **Academic Statement**:
> "The model was trained exclusively on the chronological training partition and evaluated on a temporally later validation partition. The streaming partition was not used during model training or model selection."

The model is a verified research prototype calibrated for the Kandy Route 654 transit corridor.
