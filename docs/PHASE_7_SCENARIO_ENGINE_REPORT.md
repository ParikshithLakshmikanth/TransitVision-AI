# TransitVision AI — Phase 7 Controlled Disturbance & Scenario Simulation Engine Report

## 1. Executive Summary

- **Phase**: 7 — Controlled Disturbance & Scenario Simulation Engine
- **Status**: **PASS**
- **Objective**: Provide a deterministic, spatiotemporally localized scenario disturbance layer above the Digital Transit Simulator to model real-world distribution shifts without altering the raw historical telemetry dataset.
- **Source Dataset**: `data/processed/kandy_eta_stream.parquet` ($30,102$ records, SHA-256: `332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85`, **Strictly Immutable**).
- **Production Model**: `LightGBMRegressor` (`v1.0.0`, Model ID: `model_lightgbm_v1`).
- **Scenarios Evaluated**: 7 distinct operational configurations (Baseline Control + 6 Disturbance Regimes).

---

## 2. Academic Framing

> "Synthetic disturbances are introduced exclusively within the Digital Transit Simulator to create controlled distribution-shift scenarios. They do not alter the underlying real historical dataset and are explicitly labelled as synthetic simulation events."

---

## 3. Scenario Disturbance Architecture

```
Real Telemetry Record (kandy_eta_stream.parquet)
                    │
                    ▼
       [DigitalTransitSimulator]
                    │
                    ▼
         [ScenarioEngine.apply()]
      (Temporal & Spatial Gating)
                    │
   ┌────────────────┴────────────────┐
   ▼                                 ▼
[Observable Modified Features]  [Simulated Ground Truth]
(Weather, Kinematics, Delay)    (Scenario Actual Travel Time)
   │                                 │
   ▼                                 │
[Production ETA Model]               │
(Zero Target Leakage)                │
   │                                 │
   ▼                                 ▼
[Predicted ETA (s)] ───────────► [Online Evaluation]
                                (Signed / Absolute Error)
```

---

## 4. Scenario Definitions & Physical Formulations

| Scenario ID | Name | Transformation Formulation | Spatiotemporal Scope |
| :--- | :--- | :--- | :--- |
| `BASELINE` | Baseline Replay | No modification. Control group. | Global ($30,102$ records) |
| `RUSH_HOUR` | Rush Hour Congestion | Peak indicators ($1$), delay ratio $\times (1 + 0.6 \lambda)$, travel time $\times (1 + 0.7 \lambda)$. | Diurnal Peak (07:00–09:00, 16:30–18:30) |
| `HEAVY_RAIN` | Heavy Monsoon Rain | Precip ($16\lambda\text{ mm}$), rain ($14\lambda\text{ mm}$), humidity ($88+10\lambda\%$), code ($65$), travel time $\times (1 + 0.35\lambda)$. | Global or active window |
| `CONGESTION_SURGE`| Corridor Congestion Surge | Delay ratio $\times (1 + 1.1\lambda)$, upstream kinematics $\times (1 + 0.75\lambda)$, travel time $\times (1 + 1.05\lambda)$. | Global or active window |
| `ROAD_INCIDENT` | Localized Road Incident | Congestion proxy ($1$), delay ratio $\times (1 + 1.5\lambda)$, travel time $\times (1 + 1.4\lambda) + 80\lambda\text{s}$. | Spatially localized: Segments $8, 9, 10$ |
| `DWELL_SURGE` | Passenger Dwell Surge | Cumulative trip time $+ 90\lambda\text{s}$, travel time $+ 65\lambda\text{s}$. | Global or selected stops |
| `COMBINED_DISRUPTION` | Compound Severe Disruption | Composition of Heavy Rain + Congestion Surge + Dwell Surge. | Global |

*Note: $\lambda \in [0.0, 1.0]$ represents the configurable disturbance intensity parameter.*

---

## 5. Measured Scenario Benchmark Results

All 7 scenarios were executed across the entire $30,102$ stream records ($210,714$ total predictions):

| Scenario | Configured Intensity | MAE (s) | RMSE (s) | MedAE (s) | P90 (s) | P95 (s) | Affected Events | Total Records |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Control)** | $0.00$ | **$38.77$** | **$60.83$** | **$24.54$** | **$87.61$** | **$122.28$** | $0$ | $30,102$ |
| **Rush Hour** | $0.75$ | **$54.25$** | **$86.98$** | **$33.13$** | **$123.50$** | **$174.46$** | $9,362$ | $30,102$ |
| **Heavy Monsoon Rain** | $0.80$ | **$55.62$** | **$89.18$** | **$34.17$** | **$125.70$** | **$182.29$** | $30,102$ | $30,102$ |
| **Congestion Surge** | $0.70$ | **$122.54$** | **$169.83$** | **$89.71$** | **$264.66$** | **$351.26$** | $30,102$ | $30,102$ |
| **Road Incident** | $0.85$ | **$55.99$** | **$90.43$** | **$31.26$** | **$137.37$** | **$199.84$** | $3,104$ | $30,102$ |
| **Passenger Dwell Surge** | $0.75$ | **$52.52$** | **$73.40$** | **$42.08$** | **$98.00$** | **$138.33$** | $30,102$ | $30,102$ |
| **Combined Disruption** | $0.85$ | **$301.80$** | **$359.11$** | **$249.07$** | **$563.93$** | **$689.82$** | $30,102$ | $30,102$ |

---

## 6. Findings & Distribution Shift Analysis

1. **Baseline Invariance**:
   The `BASELINE` scenario produced identical error metrics ($38.77\text{s}$ MAE) and $0$ affected events, proving complete equivalence with Phase 6 control behavior.
2. **Diurnal Rush Hour Shift**:
   In `RUSH_HOUR`, exactly $9,362$ records ($31.1\%$ of the stream) fell into the morning and evening peak windows, elevating MAE from $38.77\text{s}$ to $54.25\text{s}$ ($+39.9\%$).
3. **Spatial Localized Bottlenecks**:
   `ROAD_INCIDENT` localized delay to segments $8, 9, 10$ ($3,104$ records, $10.3\%$ of the stream). Unaffected segments $1-7$ and $11-34$ remained $100\%$ untouched with baseline delay dynamics.
4. **Severe Compound Degradation**:
   `COMBINED_DISRUPTION` exposed the static model to severe non-linear shift, increasing MAE to $301.80\text{s}$ and P95 error to $689.82\text{s}$, establishing the target evaluation benchmark for Phase 8 drift detection.

---

## 7. Data Provenance & Integrity

- Source Dataset: `data/processed/kandy_eta_stream.parquet`
- SHA-256 Before Benchmark: `332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85`
- SHA-256 After Benchmark: `332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85`
- Status: **IMMUTABLE (Unchanged)**
- Experiment Manifest: Saved to `data/metadata/scenario_experiments.json`.

---

## 8. Test Suite Summary

- **Total Tests**: 51
- **Passed**: 51 ($100\%$)
- **Failed**: 0
- **Execution Time**: $4.87\text{ seconds}$
- **Command**: `python -m pytest tests/ -v`

---

## 9. Next Steps: Phase 8 (Concept Drift Detection Engine)

With Phase 7 complete, Phase 8 will consume the streaming `EvaluationEvent` and `TelemetryEvent` logs to implement online concept-drift monitoring (Kolmogorov-Smirnov feature drift tests, Population Stability Index, and error degradation detectors) to detect when scenarios occur and trigger autonomous self-healing model updates.
