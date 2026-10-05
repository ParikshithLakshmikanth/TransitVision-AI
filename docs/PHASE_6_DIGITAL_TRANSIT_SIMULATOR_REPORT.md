# TransitVision AI — Phase 6 Digital Transit Simulator Report

## 1. Executive Summary

- **Phase**: 6 — Digital Transit Simulator: Real Historical Telemetry $\rightarrow$ Live Stream
- **Status**: **PASS**
- **Primary Source**: `data/processed/kandy_eta_stream.parquet` (Strictly Immutable, SHA-256: `332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85`)
- **Total Replayed Stream Records**: $30,102$
- **Predictions Generated**: $30,102$
- **Outcomes Resolved**: $30,102$
- **Active Vehicle Fleet**: $18$ physical GPS devices across Route 654
- **Throughput**: $3,292.9\text{ records/second}$
- **Average Inference Latency**: $0.017\text{ ms / prediction}$
- **Online Replay Error Metrics**:
  - **MAE**: $38.77\text{ seconds}$
  - **RMSE**: $60.83\text{ seconds}$
  - **Median Absolute Error**: $24.54\text{ seconds}$
  - **P90 Error**: $87.61\text{ seconds}$
  - **P95 Error**: $122.28\text{ seconds}$

---

## 2. Academic Positioning

> "A Digital Transit Simulator replays real historical bus telemetry chronologically to emulate a live operational environment, enabling controlled evaluation of real-time prediction, monitoring, drift detection and autonomous MLOps behavior without requiring proprietary live transit APIs."

The system preserves genuine observed spatiotemporal kinematics from the Kandy Route 654 corridor without synthesizing fake GPS coordinates, artificial speeds, or mock predictions.

---

## 3. Simulator Architecture & Event Lifecycle

```
Real Historical Parquet (kandy_eta_stream.parquet)
                    │
                    ▼
     [DigitalTransitSimulator Engine]
                    │
   ┌────────────────┴────────────────┐
   ▼                                 ▼
[TelemetryEvent]             [BusState Manager]
(29 Causal Features)         (Active Fleet Tracking)
   │
   ▼
[ETAPredictor] ─── (Zero Target Leakage)
   │
   ▼
[PredictionEvent] (ETA_pred, latency, model_v1.0.0)
   │
   ▼
[OutcomeEvent] (Actual Segment ETA Arrival)
   │
   ▼
[EvaluationEvent] (Signed Error, |Error|, % Error)
   │
   ▼
[Simulation Logs: JSONL Append Storage]
```

### Event Data Model
1. **`TelemetryEvent`**: Carries the 29 causal features, historical UTC timestamp, simulated wall-clock timestamp, device ID, trip ID, direction, and segment. Target columns (`eta_to_next_stop_sec`, `run_time_in_seconds`) are excluded from model inputs.
2. **`PredictionEvent`**: Encapsulates model output (`predicted_eta_sec`), raw unclipped prediction, latency in milliseconds, model ID (`model_lightgbm_v1`), version (`v1.0.0`), and algorithm (`LightGBMRegressor`).
3. **`OutcomeEvent`**: Resolves ground truth when the segment arrival occurs in the stream.
4. **`EvaluationEvent`**: Calculates online error metrics ($\text{signed\_error} = \text{actual} - \text{predicted}$, $\text{abs\_error}$, $\text{percentage\_error}$).

---

## 4. Replay Methodology & Programmatic Controls

- **Chronological Sorting**: Guaranteed strictly monotonic non-decreasing timestamp ordering.
- **Configurable Speed Multiplier**: Supports $0x$ (paused), $1x$ (real-time relative), $10x$, $50x$, $100x$, $500x$, and `MAX` non-blocking execution.
- **Lifecycle Methods**:
  - `start()`: Initializes wall-clock synchronization.
  - `pause()`: Freezes progression without dropping cursor position.
  - `resume()`: Continues replay seamlessly.
  - `reset()`: Returns cursor to index 0, clearing active bus state and online error buffers.
  - `seek(index)`: Moves cursor to a designated record offset.
  - `set_speed(speed)`: Dynamically updates replay rate.
- **Fleet Filtering**: Configurable filtering by `trip_id`, `deviceid`, `direction`, or `segment`.

---

## 5. Inference Validation Optimization

### Root Cause of Initial Per-Row Log Overhead
During initial replay, `ETAPredictor.predict()` called `LeakageGuard.validate_features()` per record, which executed an `INFO` log message (`LeakageGuard PASSED for Inference Request (29 columns verified)`) for every single prediction. Across a $30,102$-record stream, this created $30,102$ redundant I/O operations and console log lines.

### Corrective Engineering Implemented
1. **Schema-Level Initialization Validation**:
   - `LeakageGuard.validate_schema()` was implemented to inspect model feature configuration once at predictor initialization.
   - Outputs a single startup confirmation: `LeakageGuard schema validation PASSED (29 inference features).`
2. **Fast Silent Per-Request Protection**:
   - `validate_features(X, log_success=False, required_features=...)` performs fast set-based column checks and missing feature verification on every request.
   - Logs only at `DEBUG` level for successes, while raising immediate `LeakageGuardError` (at `ERROR` level) on forbidden target or future-looking columns.
3. **Predictor Reuse**:
   - `DigitalTransitSimulator` initializes `ETAPredictor` once per simulation session and reuses the booster instance across all steps.
4. **Micro-Batch Execution**:
   - `step(n)` executes model inference across micro-batches while generating individual, strictly ordered `TelemetryEvent`, `PredictionEvent`, `OutcomeEvent`, and `EvaluationEvent` objects.

### Measured Performance Comparison
- **1,000-Record Benchmark**: $33.1\text{s}$ (unoptimized per-row console logging) $\rightarrow$ **$0.301\text{s}$** ($3,323.1\text{ records/sec}$, $0.024\text{ms}$ latency).
- **Full 30,102-Record Stream**: Completed in **$9.141\text{ seconds}$** ($3,292.9\text{ records/sec}$, $0.017\text{ms}$ latency).
- **Log Pollution**: Reduced from $30,102$ repetitive log lines down to **0** per-record lines.

---

## 6. Data Integrity & Provenance Verification

- **File**: `data/processed/kandy_eta_stream.parquet`
- **Initial SHA-256**: `332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85`
- **Post-Simulation SHA-256**: `332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85`
- **Integrity Verified**: **YES (Strictly Immutable)**

---

## 7. Test Suite Verification

Executed:
```powershell
python -m pytest tests/ -v
```
- **Total Tests**: 38
- **Passed**: 38 ($100\%$)
- **Failed**: 0
- **Execution Time**: $4.50\text{ seconds}$

### Key Tests Added in Phase 6:
- `test_stream_loads_successfully`: Confirms $30,102$ stream rows.
- `test_source_immutability`: Verifies SHA-256 hash before and after execution.
- `test_chronological_ordering`: Asserts monotonic timestamp progression.
- `test_pause_and_resume` / `test_reset` / `test_seek` / `test_replay_speed_configuration`: Verifies lifecycle controls.
- `test_filtering`: Confirms device, trip, and direction filter slicing.
- `test_zero_leakage_to_prediction`: Confirms target columns are excluded from features.
- `test_missing_feature_rejected`: Confirms missing features raise `ValueError`.
- `test_no_info_log_flood_during_prediction`: Asserts 0 INFO log lines during prediction loop.
- `test_predictor_reused_in_simulator`: Confirms predictor instance reuse.
- `test_no_synthetic_disturbances_in_baseline`: Verifies baseline purity (`synthetic_disturbance=False`).

---

## 8. Preparation for Phase 7 (Concept Drift & Disturbance Injection)

The Digital Transit Simulator architecture is decoupled into layers:
- Replay Engine $\rightarrow$ Telemetry Stream $\rightarrow$ Feature Vector $\rightarrow$ Production Predictor $\rightarrow$ Evaluation Stream.
In Phase 7, a **Disturbance Injector** layer will hook into the telemetry stream to apply controlled spatiotemporal disturbances (rush-hour congestion surges, torrential rain, road blockage bottlenecks) to demonstrate concept drift and trigger autonomous self-healing model adaptation.
