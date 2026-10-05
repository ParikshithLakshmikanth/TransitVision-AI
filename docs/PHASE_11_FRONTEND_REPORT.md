# Phase 11 — TransitVision AI Frontend & Operations Command Center Report
**System**: TransitVision AI — Self-Healing Autonomous MLOps Platform  
**Target Route**: Kandy Route 654 Corridor (Kandy Goods Shed → Teldeniya)  
**Phase**: Phase 11 — Frontend Command-Center Application  
**Date**: September 2026  
**Status**: **PASS (100% Certified)**  

---

## 1. Executive Summary
Phase 11 delivers the complete command-center React frontend for TransitVision AI. Built with Vite, React, TypeScript, and a high-contrast dark design system, the interface connects directly to the Phase 10 FastAPI backend and `/ws/live` WebSocket gateway.

Every screen renders **real telemetry**, **real error evaluations**, **real 3D drift signals**, and the **real MLOps state machine**. When the backend is offline or no live data has arrived, the UI displays explicit status states (`BACKEND OFFLINE` or `WAITING FOR LIVE DATA`) without fabricating fake coordinates or mock metrics.

---

## 2. Frontend Architecture & Directory Structure

```
frontend/
├── index.html                           # App shell with Inter & JetBrains Mono fonts
├── package.json                         # React 18, Vite 5, Lucide icons, Vitest
├── tsconfig.json                        # Strict TypeScript compiler options
├── vite.config.ts                       # Dev server proxying /api and /ws to FastAPI
└── src/
    ├── main.tsx                         # React 18 createRoot entrypoint
    ├── App.tsx                          # App shell coordinating layout and active page
    ├── index.css                        # Command-center design system & tokens
    ├── types/
    │   └── api.ts                       # Strict TypeScript interfaces matching FastAPI schemas
    ├── api/
    │   ├── client.ts                    # Base fetch client with ApiError handling
    │   └── index.ts                     # health, fleet, predictions, metrics, drift, models, etc.
    ├── hooks/
    │   └── useWebSocket.ts              # Authoritative WebSocket hook with auto-reconnect
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.tsx              # Navigation sidebar with 7 operational views
    │   │   └── Topbar.tsx               # Production model, health & WebSocket indicators
    │   ├── common/
    │   │   ├── StatusBadge.tsx          # PRODUCTION, ARCHIVED, REJECTED, ROLLED_BACK, LIVE
    │   │   ├── MetricCard.tsx           # KPI display cards with unit and trend subtext
    │   │   ├── LoadingState.tsx         # Accessible loader state
    │   │   └── ErrorState.tsx           # Offline banner with retry button
    │   ├── events/
    │   │   └── EventFeed.tsx            # Live scrolling event feed with event-type badges
    │   ├── fleet/
    │   │   ├── CorridorMap.tsx          # 29-segment Route 654 topological schematic
    │   │   └── FleetTable.tsx           # Live fleet table with predicted vs actual travel time
    │   └── charts/
    │       ├── ActualVsPredictedChart.tsx # Regression correlation plot with 45° parity line
    │       └── ErrorDistributionChart.tsx # Error range histogram (0-10s, 10-20s, 20-40s...)
    ├── pages/
    │   ├── OverviewPage.tsx             # Global operations command dashboard
    │   ├── LiveFleetPage.tsx            # Corridor tracking & bus telemetry table
    │   ├── EtaAnalyticsPage.tsx         # MAE, RMSE, P95 tail errors, and correlation charts
    │   ├── DriftMonitorPage.tsx         # 3D Drift (Covariate, Performance, Concept Residuals)
    │   ├── LifecyclePage.tsx            # 11-stage autonomous MLOps closed-loop DAG
    │   ├── ModelRegistryPage.tsx        # Model catalog with hyperparameter detail modal
    │   └── ScenarioSimulatorPage.tsx    # Playback controls & Phase 7 scenario injector
    └── tests/
        └── frontend.test.ts             # Vitest unit test suite
```

---

## 3. Operational Pages Summary

### 1. Operations Overview (`/overview`)
- **Key KPIs**: Active fleet count, Online streaming MAE (seconds), Active Production Champion (`model_lightgbm_v1`), and Drift Sentinel alert count.
- **Corridor Progress Tracker**: 29-segment Route 654 schematic displaying active bus markers on their current road segment.
- **Fleet Snapshot & Real-Time Event Feed**: Dual-panel view showing active bus ETA predictions and live WebSocket events.

### 2. Live Fleet Telemetry (`/fleet`)
- Displays all active buses replaying the real Kandy stream.
- Exposes: Device ID, Trip ID, Direction (Outbound/Inbound), Segment (1..29), Timestamp, Predicted ETA, Actual ETA (when resolved), Error (s), Model Version, Scenario, and Status (`ACTIVE`).
- Updates smoothly in real time as WebSocket `PREDICTION` and `OUTCOME_RESOLVED` events arrive.

### 3. ETA Accuracy Analytics (`/analytics`)
- **Error Metrics**: MAE, RMSE, Median Absolute Error, P90 Tail Error, P95 Tail Error, Mean Signed Bias, and Average Latency (< 0.1ms).
- **Actual vs Predicted Scatter**: Visualizes prediction correlation against ideal 45° parity line.
- **Error Distribution Histogram**: Buckets errors across 0–10s, 10–20s, 20–40s, 40–60s, 60–100s, and >100s.
- **Rolling Window History Table**: Lists historical MAE and RMSE across 50-record evaluation intervals.

### 4. 3D Drift Surveillance (`/drift`)
- **3 Prominent Dimension Cards**:
  1. *Covariate / Data Drift* (PSI & Kolmogorov-Smirnov hybrid tests on 29 tabular features).
  2. *Performance Drift* (Rolling MAE ratio vs 1.60x baseline threshold).
  3. *Concept Drift* (Sequential Page-Hinkley cumulative sum on error residuals).
- **Feature Drift Frequency**: Tables ranking most frequently drifted corridor variables (e.g., meteorological features during heavy rain).
- **Sequential Drift Event Stream**: Audit log with detection latencies, test statistics, and thresholds.

### 5. Autonomous MLOps Lifecycle (`/lifecycle`)
- **11-Stage Closed-Loop DAG**:
  $$\text{DRIFT} \to \text{TRIGGER} \to \text{TRAIN} \to \text{VALIDATE} \to \text{SHADOW} \to \text{CANARY 10\%} \to \text{CANARY 50\%} \to \text{PROMOTE} \to \text{PRODUCTION} \to \text{WATCHDOG} \to \text{ROLLBACK}$$
- Highlights current production state and validates the *Exactly-One-Production* invariant.
- Displays watchdog sentinel policy ($1.25\times$ MAE threshold, 20ms latency, 0 exceptions) and the historical audit of Experiments A through F.

### 6. Model Registry & Metadata (`/models`)
- Complete table of registered models with distinct lifecycle statuses: `PRODUCTION`, `ARCHIVED`, `REJECTED`, `ROLLED_BACK`, and `CANDIDATE`.
- Detail modal allowing deep inspection of model architecture, training row counts, validation metrics, and hyperparameters.

### 7. Scenario Simulator (`/simulator`)
- **Playback Controls**: Start, Pause, Resume, Stop, Reset, Step (+1, +5, +50, +500), and Speed Multiplier (1x, 2x, 5x, 10x, 50x).
- **Disturbance Scenario Injector**: Allows selecting and activating any Phase 7 scenario (`BASELINE`, `RUSH_HOUR`, `HEAVY_RAIN`, `CONGESTION_SURGE`, `ROAD_INCIDENT`, `DWELL_SURGE`, `COMBINED_DISRUPTION`) with an interactive intensity slider.

---

## 4. Real-Time WebSocket Streaming & Resilience

- **Authoritative Hook**: Implemented via [src/hooks/useWebSocket.ts](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/frontend/src/hooks/useWebSocket.ts).
- **Ping / Pong Heartbeat**: Sends `ping` every 5 seconds to keep the socket alive.
- **Automatic Exponential Reconnection**: When the connection drops, automatically attempts reconnection at 1s, 1.5s, 2.25s up to 10s intervals.
- **Visual Status**: Topbar status changes to `CONNECTION LOST` during disconnects and returns to `LIVE` (with emerald pulse) upon reconnecting.

---

## 5. Verification & Testing

### A. Frontend Unit Tests (Vitest: 4 / 4 Passed)
```
 ✓ src/tests/frontend.test.ts (4 tests) 2ms

 Test Files  1 passed (1)
      Tests  4 passed (4)
   Duration  2.96s
```

### B. Frontend Production Build (Vite + TypeScript: 0 Errors)
```
vite v5.4.21 building for production...
✓ 1526 modules transformed.
dist/index.html                   0.81 kB │ gzip:  0.46 kB
dist/assets/index-fQ15_Z0q.css   10.91 kB │ gzip:  2.67 kB
dist/assets/index-D1WsCsxr.js   212.75 kB │ gzip: 62.00 kB
✓ built in 12.50s
```

### C. Backend Test Suite (Pytest: 105 / 105 Passed)
```
====================== 105 passed, 2 warnings in 29.08s =======================
```

### D. Protected Artifact Integrity Audit (8 / 8 Checked & Intact)
```
=== FINAL INTEGRITY AUDIT ===
[PASS] kandy_eta_training.parquet     -> 1d5af060533bd0b6821d69138b4eb99a428028c051d04b15308d0cf272e4983c
[PASS] kandy_eta_validation.parquet   -> e01635b507790e9717d1f88db538a454dd4c378ae45262b287ee820106e1f1e4
[PASS] kandy_eta_stream.parquet       -> 332479b56571e88ecfa596abd2cd412f7f51e4c1ce96f1ac016517a58606ab85
[PASS] v1.0.0 model.joblib            -> 460cf798484cce96873812ac771b7bdcfbf08d5d6297144f81bcdc911a8f9a56
[PASS] v1.0.0 preprocessor.joblib     -> 42aa29b0f8a61db6b635dd744c47e3cda83a9853b2a4223bc96323eb7a6d941c
[PASS] v1.0.0 feature_config.json     -> fded7b1e8290b7867ccc6dd7b91f637eff6f5ff3faf37c646cb8bb6c0b757e84
[PASS] v1.0.0 metadata.json           -> 8aba89f780061421c23873d07683559080fcce6b3c5fca5078bd7ace00214ec5
[PASS] drift_reference_profile.json   -> 00711473c3a51d31bcea4f75c5bafb82623e1a7c653b60f0be5f6721c507311b

ALL 8 CRITICAL ARTIFACTS VERIFIED: True
Production Model ID: model_lightgbm_v1 (v1.0.0, PRODUCTION)
```

---

## 6. How to Run the Application

### 1. Start the FastAPI Backend
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/health`
- WebSocket: `ws://localhost:8000/ws/live`

### 2. Start the React Frontend Development Server
```bash
cd frontend
npm run dev
```
- Dashboard UI: `http://localhost:5173`

---

## 7. Known Limitations
- The Kandy stream Parquet dataset contains 29 topological segment indices rather than raw latitude/longitude GPS point streams; the corridor is therefore accurately rendered via topological segment progression rather than map pinning.
- Retraining triggers via public API endpoints are restricted to dry-run mode to protect the verified production model registry from unintended mutation.

---

## 8. HARD STOP Certification
Phase 11 is complete and certified. All 11 project phases (Phases 1 through 11) have been implemented, tested, and verified. Stopping as instructed.
