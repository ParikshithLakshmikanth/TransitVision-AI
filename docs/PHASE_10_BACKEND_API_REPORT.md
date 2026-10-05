# Phase 10 — TransitVision AI Application Backend & API Report
**System**: TransitVision AI — Self-Healing Autonomous MLOps Platform  
**Target Route**: Kandy Route 654 Corridor (Kandy – Digana / Teldeniya)  
**Phase**: Phase 10 — Application Backend & API Service Layer  
**Date**: September 2026  
**Status**: **PASS (100% Verified)**  

---

## 1. Executive Summary
Phase 10 successfully establishes the production-grade application backend and real-time streaming gateway for TransitVision AI without modifying raw datasets, retraining models, or altering the protected production champion artifacts (`model_lightgbm_v1` / `v1.0.0`). The backend wraps the entire verified Phase 1–9 ML/MLOps architecture in a clean, asynchronous FastAPI and WebSocket service layer.

---

## 2. Architecture Overview

```
+----------------------------------------------------------------------------------------------------+
|                                    TRANSITVISION AI FASTAPI BACKEND                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +--------------------------+  +---------------------------+  +--------------------------------+  |
|   |  REST API Controllers    |  | WebSocket Realtime Stream |  |   Pydantic V2 Schemas & Data   |  |
|   |  /api/health             |  | /ws/live                  |  |   - Fleet & Telemetry Models   |  |
|   |  /api/fleet              |  |   * PREDICTION            |  |   - Prediction & Error Items   |  |
|   |  /api/predictions        |  |   * OUTCOME_RESOLVED      |  |   - Drift & Residual Alerts    |  |
|   |  /api/metrics            |  |   * DRIFT_DETECTED        |  |   - Lifecycle Graph Models     |  |
|   |  /api/drift              |  |   * MODEL_ROLLED_BACK     |  |   - Simulation Status Schemas  |  |
|   |  /api/models             |  |   * SIMULATION_STATUS     |  |   - Error & Exception Models   |  |
|   |  /api/simulation         |  +---------------------------+  +--------------------------------+  |
|   |  /api/scenarios          |                                                                     |
|   |  /api/retraining         |                                                                     |
|   |  /api/rollback           |                                                                     |
|   |  /api/lifecycle          |                                                                     |
|   |  /api/events             |                                                                     |
|   +-------------+------------+                                                                     |
|                 |                                                                                  |
|                 v                                                                                  |
|   +--------------------------------------------------------------------------------------------+   |
|   |                                     SERVICE LAYER                                          |   |
|   |   +----------------------+   +-----------------------+   +-----------------------------+   |   |
|   |   |  SimulatorService    |   |   MonitoringService   |   |     EventService            |   |   |
|   |   |  - ThreadSafe Replay |   |   - Covariate Drift   |   |     - Broadcast Bus         |   |   |
|   |   |  - Speed/Seek/Reset  |   |   - Performance Drift |   |     - Connection Pool       |   |   |
|   |   |  - Scenario Engine   |   |   - Residual PH Alarms|   |     - Rolling Event Queue   |   |   |
|   |   +----------+-----------+   +-----------+-----------+   +-----------------------------+   |   |
|   |              |                           |                                                 |   |
|   |   +----------+-----------+   +-----------+-----------+                                     |   |
|   |   |    ModelService      |   |   LifecycleService    |                                     |   |
|   |   |    - Read-Only Reg   |   |   - Closed-Loop Graph |                                     |   |
|   |   |    - Metadata & SHA  |   |   - Retraining History|                                     |   |
|   |   +----------------------+   +-----------------------+                                     |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
                                      |                      |
                                      v                      v
                       +-------------------------------+  +-------------------------------+
                       |  EXISTING VALIDATED ENGINES   |  |   PROTECTED DATA & ARTIFACTS  |
                       |  - DigitalTransitSimulator    |  |   - kandy_eta_stream.parquet  |
                       |  - ETAPredictor (v1.0.0)      |  |   - models/eta_model_v1.0.0/  |
                       |  - DriftEngine (Phase 8)      |  |   - drift_reference_profile   |
                       |  - LifecycleModelRegistry     |  |   - model_registry.json       |
                       |  - RollbackManager (Phase 9)  |  |   - benchmark_summary.json    |
                       +-------------------------------+  +-------------------------------+
```

---

## 3. Files Created & Modified

| Component | File Path | Type | Role |
| :--- | :--- | :--- | :--- |
| **Config** | [backend/config.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/config.py) | NEW | Pydantic Settings for server host, port, CORS origins, and simulation defaults. |
| **Schemas** | [backend/schemas.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/schemas.py) | NEW | Pydantic V2 data models for requests, responses, fleet, metrics, drift, and events. |
| **Dependencies** | [backend/dependencies.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/dependencies.py) | NEW | FastAPI dependency injection providers for thread-safe singleton services. |
| **Main App** | [backend/main.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/main.py) | NEW | FastAPI application entrypoint with lifespan context, CORS, error handlers, and `/ws/live`. |
| **Routers** | [backend/api/\_\_init\_\_.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/__init__.py) | NEW | Master router aggregating all sub-routers under `/api`. |
| **Health API** | [backend/api/health.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/health.py) | NEW | `GET /api/health` system availability and production champion check. |
| **Fleet API** | [backend/api/fleet.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/fleet.py) | NEW | `GET /api/fleet` and `GET /api/fleet/{bus_id}` active bus state snapshots. |
| **Predictions API** | [backend/api/predictions.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/predictions.py) | NEW | `GET /api/predictions` and `GET /api/predictions/{id}` error-evaluated predictions. |
| **Metrics API** | [backend/api/metrics.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/metrics.py) | NEW | `GET /api/metrics` and `GET /api/metrics/history` online error metrics. |
| **Drift API** | [backend/api/drift.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/drift.py) | NEW | `GET /api/drift`, `/summary`, `/events` statistical drift inspection. |
| **Models API** | [backend/api/models.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/models.py) | NEW | `GET /api/models`, `/production`, `/{model_id}` read-only registry access. |
| **Simulation API** | [backend/api/simulation.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/simulation.py) | NEW | Simulation controls: `start`, `pause`, `resume`, `reset`, `stop`, `step`, `seek`, `speed`. |
| **Scenarios API** | [backend/api/scenarios.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/scenarios.py) | NEW | `GET /api/scenarios` and `POST /api/scenarios/activate` Phase 7 disturbances. |
| **Retraining API** | [backend/api/retraining.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/retraining.py) | NEW | `GET /api/retraining/status`, `/history`, `POST /trigger` (dry-run guarded). |
| **Rollback API** | [backend/api/rollback.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/rollback.py) | NEW | `GET /api/rollback/status` watchdog threshold and audit inspection. |
| **Lifecycle API** | [backend/api/lifecycle.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/lifecycle.py) | NEW | `GET /api/lifecycle` unified end-to-end MLOps pipeline graph. |
| **Events API** | [backend/api/events.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/api/events.py) | NEW | `GET /api/events` normalized rolling system event queue. |
| **Event Service** | [backend/services/event_service.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/services/event_service.py) | NEW | Centralized broadcast manager and client connection pool. |
| **Simulator Service** | [backend/services/simulator_service.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/services/simulator_service.py) | NEW | Authoritative thread-safe manager wrapping `DigitalTransitSimulator`. |
| **Monitoring Service** | [backend/services/monitoring_service.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/services/monitoring_service.py) | NEW | Real-time bridge to `DriftEngine` across all 3 drift dimensions. |
| **Model Service** | [backend/services/model_service.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/services/model_service.py) | NEW | Read-only inspection of model registry, versions, and metrics. |
| **Lifecycle Service** | [backend/services/lifecycle_service.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/backend/services/lifecycle_service.py) | NEW | MLOps state aggregation and Phase 9 benchmark history parser. |
| **Tests** | [tests/test_backend_api.py](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/tests/test_backend_api.py) | NEW | 13 comprehensive pytest test cases covering all endpoints & WebSockets. |

---

## 4. Complete API Endpoint Catalog

| Method | Endpoint | Description | Request Body | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Root service health & gateway links | — | Object |
| `GET` | `/docs` | Interactive Swagger OpenAPI UI | — | HTML |
| `GET` | `/redoc` | ReDoc API documentation | — | HTML |
| `GET` | `/api/health` | Backend status & production champion | — | `HealthResponse` |
| `GET` | `/api/fleet` | Live Route 654 bus fleet state | — | `FleetResponse` |
| `GET` | `/api/fleet/{bus_id}` | Specific bus state and prediction | — | `BusStateResponse` |
| `GET` | `/api/predictions` | Recent evaluated predictions list | Query params (`limit`, `trip_id`, `deviceid`) | `PredictionsListResponse` |
| `GET` | `/api/predictions/{id}` | Prediction detail by prediction ID | — | `PredictionItem` |
| `GET` | `/api/metrics` | Current online error metrics | — | `MetricsResponse` |
| `GET` | `/api/metrics/history` | Rolling window metrics history | — | `MetricsHistoryResponse` |
| `GET` | `/api/drift` | Drift monitoring state (3D) | — | `DriftStatusResponse` |
| `GET` | `/api/drift/summary` | Drift alarms summary and counters | — | `DriftSummaryResponse` |
| `GET` | `/api/drift/events` | Recent drift alarm events | Query params (`limit`) | `List[DriftEventResponse]` |
| `GET` | `/api/models` | All registered models catalog | — | `ModelListResponse` |
| `GET` | `/api/models/production` | Active serving PRODUCTION model | — | `ModelDetail` |
| `GET` | `/api/models/{model_id}` | Model details by ID | — | `ModelDetail` |
| `GET` | `/api/simulation/status` | Replay status, cursor, progress | — | `SimulationStatusResponse` |
| `POST` | `/api/simulation/start` | Start telemetry stream replay | — | `SimulationControlResponse` |
| `POST` | `/api/simulation/pause` | Pause stream replay | — | `SimulationControlResponse` |
| `POST` | `/api/simulation/resume` | Resume playback from pause | — | `SimulationControlResponse` |
| `POST` | `/api/simulation/stop` | Terminate simulation | — | `SimulationControlResponse` |
| `POST` | `/api/simulation/reset` | Reset simulator to cursor 0 | — | `SimulationControlResponse` |
| `POST` | `/api/simulation/step` | Advance replay deterministically | `SimulationStepRequest` (`n`) | `List[PredictionItem]` |
| `POST` | `/api/simulation/seek` | Jump cursor to specific index | `SimulationSeekRequest` (`index`) | `SimulationControlResponse` |
| `POST` | `/api/simulation/speed` | Set replay speed multiplier | `SimulationSpeedRequest` (`speed`) | `SimulationControlResponse` |
| `GET` | `/api/scenarios` | List Phase 7 disturbance scenarios | — | `ScenarioListResponse` |
| `POST` | `/api/scenarios/activate` | Activate disturbance scenario | `ScenarioActivateRequest` | `ScenarioActivateResponse` |
| `GET` | `/api/retraining/status` | Current retraining pipeline state | — | `RetrainingStatusResponse` |
| `GET` | `/api/retraining/history` | Historical retraining experiments | — | Object |
| `POST` | `/api/retraining/trigger` | Dry-run trigger policy evaluation | `RetrainingTriggerRequest` | Object |
| `GET` | `/api/rollback/status` | Watchdog threshold & rollback state | — | `RollbackStatusResponse` |
| `GET` | `/api/lifecycle` | Unified MLOps pipeline graph | — | `LifecycleStateResponse` |
| `GET` | `/api/events` | Normalized system event stream | Query params (`limit`, `event_type`) | `List[SystemEvent]` |
| `WS` | `/ws/live` | Real-time live event streaming | Text `ping` | Streaming JSON |

---

## 5. Real-Time WebSocket Architecture

The WebSocket endpoint at `/ws/live` provides seamless event multiplexing:
- **Connection Handshake**: Emits immediate `CONNECTION_ESTABLISHED` acknowledgment with server timestamp.
- **Heartbeat Management**: Responds instantly to client `ping` with `pong`.
- **Event Dispatch**: Whenever the authoritative simulator advances, predictions, outcomes, drift alerts, and lifecycle transitions are structured as `SystemEvent` instances and broadcast concurrently to all active dashboard subscribers.
- **Graceful Disconnection**: Dead client sockets are safely pruned without interrupting ongoing simulations.

---

## 6. Security, Concurrency & State Protection

1. **Singleton Authority**: A single `SimulatorService` owns the authoritative `DigitalTransitSimulator` instance. Multiple REST requests or WebSocket clients observe the exact same simulation state.
2. **Thread Safety**: All simulation mutations (`start`, `step`, `seek`, `reset`, `activate_scenario`) are wrapped under `threading.Lock`.
3. **Read-Only Model Registry**: Mutating model registry states through public API routes is disabled to prevent accidental state corruption.
4. **Target Leakage Protection**: Ground-truth target column `eta_to_next_stop_sec` is never passed to the inference model and is suppressed from client telemetry payloads until outcome resolution.
5. **CORS Governance**: Configured strictly with configurable origins (`http://localhost:5173`, `http://localhost:3000`, etc.) ready for Phase 11 React integration.

---

## 7. Verification & Test Execution

### A. Full Pytest Suite (105 / 105 Passed)
```
tests/test_backend_api.py::test_root_endpoint PASSED
tests/test_backend_api.py::test_health_endpoint PASSED
tests/test_backend_api.py::test_fleet_endpoint_and_bus_lookup PASSED
tests/test_backend_api.py::test_predictions_endpoint PASSED
tests/test_backend_api.py::test_metrics_endpoint PASSED
tests/test_backend_api.py::test_drift_endpoints PASSED
tests/test_backend_api.py::test_models_registry_endpoints PASSED
tests/test_backend_api.py::test_simulation_controls PASSED
tests/test_backend_api.py::test_scenarios_endpoint PASSED
tests/test_backend_api.py::test_lifecycle_and_retraining_endpoints PASSED
tests/test_backend_api.py::test_events_endpoint PASSED
tests/test_backend_api.py::test_websocket_live_stream PASSED
tests/test_backend_api.py::test_validation_error_handler PASSED

====================== 105 passed, 2 warnings in 30.65s =======================
```

### B. Artifact Integrity Verification (8 / 8 Checksums Intact)
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

## 8. Success Criteria Evaluation

| Criterion | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **FastAPI Backend Operational** | REST + OpenAPI documentation | Functional at `/docs` & `/redoc` | **PASS** |
| **Existing Simulator Reused** | Zero synthetic telemetry generation | Real Kandy bus GPS replay used | **PASS** |
| **Real Predictions Exposed** | Live model predictions with latency | `GET /api/predictions` active | **PASS** |
| **Real Online Metrics** | MAE, RMSE, P90, P95, Latency | `GET /api/metrics` active | **PASS** |
| **Drift Monitoring Exposed** | Covariate, Performance, Concept drift | `GET /api/drift` active | **PASS** |
| **Model Registry Read-Only** | Production champion inspection | `GET /api/models` active | **PASS** |
| **MLOps Lifecycle Graph** | Full pipeline stage status | `GET /api/lifecycle` active | **PASS** |
| **WebSocket Operational** | `/ws/live` bidirectional streaming | Connection, ping/pong & events tested | **PASS** |
| **Zero Data Mutation** | Parquet & v1.0.0 unchanged | 8/8 SHA-256 hashes match baseline | **PASS** |
| **Test Suite Quality** | 100% passing tests | 105/105 tests passing | **PASS** |

---

## 9. Next Steps
Phase 10 is complete and certified. The project is ready for **Phase 11 (React Dashboard & Frontend UI Integration)** upon instruction.
*(HARD STOP enforced).*
