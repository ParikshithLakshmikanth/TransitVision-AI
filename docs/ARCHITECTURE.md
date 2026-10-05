# TransitVision AI — High-Level System Architecture

TransitVision AI is an end-to-end, self-healing MLOps platform engineered for real-time bus arrival delay (ETA) prediction and automated concept-drift adaptation in dynamic public transportation networks.

```mermaid
flowchart TB
    subgraph DataEngineering["1. Data Engineering & Provenance"]
        RawData["Real Public Transit Datasets (Dublin Bus GPS)"]
        WeatherAPI["Open-Meteo Historical Archive"]
        Validator["TransitDataValidator (Bounds, Speed, Monotonicity)"]
        CleanFeature["Feature Store & Integration Layer"]
        RawData --> Validator
        WeatherAPI --> Validator
        Validator --> CleanFeature
    end

    subgraph MLSubsystem["2. ML & Model Registry"]
        TrainSplit["Historical Training Partition"]
        Trainer["LightGBM / Regressor Pipeline"]
        Evaluator["Evaluation Engine (MAE, RMSE, MAPE)"]
        Registry["Versioned Model Registry (Candidate vs. Active Production)"]
        CleanFeature --> TrainSplit
        TrainSplit --> Trainer
        Trainer --> Evaluator
        Evaluator --> Registry
    end

    subgraph SimulationStreaming["3. Digital Transit Simulator"]
        ReplayBuffer["Historical Telemetry Stream Buffer"]
        ScenarioEngine["Disturbance Scenario Engine (Rain, Congestion, Incidents)"]
        StreamServer["Real-Time Telemetry & WebSocket Broadcaster"]
        CleanFeature --> ReplayBuffer
        ReplayBuffer --> ScenarioEngine
        ScenarioEngine --> StreamServer
    end

    subgraph InferenceMonitoring["4. Real-Time Inference & Self-Healing Loop"]
        InferenceEngine["Zero-Downtime Prediction Engine"]
        GroundTruthBuffer["Ground Truth Matcher (Actual vs Predicted)"]
        DriftDetector["Drift Monitor (Kolmogorov-Smirnov & Page-Hinkley)"]
        RetrainingGate["Shadow Validation & Promotion Gate"]
        StreamServer --> InferenceEngine
        StreamServer --> GroundTruthBuffer
        InferenceEngine --> GroundTruthBuffer
        GroundTruthBuffer --> DriftDetector
        DriftDetector -- "Drift Trigger" --> RetrainingGate
        RetrainingGate -- "Hot Swap Model" --> InferenceEngine
        RetrainingGate -. "Trigger Retrain" .-> Trainer
    end

    subgraph Dashboard["5. Observability Dashboard"]
        LiveMap["Interactive Transit Map"]
        DriftPlots["Real-time Drift & Error Metrics"]
        SimulatorControls["Scenario Injection Controls"]
    end

    StreamServer --> LiveMap
    InferenceEngine --> LiveMap
    DriftDetector --> DriftPlots
    SimulatorControls --> ScenarioEngine
```

---

## Core System Components

### 1. Data Engineering & Ingestion
- **Raw Immutable Zone (`data/raw/`)**: Stores untouched public datasets.
- **Validation Engine (`ml/data/validator.py`)**: Computes physical coordinate boundary, speed, and schema checks, generating structured audit reports.
- **Integration Engine (`ml/preprocessing/`)**: Merges transit AVL GPS telemetry with historical meteorological data and static GTFS stop topologies.

### 2. Digital Transit Simulator
- Replays historical AVL breadcrumbs at configurable speed multipliers ($1\times, 5\times, 10\times$).
- Injects controlled environmental and traffic disturbances (rush-hour surge, rainstorms, arterial accidents).
- Distinguishes strictly between real ingested telemetry and synthetic disturbance modifiers.

### 3. Real-Time Prediction Engine
- Ingests streaming bus GPS coordinates and computes instantaneous feature vectors.
- Produces sub-second ETA and arrival delay predictions.
- Supports zero-downtime hot-swapping of active model pointers during autonomous retraining.

### 4. Continuous Drift Detection & Self-Healing Retraining Loop
- **Statistical Data Drift**: Monitored per feature via two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI).
- **Concept Drift**: Monitored on prediction residuals ($|y_{\text{true}} - \hat{y}|$) via the Page-Hinkley test.
- **Automated Validation Gate**: Evaluates retrained candidate models against the active model on recent temporal validation windows before promoting to production.

### 5. Unified Configuration & APIs
- Centralized YAML configuration (`config/config.yaml`) controls all thresholds, learning rates, drift tolerances, and simulation speeds.
- Fast, asynchronous FastAPI backend exposing REST and WebSocket endpoints.
