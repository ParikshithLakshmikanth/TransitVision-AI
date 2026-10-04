"""TransitVision AI - Backend Data Schemas & API Models."""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# Enums
# -------------------------------------------------------------
class SystemEventType(str, Enum):
    PREDICTION = "PREDICTION"
    OUTCOME_RESOLVED = "OUTCOME_RESOLVED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    RETRAINING_TRIGGERED = "RETRAINING_TRIGGERED"
    RETRAINING_STARTED = "RETRAINING_STARTED"
    CANDIDATE_VALIDATED = "CANDIDATE_VALIDATED"
    SHADOW_COMPLETED = "SHADOW_COMPLETED"
    CANARY_STARTED = "CANARY_STARTED"
    CANARY_COMPLETED = "CANARY_COMPLETED"
    MODEL_PROMOTED = "MODEL_PROMOTED"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"
    ROLLBACK_TRIGGERED = "ROLLBACK_TRIGGERED"
    MODEL_ROLLED_BACK = "MODEL_ROLLED_BACK"
    SIMULATION_STATUS = "SIMULATION_STATUS"


# -------------------------------------------------------------
# Health Schemas
# -------------------------------------------------------------
class ServiceStatus(BaseModel):
    available: bool
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp_utc: str
    production_model_id: str
    production_model_version: str
    simulator_status: str
    services: Dict[str, ServiceStatus]


# -------------------------------------------------------------
# Fleet & Telemetry Schemas
# -------------------------------------------------------------
class BusStateResponse(BaseModel):
    deviceid: str
    trip_id: str
    direction: int
    current_segment: int
    last_event_timestamp_utc: str
    last_simulation_timestamp_utc: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    predicted_eta_sec: Optional[float] = None
    actual_eta_sec: Optional[float] = None
    prediction_error_sec: Optional[float] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    scenario_id: str
    scenario_name: str
    disturbance_intensity: float
    status: str


class FleetResponse(BaseModel):
    total_active_buses: int
    timestamp_utc: str
    buses: List[BusStateResponse]


# -------------------------------------------------------------
# Prediction Schemas
# -------------------------------------------------------------
class PredictionItem(BaseModel):
    prediction_id: str
    event_id: str
    trip_id: str
    deviceid: str
    direction: int
    segment: int
    predicted_eta_sec: float
    raw_prediction_sec: Optional[float] = None
    actual_eta_sec: Optional[float] = None
    signed_error_sec: Optional[float] = None
    absolute_error_sec: Optional[float] = None
    percentage_error: Optional[float] = None
    prediction_timestamp_utc: str
    model_id: str
    model_version: str
    algorithm: str
    latency_ms: float
    scenario_id: str
    synthetic_disturbance: bool


class PredictionsListResponse(BaseModel):
    total_predictions: int
    limit: int
    predictions: List[PredictionItem]


# -------------------------------------------------------------
# Simulation Control Schemas
# -------------------------------------------------------------
class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: str
    cursor: int
    total_records: int
    progress_pct: float
    replay_speed: float
    current_source_timestamp_utc: Optional[str] = None
    simulation_timestamp_utc: str
    records_emitted: int
    predictions_generated: int
    outcomes_resolved: int
    active_buses_count: int
    model_version: str
    scenario_id: str
    scenario_name: str
    synthetic_disturbance: bool
    disturbance_intensity: float
    affected_events: int
    unaffected_events: int
    source_dataset: str
    source_hash_verified: bool


class SimulationStepRequest(BaseModel):
    n: int = Field(default=1, ge=1, le=5000, description="Number of simulation steps/records to advance")


class SimulationSeekRequest(BaseModel):
    index: int = Field(ge=0, description="Target record index in stream")


class SimulationSpeedRequest(BaseModel):
    speed: float = Field(ge=0.0, le=100.0, description="Replay speed multiplier")


class SimulationControlResponse(BaseModel):
    success: bool
    message: str
    simulator_state: SimulationStatusResponse


# -------------------------------------------------------------
# Scenario Schemas
# -------------------------------------------------------------
class ScenarioInfo(BaseModel):
    scenario_id: str
    name: str
    description: str
    default_intensity: float
    is_synthetic: bool
    affected_segments: Optional[List[int]] = None


class ScenarioListResponse(BaseModel):
    active_scenario_id: str
    active_scenario_intensity: float
    available_scenarios: List[ScenarioInfo]


class ScenarioActivateRequest(BaseModel):
    scenario_id: str
    intensity: float = Field(default=0.75, ge=0.0, le=1.0, description="Disturbance intensity factor (0.0 to 1.0)")


class ScenarioActivateResponse(BaseModel):
    success: bool
    message: str
    scenario_id: str
    scenario_name: str
    intensity: float
    is_synthetic: bool


# -------------------------------------------------------------
# Metrics Schemas
# -------------------------------------------------------------
class OnlineMetrics(BaseModel):
    mae_sec: float
    rmse_sec: float
    median_ae_sec: float
    p90_error_sec: float
    p95_error_sec: float
    mean_signed_bias_sec: float = 0.0
    avg_inference_latency_ms: float


class MetricsResponse(BaseModel):
    simulation_id: str
    status: str
    active_model_id: str
    active_model_version: str
    records_emitted: int
    predictions_generated: int
    outcomes_resolved: int
    affected_events: int
    unaffected_events: int
    online_metrics: OnlineMetrics


class MetricsHistoryItem(BaseModel):
    window_index: int
    sample_count: int
    mae_sec: float
    rmse_sec: float
    median_ae_sec: float
    p90_error_sec: float
    scenario_id: str
    model_version: str
    timestamp_utc: str


class MetricsHistoryResponse(BaseModel):
    history: List[MetricsHistoryItem]


# -------------------------------------------------------------
# Drift Schemas
# -------------------------------------------------------------
class DriftEventResponse(BaseModel):
    event_id: str
    timestamp_utc: str
    drift_type: str
    severity: str
    detector: str
    scenario_id: str
    scenario_name: str
    synthetic_disturbance: bool
    disturbance_intensity: float
    source_type: str
    model_id: str
    model_version: str
    window_start_idx: int
    window_end_idx: int
    sample_count: int
    statistic: float
    threshold: float
    p_value: Optional[float] = None
    baseline_metric: float
    current_metric: float
    affected_features: List[str]
    evidence: Dict[str, Any]


class DriftSummaryResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    total_records_monitored: int
    windows_evaluated: int
    total_alerts: int
    data_drift_alerts: int
    performance_drift_alerts: int
    concept_drift_alerts: int
    first_detection_index: Optional[int] = None
    first_detection_delay_records: Optional[int] = None
    first_detection_timestamp: Optional[str] = None
    first_detection_type: Optional[str] = None
    first_detector_name: Optional[str] = None
    max_severity: str
    frequently_drifted_features: List[List[Any]]


class DriftStatusResponse(BaseModel):
    is_data_drift_active: bool
    is_performance_drift_active: bool
    is_concept_drift_active: bool
    latest_severity: str
    active_scenario: str
    summary: DriftSummaryResponse
    recent_events: List[DriftEventResponse]


# -------------------------------------------------------------
# Model Registry Schemas
# -------------------------------------------------------------
class ModelArtifactInfo(BaseModel):
    artifact_path: Optional[str] = None
    sha256: Optional[str] = None
    exists: bool = True


class ModelDetail(BaseModel):
    model_id: str
    version: str
    algorithm: str
    status: str
    parent_model_id: Optional[str] = None
    created_at_utc: Optional[str] = None
    promoted_at_utc: Optional[str] = None
    training_rows: Optional[int] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ModelListResponse(BaseModel):
    production_model_id: str
    total_models: int
    models: List[ModelDetail]


# -------------------------------------------------------------
# MLOps Lifecycle & Retraining Schemas
# -------------------------------------------------------------
class LifecycleStateResponse(BaseModel):
    production_model_id: str
    production_model_version: str
    production_algorithm: str
    current_lifecycle_state: str
    latest_candidate_id: Optional[str] = None
    latest_candidate_version: Optional[str] = None
    latest_candidate_status: Optional[str] = None
    latest_retraining_trigger_reason: Optional[str] = None
    latest_validation_result: Optional[Dict[str, Any]] = None
    shadow_status: Optional[str] = None
    canary_status: Optional[str] = None
    rollback_status: Optional[str] = None
    recent_lifecycle_events: List[Dict[str, Any]] = []


class RetrainingStatusResponse(BaseModel):
    is_retraining_in_progress: bool
    active_production_model_id: str
    last_trigger_decision: Optional[Dict[str, Any]] = None
    last_retraining_result: Optional[Dict[str, Any]] = None


class RetrainingTriggerRequest(BaseModel):
    candidate_version: str = Field(default="v1.2.0-demo", description="Candidate model version tag")
    algorithm: str = Field(default="LightGBMRegressor", description="Algorithm for candidate")
    dry_run: bool = Field(default=True, description="If True, evaluates gates without modifying production")


class RollbackStatusResponse(BaseModel):
    production_model_id: str
    production_model_version: str
    watchdog_window_size: int
    max_degradation_ratio: float
    max_latency_ms: float
    max_allowed_exceptions: int
    pre_promotion_baseline_mae_sec: float
    rollback_history: List[Dict[str, Any]] = []


# -------------------------------------------------------------
# System Event & WebSocket Schemas
# -------------------------------------------------------------
class SystemEvent(BaseModel):
    event_id: str
    timestamp_utc: str
    event_type: SystemEventType
    component: str
    model_version: str
    scenario_id: str
    payload: Dict[str, Any]
