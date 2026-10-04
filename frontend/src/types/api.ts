/**
 * TransitVision AI - TypeScript API Definitions & Event Types
 * Matching FastAPI Pydantic V2 schemas strictly.
 */

export type SystemEventType =
  | 'PREDICTION'
  | 'OUTCOME_RESOLVED'
  | 'DRIFT_DETECTED'
  | 'RETRAINING_TRIGGERED'
  | 'RETRAINING_STARTED'
  | 'CANDIDATE_VALIDATED'
  | 'SHADOW_COMPLETED'
  | 'CANARY_STARTED'
  | 'CANARY_COMPLETED'
  | 'MODEL_PROMOTED'
  | 'REGRESSION_DETECTED'
  | 'ROLLBACK_TRIGGERED'
  | 'MODEL_ROLLED_BACK'
  | 'SIMULATION_STATUS';

export interface ServiceStatus {
  available: boolean;
  details?: string;
}

export interface HealthResponse {
  status: string;
  timestamp_utc: string;
  production_model_id: string;
  production_model_version: string;
  simulator_status: string;
  services: Record<string, ServiceStatus>;
}

export interface BusState {
  deviceid: string;
  trip_id: string;
  direction: number;
  current_segment: number;
  last_event_timestamp_utc: string;
  last_simulation_timestamp_utc: string;
  latitude?: number | null;
  longitude?: number | null;
  predicted_eta_sec?: number | null;
  actual_eta_sec?: number | null;
  prediction_error_sec?: number | null;
  model_id?: string | null;
  model_version?: string | null;
  scenario_id: string;
  scenario_name: string;
  disturbance_intensity: number;
  status: string;
}

export interface FleetResponse {
  total_active_buses: number;
  timestamp_utc: string;
  buses: BusState[];
}

export interface PredictionItem {
  prediction_id: string;
  event_id: string;
  trip_id: string;
  deviceid: string;
  direction: number;
  segment: number;
  predicted_eta_sec: number;
  raw_prediction_sec?: number | null;
  actual_eta_sec?: number | null;
  signed_error_sec?: number | null;
  absolute_error_sec?: number | null;
  percentage_error?: number | null;
  prediction_timestamp_utc: string;
  model_id: string;
  model_version: string;
  algorithm: string;
  latency_ms: number;
  scenario_id: string;
  synthetic_disturbance: boolean;
}

export interface PredictionsListResponse {
  total_predictions: number;
  limit: number;
  predictions: PredictionItem[];
}

export interface OnlineMetrics {
  mae_sec: number;
  rmse_sec: number;
  median_ae_sec: number;
  p90_error_sec: number;
  p95_error_sec: number;
  mean_signed_bias_sec: number;
  avg_inference_latency_ms: number;
}

export interface MetricsResponse {
  simulation_id: string;
  status: string;
  active_model_id: string;
  active_model_version: string;
  records_emitted: number;
  predictions_generated: number;
  outcomes_resolved: number;
  affected_events: number;
  unaffected_events: number;
  online_metrics: OnlineMetrics;
}

export interface MetricsHistoryItem {
  window_index: number;
  sample_count: number;
  mae_sec: number;
  rmse_sec: number;
  median_ae_sec: number;
  p90_error_sec: number;
  scenario_id: string;
  model_version: string;
  timestamp_utc: string;
}

export interface MetricsHistoryResponse {
  history: MetricsHistoryItem[];
}

export interface DriftEvent {
  event_id: string;
  timestamp_utc: string;
  drift_type: string;
  severity: string;
  detector: string;
  scenario_id: string;
  scenario_name: string;
  synthetic_disturbance: boolean;
  disturbance_intensity: number;
  source_type: string;
  model_id: string;
  model_version: string;
  window_start_idx: number;
  window_end_idx: number;
  sample_count: number;
  statistic: number;
  threshold: number;
  p_value?: number | null;
  baseline_metric: number;
  current_metric: number;
  affected_features: string[];
  evidence: Record<string, any>;
}

export interface DriftSummaryResponse {
  scenario_id: string;
  scenario_name: string;
  total_records_monitored: number;
  windows_evaluated: number;
  total_alerts: number;
  data_drift_alerts: number;
  performance_drift_alerts: number;
  concept_drift_alerts: number;
  first_detection_index?: number | null;
  first_detection_delay_records?: number | null;
  first_detection_timestamp?: string | null;
  first_detection_type?: string | null;
  first_detector_name?: string | null;
  max_severity: string;
  frequently_drifted_features: [string, number][];
}

export interface DriftStatusResponse {
  is_data_drift_active: boolean;
  is_performance_drift_active: boolean;
  is_concept_drift_active: boolean;
  latest_severity: string;
  active_scenario: string;
  summary: DriftSummaryResponse;
  recent_events: DriftEvent[];
}

export interface ModelDetail {
  model_id: string;
  version: string;
  algorithm: string;
  status: string;
  parent_model_id?: string | null;
  created_at_utc?: string | null;
  promoted_at_utc?: string | null;
  training_rows?: number | null;
  validation_metrics?: Record<string, any> | null;
  hyperparameters?: Record<string, any> | null;
  notes?: string | null;
}

export interface ModelListResponse {
  production_model_id: string;
  total_models: number;
  models: ModelDetail[];
}

export interface SimulationStatusResponse {
  simulation_id: string;
  status: string;
  cursor: number;
  total_records: number;
  progress_pct: number;
  replay_speed: number;
  current_source_timestamp_utc?: string | null;
  simulation_timestamp_utc: string;
  records_emitted: number;
  predictions_generated: number;
  outcomes_resolved: number;
  active_buses_count: number;
  model_version: string;
  scenario_id: string;
  scenario_name: string;
  synthetic_disturbance: boolean;
  disturbance_intensity: number;
  affected_events: number;
  unaffected_events: number;
  source_dataset: string;
  source_hash_verified: boolean;
}

export interface SimulationControlResponse {
  success: boolean;
  message: string;
  simulator_state: SimulationStatusResponse;
}

export interface ScenarioInfo {
  scenario_id: string;
  name: string;
  description: string;
  default_intensity: number;
  is_synthetic: boolean;
  affected_segments?: number[] | null;
}

export interface ScenarioListResponse {
  active_scenario_id: string;
  active_scenario_intensity: number;
  available_scenarios: ScenarioInfo[];
}

export interface ScenarioActivateResponse {
  success: boolean;
  message: string;
  scenario_id: string;
  scenario_name: string;
  intensity: number;
  is_synthetic: boolean;
}


export interface LifecycleStateResponse {
  production_model_id: string;
  production_model_version: string;
  production_algorithm: string;
  current_lifecycle_state: string;
  latest_candidate_id?: string | null;
  latest_candidate_version?: string | null;
  latest_candidate_status?: string | null;
  latest_retraining_trigger_reason?: string | null;
  latest_validation_result?: Record<string, any> | null;
  shadow_status?: string | null;
  canary_status?: string | null;
  rollback_status?: string | null;
  recent_lifecycle_events: Record<string, any>[];
}

export interface RetrainingStatusResponse {
  is_retraining_in_progress: boolean;
  active_production_model_id: string;
  last_trigger_decision?: Record<string, any> | null;
  last_retraining_result?: Record<string, any> | null;
}

export interface RollbackStatusResponse {
  production_model_id: string;
  production_model_version: string;
  watchdog_window_size: number;
  max_degradation_ratio: number;
  max_latency_ms: number;
  max_allowed_exceptions: number;
  pre_promotion_baseline_mae_sec: number;
  rollback_history: Record<string, any>[];
}

export interface SystemEvent {
  event_id: string;
  timestamp_utc: string;
  event_type: SystemEventType;
  component: string;
  model_version: string;
  scenario_id: string;
  payload: Record<string, any>;
}
