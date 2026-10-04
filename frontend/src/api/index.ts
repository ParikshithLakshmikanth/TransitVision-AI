import { apiClient } from './client';
import {
  HealthResponse,
  FleetResponse,
  BusState,
  PredictionsListResponse,
  PredictionItem,
  MetricsResponse,
  MetricsHistoryResponse,
  DriftStatusResponse,
  DriftSummaryResponse,
  DriftEvent,
  ModelListResponse,
  ModelDetail,
  SimulationStatusResponse,
  SimulationControlResponse,
  ScenarioListResponse,
  ScenarioActivateResponse,
  LifecycleStateResponse,
  RetrainingStatusResponse,
  RollbackStatusResponse,
  SystemEvent,
  SystemEventType,
} from '../types/api';

export const healthApi = {
  getHealth: () => apiClient<HealthResponse>('/health'),
};

export const fleetApi = {
  getFleet: () => apiClient<FleetResponse>('/fleet'),
  getBusState: (busId: string) => apiClient<BusState>(`/fleet/${encodeURIComponent(busId)}`),
};

export const predictionsApi = {
  getPredictions: (limit = 50, tripId?: string, deviceid?: string) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (tripId) params.append('trip_id', tripId);
    if (deviceid) params.append('deviceid', deviceid);
    return apiClient<PredictionsListResponse>(`/predictions?${params.toString()}`);
  },
  getPredictionById: (id: string) => apiClient<PredictionItem>(`/predictions/${encodeURIComponent(id)}`),
};

export const metricsApi = {
  getMetrics: () => apiClient<MetricsResponse>('/metrics'),
  getMetricsHistory: () => apiClient<MetricsHistoryResponse>('/metrics/history'),
};

export const driftApi = {
  getDriftStatus: () => apiClient<DriftStatusResponse>('/drift'),
  getDriftSummary: () => apiClient<DriftSummaryResponse>('/drift/summary'),
  getDriftEvents: (limit = 50) => apiClient<DriftEvent[]>(`/drift/events?limit=${limit}`),
};

export const modelsApi = {
  getModels: () => apiClient<ModelListResponse>('/models'),
  getProductionModel: () => apiClient<ModelDetail>('/models/production'),
  getModelById: (modelId: string) => apiClient<ModelDetail>(`/models/${encodeURIComponent(modelId)}`),
};

export const simulationApi = {
  getStatus: () => apiClient<SimulationStatusResponse>('/simulation/status'),
  start: () => apiClient<SimulationControlResponse>('/simulation/start', { method: 'POST' }),
  pause: () => apiClient<SimulationControlResponse>('/simulation/pause', { method: 'POST' }),
  resume: () => apiClient<SimulationControlResponse>('/simulation/resume', { method: 'POST' }),
  stop: () => apiClient<SimulationControlResponse>('/simulation/stop', { method: 'POST' }),
  reset: () => apiClient<SimulationControlResponse>('/simulation/reset', { method: 'POST' }),
  step: (n = 1) => apiClient<PredictionItem[]>('/simulation/step', {
    method: 'POST',
    body: JSON.stringify({ n }),
  }),
  seek: (index: number) => apiClient<SimulationControlResponse>('/simulation/seek', {
    method: 'POST',
    body: JSON.stringify({ index }),
  }),
  setSpeed: (speed: number) => apiClient<SimulationControlResponse>('/simulation/speed', {
    method: 'POST',
    body: JSON.stringify({ speed }),
  }),
};

export const scenariosApi = {
  getScenarios: () => apiClient<ScenarioListResponse>('/scenarios'),
  activateScenario: (scenarioId: string, intensity = 0.75) => apiClient<ScenarioActivateResponse>('/scenarios/activate', {
    method: 'POST',
    body: JSON.stringify({ scenario_id: scenarioId, intensity }),
  }),
};

export const lifecycleApi = {
  getLifecycleState: () => apiClient<LifecycleStateResponse>('/lifecycle'),
  getRetrainingStatus: () => apiClient<RetrainingStatusResponse>('/retraining/status'),
  getRetrainingHistory: () => apiClient<Record<string, any>>('/retraining/history'),
  triggerDryRunRetraining: (candidateVersion = 'v1.2.0-demo') => apiClient<any>('/retraining/trigger', {
    method: 'POST',
    body: JSON.stringify({ candidate_version: candidateVersion, dry_run: true }),
  }),
  getRollbackStatus: () => apiClient<RollbackStatusResponse>('/rollback/status'),
};

export const eventsApi = {
  getRecentEvents: (limit = 50, eventType?: SystemEventType) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (eventType) params.append('event_type', eventType);
    return apiClient<SystemEvent[]>(`/events?${params.toString()}`);
  },
};
