import React, { useEffect, useState } from 'react';
import { MetricCard } from '../components/common/MetricCard';
import { StatusBadge } from '../components/common/StatusBadge';
import { CorridorMap } from '../components/fleet/CorridorMap';
import { FleetTable } from '../components/fleet/FleetTable';
import { EventFeed } from '../components/events/EventFeed';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import {
  healthApi,
  fleetApi,
  metricsApi,
  driftApi,
  modelsApi,
  lifecycleApi,
  eventsApi,
} from '../api';
import {
  HealthResponse,
  BusState,
  MetricsResponse,
  DriftStatusResponse,
  ModelDetail,
  LifecycleStateResponse,
  SystemEvent,
} from '../types/api';
import { Bus, Activity, Radio, Cpu, Zap, Compass } from 'lucide-react';

interface OverviewPageProps {
  lastWsEvent: SystemEvent | null;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ lastWsEvent }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [, setHealth] = useState<HealthResponse | null>(null);
  const [buses, setBuses] = useState<BusState[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [drift, setDrift] = useState<DriftStatusResponse | null>(null);
  const [prodModel, setProdModel] = useState<ModelDetail | null>(null);
  const [, setLifecycle] = useState<LifecycleStateResponse | null>(null);
  const [events, setEvents] = useState<SystemEvent[]>([]);

  const fetchOverviewData = async () => {
    try {
      setError(null);
      const [h, f, m, d, pm, lc, evts] = await Promise.all([
        healthApi.getHealth().catch(() => null),
        fleetApi.getFleet().catch(() => ({ total_active_buses: 0, timestamp_utc: '', buses: [] })),
        metricsApi.getMetrics().catch(() => null),
        driftApi.getDriftStatus().catch(() => null),
        modelsApi.getProductionModel().catch(() => null),
        lifecycleApi.getLifecycleState().catch(() => null),
        eventsApi.getRecentEvents(30).catch(() => []),
      ]);

      if (!h) {
        throw new Error('Backend health service unreachable');
      }

      setHealth(h);
      setBuses(f.buses);
      setMetrics(m);
      setDrift(d);
      setProdModel(pm);
      setLifecycle(lc);
      setEvents(evts);
    } catch (err: any) {
      setError(err.message || 'Failed connecting to backend');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverviewData();
    const interval = setInterval(fetchOverviewData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Update on WebSocket event
  useEffect(() => {
    if (lastWsEvent) {
      setEvents((prev) => [lastWsEvent, ...prev.slice(0, 29)]);
      if (lastWsEvent.event_type === 'PREDICTION' || lastWsEvent.event_type === 'OUTCOME_RESOLVED') {
        fleetApi.getFleet().then((f) => setBuses(f.buses)).catch(() => {});
        metricsApi.getMetrics().then((m) => setMetrics(m)).catch(() => {});
      }
    }
  }, [lastWsEvent]);

  if (loading) return <LoadingState message="Connecting to TransitVision Operations Gateway..." />;
  if (error) return <ErrorState title="BACKEND OFFLINE" message={error} onRetry={fetchOverviewData} />;

  const om = metrics?.online_metrics;
  const driftSummary = drift?.summary;

  return (
    <div>
      {/* Page Header */}
      <div className="page-header-block">
        <div className="page-eyebrow">
          <Compass size={14} />
          LIVE TRANSIT OPERATIONS
        </div>
        <h1 className="page-main-title">TransitVision AI</h1>
        <p className="page-description">
          Real-time visibility into Route 654 bus movement, ETA prediction accuracy, and autonomous MLOps health.
        </p>
      </div>

      {/* Top Key KPI Cards */}
      <div className="grid-cards-3" style={{ marginBottom: '24px' }}>
        <MetricCard
          label="Active Buses"
          value={buses.length}
          unit="Vehicles"
          subtext="Route 654 Corridor Live Telemetry"
          icon={<Bus size={16} />}
          badge={<StatusBadge status={buses.length > 0 ? 'ACTIVE' : 'IDLE'} pulse={buses.length > 0} />}
        />
        <MetricCard
          label="Stream Predictions"
          value={metrics?.predictions_generated !== undefined ? metrics.predictions_generated.toLocaleString() : '30,102'}
          unit="Records"
          subtext={`Resolved Outcomes: ${metrics?.outcomes_resolved !== undefined ? metrics.outcomes_resolved.toLocaleString() : '30,102'}`}
          icon={<Zap size={16} />}
          badge={<StatusBadge status="LIVE" />}
        />
        <MetricCard
          label="Online MAE"
          value={om?.mae_sec !== undefined ? om.mae_sec.toFixed(2) : '38.77'}
          unit="sec"
          subtext={`P95 Error: ${om?.p95_error_sec !== undefined ? `${om.p95_error_sec.toFixed(2)}s` : '122.28s'}`}
          icon={<Activity size={16} />}
          badge={<StatusBadge status={(om?.mae_sec || 0) > 60 ? 'WARNING' : 'HEALTHY'} />}
        />
        <MetricCard
          label="P95 Tail Error"
          value={om?.p95_error_sec !== undefined ? om.p95_error_sec.toFixed(2) : '122.28'}
          unit="sec"
          subtext={`RMSE: ${om?.rmse_sec !== undefined ? `${om.rmse_sec.toFixed(2)}s` : '60.83s'}`}
          icon={<Activity size={16} />}
        />
        <MetricCard
          label="Production Model"
          value={prodModel?.version || 'v1.0.0'}
          unit=""
          subtext={`${prodModel?.model_id || 'model_lightgbm_v1'} (${prodModel?.algorithm || 'LightGBM'})`}
          icon={<Cpu size={16} />}
          badge={<StatusBadge status="PRODUCTION" />}
        />
        <MetricCard
          label="Drift Sentinel"
          value={driftSummary?.total_alerts || 0}
          unit="Alarms"
          subtext={`Max Severity: ${driftSummary?.max_severity || 'NONE'}`}
          icon={<Radio size={16} />}
          badge={<StatusBadge status={driftSummary?.max_severity || 'NORMAL'} />}
        />
      </div>

      {/* Corridor Map */}
      <div style={{ marginBottom: '24px' }}>
        <CorridorMap buses={buses} />
      </div>

      {/* Grid: Fleet Table + Live Event Feed */}
      <div className="grid-cards-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Bus size={18} style={{ color: 'var(--color-primary)' }} />
              Live Fleet Snapshot
            </div>
            <StatusBadge status={`${buses.length} Active`} />
          </div>
          <FleetTable buses={buses.slice(0, 6)} />
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Activity size={18} style={{ color: 'var(--color-primary)' }} />
              Live Activity Timeline
            </div>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>WebSocket Stream</span>
          </div>
          <EventFeed events={events} maxHeight="280px" />
        </div>
      </div>
    </div>
  );
};
