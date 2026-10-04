import React, { useEffect, useState } from 'react';
import { metricsApi, predictionsApi } from '../api';
import { MetricsResponse, MetricsHistoryResponse, PredictionItem } from '../types/api';
import { MetricCard } from '../components/common/MetricCard';
import { StatusBadge } from '../components/common/StatusBadge';
import { ActualVsPredictedChart } from '../components/charts/ActualVsPredictedChart';
import { ErrorDistributionChart } from '../components/charts/ErrorDistributionChart';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import { Activity, Clock, BarChart3, TrendingUp, Compass } from 'lucide-react';

export const EtaAnalyticsPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [history, setHistory] = useState<MetricsHistoryResponse | null>(null);
  const [predictions, setPredictions] = useState<PredictionItem[]>([]);

  const fetchAnalytics = async () => {
    try {
      setError(null);
      const [m, h, p] = await Promise.all([
        metricsApi.getMetrics(),
        metricsApi.getMetricsHistory(),
        predictionsApi.getPredictions(100),
      ]);
      setMetrics(m);
      setHistory(h);
      setPredictions(p.predictions);
    } catch (err: any) {
      setError(err.message || 'Failed loading ETA analytics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 4000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <LoadingState message="Calculating real-time ETA accuracy metrics..." />;
  if (error) return <ErrorState title="ANALYTICS UNAVAILABLE" message={error} onRetry={fetchAnalytics} />;

  const om = metrics?.online_metrics;

  return (
    <div>
      {/* Page Header */}
      <div className="page-header-block">
        <div className="page-eyebrow">
          <Compass size={14} />
          ETA PERFORMANCE
        </div>
        <h1 className="page-main-title">ETA Analytics</h1>
        <p className="page-description">
          Continuous accuracy benchmarks, residual distribution, and rolling window error tracking across live stream replays.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid-cards-4">
        <MetricCard
          label="Mean Absolute Error"
          value={om?.mae_sec !== undefined ? om.mae_sec.toFixed(2) : '—'}
          unit="sec"
          subtext="Online streaming MAE"
          icon={<Activity size={16} />}
        />
        <MetricCard
          label="Root Mean Squared Error"
          value={om?.rmse_sec !== undefined ? om.rmse_sec.toFixed(2) : '—'}
          unit="sec"
          subtext="Penalizes large tail deviations"
          icon={<TrendingUp size={16} />}
        />
        <MetricCard
          label="P95 Tail Error"
          value={om?.p95_error_sec !== undefined ? om.p95_error_sec.toFixed(2) : '—'}
          unit="sec"
          subtext={`P90 Error: ${om?.p90_error_sec !== undefined ? `${om.p90_error_sec.toFixed(1)}s` : '—'}`}
          icon={<BarChart3 size={16} />}
        />
        <MetricCard
          label="Inference Latency"
          value={om?.avg_inference_latency_ms !== undefined ? om.avg_inference_latency_ms.toFixed(2) : '—'}
          unit="ms"
          subtext="Sub-millisecond model inference"
          icon={<Clock size={16} />}
          badge={<StatusBadge status="< 5ms" />}
        />
      </div>

      {/* Charts Grid */}
      <div className="grid-cards-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Actual vs Predicted ETA Correlation</div>
              <div className="card-subtitle">Ideal 45° parity dashed line with observed points</div>
            </div>
          </div>
          <ActualVsPredictedChart predictions={predictions} />
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Streaming Error Distribution</div>
              <div className="card-subtitle">Absolute error bucketing (seconds)</div>
            </div>
          </div>
          <ErrorDistributionChart predictions={predictions} />
        </div>
      </div>

      {/* Historical Performance Trend Table */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Rolling Window Performance Metrics History</div>
            <div className="card-subtitle">Aggregated 50-record window snapshots</div>
          </div>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Window Index</th>
                <th>Sample Count</th>
                <th>Window MAE</th>
                <th>Window RMSE</th>
                <th>Median AE</th>
                <th>P90 Error</th>
                <th>Scenario</th>
                <th>Model Version</th>
                <th>Snapshot Time</th>
              </tr>
            </thead>
            <tbody>
              {history && history.history.length > 0 ? (
                history.history.slice().reverse().map((h) => (
                  <tr key={h.window_index}>
                    <td className="mono">W#{h.window_index}</td>
                    <td className="mono">{h.sample_count}</td>
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{h.mae_sec.toFixed(2)}s</td>
                    <td className="mono">{h.rmse_sec.toFixed(2)}s</td>
                    <td className="mono">{h.median_ae_sec.toFixed(2)}s</td>
                    <td className="mono">{h.p90_error_sec.toFixed(2)}s</td>
                    <td><StatusBadge status={h.scenario_id} /></td>
                    <td className="mono" style={{ fontSize: '0.78rem' }}>{h.model_version}</td>
                    <td className="mono" style={{ fontSize: '0.78rem' }}>
                      {new Date(h.timestamp_utc).toLocaleTimeString()}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '28px' }}>
                    No rolling window metrics recorded yet. Advance the simulator to generate window snapshots.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
