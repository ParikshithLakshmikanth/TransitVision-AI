import React, { useEffect, useState } from 'react';
import { driftApi } from '../api';
import { DriftStatusResponse, DriftEvent } from '../types/api';
import { StatusBadge } from '../components/common/StatusBadge';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import { Radio, Activity, Layers, Compass } from 'lucide-react';

export const DriftMonitorPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [driftStatus, setDriftStatus] = useState<DriftStatusResponse | null>(null);
  const [events, setEvents] = useState<DriftEvent[]>([]);

  const fetchDrift = async () => {
    try {
      setError(null);
      const [statusRes, eventsRes] = await Promise.all([
        driftApi.getDriftStatus(),
        driftApi.getDriftEvents(50),
      ]);
      setDriftStatus(statusRes);
      setEvents(eventsRes);
    } catch (err: any) {
      setError(err.message || 'Failed connecting to Drift Engine');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDrift();
    const interval = setInterval(fetchDrift, 4000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <LoadingState message="Querying Statistical Drift Detectors..." />;
  if (error) return <ErrorState title="DRIFT MONITOR UNAVAILABLE" message={error} onRetry={fetchDrift} />;

  const s = driftStatus?.summary;

  return (
    <div>
      {/* Page Header */}
      <div className="page-header-block">
        <div className="page-eyebrow">
          <Compass size={14} />
          MODEL MONITORING
        </div>
        <h1 className="page-main-title">Drift Monitor</h1>
        <p className="page-description">
          Detect covariate distribution shifts, performance degradation, and concept drift before transit service quality is impacted.
        </p>
      </div>

      {/* 3 Prominent Monitoring Dimension Cards */}
      <div className="grid-cards-3">
        {/* Dimension 1: Data Drift */}
        <div className="card" style={{ borderTop: '3px solid var(--color-primary)' }}>
          <div className="card-header">
            <div className="card-title">
              <Layers size={18} style={{ color: 'var(--color-primary)' }} />
              Covariate / Data Drift
            </div>
            <StatusBadge status={driftStatus?.is_data_drift_active ? 'DRIFT DETECTED' : 'STABLE'} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Detects input feature distribution shifts using Population Stability Index (PSI) & Kolmogorov-Smirnov (KS) tests vs baseline.
            </div>
            <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '4px' }}>
              {s?.data_drift_alerts || 0} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Alarms</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Detector: <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>PSI_KS_HYBRID (Threshold = 0.20)</span>
            </div>
          </div>
        </div>

        {/* Dimension 2: Performance Drift */}
        <div className="card" style={{ borderTop: '3px solid var(--color-warning)' }}>
          <div className="card-header">
            <div className="card-title">
              <Activity size={18} style={{ color: 'var(--color-warning)' }} />
              Performance Drift
            </div>
            <StatusBadge status={driftStatus?.is_performance_drift_active ? 'DEGRADATION' : 'OPTIMAL'} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Monitors online MAE and RMSE degradation over rolling 500-record sliding prediction windows.
            </div>
            <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '4px' }}>
              {s?.performance_drift_alerts || 0} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Alarms</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Detector: <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>RollingPerformanceMonitor (1.60x MAE)</span>
            </div>
          </div>
        </div>

        {/* Dimension 3: Concept Drift */}
        <div className="card" style={{ borderTop: '3px solid var(--color-purple)' }}>
          <div className="card-header">
            <div className="card-title">
              <Radio size={18} style={{ color: 'var(--color-purple)' }} />
              Concept Drift
            </div>
            <StatusBadge status={driftStatus?.is_concept_drift_active ? 'SHIFT DETECTED' : 'CALIBRATED'} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Sequential residual monitoring via Page-Hinkley cumulative sum tests on prediction errors.
            </div>
            <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '4px' }}>
              {s?.concept_drift_alerts || 0} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Alarms</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Detector: <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Page-Hinkley (Lambda = 50.0)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Frequently Drifted Features & Detection Latency */}
      <div className="grid-cards-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Top Drifted Corridor Features</div>
              <div className="card-subtitle">Frequency of statistical distribution alerts</div>
            </div>
          </div>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature Name</th>
                  <th>Drift Occurrence Count</th>
                  <th>Classification</th>
                </tr>
              </thead>
              <tbody>
                {s && s.frequently_drifted_features.length > 0 ? (
                  s.frequently_drifted_features.map(([feat, count]) => (
                    <tr key={feat}>
                      <td className="mono" style={{ fontWeight: 600, color: 'var(--color-primary)' }}>
                        {feat}
                      </td>
                      <td className="mono">{count} events</td>
                      <td>
                        <StatusBadge
                          status={feat.includes('rain') || feat.includes('temp') || feat.includes('wind') ? 'METEOROLOGICAL' : 'TRAFFIC'}
                        />
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px' }}>
                      No feature drift events detected yet (Nominal Stream Baseline).
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Detection Latency & Sentinel Metrics</div>
              <div className="card-subtitle">Earliest alarm observation benchmarks</div>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Total Stream Records Monitored:</span>
              <span className="mono" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{s?.total_records_monitored || 0}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Sliding Windows Evaluated:</span>
              <span className="mono" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{s?.windows_evaluated || 0}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>First Detection Delay:</span>
              <span className="mono" style={{ fontWeight: 700, color: 'var(--color-warning)' }}>
                {s?.first_detection_delay_records ? `${s.first_detection_delay_records} records` : 'None (No Drift)'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>First Responding Detector:</span>
              <span className="mono" style={{ color: 'var(--text-primary)' }}>{s?.first_detector_name || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Peak Severity Level:</span>
              <StatusBadge status={s?.max_severity || 'NONE'} />
            </div>
          </div>
        </div>
      </div>

      {/* Chronological Drift Event Stream */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Sequential Drift Alarm Event Log</div>
            <div className="card-subtitle">Emitted when statistical or performance thresholds are breached</div>
          </div>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Event ID</th>
                <th>Type</th>
                <th>Severity</th>
                <th>Detector</th>
                <th>Window Range</th>
                <th>Statistic</th>
                <th>Threshold</th>
                <th>Scenario</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {events.length > 0 ? (
                events.slice().reverse().map((e) => (
                  <tr key={e.event_id}>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>{e.event_id}</td>
                    <td><StatusBadge status={e.drift_type} /></td>
                    <td><StatusBadge status={e.severity} /></td>
                    <td className="mono" style={{ fontSize: '0.78rem' }}>{e.detector}</td>
                    <td className="mono">[{e.window_start_idx}..{e.window_end_idx}]</td>
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{e.statistic.toFixed(4)}</td>
                    <td className="mono">{e.threshold.toFixed(4)}</td>
                    <td><span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{e.scenario_name}</span></td>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>
                      {new Date(e.timestamp_utc).toLocaleTimeString()}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '28px' }}>
                    No drift alarms recorded. The corridor model is serving within certified baseline tolerances.
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
