import React, { useEffect, useState } from 'react';
import { simulationApi, scenariosApi } from '../api';
import { SimulationStatusResponse, ScenarioInfo } from '../types/api';
import { StatusBadge } from '../components/common/StatusBadge';
import { MetricCard } from '../components/common/MetricCard';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import {
  Play,
  Pause,
  RotateCcw,
  Square,
  Sliders,
  CloudRain,
  Zap,
  Activity,
  Compass,
} from 'lucide-react';

export const ScenarioSimulatorPage: React.FC = () => {
  const [status, setStatus] = useState<SimulationStatusResponse | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [selectedIntensity, setSelectedIntensity] = useState<number>(0.75);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      setError(null);
      const [st, sc] = await Promise.all([
        simulationApi.getStatus(),
        scenariosApi.getScenarios(),
      ]);
      setStatus(st);
      setScenarios(sc.available_scenarios);
      setSelectedIntensity(st.disturbance_intensity || 0.75);
    } catch (err: any) {
      setError(err.message || 'Failed connecting to Transit Simulator');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2500);
    return () => clearInterval(interval);
  }, []);

  const handleControl = async (action: () => Promise<any>, successMsg: string) => {
    try {
      setActionLoading(true);
      await action();
      setFeedback(successMsg);
      await fetchStatus();
      setTimeout(() => setFeedback(null), 3000);
    } catch (err: any) {
      setFeedback(`Error: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleActivateScenario = async (scenarioId: string) => {
    try {
      setActionLoading(true);
      const intensity = scenarioId === 'BASELINE' ? 0.0 : selectedIntensity;
      const res = await scenariosApi.activateScenario(scenarioId, intensity);
      setFeedback(`Activated scenario: ${res.scenario_name} (Intensity: ${res.intensity})`);
      await fetchStatus();
      setTimeout(() => setFeedback(null), 4000);
    } catch (err: any) {
      setFeedback(`Error activating scenario: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) return <LoadingState message="Loading Scenario Simulator Gateway..." />;
  if (error) return <ErrorState title="SIMULATOR UNAVAILABLE" message={error} onRetry={fetchStatus} />;

  const isRunning = status?.status === 'RUNNING';
  const isPaused = status?.status === 'PAUSED';

  return (
    <div>
      {/* Page Header */}
      <div className="page-header-block">
        <div className="page-eyebrow">
          <Compass size={14} />
          TRANSIT EXPERIMENTATION
        </div>
        <h1 className="page-main-title">Scenario Simulator</h1>
        <p className="page-description">
          Evaluate ETA prediction robustness and trigger closed-loop MLOps adaptation under controlled Route 654 corridor disturbances.
        </p>
      </div>

      {feedback && (
        <div
          style={{
            padding: '12px 18px',
            backgroundColor: 'var(--color-primary-subtle)',
            border: '1px solid var(--color-primary-border)',
            borderRadius: '8px',
            marginBottom: '20px',
            fontSize: '0.85rem',
            color: 'var(--color-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontWeight: 500,
          }}
        >
          <Zap size={16} />
          {feedback}
        </div>
      )}

      {/* Top Playback Status Cards */}
      <div className="grid-cards-4">
        <MetricCard
          label="Simulation Status"
          value={status?.status || 'INITIALIZED'}
          unit=""
          subtext={`Replay Speed: ${status?.replay_speed || 1}x`}
          icon={<Activity size={16} />}
          badge={<StatusBadge status={status?.status || 'INITIALIZED'} pulse={isRunning} />}
        />
        <MetricCard
          label="Stream Progress"
          value={`${status?.progress_pct?.toFixed(1) || 0}%`}
          unit=""
          subtext={`Cursor: ${status?.cursor?.toLocaleString() || 0} / ${status?.total_records?.toLocaleString() || 0}`}
          icon={<Sliders size={16} />}
        />
        <MetricCard
          label="Active Scenario"
          value={status?.scenario_name || 'Baseline'}
          unit=""
          subtext={`Intensity: ${status?.disturbance_intensity?.toFixed(2) || '0.00'}`}
          icon={<CloudRain size={16} />}
          badge={<StatusBadge status={status?.synthetic_disturbance ? 'DISTURBED' : 'BASELINE'} />}
        />
        <MetricCard
          label="Telemetry Emitted"
          value={status?.records_emitted?.toLocaleString() || 0}
          unit="Events"
          subtext={`Predictions: ${status?.predictions_generated?.toLocaleString() || 0}`}
          icon={<Zap size={16} />}
        />
      </div>

      {/* Playback Controls Card */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <div className="card-title">
            <Sliders size={18} style={{ color: 'var(--color-primary)' }} />
            Simulation Playback Controls
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Dataset: <span className="mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{status?.source_dataset}</span> (Hash Verified)
          </span>
        </div>

        {/* Progress Bar */}
        <div style={{ width: '100%', height: '8px', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', overflow: 'hidden', marginBottom: '20px' }}>
          <div
            style={{
              width: `${status?.progress_pct || 0}%`,
              height: '100%',
              backgroundColor: 'var(--color-primary)',
              transition: 'width 0.3s ease',
            }}
          />
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center', marginBottom: '20px' }}>
          {!isRunning ? (
            <button
              className="btn btn-primary"
              disabled={actionLoading}
              onClick={() => handleControl(isPaused ? simulationApi.resume : simulationApi.start, 'Simulation started')}
            >
              <Play size={16} />
              {isPaused ? 'Resume Replay' : 'Start Replay'}
            </button>
          ) : (
            <button
              className="btn btn-secondary"
              disabled={actionLoading}
              onClick={() => handleControl(simulationApi.pause, 'Simulation paused')}
            >
              <Pause size={16} />
              Pause
            </button>
          )}

          <button
            className="btn btn-secondary"
            disabled={actionLoading}
            onClick={() => handleControl(simulationApi.stop, 'Simulation stopped')}
          >
            <Square size={16} />
            Stop
          </button>

          <button
            className="btn btn-secondary"
            disabled={actionLoading}
            onClick={() => handleControl(simulationApi.reset, 'Simulation reset to record 0')}
          >
            <RotateCcw size={16} />
            Reset
          </button>

          <div style={{ height: '24px', width: '1px', background: 'var(--border-color)', margin: '0 8px' }} />

          {/* Step Buttons */}
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>STEP:</span>
          <button
            className="btn btn-secondary btn-sm"
            disabled={actionLoading}
            onClick={() => handleControl(() => simulationApi.step(1), 'Stepped +1 record')}
          >
            +1
          </button>
          <button
            className="btn btn-secondary btn-sm"
            disabled={actionLoading}
            onClick={() => handleControl(() => simulationApi.step(5), 'Stepped +5 records')}
          >
            +5
          </button>
          <button
            className="btn btn-secondary btn-sm"
            disabled={actionLoading}
            onClick={() => handleControl(() => simulationApi.step(50), 'Stepped +50 records')}
          >
            +50
          </button>
          <button
            className="btn btn-secondary btn-sm"
            disabled={actionLoading}
            onClick={() => handleControl(() => simulationApi.step(500), 'Stepped +500 records')}
          >
            +500
          </button>

          <div style={{ height: '24px', width: '1px', background: 'var(--border-color)', margin: '0 8px' }} />

          {/* Speed Buttons */}
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>SPEED:</span>
          {[1.0, 2.0, 5.0, 10.0, 50.0].map((spd) => (
            <button
              key={spd}
              className={`btn btn-sm ${status?.replay_speed === spd ? 'btn-primary' : 'btn-secondary'}`}
              disabled={actionLoading}
              onClick={() => handleControl(() => simulationApi.setSpeed(spd), `Replay speed set to ${spd}x`)}
            >
              {spd}x
            </button>
          ))}
        </div>
      </div>

      {/* Scenario Injection Section */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <CloudRain size={18} style={{ color: 'var(--color-primary)' }} />
            Phase 7 Transit Disturbance Scenario Injection
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Disturbance Intensity:</span>
            <input
              type="range"
              min="0.10"
              max="1.00"
              step="0.05"
              value={selectedIntensity}
              onChange={(e) => setSelectedIntensity(parseFloat(e.target.value))}
              className="slider-range"
              style={{ width: '130px' }}
            />
            <span className="mono" style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--color-primary)', minWidth: '40px' }}>
              {(selectedIntensity * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="grid-cards-3" style={{ marginBottom: 0 }}>
          {scenarios.map((sc) => {
            const isActive = status?.scenario_id === sc.scenario_id;

            return (
              <div
                key={sc.scenario_id}
                style={{
                  background: isActive ? 'var(--color-primary-subtle)' : '#ffffff',
                  border: `1px solid ${isActive ? 'var(--color-primary)' : 'var(--border-color)'}`,
                  borderRadius: '10px',
                  padding: '18px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '14px',
                  boxShadow: isActive ? '0 2px 4px rgba(0, 87, 184, 0.1)' : 'var(--shadow-sm)',
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                    <div style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-primary)' }}>{sc.name}</div>
                    <StatusBadge status={sc.is_synthetic ? 'SYNTHETIC' : 'REAL'} />
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    {sc.description}
                  </p>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px', borderTop: '1px solid var(--border-color)' }}>
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    {sc.affected_segments ? `Segments: [${sc.affected_segments.join(',')}]` : 'Global Corridor'}
                  </span>
                  <button
                    className={`btn btn-sm ${isActive ? 'btn-primary' : 'btn-secondary'}`}
                    disabled={actionLoading || isActive}
                    onClick={() => handleActivateScenario(sc.scenario_id)}
                  >
                    {isActive ? 'Active Now' : 'Inject Scenario'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
