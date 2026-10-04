import React, { useEffect, useState } from 'react';
import { lifecycleApi } from '../api';
import { LifecycleStateResponse, RollbackStatusResponse } from '../types/api';
import { StatusBadge } from '../components/common/StatusBadge';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import {
  GitBranch,
  RotateCcw,
  ArrowRight,
  Cpu,
  Compass,
} from 'lucide-react';

export const LifecyclePage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [lifecycle, setLifecycle] = useState<LifecycleStateResponse | null>(null);
  const [rollback, setRollback] = useState<RollbackStatusResponse | null>(null);
  const [, setHistory] = useState<Record<string, any>>({});

  const fetchLifecycle = async () => {
    try {
      setError(null);
      const [lc, rb, hist] = await Promise.all([
        lifecycleApi.getLifecycleState(),
        lifecycleApi.getRollbackStatus(),
        lifecycleApi.getRetrainingHistory(),
      ]);
      setLifecycle(lc);
      setRollback(rb);
      setHistory(hist);
    } catch (err: any) {
      setError(err.message || 'Failed connecting to MLOps Lifecycle Controller');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLifecycle();
    const interval = setInterval(fetchLifecycle, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <LoadingState message="Loading Autonomous MLOps Pipeline State..." />;
  if (error) return <ErrorState title="LIFECYCLE DATA UNAVAILABLE" message={error} onRetry={fetchLifecycle} />;

  // 11 Pipeline Stages
  const pipelineStages = [
    { id: 'DRIFT_DETECTED', label: '1. Drift Sentinel', status: 'MONITORED' },
    { id: 'RETRAINING_TRIGGER', label: '2. Trigger Policy', status: 'EVALUATED' },
    { id: 'CANDIDATE_TRAINING', label: '3. Data & Train', status: 'READY' },
    { id: 'VALIDATION', label: '4. Dual Holdout', status: 'GATED' },
    { id: 'SHADOW', label: '5. Shadow Eval', status: 'PASSED' },
    { id: 'CANARY_10', label: '6. Canary 10%', status: 'PASSED' },
    { id: 'CANARY_50', label: '7. Canary 50%', status: 'PASSED' },
    { id: 'PROMOTION', label: '8. Hot Swap', status: 'VERIFIED' },
    { id: 'PRODUCTION', label: '9. Production', status: 'ACTIVE', isCurrent: true },
    { id: 'WATCHDOG', label: '10. Watchdog', status: 'GUARDED' },
    { id: 'ROLLBACK', label: '11. Rollback', status: 'STANDBY' },
  ];

  return (
    <div>
      {/* Page Header */}
      <div className="page-header-block">
        <div className="page-eyebrow">
          <Compass size={14} />
          AUTONOMOUS MLOPS
        </div>
        <h1 className="page-main-title">MLOps Lifecycle</h1>
        <p className="page-description">
          Deterministic self-healing pipeline: Continuous drift surveillance, dual-holdout validation gates, canary traffic routing, atomic hot-swap, and automated watchdog rollback.
        </p>
      </div>

      {/* Hero Pipeline Stage Graph */}
      <div className="card" style={{ marginBottom: '24px', overflowX: 'auto' }}>
        <div className="card-header">
          <div className="card-title">
            <GitBranch size={18} style={{ color: 'var(--color-primary)' }} />
            11-Stage Self-Healing Closed-Loop Workflow
          </div>
          <StatusBadge status="ACTIVE PRODUCTION CHAMPION" pulse />
        </div>

        <div className="pipeline-container">
          {pipelineStages.map((st, idx) => (
            <React.Fragment key={st.id}>
              <div className={`pipeline-node ${st.isCurrent ? 'active-stage' : 'passed'}`}>
                <div className="pipeline-node-title">{st.label}</div>
                <div className="pipeline-node-status">
                  <StatusBadge status={st.status} />
                </div>
              </div>
              {idx < pipelineStages.length - 1 && (
                <ArrowRight size={16} className="pipeline-arrow" />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Grid: Production State & Watchdog Rollback Policy */}
      <div className="grid-cards-2">
        {/* Active Production Champion */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Cpu size={18} style={{ color: 'var(--color-primary)' }} />
              Active Production Champion
            </div>
            <StatusBadge status="PRODUCTION" pulse />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Active Model ID:</span>
              <span className="mono" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{lifecycle?.production_model_id}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Model Version:</span>
              <span className="mono" style={{ color: 'var(--color-primary)', fontWeight: 600 }}>{lifecycle?.production_model_version}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Algorithm Architecture:</span>
              <span className="mono" style={{ color: 'var(--text-primary)' }}>{lifecycle?.production_algorithm}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Serving Invariant:</span>
              <span style={{ color: 'var(--color-live)', fontWeight: 600, fontSize: '0.85rem' }}>EXACTLY ONE PRODUCTION MODEL</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Feature Preprocessor Contract:</span>
              <span className="mono" style={{ color: 'var(--text-secondary)' }}>29 Validated Tabular Columns</span>
            </div>
          </div>
        </div>

        {/* Watchdog Rollback Policy */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <RotateCcw size={18} style={{ color: 'var(--color-danger)' }} />
              Post-Promotion Watchdog Sentinel Policy
            </div>
            <StatusBadge status="WATCHDOG ACTIVE" />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Watchdog Window:</span>
              <span className="mono" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{rollback?.watchdog_window_size} streaming samples</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Max Degradation Threshold:</span>
              <span className="mono" style={{ color: 'var(--color-warning)', fontWeight: 600 }}>MAE &gt; {rollback?.max_degradation_ratio}x baseline</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Pre-Promotion Baseline MAE:</span>
              <span className="mono" style={{ color: 'var(--text-primary)' }}>{rollback?.pre_promotion_baseline_mae_sec} seconds</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Latency Threshold:</span>
              <span className="mono" style={{ color: 'var(--text-primary)' }}>&le; {rollback?.max_latency_ms} ms</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Runtime Exceptions Allowed:</span>
              <span className="mono" style={{ color: 'var(--color-live)', fontWeight: 600 }}>0 (Zero Tolerance)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Retraining Benchmark Experiment Audit Log */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Phase 9 Retraining & Lifecycle Benchmark Log</div>
            <div className="card-subtitle">Formal audit trail of verified experimental runs</div>
          </div>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Scenario / Purpose</th>
                <th>Observed Outcome</th>
                <th>Final Model Action</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="mono" style={{ fontWeight: 700, color: 'var(--color-primary)' }}>EXPERIMENT A</td>
                <td>Nominal Kandy Stream Baseline Control</td>
                <td>MAE 38.77s | 0 Performance Drift | 0 Retraining Triggers</td>
                <td>v1.0.0 Maintained in PRODUCTION</td>
                <td><StatusBadge status="PASS" /></td>
              </tr>
              <tr>
                <td className="mono" style={{ fontWeight: 700, color: 'var(--color-primary)' }}>EXPERIMENT B</td>
                <td>Mild Rush Hour Corridor Disturbance (0.75)</td>
                <td>MAE 54.25s (+39.9%) | Retraining Triggered</td>
                <td>Downstream Control Isolated</td>
                <td><StatusBadge status="PASS" /></td>
              </tr>
              <tr>
                <td className="mono" style={{ fontWeight: 700, color: 'var(--color-primary)' }}>EXPERIMENT C</td>
                <td>Severe Corridor Disruption & Adaptation</td>
                <td>Passed Holdouts, Shadow, Canary 10%, Canary 50%</td>
                <td>v1.1.0 Promoted to PRODUCTION</td>
                <td><StatusBadge status="PASS" /></td>
              </tr>
              <tr>
                <td className="mono" style={{ fontWeight: 700, color: 'var(--color-primary)' }}>EXPERIMENT D</td>
                <td>Bad Underfitting Candidate Safety Rejection</td>
                <td>Failed Gate A (MAE 96s) & Gate B (MAE 98s)</td>
                <td>Candidate Quarantined to REJECTED</td>
                <td><StatusBadge status="PASS" /></td>
              </tr>
              <tr>
                <td className="mono" style={{ fontWeight: 700, color: 'var(--color-primary)' }}>EXPERIMENT E</td>
                <td>State Machine Legal / Illegal Transition Audit</td>
                <td>14/14 Legal Accepted | 11/11 Illegal Transitions Blocked</td>
                <td>Registry State Machine Certified</td>
                <td><StatusBadge status="PASS" /></td>
              </tr>
              <tr>
                <td className="mono" style={{ fontWeight: 700, color: 'var(--color-primary)' }}>EXPERIMENT F</td>
                <td>Post-Promotion Heavy Rain Degradation (1.97x)</td>
                <td>Watchdog Breached 1.25x MAE Threshold at 500 Samples</td>
                <td>v1.0.0 Restored to PRODUCTION (v1.1.0 ROLLED_BACK)</td>
                <td><StatusBadge status="PASS" /></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
