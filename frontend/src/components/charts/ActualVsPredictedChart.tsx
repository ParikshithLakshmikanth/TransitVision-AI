import React from 'react';
import { PredictionItem } from '../../types/api';

interface ActualVsPredictedChartProps {
  predictions: PredictionItem[];
}

export const ActualVsPredictedChart: React.FC<ActualVsPredictedChartProps> = ({ predictions }) => {
  const validPoints = predictions.filter(
    (p) => p.actual_eta_sec !== null && p.actual_eta_sec !== undefined && p.actual_eta_sec > 0
  );

  if (validPoints.length === 0) {
    return (
      <div style={{ height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        Waiting for resolved ground-truth predictions...
      </div>
    );
  }

  const width = 520;
  const height = 250;
  const padding = 42;

  const maxVal = Math.max(
    ...validPoints.map((p) => Math.max(p.predicted_eta_sec, p.actual_eta_sec || 0)),
    100
  );

  const scaleX = (val: number) => padding + (val / maxVal) * (width - 2 * padding);
  const scaleY = (val: number) => height - padding - (val / maxVal) * (height - 2 * padding);

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{
          width: '100%',
          height: 'auto',
          background: 'var(--bg-secondary)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)',
        }}
      >
        {/* Grid Background Lines */}
        <line x1={padding} y1={scaleY(maxVal * 0.5)} x2={width - padding} y2={scaleY(maxVal * 0.5)} stroke="#e2e8f0" strokeDasharray="3 3" />
        <line x1={scaleX(maxVal * 0.5)} y1={padding} x2={scaleX(maxVal * 0.5)} y2={height - padding} stroke="#e2e8f0" strokeDasharray="3 3" />

        {/* Main Axes */}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#94a3b8" strokeWidth="1.5" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#94a3b8" strokeWidth="1.5" />

        {/* 45-degree Reference Parity Line */}
        <line
          x1={scaleX(0)}
          y1={scaleY(0)}
          x2={scaleX(maxVal)}
          y2={scaleY(maxVal)}
          stroke="#0057b8"
          strokeWidth="1.5"
          strokeDasharray="4 4"
          opacity="0.4"
        />

        {/* Data Points */}
        {validPoints.map((p) => {
          const cx = scaleX(p.actual_eta_sec || 0);
          const cy = scaleY(p.predicted_eta_sec);
          const err = Math.abs((p.actual_eta_sec || 0) - p.predicted_eta_sec);
          const isWarning = err > 30;
          const color = isWarning ? 'var(--color-warning)' : 'var(--color-primary)';

          return (
            <circle
              key={p.prediction_id}
              cx={cx}
              cy={cy}
              r="3.5"
              fill={color}
              opacity="0.85"
              stroke="#ffffff"
              strokeWidth="0.5"
            >
              <title>{`Actual: ${p.actual_eta_sec}s | Pred: ${p.predicted_eta_sec.toFixed(1)}s | Error: ${p.signed_error_sec}s`}</title>
            </circle>
          );
        })}

        {/* Axis Labels */}
        <text x={width / 2} y={height - 10} fill="var(--text-secondary)" fontSize="11" fontWeight="500" textAnchor="middle">
          Actual Observed Travel Time (seconds)
        </text>
        <text
          x={14}
          y={height / 2}
          fill="var(--text-secondary)"
          fontSize="11"
          fontWeight="500"
          textAnchor="middle"
          transform={`rotate(-90 14 ${height / 2})`}
        >
          Predicted ETA (s)
        </text>
      </svg>
    </div>
  );
};
