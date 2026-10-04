import React from 'react';
import { PredictionItem } from '../../types/api';

interface ErrorDistributionChartProps {
  predictions: PredictionItem[];
}

export const ErrorDistributionChart: React.FC<ErrorDistributionChartProps> = ({ predictions }) => {
  const errors = predictions
    .map((p) => p.absolute_error_sec)
    .filter((e): e is number => e !== null && e !== undefined);

  if (errors.length === 0) {
    return (
      <div style={{ height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        Waiting for prediction error samples...
      </div>
    );
  }

  // Group errors into 6 buckets: 0-10s, 10-20s, 20-40s, 40-60s, 60-100s, >100s
  const buckets = [
    { label: '0–10s', count: 0, min: 0, max: 10 },
    { label: '10–20s', count: 0, min: 10, max: 20 },
    { label: '20–40s', count: 0, min: 20, max: 40 },
    { label: '40–60s', count: 0, min: 40, max: 60 },
    { label: '60–100s', count: 0, min: 60, max: 100 },
    { label: '>100s', count: 0, min: 100, max: Infinity },
  ];

  errors.forEach((err) => {
    for (const b of buckets) {
      if (err >= b.min && err < b.max) {
        b.count++;
        break;
      }
    }
  });

  const maxCount = Math.max(...buckets.map((b) => b.count), 1);
  const width = 520;
  const height = 250;
  const padding = 38;
  const barWidth = 48;

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
        {/* Baseline Axis */}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#94a3b8" strokeWidth="1.5" />

        {buckets.map((b, idx) => {
          const x = padding + idx * ((width - 2 * padding) / buckets.length) + 12;
          const barHeight = (b.count / maxCount) * (height - 2 * padding - 20);
          const y = height - padding - barHeight;

          return (
            <g key={b.label}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill="var(--color-primary)"
                opacity="0.9"
                rx="4"
              />
              {b.count > 0 && (
                <text
                  x={x + barWidth / 2}
                  y={y - 6}
                  fill="var(--text-primary)"
                  fontSize="11"
                  fontWeight="600"
                  textAnchor="middle"
                >
                  {b.count}
                </text>
              )}
              <text
                x={x + barWidth / 2}
                y={height - padding + 16}
                fill="var(--text-secondary)"
                fontSize="11"
                fontWeight="500"
                textAnchor="middle"
              >
                {b.label}
              </text>
            </g>
          );
        })}

        <text x={width / 2} y={height - 4} fill="var(--text-secondary)" fontSize="11" fontWeight="500" textAnchor="middle">
          Absolute Error Range (seconds)
        </text>
      </svg>
    </div>
  );
};
