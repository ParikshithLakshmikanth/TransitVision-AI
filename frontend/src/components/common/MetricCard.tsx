import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
  icon?: React.ReactNode;
  badge?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  subtext,
  icon,
  badge,
}) => {
  return (
    <div className="metric-card">
      <div className="metric-label">
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {icon}
          {label}
        </span>
        {badge}
      </div>
      <div className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      {subtext && <div className="metric-footer">{subtext}</div>}
    </div>
  );
};
