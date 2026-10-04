import React from 'react';

interface StatusBadgeProps {
  status: string;
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, pulse = false }) => {
  const norm = (status || 'UNKNOWN').toUpperCase();

  let badgeClass = 'badge-neutral';
  if (['PRODUCTION', 'HEALTHY', 'ACTIVE', 'LIVE', 'SUCCESS', 'ONLINE', 'PASSED', 'COMPLETED', 'OPEN'].includes(norm)) {
    badgeClass = 'badge-live';
  } else if (['WARNING', 'DEGRADED', 'PAUSED', 'MONITORING', 'SHADOW', 'CANARY_10', 'CANARY_50', 'VALIDATED'].includes(norm)) {
    badgeClass = 'badge-warning';
  } else if (['CRITICAL', 'DANGER', 'ERROR', 'FAILED', 'REJECTED', 'ROLLED_BACK', 'CLOSED', 'CONNECTION LOST'].includes(norm)) {
    badgeClass = 'badge-danger';
  } else if (['CANDIDATE', 'PROMOTION_CANDIDATE'].includes(norm)) {
    badgeClass = 'badge-purple';
  } else if (['INITIALIZED', 'RUNNING'].includes(norm)) {
    badgeClass = 'badge-cyan';
  }

  return (
    <span className={`badge ${badgeClass}`}>
      {pulse && <span className="pulse-dot" />}
      {status}
    </span>
  );
};
