import React from 'react';
import { SystemEvent } from '../../types/api';
import { StatusBadge } from '../common/StatusBadge';
import { Radio, AlertTriangle, ArrowRightLeft, CheckCircle2, RotateCcw, Zap } from 'lucide-react';

interface EventFeedProps {
  events: SystemEvent[];
  maxHeight?: string;
}

export const EventFeed: React.FC<EventFeedProps> = ({ events, maxHeight = '360px' }) => {
  if (!events || events.length === 0) {
    return (
      <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        Waiting for live WebSocket events...
      </div>
    );
  }

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'DRIFT_DETECTED':
        return <Radio size={14} style={{ color: 'var(--color-warning)' }} />;
      case 'ROLLBACK_TRIGGERED':
      case 'MODEL_ROLLED_BACK':
      case 'REGRESSION_DETECTED':
        return <RotateCcw size={14} style={{ color: 'var(--color-danger)' }} />;
      case 'MODEL_PROMOTED':
      case 'CANDIDATE_VALIDATED':
        return <CheckCircle2 size={14} style={{ color: 'var(--color-live)' }} />;
      case 'OUTCOME_RESOLVED':
        return <ArrowRightLeft size={14} style={{ color: 'var(--color-primary)' }} />;
      case 'PREDICTION':
        return <Zap size={14} style={{ color: 'var(--color-primary)' }} />;
      default:
        return <AlertTriangle size={14} style={{ color: 'var(--text-secondary)' }} />;
    }
  };

  const formatTime = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="event-feed-list" style={{ maxHeight }}>
      {events.slice().reverse().map((evt) => (
        <div key={evt.event_id} className="event-feed-item">
          <div className="event-feed-time">{formatTime(evt.timestamp_utc)}</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {getEventIcon(evt.event_type)}
            <StatusBadge status={evt.event_type} />
          </div>
          <div style={{ flex: 1, color: 'var(--text-secondary)' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)', marginRight: '6px' }}>
              [{evt.component}]
            </span>
            {evt.payload.deviceid && `Bus ${evt.payload.deviceid} · `}
            {evt.payload.predicted_eta_sec !== undefined && `Pred ETA: ${evt.payload.predicted_eta_sec}s `}
            {evt.payload.actual_eta_sec !== undefined && `| Actual: ${evt.payload.actual_eta_sec}s `}
            {evt.payload.detector && `Detector: ${evt.payload.detector} `}
            {evt.payload.message || ''}
          </div>
        </div>
      ))}
    </div>
  );
};
