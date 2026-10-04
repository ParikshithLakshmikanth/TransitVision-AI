import React from 'react';
import { BusState } from '../../types/api';
import { StatusBadge } from '../common/StatusBadge';

interface FleetTableProps {
  buses: BusState[];
}

export const FleetTable: React.FC<FleetTableProps> = ({ buses }) => {
  if (!buses || buses.length === 0) {
    return (
      <div style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
        Waiting for active Route 654 bus telemetry...
      </div>
    );
  }

  const formatTime = (ts: string) => {
    try {
      return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch {
      return ts;
    }
  };

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Bus / Device</th>
            <th>Trip ID</th>
            <th>Direction</th>
            <th>Segment</th>
            <th>Timestamp</th>
            <th>Predicted ETA</th>
            <th>Actual Travel</th>
            <th>Error (s)</th>
            <th>Model</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {buses.map((bus) => (
            <tr key={bus.deviceid}>
              <td className="mono" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                {bus.deviceid}
              </td>
              <td className="mono" style={{ fontSize: '0.78rem' }}>{bus.trip_id}</td>
              <td>{bus.direction === 1 ? 'Outbound →' : '← Inbound'}</td>
              <td className="mono" style={{ color: 'var(--color-primary)', fontWeight: 600 }}>
                Segment {bus.current_segment}
              </td>
              <td className="mono" style={{ fontSize: '0.78rem' }}>
                {formatTime(bus.last_event_timestamp_utc)}
              </td>
              <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                {bus.predicted_eta_sec !== null && bus.predicted_eta_sec !== undefined
                  ? `${bus.predicted_eta_sec.toFixed(1)}s`
                  : '—'}
              </td>
              <td className="mono">
                {bus.actual_eta_sec !== null && bus.actual_eta_sec !== undefined
                  ? `${bus.actual_eta_sec.toFixed(1)}s`
                  : '—'}
              </td>
              <td className="mono">
                {bus.prediction_error_sec !== null && bus.prediction_error_sec !== undefined ? (
                  <span style={{ color: Math.abs(bus.prediction_error_sec) > 30 ? 'var(--color-warning)' : 'var(--color-live)', fontWeight: 600 }}>
                    {bus.prediction_error_sec > 0 ? `+${bus.prediction_error_sec.toFixed(1)}s` : `${bus.prediction_error_sec.toFixed(1)}s`}
                  </span>
                ) : '—'}
              </td>
              <td>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  {bus.model_version || 'v1.0.0'}
                </span>
              </td>
              <td>
                <StatusBadge status={bus.status} pulse={bus.status === 'ACTIVE'} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
