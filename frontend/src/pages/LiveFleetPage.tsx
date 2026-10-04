import React, { useEffect, useState } from 'react';
import { fleetApi } from '../api';
import { BusState, SystemEvent } from '../types/api';
import { CorridorMap } from '../components/fleet/CorridorMap';
import { FleetTable } from '../components/fleet/FleetTable';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import { Bus, RefreshCw, Radio } from 'lucide-react';

interface LiveFleetPageProps {
  lastWsEvent: SystemEvent | null;
}

export const LiveFleetPage: React.FC<LiveFleetPageProps> = ({ lastWsEvent }) => {
  const [buses, setBuses] = useState<BusState[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchFleet = async () => {
    try {
      setError(null);
      const res = await fleetApi.getFleet();
      setBuses(res.buses);
    } catch (err: any) {
      setError(err.message || 'Failed fetching fleet data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFleet();
    const interval = setInterval(fetchFleet, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (lastWsEvent && (lastWsEvent.event_type === 'PREDICTION' || lastWsEvent.event_type === 'OUTCOME_RESOLVED')) {
      fleetApi.getFleet().then((res) => setBuses(res.buses)).catch(() => {});
    }
  }, [lastWsEvent]);

  if (loading) return <LoadingState message="Loading live Route 654 fleet telemetry..." />;
  if (error) return <ErrorState title="FLEET TELEMETRY UNAVAILABLE" message={error} onRetry={fetchFleet} />;

  return (
    <div>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '28px' }}>
        <div className="page-header-block" style={{ marginBottom: 0 }}>
          <div className="page-eyebrow">
            <Radio size={14} />
            LIVE TRANSIT
          </div>
          <h1 className="page-main-title">Live Fleet</h1>
          <p className="page-description">
            Monitor active Route 654 vehicles, segment transitions, and real-time ETA predictions along the Kandy – Teldeniya line.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={fetchFleet}>
          <RefreshCw size={14} />
          Refresh Fleet
        </button>
      </div>

      <div style={{ marginBottom: '24px' }}>
        <CorridorMap buses={buses} />
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Bus size={18} style={{ color: 'var(--color-primary)' }} />
            Active Fleet Register ({buses.length} Vehicles Online)
          </div>
        </div>
        <FleetTable buses={buses} />
      </div>
    </div>
  );
};
