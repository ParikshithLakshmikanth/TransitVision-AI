import React from 'react';
import { StatusBadge } from '../common/StatusBadge';
import { WebSocketStatus } from '../../hooks/useWebSocket';

interface TopbarProps {
  pageTitle: string;
  wsStatus: WebSocketStatus;
  productionModel: string;
  productionVersion: string;
  scenarioName?: string;
  onReconnect?: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({
  pageTitle,
  wsStatus,
  productionModel,
  productionVersion,
  scenarioName,
}) => {
  const isLive = wsStatus === 'OPEN';

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1 className="page-title">{pageTitle}</h1>
      </div>

      <div className="topbar-right">
        {scenarioName && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SCENARIO:</span>
            <StatusBadge status={scenarioName} />
          </div>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MODEL:</span>
          <StatusBadge status={`${productionModel} (${productionVersion})`} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <StatusBadge status={isLive ? 'LIVE' : 'CONNECTION LOST'} pulse={isLive} />
        </div>
      </div>
    </header>
  );
};
