import React from 'react';
import { StatusBadge } from '../common/StatusBadge';
import { WebSocketStatus } from '../../hooks/useWebSocket';
import {
  LayoutDashboard,
  Bus,
  Activity,
  Radio,
  GitBranch,
  Database,
  Sliders,
  ExternalLink,
} from 'lucide-react';

export type PageId =
  | 'overview'
  | 'fleet'
  | 'analytics'
  | 'drift'
  | 'lifecycle'
  | 'models'
  | 'simulator';

interface NavbarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
  wsStatus: WebSocketStatus;
  productionModel: string;
  productionVersion: string;
  backendHealthy: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activePage,
  onSelectPage,
  wsStatus,
  productionVersion,
  backendHealthy,
}) => {
  const isLive = wsStatus === 'OPEN';

  const navItems: { id: PageId; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={15} /> },
    { id: 'fleet', label: 'Live Fleet', icon: <Bus size={15} /> },
    { id: 'analytics', label: 'ETA Analytics', icon: <Activity size={15} /> },
    { id: 'drift', label: 'Drift Monitor', icon: <Radio size={15} /> },
    { id: 'lifecycle', label: 'MLOps', icon: <GitBranch size={15} /> },
    { id: 'models', label: 'Models', icon: <Database size={15} /> },
    { id: 'simulator', label: 'Simulator', icon: <Sliders size={15} /> },
  ];

  return (
    <header className="top-header">
      {/* Brand & Logo */}
      <div className="brand-section" onClick={() => onSelectPage('overview')}>
        <div className="brand-logo-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
            <line x1="4" y1="22" x2="4" y2="15" />
          </svg>
        </div>
        <div className="brand-text-wrapper">
          <div className="brand-title">
            TRANSITVISION <span>AI</span>
          </div>
          <div className="brand-subtitle">INTELLIGENT TRANSIT OPERATIONS</div>
        </div>
      </div>

      {/* Navigation Pills */}
      <nav className="header-nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-pill ${activePage === item.id ? 'active' : ''}`}
            onClick={() => onSelectPage(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Right Stats & Meta */}
      <div className="header-meta">
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="header-link"
          title="Open FastAPI Swagger Documentation"
        >
          <span>API Docs</span>
          <ExternalLink size={12} />
        </a>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <StatusBadge status={backendHealthy ? 'HEALTHY' : 'OFFLINE'} pulse={backendHealthy} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <StatusBadge status={isLive ? 'LIVE' : 'CONNECTION LOST'} pulse={isLive} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>PROD:</span>
          <StatusBadge status={productionVersion} />
        </div>
      </div>
    </header>
  );
};
