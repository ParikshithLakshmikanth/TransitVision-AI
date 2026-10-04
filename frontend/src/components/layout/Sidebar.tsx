import React from 'react';
import {
  LayoutDashboard,
  Bus,
  Activity,
  GitBranch,
  Layers,
  Database,
  Sliders,
  Radio,
} from 'lucide-react';

export type PageId =
  | 'overview'
  | 'fleet'
  | 'analytics'
  | 'drift'
  | 'lifecycle'
  | 'models'
  | 'simulator';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activePage, onSelectPage }) => {
  const navItems: { id: PageId; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={18} /> },
    { id: 'fleet', label: 'Live Fleet', icon: <Bus size={18} /> },
    { id: 'analytics', label: 'ETA Analytics', icon: <Activity size={18} /> },
    { id: 'drift', label: 'Drift Monitor', icon: <Radio size={18} /> },
    { id: 'lifecycle', label: 'MLOps Lifecycle', icon: <GitBranch size={18} /> },
    { id: 'models', label: 'Model Registry', icon: <Database size={18} /> },
    { id: 'simulator', label: 'Scenario Simulator', icon: <Sliders size={18} /> },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-title">
          <span>TRANSITVISION</span> AI
        </div>
        <div className="brand-subtitle">Route 654 Corridor Operations</div>
      </div>

      <ul className="nav-list">
        {navItems.map((item) => (
          <li
            key={item.id}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => onSelectPage(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </li>
        ))}
      </ul>

      <div className="sidebar-footer">
        <div>Kandy – Teldeniya Line</div>
        <div style={{ color: 'var(--text-secondary)', marginTop: '2px' }}>v1.0.0 Production</div>
      </div>
    </aside>
  );
};
