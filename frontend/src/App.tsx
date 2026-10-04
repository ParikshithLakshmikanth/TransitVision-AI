import React, { useState, useEffect } from 'react';
import { Navbar, PageId } from './components/layout/Navbar';
import { useWebSocket } from './hooks/useWebSocket';
import { healthApi, simulationApi } from './api';
import { HealthResponse, SimulationStatusResponse } from './types/api';

// Pages
import { OverviewPage } from './pages/OverviewPage';
import { LiveFleetPage } from './pages/LiveFleetPage';
import { EtaAnalyticsPage } from './pages/EtaAnalyticsPage';
import { DriftMonitorPage } from './pages/DriftMonitorPage';
import { LifecyclePage } from './pages/LifecyclePage';
import { ModelRegistryPage } from './pages/ModelRegistryPage';
import { ScenarioSimulatorPage } from './pages/ScenarioSimulatorPage';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [, setSimStatus] = useState<SimulationStatusResponse | null>(null);

  const { status: wsStatus, lastEvent } = useWebSocket(200);

  const fetchGlobalState = async () => {
    try {
      const [h, s] = await Promise.all([
        healthApi.getHealth().catch(() => null),
        simulationApi.getStatus().catch(() => null),
      ]);
      setHealth(h);
      setSimStatus(s);
    } catch {
      // Handled via offline status in Navbar
    }
  };

  useEffect(() => {
    fetchGlobalState();
    const interval = setInterval(fetchGlobalState, 5000);
    return () => clearInterval(interval);
  }, []);

  const renderActivePage = () => {
    switch (activePage) {
      case 'overview':
        return <OverviewPage lastWsEvent={lastEvent} />;
      case 'fleet':
        return <LiveFleetPage lastWsEvent={lastEvent} />;
      case 'analytics':
        return <EtaAnalyticsPage />;
      case 'drift':
        return <DriftMonitorPage />;
      case 'lifecycle':
        return <LifecyclePage />;
      case 'models':
        return <ModelRegistryPage />;
      case 'simulator':
        return <ScenarioSimulatorPage />;
      default:
        return <OverviewPage lastWsEvent={lastEvent} />;
    }
  };

  return (
    <div className="app-container">
      <Navbar
        activePage={activePage}
        onSelectPage={setActivePage}
        wsStatus={wsStatus}
        productionModel={health?.production_model_id || 'model_lightgbm_v1'}
        productionVersion={health?.production_model_version || 'v1.0.0'}
        backendHealthy={!!health}
      />

      <main className="content-area">
        {renderActivePage()}
      </main>
    </div>
  );
};

export default App;
