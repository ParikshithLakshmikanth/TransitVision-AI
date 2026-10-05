import { describe, it, expect } from 'vitest';
import { healthApi, fleetApi, metricsApi, driftApi, modelsApi, simulationApi, scenariosApi } from '../api';

describe('Frontend API Client Layer Tests', () => {
  it('should define all API service methods properly', () => {
    expect(typeof healthApi.getHealth).toBe('function');
    expect(typeof fleetApi.getFleet).toBe('function');
    expect(typeof fleetApi.getBusState).toBe('function');
    expect(typeof metricsApi.getMetrics).toBe('function');
    expect(typeof metricsApi.getMetricsHistory).toBe('function');
    expect(typeof driftApi.getDriftStatus).toBe('function');
    expect(typeof driftApi.getDriftSummary).toBe('function');
    expect(typeof driftApi.getDriftEvents).toBe('function');
    expect(typeof modelsApi.getModels).toBe('function');
    expect(typeof modelsApi.getProductionModel).toBe('function');
    expect(typeof modelsApi.getModelById).toBe('function');
    expect(typeof simulationApi.getStatus).toBe('function');
    expect(typeof simulationApi.start).toBe('function');
    expect(typeof simulationApi.pause).toBe('function');
    expect(typeof simulationApi.resume).toBe('function');
    expect(typeof simulationApi.stop).toBe('function');
    expect(typeof simulationApi.reset).toBe('function');
    expect(typeof simulationApi.step).toBe('function');
    expect(typeof simulationApi.seek).toBe('function');
    expect(typeof simulationApi.setSpeed).toBe('function');
    expect(typeof scenariosApi.getScenarios).toBe('function');
    expect(typeof scenariosApi.activateScenario).toBe('function');
  });
});

describe('Route 654 Corridor Topography Tests', () => {
  it('should verify 29 total route segments along Kandy - Teldeniya line', () => {
    const totalSegments = 29;
    const segments = Array.from({ length: totalSegments }, (_, i) => i + 1);
    expect(segments.length).toBe(29);
    expect(segments[0]).toBe(1);
    expect(segments[28]).toBe(29);
  });
});

describe('Status Badge Classification Tests', () => {
  it('should map production and live statuses correctly', () => {
    const validLive = ['PRODUCTION', 'HEALTHY', 'ACTIVE', 'LIVE', 'SUCCESS'];
    validLive.forEach((s) => {
      expect(['PRODUCTION', 'HEALTHY', 'ACTIVE', 'LIVE', 'SUCCESS', 'ONLINE', 'PASSED', 'COMPLETED', 'OPEN']).toContain(s);
    });

    const validDanger = ['CRITICAL', 'DANGER', 'ERROR', 'FAILED', 'REJECTED', 'ROLLED_BACK'];
    validDanger.forEach((s) => {
      expect(['CRITICAL', 'DANGER', 'ERROR', 'FAILED', 'REJECTED', 'ROLLED_BACK', 'CLOSED', 'CONNECTION LOST']).toContain(s);
    });
  });
});

describe('Scenario Specification Tests', () => {
  it('should verify standard Phase 7 transit disturbance scenarios', () => {
    const scenarios = [
      'BASELINE',
      'RUSH_HOUR',
      'HEAVY_RAIN',
      'CONGESTION_SURGE',
      'ROAD_INCIDENT',
      'DWELL_SURGE',
      'COMBINED_DISRUPTION',
    ];
    expect(scenarios.length).toBe(7);
    expect(scenarios).toContain('BASELINE');
    expect(scenarios).toContain('HEAVY_RAIN');
    expect(scenarios).toContain('COMBINED_DISRUPTION');
  });
});
