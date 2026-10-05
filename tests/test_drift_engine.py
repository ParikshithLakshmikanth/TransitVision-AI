"""Integration tests for Phase 8 Unified DriftEngine and Pipeline."""
import hashlib
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import PROCESSED_DATA_DIR, METADATA_DIR
from simulator.event_models import SimulationConfig
from simulator.transit_simulator import DigitalTransitSimulator
from monitoring.drift_models import DriftConfig, DriftType, DriftSeverity, DriftEvent
from monitoring.drift_engine import DriftEngine
from monitoring.drift_detectors import ReferenceProfiler


@pytest.fixture(scope="module")
def reference_profile():
    """Builds reference profile once for integration tests."""
    profiler = ReferenceProfiler()
    return profiler.build_profile()


def test_drift_engine_baseline_no_false_critical_alarms(reference_profile):
    """Verify Baseline replay does not trigger false CRITICAL data or performance drift."""
    config = DriftConfig(window_size=300, step_size=150, min_window_size=100)
    engine = DriftEngine(config=config, reference_profile=reference_profile)

    sim = DigitalTransitSimulator(config=SimulationConfig(scenario_id="baseline", auto_predict=True))
    sim.start()
    
    # Run 1000 records
    step_results = sim.step(1000)
    for telemetry, pred, outcome, eval_event in step_results:
        cur_idx = int(telemetry.event_id.split("_")[-1])
        engine.process_record(
            record_idx=cur_idx,
            features=telemetry.features,
            predicted_eta=pred.predicted_eta_sec,
            actual_eta=outcome.actual_eta_sec,
            timestamp_utc=telemetry.event_timestamp_utc
        )

    engine.flush()
    summary = engine.get_summary()

    # Baseline should not trigger CRITICAL drift
    critical_alerts = [e for e in engine.events_history if e.severity == DriftSeverity.CRITICAL]
    assert len(critical_alerts) == 0
    assert summary["windows_evaluated"] > 0


def test_drift_engine_detects_heavy_rain_covariate_drift(reference_profile):
    """Verify Heavy Rain scenario triggers DATA_DRIFT on meteorological features."""
    config = DriftConfig(window_size=300, step_size=150, min_window_size=100)
    engine = DriftEngine(config=config, reference_profile=reference_profile)

    sim = DigitalTransitSimulator(config=SimulationConfig(scenario_id="heavy_rain", scenario_intensity=0.85, auto_predict=True))
    engine.set_scenario_context("HEAVY_RAIN", "Heavy Rain", synthetic_disturbance=True, intensity=0.85)
    sim.start()

    step_results = sim.step(600)
    for telemetry, pred, outcome, eval_event in step_results:
        cur_idx = int(telemetry.event_id.split("_")[-1])
        engine.process_record(
            record_idx=cur_idx,
            features=telemetry.features,
            predicted_eta=pred.predicted_eta_sec,
            actual_eta=outcome.actual_eta_sec,
            timestamp_utc=telemetry.event_timestamp_utc
        )

    engine.flush()
    summary = engine.get_summary()

    assert summary["data_drift_alerts"] > 0
    data_alerts = [e for e in engine.events_history if e.drift_type == DriftType.DATA_DRIFT]
    affected_feats = []
    for a in data_alerts:
        affected_feats.extend(a.affected_features)

    assert "precipitation" in affected_feats or "rain" in affected_feats or "relative_humidity_2m" in affected_feats


def test_drift_engine_detection_latency_tracking(reference_profile):
    """Verify first detection index and timestamp are accurately recorded."""
    config = DriftConfig(window_size=200, step_size=100, min_window_size=50)
    engine = DriftEngine(config=config, reference_profile=reference_profile)

    sim = DigitalTransitSimulator(config=SimulationConfig(scenario_id="congestion", scenario_intensity=0.80, auto_predict=True))
    engine.set_scenario_context("CONGESTION_SURGE", "Congestion Surge", synthetic_disturbance=True, intensity=0.80)
    sim.start()

    step_results = sim.step(500)
    for telemetry, pred, outcome, eval_event in step_results:
        cur_idx = int(telemetry.event_id.split("_")[-1])
        engine.process_record(
            record_idx=cur_idx,
            features=telemetry.features,
            predicted_eta=pred.predicted_eta_sec,
            actual_eta=outcome.actual_eta_sec,
            timestamp_utc=telemetry.event_timestamp_utc
        )

    summary = engine.get_summary()
    assert summary["first_detection_index"] is not None
    assert summary["first_detection_index"] <= 500
    assert summary["first_detection_type"] is not None


def test_drift_event_serialization(tmp_path, reference_profile):
    """Verify DriftEvents serialize to valid JSON schema."""
    engine = DriftEngine(output_events_path=tmp_path / "test_drift_events.json", reference_profile=reference_profile)
    evt = DriftEvent(
        event_id="TEST_EVT_001",
        timestamp_utc="2022-10-01T08:00:00Z",
        drift_type=DriftType.DATA_DRIFT,
        severity=DriftSeverity.HIGH,
        detector="PSI_KS_HYBRID",
        scenario_id="HEAVY_RAIN",
        scenario_name="Heavy Rain",
        synthetic_disturbance=True,
        disturbance_intensity=0.8,
        source_type="SYNTHETIC_SIMULATION",
        model_id="eta_model_v1",
        model_version="v1.0.0",
        window_start_idx=1,
        window_end_idx=500,
        sample_count=500,
        statistic=0.45,
        threshold=0.20,
        p_value=0.0001,
        baseline_metric=0.0,
        current_metric=0.45,
        affected_features=["precipitation", "rain"],
        evidence={"precipitation": 0.45}
    )
    engine._record_alert(evt)
    engine.save_events()

    saved_file = tmp_path / "test_drift_events.json"
    assert saved_file.exists()
    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["event_id"] == "TEST_EVT_001"
    assert data[0]["drift_type"] == "DATA_DRIFT"


def test_source_parquet_immutability_during_drift_monitoring(reference_profile):
    """Verify source parquet hash remains identical before and after drift monitoring."""
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    with open(stream_path, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest()

    engine = DriftEngine(reference_profile=reference_profile)
    sim = DigitalTransitSimulator(config=SimulationConfig(scenario_id="combined", scenario_intensity=0.90, auto_predict=True))
    sim.start()
    step_results = sim.step(50)
    for telemetry, pred, outcome, eval_event in step_results:
        cur_idx = int(telemetry.event_id.split("_")[-1])
        engine.process_record(
            record_idx=cur_idx,
            features=telemetry.features,
            predicted_eta=pred.predicted_eta_sec,
            actual_eta=outcome.actual_eta_sec,
            timestamp_utc=telemetry.event_timestamp_utc
        )

    with open(stream_path, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest()

    assert hash_before == hash_after
