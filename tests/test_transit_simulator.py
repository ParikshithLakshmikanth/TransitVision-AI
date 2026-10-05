"""Comprehensive unit and integration tests for Digital Transit Simulator (Phase 6)."""
import hashlib
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import PROCESSED_DATA_DIR, SIMULATION_DATA_DIR, ROOT_DIR
from simulator.event_models import SimulationConfig, TelemetryEvent, BusState
from simulator.transit_simulator import DigitalTransitSimulator
from ml.training.leak_guard import LeakageGuardError


@pytest.fixture
def stream_file_hash():
    """Calculates SHA256 of kandy_eta_stream.parquet."""
    path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_stream_loads_successfully():
    """Verify simulator loads genuine stream dataset."""
    sim = DigitalTransitSimulator()
    assert sim.total_records == 30102
    assert sim.status == "INITIALIZED"
    assert sim.cursor == 0


def test_source_immutability(stream_file_hash):
    """Verify simulator leaves source parquet file strictly unmodified (provenance test)."""
    sim = DigitalTransitSimulator()
    sim.start()
    sim.step(10)
    sim.stop()

    assert sim.verify_source_immutability() is True
    # Re-calculate hash directly
    path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    with open(path, "rb") as f:
        new_hash = hashlib.sha256(f.read()).hexdigest()
    assert new_hash == stream_file_hash


def test_chronological_ordering():
    """Verify simulator processes events in strictly increasing timestamp order."""
    sim = DigitalTransitSimulator()
    sim.start()
    results = sim.step(25)
    
    timestamps = [pd.to_datetime(t[0].event_timestamp_utc) for t in results]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], "Stream records must be monotonically non-decreasing in time"


def test_replay_cursor_advances():
    """Verify step advances cursor by exactly requested number of records."""
    sim = DigitalTransitSimulator()
    sim.start()
    sim.step(5)
    assert sim.cursor == 5
    assert sim.records_emitted == 5

    sim.step(10)
    assert sim.cursor == 15
    assert sim.records_emitted == 15


def test_pause_and_resume():
    """Verify pause freezes progression and resume continues from same cursor."""
    sim = DigitalTransitSimulator()
    sim.start()
    sim.step(3)
    assert sim.cursor == 3

    sim.pause()
    assert sim.status == "PAUSED"
    # While paused, step should not advance
    res_paused = sim.step(5)
    assert len(res_paused) == 0
    assert sim.cursor == 3

    sim.resume()
    assert sim.status == "RUNNING"
    sim.step(4)
    assert sim.cursor == 7


def test_reset():
    """Verify reset returns simulator to initial index 0 and clears metrics."""
    sim = DigitalTransitSimulator()
    sim.start()
    sim.step(10)
    assert sim.cursor == 10
    assert sim.records_emitted == 10

    sim.reset()
    assert sim.cursor == 0
    assert sim.records_emitted == 0
    assert len(sim.active_buses) == 0
    assert sim.status == "INITIALIZED"


def test_seek():
    """Verify seeking to a valid index positions cursor accurately."""
    sim = DigitalTransitSimulator()
    sim.seek(100)
    assert sim.cursor == 100

    # Test out of bounds seek
    with pytest.raises(IndexError):
        sim.seek(999999)


def test_replay_speed_configuration():
    """Verify replay speed can be dynamically updated."""
    sim = DigitalTransitSimulator(config=SimulationConfig(replay_speed=50.0))
    assert sim.replay_speed == 50.0

    sim.set_speed(250.0)
    assert sim.replay_speed == 250.0


def test_filtering():
    """Verify simulator filters stream dataset by trip_id, deviceid, direction, segment."""
    # Filter by deviceid '1166'
    config = SimulationConfig(filter_deviceid="1166")
    sim = DigitalTransitSimulator(config=config)
    assert sim.total_records > 0
    assert (sim.stream_df["deviceid"] == "1166").all()

    # Filter by direction 1
    config_dir = SimulationConfig(filter_direction=1)
    sim_dir = DigitalTransitSimulator(config=config_dir)
    assert (sim_dir.stream_df["direction"] == 1).all()


def test_prediction_integration():
    """Verify production model is invoked and returns valid ETA predictions."""
    sim = DigitalTransitSimulator()
    sim.start()
    res = sim.step(5)
    assert len(res) == 5

    for telemetry, pred, outcome, eval_event in res:
        assert pred is not None
        assert pred.model_version.startswith("v1.")
        assert pred.algorithm == "LightGBMRegressor"
        assert 3.0 <= pred.predicted_eta_sec <= 3600.0
        assert pred.latency_ms >= 0.0


def test_zero_leakage_to_prediction():
    """Verify features passed to model inference contain ZERO target ground truth."""
    sim = DigitalTransitSimulator()
    sim.start()
    res = sim.step(3)

    for telemetry, pred, outcome, eval_event in res:
        # Telemetry event features must not contain target
        assert "eta_to_next_stop_sec" not in telemetry.features
        assert "run_time_in_seconds" not in telemetry.features
        assert "future_segment_run_time" not in telemetry.features


def test_outcome_and_evaluation_resolution():
    """Verify outcomes resolve ground truth and compute exact signed/absolute errors."""
    sim = DigitalTransitSimulator()
    sim.start()
    res = sim.step(5)

    for telemetry, pred, outcome, eval_event in res:
        assert outcome is not None
        assert eval_event is not None
        assert outcome.actual_eta_sec == telemetry._ground_truth_eta_sec
        # Mathematical verification of signed and absolute error
        expected_signed = round(outcome.actual_eta_sec - pred.predicted_eta_sec, 2)
        assert eval_event.signed_error_sec == expected_signed
        assert eval_event.absolute_error_sec == round(abs(expected_signed), 2)


def test_bus_state_tracking():
    """Verify live bus state dictionary tracks active devices accurately."""
    sim = DigitalTransitSimulator()
    sim.start()
    sim.step(10)

    active_buses = sim.get_active_buses()
    assert len(active_buses) > 0

    for dev_id, state in active_buses.items():
        assert "deviceid" in state
        assert "trip_id" in state
        assert "current_segment" in state
        assert state["status"] == "ACTIVE"
        assert state["last_predicted_eta_sec"] is not None


def test_unique_simulation_session_ids():
    """Verify every simulation instantiation generates a distinct session ID."""
    sim1 = DigitalTransitSimulator()
    sim2 = DigitalTransitSimulator()
    assert sim1.simulation_id != sim2.simulation_id
    assert sim1.simulation_id.startswith("SIM_")


def test_session_metadata_persisted():
    """Verify session is written to data/simulation/simulation_sessions.json."""
    sim = DigitalTransitSimulator()
    sim.start()
    sim.step(5)
    sim.stop()

    meta_file = sim.sessions_meta_file
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    matching = [s for s in sessions if s["simulation_id"] == sim.simulation_id]
    assert len(matching) == 1
    assert matching[0]["records_emitted"] == 5


def test_empty_filter_handling():
    """Verify simulator handles filters matching zero records safely without crashing."""
    config = SimulationConfig(filter_trip_id="NON_EXISTENT_TRIP_999999")
    sim = DigitalTransitSimulator(config=config)
    assert sim.total_records == 0
    sim.start()
    res = sim.step(5)
    assert len(res) == 0


def test_no_synthetic_disturbances_in_baseline():
    """Verify baseline replay contains zero synthetic disturbances."""
    sim = DigitalTransitSimulator()
    sim.start()
    res = sim.step(5)
    for telemetry, pred, outcome, eval_event in res:
        assert telemetry.synthetic_disturbance is False
        assert telemetry.scenario_id == "BASELINE"
        assert telemetry.source_type == "REAL_REPLAY"


def test_missing_feature_rejected():
    """Verify removing a required inference feature fails validation."""
    from ml.inference.predictor import ETAPredictor
    from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS
    predictor = ETAPredictor()

    val_path = PROCESSED_DATA_DIR / "kandy_eta_validation.parquet"
    sample = pd.read_parquet(val_path).iloc[0][ALL_FEATURE_COLUMNS].to_dict()
    
    # Remove one required feature
    del sample["segment_length_km"]

    with pytest.raises(ValueError, match="MISSING REQUIRED FEATURES"):
        predictor.predict(sample)


def test_no_info_log_flood_during_prediction(caplog):
    """Verify inference calls do not emit repeated INFO logs for each record."""
    import logging
    from ml.inference.predictor import ETAPredictor
    from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS
    predictor = ETAPredictor()

    val_path = PROCESSED_DATA_DIR / "kandy_eta_validation.parquet"
    sample = pd.read_parquet(val_path).iloc[0][ALL_FEATURE_COLUMNS].to_dict()

    with caplog.at_level(logging.INFO, logger="TransitVision.LeakGuard"):
        caplog.clear()
        for _ in range(50):
            predictor.predict(sample)

        # There should be 0 INFO logs from LeakGuard during inference loop
        leakguard_info_logs = [r for r in caplog.records if r.name == "TransitVision.LeakGuard" and r.levelno == logging.INFO]
        assert len(leakguard_info_logs) == 0


def test_predictor_reused_in_simulator():
    """Verify simulator reuses the predictor instance rather than reloading it."""
    sim = DigitalTransitSimulator()
    initial_predictor = sim.predictor
    sim.start()
    sim.step(10)
    assert sim.predictor is initial_predictor
    sim.step(10)
    assert sim.predictor is initial_predictor

