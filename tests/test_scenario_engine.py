"""Comprehensive unit and integration tests for Controlled Disturbance Scenario Engine (Phase 7)."""
import hashlib
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import PROCESSED_DATA_DIR, METADATA_DIR
from simulator.event_models import SimulationConfig
from simulator.scenarios import (
    BaseScenario,
    BaselineScenario,
    RushHourScenario,
    HeavyRainScenario,
    CongestionSurgeScenario,
    RoadIncidentScenario,
    DwellTimeScenario,
    CombinedDisruptionScenario
)
from simulator.scenario_engine import ScenarioEngine
from simulator.transit_simulator import DigitalTransitSimulator
from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS


@pytest.fixture
def sample_feature_record():
    """Provides a realistic single feature dictionary and ground truth."""
    val_path = PROCESSED_DATA_DIR / "kandy_eta_validation.parquet"
    row = pd.read_parquet(val_path).iloc[0]
    features = {col: row[col] for col in ALL_FEATURE_COLUMNS if col in row}
    ground_truth = float(row["eta_to_next_stop_sec"])
    event_ts = pd.to_datetime(row["timestamp_utc"]).isoformat()
    segment = int(row["segment"])
    return features, ground_truth, event_ts, segment


def test_baseline_leaves_events_unchanged(sample_feature_record):
    """Verify BaselineScenario produces identical features and ground truth."""
    features, gt, ts, seg = sample_feature_record
    scenario = BaselineScenario()

    mod_feat, mod_gt, applied = scenario.apply(features, gt, ts, seg)
    assert applied is False
    assert mod_feat == features
    assert mod_gt == gt
    assert scenario.is_synthetic is False


def test_intensity_zero_produces_no_modification(sample_feature_record):
    """Verify any scenario with intensity 0.0 applies zero modification."""
    features, gt, ts, seg = sample_feature_record
    for scenario_cls in [RushHourScenario, HeavyRainScenario, CongestionSurgeScenario, RoadIncidentScenario, DwellTimeScenario, CombinedDisruptionScenario]:
        scenario = scenario_cls(intensity=0.0)
        mod_feat, mod_gt, applied = scenario.apply(features, gt, ts, seg)
        assert applied is False
        assert mod_feat == features
        assert mod_gt == gt


def test_invalid_intensity_rejected():
    """Verify intensity outside [0.0, 1.0] raises ValueError."""
    with pytest.raises(ValueError):
        RushHourScenario(intensity=-0.1)
    with pytest.raises(ValueError):
        HeavyRainScenario(intensity=1.5)


def test_temporal_window_containment(sample_feature_record):
    """Verify scenario applies inside temporal window and stays inactive outside."""
    features, gt, _, seg = sample_feature_record
    
    # Define active window
    start_ts = "2022-10-01T08:00:00+00:00"
    end_ts = "2022-10-01T10:00:00+00:00"
    scenario = CongestionSurgeScenario(intensity=0.8, start_time_utc=start_ts, end_time_utc=end_ts)

    # 1. Inside window
    inside_ts = "2022-10-01T08:30:00+00:00"
    _, _, applied_inside = scenario.apply(features, gt, inside_ts, seg)
    assert applied_inside is True

    # 2. Before window
    before_ts = "2022-10-01T07:30:00+00:00"
    mod_feat_b, mod_gt_b, applied_before = scenario.apply(features, gt, before_ts, seg)
    assert applied_before is False
    assert mod_feat_b == features
    assert mod_gt_b == gt

    # 3. After window
    after_ts = "2022-10-01T10:30:00+00:00"
    mod_feat_a, mod_gt_a, applied_after = scenario.apply(features, gt, after_ts, seg)
    assert applied_after is False
    assert mod_feat_a == features


def test_road_incident_spatial_localization(sample_feature_record):
    """Verify incident affects ONLY designated segments, leaving other segments 100% untouched."""
    features, gt, ts, _ = sample_feature_record
    affected_segs = [8, 9, 10]
    scenario = RoadIncidentScenario(intensity=0.85, affected_segments=affected_segs)

    # Test affected segment 8
    mod_feat_8, mod_gt_8, applied_8 = scenario.apply(features, gt, ts, segment=8)
    assert applied_8 is True
    assert mod_gt_8 > gt
    assert mod_feat_8["congestion_proxy"] == 1

    # Test unaffected segment 5
    mod_feat_5, mod_gt_5, applied_5 = scenario.apply(features, gt, ts, segment=5)
    assert applied_5 is False
    assert mod_feat_5 == features
    assert mod_gt_5 == gt


def test_heavy_rain_modifies_meteorological_features(sample_feature_record):
    """Verify HeavyRainScenario modifies only observable weather variables and delay."""
    features, gt, ts, seg = sample_feature_record
    scenario = HeavyRainScenario(intensity=0.80)

    mod_feat, mod_gt, applied = scenario.apply(features, gt, ts, seg)
    assert applied is True
    assert mod_feat["precipitation"] >= 12.0
    assert mod_feat["rain"] >= 10.0
    assert mod_feat["relative_humidity_2m"] >= 90.0
    assert mod_feat["weather_code"] in [63, 65]
    assert mod_gt > gt


def test_dwell_time_surge(sample_feature_record):
    """Verify DwellTimeScenario adds dwell latency accurately."""
    features, gt, ts, seg = sample_feature_record
    scenario = DwellTimeScenario(intensity=0.75)

    mod_feat, mod_gt, applied = scenario.apply(features, gt, ts, seg)
    assert applied is True
    assert mod_gt == round(gt + 65.0 * 0.75, 2)
    assert mod_feat["cumulative_trip_time_sec"] > features.get("cumulative_trip_time_sec", 0.0)


def test_combined_disruption_composition(sample_feature_record):
    """Verify CombinedDisruptionScenario composes rain, congestion, and dwell effects."""
    features, gt, ts, seg = sample_feature_record
    scenario = CombinedDisruptionScenario(intensity=0.85)

    mod_feat, mod_gt, applied = scenario.apply(features, gt, ts, seg)
    assert applied is True
    assert mod_feat["precipitation"] >= 12.0
    assert mod_feat["congestion_proxy"] == 1
    assert mod_gt > gt * 1.5  # Substantial compound delay


def test_scenario_transformation_determinism(sample_feature_record):
    """Verify identical inputs and configuration produce exact identical outputs."""
    features, gt, ts, seg = sample_feature_record
    sc1 = CongestionSurgeScenario(intensity=0.75)
    sc2 = CongestionSurgeScenario(intensity=0.75)

    f1, gt1, a1 = sc1.apply(features, gt, ts, seg)
    f2, gt2, a2 = sc2.apply(features, gt, ts, seg)

    assert f1 == f2
    assert gt1 == gt2
    assert a1 == a2 == True


def test_monotonic_intensity_scaling(sample_feature_record):
    """Verify higher intensity produces monotonically higher delay and transformation."""
    features, gt, ts, seg = sample_feature_record
    sc_mild = CongestionSurgeScenario(intensity=0.25)
    sc_severe = CongestionSurgeScenario(intensity=0.75)

    f_mild, gt_mild, _ = sc_mild.apply(features, gt, ts, seg)
    f_severe, gt_severe, _ = sc_severe.apply(features, gt, ts, seg)

    assert gt_severe > gt_mild > gt
    assert f_severe["segment_delay_ratio"] > f_mild["segment_delay_ratio"]


def test_scenario_engine_factory():
    """Verify ScenarioEngine instantiates valid scenarios and rejects unknown IDs."""
    sc_base = ScenarioEngine.get_scenario("baseline")
    assert isinstance(sc_base, BaselineScenario)

    sc_rain = ScenarioEngine.get_scenario("heavy_rain", intensity=0.8)
    assert isinstance(sc_rain, HeavyRainScenario)
    assert sc_rain.intensity == 0.8

    with pytest.raises(ValueError):
        ScenarioEngine.get_scenario("UNKNOWN_NON_EXISTENT_SCENARIO")


def test_simulation_with_scenario_execution():
    """Verify DigitalTransitSimulator executes scenario, marks synthetic events, and tracks metrics."""
    config = SimulationConfig(
        scenario_id="HEAVY_RAIN",
        scenario_intensity=0.80,
        replay_speed=10000.0
    )
    sim = DigitalTransitSimulator(config=config)
    assert sim.scenario.scenario_id == "HEAVY_RAIN"
    assert sim.scenario.is_synthetic is True

    sim.start()
    res = sim.step(10)
    assert len(res) == 10

    for telemetry, pred, outcome, eval_event in res:
        assert telemetry.synthetic_disturbance is True
        assert telemetry.scenario_id == "HEAVY_RAIN"
        assert telemetry.features["precipitation"] >= 10.0
        assert outcome.actual_eta_sec >= outcome.baseline_actual_eta_sec
        assert eval_event.synthetic_disturbance is True

    summary = sim.get_performance_summary()
    assert summary["affected_events"] == 10
    assert summary["synthetic_disturbance"] is True


def test_source_stream_immutability_after_scenario_execution():
    """Verify source parquet hash remains strictly identical after scenario execution."""
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    with open(stream_path, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest()

    config = SimulationConfig(
        scenario_id="COMBINED_DISRUPTION",
        scenario_intensity=0.90,
        replay_speed=10000.0
    )
    sim = DigitalTransitSimulator(config=config)
    sim.start()
    sim.step(20)
    sim.stop()

    with open(stream_path, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest()

    assert hash_after == hash_before
