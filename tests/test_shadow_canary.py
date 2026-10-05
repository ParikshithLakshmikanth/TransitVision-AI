"""TransitVision AI - Unit Tests for Simulation Shadow and Canary Deployments."""
from pathlib import Path
from retraining.canary_controller import CanaryController, deterministic_canary_route
from retraining.shadow_validator import ShadowValidator
from simulator.transit_simulator import DigitalTransitSimulator


def test_deterministic_canary_routing():
    """Verify trip hash partitioning is deterministic."""
    trip_1 = "trip_kandy_001"
    trip_2 = "trip_kandy_002"

    # Same trip produces identical routing decision across calls
    r1 = deterministic_canary_route(trip_1, canary_percentage=10)
    r2 = deterministic_canary_route(trip_1, canary_percentage=10)
    assert r1 == r2

    # 0% is always False, 100% is always True
    assert not deterministic_canary_route(trip_1, 0)
    assert deterministic_canary_route(trip_1, 100)


def test_shadow_safety_check_execution():
    """Verify shadow safety check runs without interfering with champion."""
    shadow = ShadowValidator(shadow_window_size=50)
    sim = DigitalTransitSimulator()
    chall_dir = str(sim.predictor.model_dir)

    res = shadow.run_shadow_safety_check(
        challenger_model_dir=chall_dir,
        simulator=sim,
        start_cursor=0,
    )

    assert res.passed
    assert res.exceptions_count == 0
    assert res.sample_count == 50
    assert res.champion_unaffected


def test_canary_stage_identical_record_evaluation():
    """Verify canary stage evaluates Champion and Challenger on identical records."""
    canary = CanaryController(records_per_stage=100, min_canary_samples=5)
    sim = DigitalTransitSimulator()
    chall_dir = str(sim.predictor.model_dir)

    res = canary.run_canary_stage(
        challenger_model_dir=chall_dir,
        canary_percentage=50,
        simulator=sim,
        start_cursor=0,
    )

    assert res.passed
    assert res.exceptions_count == 0
    assert res.record_ids_match
    assert res.champion_evaluation_count == res.challenger_evaluation_count
    assert res.selected_canary_records > 0
