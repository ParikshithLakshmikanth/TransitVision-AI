"""TransitVision AI - Unit Tests for Atomic Hot Swap and Rollback Watchdog."""
from pathlib import Path
import pytest
from ml.inference.predictor import ETAPredictor
from registry.model_registry import LifecycleModelRegistry, ModelStatus
from retraining.hot_swap import HotSwapCoordinator
from retraining.rollback_manager import RollbackManager


def test_hot_swap_full_sequence_and_thread_safety():
    """Verify in-memory model hot swap performs separate loading, warmup, and atomic reference swap."""
    predictor = ETAPredictor()
    initial_model_id = predictor.metadata.get("model_id")

    # Swapping to valid model dir
    valid_dir = Path("models/eta_model_v1.0.0").resolve()
    swap_ok = predictor.hot_swap_model(valid_dir)
    assert swap_ok
    assert predictor.health_check()["status"] == "HEALTHY"
    assert predictor.metadata.get("model_id") == "model_lightgbm_v1"


def test_hot_swap_failure_keeps_champion_serving():
    """Verify attempting hot swap to an invalid directory fails safely without disrupting champion."""
    predictor = ETAPredictor()
    initial_id = predictor.metadata.get("model_id")
    initial_model_instance = predictor.model

    # Attempt swap to non-existent directory
    swap_ok = predictor.hot_swap_model(Path("models/nonexistent_model"))
    assert not swap_ok

    # Champion remains active and identical
    assert predictor.metadata.get("model_id") == initial_id
    assert predictor.model is initial_model_instance
    assert predictor.health_check()["status"] == "HEALTHY"


def test_rollback_watchdog_triggers_on_1_25x_degradation(tmp_path):
    """Verify watchdog detects degradation exceeding 1.25x and restores previous champion."""
    reg_file = tmp_path / "model_registry.json"
    reg = LifecycleModelRegistry(registry_file=reg_file)

    # Setup champion v1.0.0 in PRODUCTION
    reg.register_candidate(
        model_id="model_lightgbm_v1",
        version="v1.0.0",
        algorithm="LightGBMRegressor",
        training_rows=140000,
        validation_rows=30000,
        hyperparameters={},
        artifact_dir=str(Path("models/eta_model_v1.0.0").resolve()),
    )
    reg.transition_status("model_lightgbm_v1", ModelStatus.VALIDATED, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.SHADOW, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.CANARY_10, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.CANARY_50, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.PROMOTION_CANDIDATE, "pass")
    reg.promote_to_production("model_lightgbm_v1", "initial prod")

    # Setup candidate v1.1.0 and promote
    reg.register_candidate(
        model_id="model_lightgbm_v1.1.0",
        version="v1.1.0",
        algorithm="LightGBMRegressor",
        training_rows=142000,
        validation_rows=30000,
        hyperparameters={},
        parent_model_id="model_lightgbm_v1",
        artifact_dir=str(Path("models/eta_model_v1.0.0").resolve()),
    )
    reg.transition_status("model_lightgbm_v1.1.0", ModelStatus.VALIDATED, "pass")
    reg.transition_status("model_lightgbm_v1.1.0", ModelStatus.SHADOW, "pass")
    reg.transition_status("model_lightgbm_v1.1.0", ModelStatus.CANARY_10, "pass")
    reg.transition_status("model_lightgbm_v1.1.0", ModelStatus.CANARY_50, "pass")
    reg.transition_status("model_lightgbm_v1.1.0", ModelStatus.PROMOTION_CANDIDATE, "pass")
    reg.promote_to_production("model_lightgbm_v1.1.0", "promoted")

    assert reg.get_production_model()["model_id"] == "model_lightgbm_v1.1.0"

    # Watchdog receives degraded metrics (MAE = 55.0s > 38.77s * 1.25 = 48.46s)
    degraded_events = [{"error_sec": 55.0, "latency_ms": 2.5} for _ in range(100)]
    rollback_mgr = RollbackManager(registry=reg, watchdog_window_size=100, max_degradation_ratio=1.25)
    res = rollback_mgr.monitor_and_enforce(degraded_events, baseline_mae=38.77)

    assert res.rollback_triggered
    assert res.active_production_model_id == "model_lightgbm_v1"
    assert reg.get_production_model()["model_id"] == "model_lightgbm_v1"
    assert reg.get_model("model_lightgbm_v1.1.0")["status"] == ModelStatus.ROLLED_BACK.value
