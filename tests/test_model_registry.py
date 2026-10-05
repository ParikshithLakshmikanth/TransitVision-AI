"""TransitVision AI - Unit Tests for Lifecycle Model Registry."""
import pytest
from registry.model_registry import LifecycleModelRegistry, ModelStatus


def test_registry_lifecycle_happy_path(tmp_path):
    """Verify standard candidate promotion lifecycle transitions."""
    reg_file = tmp_path / "model_registry.json"
    reg = LifecycleModelRegistry(registry_file=reg_file)

    # 1. Register candidate
    reg.register_candidate(
        model_id="test_model_v1.1.0",
        version="v1.1.0",
        algorithm="LightGBMRegressor",
        training_rows=140000,
        validation_rows=30000,
        hyperparameters={},
        parent_model_id="model_lightgbm_v1",
    )
    assert reg.get_model("test_model_v1.1.0")["status"] == ModelStatus.CANDIDATE.value

    # 2. VALIDATED
    reg.transition_status("test_model_v1.1.0", ModelStatus.VALIDATED, "Passed Dual-Holdout Gate")
    assert reg.get_model("test_model_v1.1.0")["status"] == ModelStatus.VALIDATED.value

    # 3. SHADOW
    reg.transition_status("test_model_v1.1.0", ModelStatus.SHADOW, "Starting 300-record shadow")
    assert reg.get_model("test_model_v1.1.0")["status"] == ModelStatus.SHADOW.value

    # 4. CANARY_10
    reg.transition_status("test_model_v1.1.0", ModelStatus.CANARY_10, "Starting 10% canary")
    assert reg.get_model("test_model_v1.1.0")["status"] == ModelStatus.CANARY_10.value

    # 5. CANARY_50
    reg.transition_status("test_model_v1.1.0", ModelStatus.CANARY_50, "Promoting to 50% canary")
    assert reg.get_model("test_model_v1.1.0")["status"] == ModelStatus.CANARY_50.value

    # 6. PROMOTION_CANDIDATE
    reg.transition_status("test_model_v1.1.0", ModelStatus.PROMOTION_CANDIDATE, "Canary stages passed")
    assert reg.get_model("test_model_v1.1.0")["status"] == ModelStatus.PROMOTION_CANDIDATE.value

    # 7. PRODUCTION
    reg.promote_to_production("test_model_v1.1.0", "Hot swap successful")
    assert reg.get_production_model()["model_id"] == "test_model_v1.1.0"


def test_registry_illegal_direct_jumps_rejected(tmp_path):
    """Verify illegal transitions (e.g. CANDIDATE/VALIDATED/REJECTED/ROLLED_BACK -> PRODUCTION) raise ValueError."""
    reg_file = tmp_path / "model_registry.json"
    reg = LifecycleModelRegistry(registry_file=reg_file)

    # Register initial base champion
    reg.register_candidate(
        model_id="model_lightgbm_v1",
        version="v1.0.0",
        algorithm="LightGBMRegressor",
        training_rows=140000,
        validation_rows=30000,
        hyperparameters={},
    )
    reg.transition_status("model_lightgbm_v1", ModelStatus.VALIDATED, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.SHADOW, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.CANARY_10, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.CANARY_50, "pass")
    reg.transition_status("model_lightgbm_v1", ModelStatus.PROMOTION_CANDIDATE, "pass")
    reg.promote_to_production("model_lightgbm_v1", "initial prod")

    reg.register_candidate(
        model_id="test_model_bad",
        version="v1.1.0",
        algorithm="LightGBMRegressor",
        training_rows=140000,
        validation_rows=30000,
        hyperparameters={},
        parent_model_id="model_lightgbm_v1",
    )

    # 1. CANDIDATE -> PRODUCTION (illegal)
    with pytest.raises(ValueError):
        reg.transition_status("test_model_bad", ModelStatus.PRODUCTION, "Illegal direct jump")

    # 2. VALIDATED -> PRODUCTION (illegal, skipping shadow/canary)
    reg.transition_status("test_model_bad", ModelStatus.VALIDATED, "Pass validation")
    with pytest.raises(ValueError):
        reg.transition_status("test_model_bad", ModelStatus.PRODUCTION, "Illegal skip of shadow/canary")

    # 3. REJECTED -> PRODUCTION (illegal)
    reg.transition_status("test_model_bad", ModelStatus.REJECTED, "Failed shadow")
    with pytest.raises(ValueError):
        reg.transition_status("test_model_bad", ModelStatus.PRODUCTION, "Illegal jump from rejected")

    # 4. ROLLED_BACK -> PRODUCTION (illegal without creating a new candidate)
    reg.register_candidate(
        model_id="test_model_rolled_back",
        version="v1.2.0",
        algorithm="LightGBMRegressor",
        training_rows=140000,
        validation_rows=30000,
        hyperparameters={},
        parent_model_id="model_lightgbm_v1",
    )
    reg.transition_status("test_model_rolled_back", ModelStatus.VALIDATED, "pass")
    reg.transition_status("test_model_rolled_back", ModelStatus.SHADOW, "pass")
    reg.transition_status("test_model_rolled_back", ModelStatus.CANARY_10, "pass")
    reg.transition_status("test_model_rolled_back", ModelStatus.CANARY_50, "pass")
    reg.transition_status("test_model_rolled_back", ModelStatus.PROMOTION_CANDIDATE, "pass")
    reg.promote_to_production("test_model_rolled_back", "promoted")
    reg.rollback_production("Watchdog failure")

    assert reg.get_model("test_model_rolled_back")["status"] == ModelStatus.ROLLED_BACK.value
    with pytest.raises(ValueError):
        reg.transition_status("test_model_rolled_back", ModelStatus.PRODUCTION, "Illegal revive of rolled back model")


def test_registry_rejection_transition_from_any_stage(tmp_path):
    """Verify any pre-production state can legally transition to REJECTED."""
    reg_file = tmp_path / "model_registry.json"
    reg = LifecycleModelRegistry(registry_file=reg_file)

    stages = [
        ModelStatus.CANDIDATE,
        ModelStatus.VALIDATED,
        ModelStatus.SHADOW,
        ModelStatus.CANARY_10,
        ModelStatus.CANARY_50,
        ModelStatus.PROMOTION_CANDIDATE,
    ]

    for idx, stage in enumerate(stages):
        m_id = f"test_model_stage_{idx}"
        reg.register_candidate(
            model_id=m_id,
            version="v1.0",
            algorithm="LightGBMRegressor",
            training_rows=1000,
            validation_rows=100,
            hyperparameters={},
        )
        # Advance to stage
        if stage != ModelStatus.CANDIDATE:
            for st in stages[1 : stages.index(stage) + 1]:
                reg.transition_status(m_id, st, f"Advance to {st.value}")

        # Transition to REJECTED
        reg.transition_status(m_id, ModelStatus.REJECTED, f"Reject at {stage.value}")
        assert reg.get_model(m_id)["status"] == ModelStatus.REJECTED.value


def test_registry_comprehensive_illegal_transitions(tmp_path):
    """Verify all 10 specified illegal state transitions are strictly rejected by the state machine."""
    reg_file = tmp_path / "model_registry.json"
    reg = LifecycleModelRegistry(registry_file=reg_file)

    def _create_candidate(m_id):
        return reg.register_candidate(
            model_id=m_id,
            version="v1.0",
            algorithm="LightGBMRegressor",
            training_rows=1000,
            validation_rows=100,
            hyperparameters={},
        )

    # 1. CANDIDATE -> PRODUCTION (illegal)
    _create_candidate("m1")
    with pytest.raises(ValueError):
        reg.transition_status("m1", ModelStatus.PRODUCTION, "illegal")

    # 2. CANDIDATE -> CANARY_50 (illegal)
    _create_candidate("m2")
    with pytest.raises(ValueError):
        reg.transition_status("m2", ModelStatus.CANARY_50, "illegal")

    # 3. CANDIDATE -> PROMOTION_CANDIDATE (illegal)
    _create_candidate("m3")
    with pytest.raises(ValueError):
        reg.transition_status("m3", ModelStatus.PROMOTION_CANDIDATE, "illegal")

    # 4. REJECTED -> PRODUCTION (illegal)
    _create_candidate("m4")
    reg.transition_status("m4", ModelStatus.REJECTED, "reject")
    with pytest.raises(ValueError):
        reg.transition_status("m4", ModelStatus.PRODUCTION, "illegal")

    # 5. SHADOW -> PRODUCTION (illegal)
    _create_candidate("m5")
    reg.transition_status("m5", ModelStatus.VALIDATED, "valid")
    reg.transition_status("m5", ModelStatus.SHADOW, "shadow")
    with pytest.raises(ValueError):
        reg.transition_status("m5", ModelStatus.PRODUCTION, "illegal")

    # 6. CANARY_10 -> PRODUCTION (illegal)
    _create_candidate("m6")
    reg.transition_status("m6", ModelStatus.VALIDATED, "valid")
    reg.transition_status("m6", ModelStatus.SHADOW, "shadow")
    reg.transition_status("m6", ModelStatus.CANARY_10, "canary10")
    with pytest.raises(ValueError):
        reg.transition_status("m6", ModelStatus.PRODUCTION, "illegal")

    # 7. ARCHIVED -> CANARY_50 or SHADOW (illegal)
    _create_candidate("m7")
    reg.transition_status("m7", ModelStatus.VALIDATED, "valid")
    reg.transition_status("m7", ModelStatus.SHADOW, "shadow")
    reg.transition_status("m7", ModelStatus.CANARY_10, "canary10")
    reg.transition_status("m7", ModelStatus.CANARY_50, "canary50")
    reg.transition_status("m7", ModelStatus.PROMOTION_CANDIDATE, "pc")
    reg.promote_to_production("m7", "promote")
    reg.transition_status("m7", ModelStatus.ARCHIVED, "archive")
    with pytest.raises(ValueError):
        reg.transition_status("m7", ModelStatus.SHADOW, "illegal revive to shadow")
    with pytest.raises(ValueError):
        reg.transition_status("m7", ModelStatus.CANARY_50, "illegal revive to canary50")

    # 8. Direct promote_to_production on unvalidated CANDIDATE (illegal)
    _create_candidate("m8")
    with pytest.raises(ValueError):
        reg.promote_to_production("m8", "illegal direct promotion")

    # 9. VALIDATED -> CANARY_50 (illegal skip of shadow & canary10)
    _create_candidate("m9")
    reg.transition_status("m9", ModelStatus.VALIDATED, "valid")
    with pytest.raises(ValueError):
        reg.transition_status("m9", ModelStatus.CANARY_50, "illegal skip")

    # 10. REJECTED -> VALIDATED or SHADOW (illegal revival)
    _create_candidate("m10")
    reg.transition_status("m10", ModelStatus.REJECTED, "reject")
    with pytest.raises(ValueError):
        reg.transition_status("m10", ModelStatus.VALIDATED, "illegal revive")
    with pytest.raises(ValueError):
        reg.transition_status("m10", ModelStatus.SHADOW, "illegal revive")


def test_registry_exactly_one_production_model_invariant(tmp_path):
    """Verify that promoting a new model atomically archives the predecessor, maintaining exactly 1 PRODUCTION model."""
    reg_file = tmp_path / "model_registry.json"
    reg = LifecycleModelRegistry(registry_file=reg_file)

    for m_id in ["model_v1", "model_v2", "model_v3"]:
        reg.register_candidate(
            model_id=m_id,
            version=m_id,
            algorithm="LightGBMRegressor",
            training_rows=1000,
            validation_rows=100,
            hyperparameters={},
        )
        reg.transition_status(m_id, ModelStatus.VALIDATED, "valid")
        reg.transition_status(m_id, ModelStatus.SHADOW, "shadow")
        reg.transition_status(m_id, ModelStatus.CANARY_10, "canary10")
        reg.transition_status(m_id, ModelStatus.CANARY_50, "canary50")
        reg.transition_status(m_id, ModelStatus.PROMOTION_CANDIDATE, "pc")
        reg.promote_to_production(m_id, f"Promoting {m_id}")

        # Check invariant: exactly 1 PRODUCTION model
        prod_models = [m for m in reg.list_models() if m.get("status") == ModelStatus.PRODUCTION.value]
        assert len(prod_models) == 1
        assert prod_models[0]["model_id"] == m_id
        assert reg.get_production_model()["model_id"] == m_id
