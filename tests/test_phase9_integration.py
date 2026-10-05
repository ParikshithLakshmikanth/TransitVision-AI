"""TransitVision AI - End-to-End Integration Tests for Phase 9 Closed-Loop Retraining."""
from pathlib import Path
import json
import pandas as pd
from config.settings import PROCESSED_DATA_DIR, METADATA_DIR
from registry.model_registry import LifecycleModelRegistry, ModelStatus
from retraining.trainer import CandidateTrainer
from retraining.retraining_orchestrator import RetrainingOrchestrator


def test_phase9_end_to_end_candidate_promotion_and_rejection(tmp_path):
    """Verify closed-loop candidate training, dual-holdout gating, shadow, canary, and promotion in isolated environment."""
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    assert stream_path.exists()
    stream_df = pd.read_parquet(stream_path).iloc[:2000].copy()

    # Initialize isolated test registry copying base Phase 5 metadata
    test_reg_file = tmp_path / "model_registry.json"
    live_reg_file = METADATA_DIR / "model_registry.json"
    with open(live_reg_file, "r", encoding="utf-8") as f:
        live_data = json.load(f)

    # Ensure clean starting champion for isolated test run
    live_data["production_model_id"] = "model_lightgbm_v1"
    filtered_models = []
    for m in live_data.get("models", []):
        if m["model_id"] == "model_lightgbm_v1":
            m["status"] = "PRODUCTION"
            m["artifact_dir"] = "models/eta_model_v1.0.0"
            filtered_models.append(m)
        elif not m["model_id"].startswith("model_lightgbm_v1.1"):
            filtered_models.append(m)
    live_data["models"] = filtered_models

    with open(test_reg_file, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2)

    reg = LifecycleModelRegistry(registry_file=test_reg_file)
    cand_dir = tmp_path / "candidates"
    trainer = CandidateTrainer(candidates_root_dir=cand_dir)
    from ml.inference.predictor import ETAPredictor
    champ_pred = ETAPredictor(model_dir=Path("models/eta_model_v1.0.0"))
    orchestrator = RetrainingOrchestrator(registry=reg, trainer=trainer, predictor=champ_pred)

    # 1. Successful candidate promotion
    res = orchestrator.run_retraining_cycle(
        drift_events=[],
        resolved_stream_df=stream_df,
        current_stream_index=2000,
        candidate_version="v1.1.0_integration",
        force_trigger=True,
    )

    assert res.success
    assert res.lifecycle_final_status == ModelStatus.PRODUCTION.value
    assert res.candidate_model_id == "model_lightgbm_v1.1.0_integration"
    assert res.active_production_model_id == "model_lightgbm_v1.1.0_integration"
    assert res.validation_gate_result["passed"]
    assert res.shadow_result["passed"]
    assert len(res.canary_results) == 2
    assert res.canary_results[0]["passed"]
    assert res.canary_results[1]["passed"]

    # 2. Rejection of bad candidate
    bad_res = orchestrator.run_retraining_cycle(
        drift_events=[],
        resolved_stream_df=stream_df,
        current_stream_index=2000,
        candidate_version="v1.1.0_bad_integration",
        algorithm="Ridge",
        hyperparameters={"alpha": 1e8},
        force_trigger=True,
    )

    assert not bad_res.success
    assert bad_res.lifecycle_final_status == ModelStatus.REJECTED.value
    # Active production remains previous promoted model
    assert reg.get_production_model()["model_id"] == "model_lightgbm_v1.1.0_integration"
