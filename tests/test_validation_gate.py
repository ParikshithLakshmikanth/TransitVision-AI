"""TransitVision AI - Unit Tests for Dual-Holdout Promotion Gate."""
from pathlib import Path
import pandas as pd
import pytest
from ml.inference.predictor import ETAPredictor
from retraining.promotion_gate import PromotionGate
from retraining.trainer import CandidateTrainer
from retraining.dataset_builder import DatasetBuilder
from config.settings import PROCESSED_DATA_DIR


def test_promotion_gate_evaluates_dual_holdouts():
    """Verify PromotionGate evaluates Gate A, Gate B, and Gate C correctly."""
    builder = DatasetBuilder()
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    sample_stream = pd.read_parquet(stream_path).iloc[:2000].copy()

    cand_train, recent_holdout, orig_holdout, manifest = builder.build_candidate_dataset(
        resolved_stream_df=sample_stream
    )

    trainer = CandidateTrainer()
    train_res = trainer.train_candidate(
        candidate_train_df=cand_train,
        candidate_version="v1.1.0_test",
        parent_model_id="model_lightgbm_v1",
        provenance_manifest=manifest,
        algorithm="LightGBMRegressor",
    )

    champ_predictor = ETAPredictor(model_dir=Path("models/eta_model_v1.0.0"))
    gate = PromotionGate()  # Default reconciled thresholds: 1.01x Gate A, 1.00x Gate B, 1.05x RMSE, 1.08x P95
    gate_res = gate.evaluate_candidate(
        champion_model=champ_predictor.model,
        champion_preprocessor=champ_predictor.preprocessor,
        candidate_model=train_res["model"],
        candidate_preprocessor=train_res["preprocessor"],
        original_holdout_df=orig_holdout,
        recent_holdout_df=recent_holdout,
        champion_model_id="model_lightgbm_v1",
        candidate_model_id="model_lightgbm_v1.1.0_test",
    )

    assert gate_res.gate_a_passed
    assert gate_res.gate_b_passed
    assert gate_res.gate_c_passed
    assert gate_res.passed
    assert gate_res.applied_thresholds["max_original_mae_degradation_ratio"] == 1.01
    assert gate_res.applied_thresholds["max_recent_mae_degradation_ratio"] == 1.00


def test_promotion_gate_rejects_bad_candidate():
    """Verify PromotionGate strictly fails an intentionally degraded model."""
    builder = DatasetBuilder()
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    sample_stream = pd.read_parquet(stream_path).iloc[:1000].copy()

    cand_train, recent_holdout, orig_holdout, manifest = builder.build_candidate_dataset(
        resolved_stream_df=sample_stream
    )

    # Train poor model
    trainer = CandidateTrainer()
    bad_train_res = trainer.train_candidate(
        candidate_train_df=cand_train,
        candidate_version="v1.1.0_bad_test",
        parent_model_id="model_lightgbm_v1",
        provenance_manifest=manifest,
        algorithm="Ridge",
        hyperparameters={"alpha": 1e8},
    )

    champ_predictor = ETAPredictor()
    gate = PromotionGate()
    gate_res = gate.evaluate_candidate(
        champion_model=champ_predictor.model,
        champion_preprocessor=champ_predictor.preprocessor,
        candidate_model=bad_train_res["model"],
        candidate_preprocessor=bad_train_res["preprocessor"],
        original_holdout_df=orig_holdout,
        recent_holdout_df=recent_holdout,
        champion_model_id="model_lightgbm_v1",
        candidate_model_id="model_ridge_v1.1.0_bad_test",
    )

    assert not gate_res.passed
    assert len(gate_res.failure_reasons) > 0
