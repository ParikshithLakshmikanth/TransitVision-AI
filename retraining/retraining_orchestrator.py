"""TransitVision AI - Autonomous Retraining Closed-Loop Orchestrator.
Coordinates the full lifecycle from drift trigger to dataset construction, candidate
training, dual-holdout validation, shadow/canary deployments, hot swap, and rollback.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from ml.inference.predictor import ETAPredictor
from registry.model_registry import LifecycleModelRegistry, ModelStatus
from retraining.canary_controller import CanaryController, CanaryStageResult
from retraining.dataset_builder import DatasetBuilder, DatasetProvenanceManifest
from retraining.hot_swap import HotSwapCoordinator
from retraining.promotion_gate import PromotionGate, PromotionGateResult
from retraining.rollback_manager import RollbackManager, WatchdogEvaluationResult
from retraining.shadow_validator import ShadowValidator, ShadowValidationResult
from retraining.trainer import CandidateTrainer
from retraining.trigger_policy import RetrainingTriggerPolicy, TriggerDecision
from simulator.transit_simulator import DigitalTransitSimulator

logger = logging.getLogger("TransitVision.Orchestrator")


@dataclass
class OrchestratorResult:
    """Complete summary of a retraining loop execution."""
    success: bool
    trigger_decision: Dict[str, Any]
    candidate_model_id: Optional[str] = None
    candidate_version: Optional[str] = None
    lifecycle_final_status: Optional[str] = None
    active_production_model_id: str = "model_lightgbm_v1"
    dataset_manifest: Optional[Dict[str, Any]] = None
    validation_gate_result: Optional[Dict[str, Any]] = None
    shadow_result: Optional[Dict[str, Any]] = None
    canary_results: List[Dict[str, Any]] = field(default_factory=list)
    hot_swap_result: Optional[Dict[str, Any]] = None
    watchdog_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class RetrainingOrchestrator:
    """
    Closed-loop MLOps pipeline orchestrator.
    """

    def __init__(
        self,
        registry: Optional[LifecycleModelRegistry] = None,
        trigger_policy: Optional[RetrainingTriggerPolicy] = None,
        dataset_builder: Optional[DatasetBuilder] = None,
        trainer: Optional[CandidateTrainer] = None,
        promotion_gate: Optional[PromotionGate] = None,
        shadow_validator: Optional[ShadowValidator] = None,
        canary_controller: Optional[CanaryController] = None,
        hot_swap_coordinator: Optional[HotSwapCoordinator] = None,
        rollback_manager: Optional[RollbackManager] = None,
        predictor: Optional[ETAPredictor] = None,
    ):
        self.registry = registry or LifecycleModelRegistry()
        self.trigger_policy = trigger_policy or RetrainingTriggerPolicy()
        self.dataset_builder = dataset_builder or DatasetBuilder()
        self.trainer = trainer or CandidateTrainer()
        self.promotion_gate = promotion_gate or PromotionGate()
        self.shadow_validator = shadow_validator or ShadowValidator()
        self.canary_controller = canary_controller or CanaryController()
        self.hot_swap_coordinator = hot_swap_coordinator or HotSwapCoordinator(registry=self.registry)
        self.rollback_manager = rollback_manager or RollbackManager(registry=self.registry)
        self.predictor = predictor or ETAPredictor()

    def run_retraining_cycle(
        self,
        drift_events: List[Dict[str, Any]],
        resolved_stream_df: pd.DataFrame,
        current_stream_index: int,
        candidate_version: str,
        rolling_metrics_history: Optional[List[Dict[str, Any]]] = None,
        algorithm: str = "LightGBMRegressor",
        hyperparameters: Optional[Dict[str, Any]] = None,
        force_trigger: bool = False,
    ) -> OrchestratorResult:
        """
        Executes complete retraining cycle through all gates and stages.
        """
        logger.info(f"Starting closed-loop retraining cycle for candidate version '{candidate_version}'")

        # 1. Trigger Policy Evaluation
        trigger_decision = self.trigger_policy.evaluate(
            drift_events=drift_events,
            current_stream_index=current_stream_index,
            resolved_stream_buffer_size=len(resolved_stream_df),
            rolling_metrics_history=rolling_metrics_history,
        )

        if not trigger_decision.should_retrain and not force_trigger:
            logger.info(f"Retraining cycle terminated: {trigger_decision.trigger_reason}")
            prod = self.registry.get_production_model()
            return OrchestratorResult(
                success=True,
                trigger_decision=trigger_decision.__dict__,
                active_production_model_id=prod["model_id"] if prod else "model_lightgbm_v1",
                lifecycle_final_status="NO_TRIGGER",
            )

        # 2. Dataset Construction & Non-Contamination Dual-Holdout Split
        current_prod = self.registry.get_production_model()
        parent_id = current_prod["model_id"] if current_prod else "model_lightgbm_v1"

        try:
            cand_train_df, recent_holdout_df, orig_holdout_df, manifest = self.dataset_builder.build_candidate_dataset(
                resolved_stream_df=resolved_stream_df,
                trigger_reason=trigger_decision.trigger_reason,
                source_model_version=current_prod.get("version", "v1.0.0") if current_prod else "v1.0.0",
            )
        except Exception as e:
            logger.error(f"Dataset construction failed: {e}")
            return OrchestratorResult(
                success=False,
                trigger_decision=trigger_decision.__dict__,
                error_message=f"Dataset construction failed: {e}",
            )

        # 3. Candidate Model Training
        try:
            train_res = self.trainer.train_candidate(
                candidate_train_df=cand_train_df,
                candidate_version=candidate_version,
                parent_model_id=parent_id,
                provenance_manifest=manifest,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
            )
            candidate_model_id = train_res["model_id"]
            cand_artifact_dir = train_res["artifact_dir"]
            candidate_model = train_res["model"]
            candidate_preproc = train_res["preprocessor"]
        except Exception as e:
            logger.error(f"Candidate training failed: {e}")
            return OrchestratorResult(
                success=False,
                trigger_decision=trigger_decision.__dict__,
                dataset_manifest=manifest.to_dict(),
                error_message=f"Candidate training failed: {e}",
            )

        # 4. Register in Lifecycle Model Registry as CANDIDATE
        self.registry.register_candidate(
            model_id=candidate_model_id,
            version=candidate_version,
            algorithm=algorithm,
            training_rows=len(cand_train_df),
            validation_rows=len(orig_holdout_df),
            hyperparameters=train_res["metadata"]["hyperparameters"],
            parent_model_id=parent_id,
            artifact_dir=cand_artifact_dir,
            artifact_hashes=train_res["artifact_hashes"],
            dataset_provenance_hash=manifest.candidate_train_data_hash,
        )

        # 5. Dual-Holdout Multi-Tier Validation Gate
        champ_dir = Path(current_prod["artifact_dir"]) if current_prod and "artifact_dir" in current_prod and current_prod["artifact_dir"] else Path("models/eta_model_v1.0.0")
        champ_predictor = ETAPredictor(model_dir=champ_dir)

        gate_res = self.promotion_gate.evaluate_candidate(
            champion_model=champ_predictor.model,
            champion_preprocessor=champ_predictor.preprocessor,
            candidate_model=candidate_model,
            candidate_preprocessor=candidate_preproc,
            original_holdout_df=orig_holdout_df,
            recent_holdout_df=recent_holdout_df,
            champion_model_id=parent_id,
            candidate_model_id=candidate_model_id,
        )

        if not gate_res.passed:
            # Candidate REJECTED
            rejection_reason = f"Candidate {candidate_model_id} REJECTED due to failure in: " + ", ".join(
                [f for f in ["Gate A (Generalization)" if not gate_res.gate_a_passed else None,
                             "Gate B (Adaptation)" if not gate_res.gate_b_passed else None,
                             "Gate C (Safety/Tail Risk)" if not gate_res.gate_c_passed else None] if f]
            )
            self.registry.transition_status(
                candidate_model_id,
                ModelStatus.REJECTED,
                reason=rejection_reason,
                metrics_update=gate_res.to_dict(),
            )
            return OrchestratorResult(
                success=False,
                trigger_decision=trigger_decision.__dict__,
                candidate_model_id=candidate_model_id,
                candidate_version=candidate_version,
                lifecycle_final_status=ModelStatus.REJECTED.value,
                active_production_model_id=parent_id,
                dataset_manifest=manifest.to_dict(),
                validation_gate_result=gate_res.to_dict(),
                error_message=rejection_reason,
            )

        # Transition to VALIDATED
        self.registry.transition_status(
            candidate_model_id,
            ModelStatus.VALIDATED,
            reason=f"Candidate {candidate_model_id} passed all validation gates (Gate A Generalization: {gate_res.candidate_original_metrics['mae_sec']}s vs {gate_res.champion_original_metrics['mae_sec']}s; Gate B Adaptation: {gate_res.candidate_recent_metrics['mae_sec']}s vs {gate_res.champion_recent_metrics['mae_sec']}s). Approved for Shadow Safety Replay.",
            metrics_update=gate_res.to_dict(),
        )

        # 6. Shadow Safety Replay (300 records)
        self.registry.transition_status(
            candidate_model_id,
            ModelStatus.SHADOW,
            reason="Starting 300-record simulation shadow safety check.",
        )
        shadow_res = self.shadow_validator.run_shadow_safety_check(
            challenger_model_dir=cand_artifact_dir,
            start_cursor=0,
        )

        if not shadow_res.passed:
            rejection_reason = f"Candidate {candidate_model_id} REJECTED during Shadow Safety Check: " + "; ".join(shadow_res.failure_reasons)
            self.registry.transition_status(
                candidate_model_id,
                ModelStatus.REJECTED,
                reason=rejection_reason,
                metrics_update={"shadow_metrics": shadow_res.to_dict()},
            )
            return OrchestratorResult(
                success=False,
                trigger_decision=trigger_decision.__dict__,
                candidate_model_id=candidate_model_id,
                candidate_version=candidate_version,
                lifecycle_final_status=ModelStatus.REJECTED.value,
                active_production_model_id=parent_id,
                dataset_manifest=manifest.to_dict(),
                validation_gate_result=gate_res.to_dict(),
                shadow_result=shadow_res.to_dict(),
                error_message=rejection_reason,
            )

        # 7. Canary Deployment Progression (10% -> 50%)
        canary_stage_results = []
        for stage_pct in [10, 50]:
            target_status = ModelStatus.CANARY_10 if stage_pct == 10 else ModelStatus.CANARY_50
            if stage_pct == 10:
                self.registry.transition_status(
                    candidate_model_id,
                    ModelStatus.CANARY_10,
                    reason="Starting 10% canary traffic rollout.",
                )
            elif stage_pct == 50:
                self.registry.transition_status(
                    candidate_model_id,
                    ModelStatus.CANARY_50,
                    reason="Promoting to 50% canary traffic rollout.",
                )

            c_res = self.canary_controller.run_canary_stage(
                challenger_model_dir=cand_artifact_dir,
                canary_percentage=stage_pct,
                start_cursor=300 if stage_pct == 10 else 600,
            )
            canary_stage_results.append(c_res.to_dict())

            if not c_res.passed:
                rejection_reason = f"Candidate {candidate_model_id} REJECTED during {c_res.stage_name} Canary: " + "; ".join(c_res.failure_reasons)
                self.registry.transition_status(
                    candidate_model_id,
                    ModelStatus.REJECTED,
                    reason=rejection_reason,
                    metrics_update={"canary_stages": canary_stage_results},
                )
                return OrchestratorResult(
                    success=False,
                    trigger_decision=trigger_decision.__dict__,
                    candidate_model_id=candidate_model_id,
                    candidate_version=candidate_version,
                    lifecycle_final_status=ModelStatus.REJECTED.value,
                    active_production_model_id=parent_id,
                    dataset_manifest=manifest.to_dict(),
                    validation_gate_result=gate_res.to_dict(),
                    shadow_result=shadow_res.to_dict(),
                    canary_results=canary_stage_results,
                    error_message=rejection_reason,
                )

        # 8. Transition to PROMOTION_CANDIDATE
        self.registry.transition_status(
            candidate_model_id,
            ModelStatus.PROMOTION_CANDIDATE,
            reason="Successfully passed 10% and 50% Canary stages. Ready for atomic hot swap.",
            metrics_update={"canary_stages": canary_stage_results},
        )

        # 9. Atomic Zero-Downtime Hot Swap to PRODUCTION (100%)
        try:
            hot_swap_res = self.hot_swap_coordinator.execute_hot_swap(candidate_model_id)
            self.trigger_policy.record_retraining_execution(current_stream_index)
        except Exception as e:
            logger.error(f"Hot swap failed: {e}")
            return OrchestratorResult(
                success=False,
                trigger_decision=trigger_decision.__dict__,
                candidate_model_id=candidate_model_id,
                candidate_version=candidate_version,
                lifecycle_final_status=ModelStatus.PROMOTION_CANDIDATE.value,
                active_production_model_id=parent_id,
                dataset_manifest=manifest.to_dict(),
                validation_gate_result=gate_res.to_dict(),
                shadow_result=shadow_res.to_dict(),
                canary_results=canary_stage_results,
                error_message=f"Hot swap failed: {e}",
            )

        logger.info(f"Retraining cycle COMPLETED successfully. Candidate {candidate_model_id} is now PRODUCTION.")
        return OrchestratorResult(
            success=True,
            trigger_decision=trigger_decision.__dict__,
            candidate_model_id=candidate_model_id,
            candidate_version=candidate_version,
            lifecycle_final_status=ModelStatus.PRODUCTION.value,
            active_production_model_id=candidate_model_id,
            dataset_manifest=manifest.to_dict(),
            validation_gate_result=gate_res.to_dict(),
            shadow_result=shadow_res.to_dict(),
            canary_results=canary_stage_results,
            hot_swap_result=hot_swap_res,
        )
