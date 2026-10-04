"""TransitVision AI - Autonomous Retraining Package."""
from retraining.trigger_policy import RetrainingTriggerPolicy, TriggerDecision
from retraining.dataset_builder import DatasetBuilder, DatasetProvenanceManifest
from retraining.trainer import CandidateTrainer
from retraining.validator import DualHoldoutValidator, EvaluationMetrics
from retraining.promotion_gate import PromotionGate, PromotionGateResult
from retraining.shadow_validator import ShadowValidator, ShadowValidationResult
from retraining.canary_controller import CanaryController, CanaryStageResult
from retraining.hot_swap import HotSwapCoordinator
from retraining.rollback_manager import RollbackManager, WatchdogEvaluationResult
from retraining.retraining_orchestrator import RetrainingOrchestrator, OrchestratorResult

__all__ = [
    "RetrainingTriggerPolicy",
    "TriggerDecision",
    "DatasetBuilder",
    "DatasetProvenanceManifest",
    "CandidateTrainer",
    "DualHoldoutValidator",
    "EvaluationMetrics",
    "PromotionGate",
    "PromotionGateResult",
    "ShadowValidator",
    "ShadowValidationResult",
    "CanaryController",
    "CanaryStageResult",
    "HotSwapCoordinator",
    "RollbackManager",
    "WatchdogEvaluationResult",
    "RetrainingOrchestrator",
    "OrchestratorResult",
]
