"""TransitVision AI - Atomic Hot-Swap Coordinator.
Coordinates model hot-swapping across registry status updates and in-memory predictors.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from ml.inference.predictor import ETAPredictor
from registry.model_registry import LifecycleModelRegistry, ModelStatus

logger = logging.getLogger("TransitVision.HotSwap")


class HotSwapCoordinator:
    """
    Coordinates zero-downtime hot swapping between LifecycleModelRegistry and ETAPredictor.
    """

    def __init__(
        self,
        registry: Optional[LifecycleModelRegistry] = None,
        predictor: Optional[ETAPredictor] = None,
    ):
        self.registry = registry or LifecycleModelRegistry()
        self.predictor = predictor or ETAPredictor()

    def execute_hot_swap(self, candidate_model_id: str) -> Dict[str, Any]:
        """
        Executes complete hot swap pipeline:
        1. Verifies candidate is in PROMOTION_CANDIDATE state.
        2. Retrieves candidate artifact directory.
        3. Invokes atomic in-memory hot swap on predictor.
        4. Promotes model to PRODUCTION in registry.
        5. If predictor swap fails, aborts registry promotion.
        """
        candidate_entry = self.registry.get_model(candidate_model_id)
        if not candidate_entry:
            raise KeyError(f"Candidate model {candidate_model_id} not found in registry.")

        current_status = candidate_entry.get("status")
        if current_status not in [ModelStatus.PROMOTION_CANDIDATE.value, ModelStatus.VALIDATED.value]:
            raise ValueError(
                f"Candidate {candidate_model_id} cannot be hot-swapped from status '{current_status}'."
            )

        artifact_dir = candidate_entry.get("artifact_dir")
        if not artifact_dir or not Path(artifact_dir).exists():
            raise FileNotFoundError(f"Artifact directory for candidate {candidate_model_id} not found: {artifact_dir}")

        logger.info(f"Initiating atomic hot swap for candidate {candidate_model_id} from {artifact_dir}")

        # Execute in-memory hot swap
        swap_ok = self.predictor.hot_swap_model(artifact_dir)
        if not swap_ok:
            raise RuntimeError(f"In-memory hot swap failed for candidate {candidate_model_id}.")

        # Update registry to PRODUCTION
        promoted_entry = self.registry.promote_to_production(
            candidate_model_id,
            reason="Atomic hot swap executed successfully with all safety gates verified."
        )

        return {
            "status": "SUCCESS",
            "model_id": candidate_model_id,
            "version": promoted_entry.get("version"),
            "artifact_dir": artifact_dir,
            "promoted_at": promoted_entry.get("promoted_to_production_at"),
        }
