"""TransitVision AI - MLOps Closed-Loop Lifecycle & Retraining Service."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import METADATA_DIR, REPORTS_DIR
from registry.model_registry import LifecycleModelRegistry, ModelStatus
from retraining.retraining_orchestrator import RetrainingOrchestrator
from retraining.rollback_manager import RollbackManager
from backend.schemas import (
    LifecycleStateResponse,
    RetrainingStatusResponse,
    RollbackStatusResponse,
)

logger = logging.getLogger("TransitVision.LifecycleService")


class LifecycleService:
    """
    Coordinates MLOps closed-loop lifecycle introspection,
    retraining status, historical benchmark summaries, and rollback status.
    """

    def __init__(
        self,
        registry: Optional[LifecycleModelRegistry] = None,
        orchestrator: Optional[RetrainingOrchestrator] = None,
        rollback_manager: Optional[RollbackManager] = None,
    ):
        self.registry = registry or LifecycleModelRegistry()
        self.orchestrator = orchestrator or RetrainingOrchestrator(registry=self.registry)
        self.rollback_manager = rollback_manager or RollbackManager(registry=self.registry)
        self.benchmark_summary_path = REPORTS_DIR / "phase9_retraining_benchmark_summary.json"

    def get_lifecycle_state(self) -> LifecycleStateResponse:
        """Constructs comprehensive lifecycle response for frontend rendering."""
        prod = self.registry.get_production_model()
        prod_id = prod.get("model_id", "model_lightgbm_v1") if prod else "model_lightgbm_v1"
        prod_ver = prod.get("version", "v1.0.0") if prod else "v1.0.0"
        prod_alg = prod.get("algorithm", "LightGBMRegressor") if prod else "LightGBMRegressor"

        # Find latest candidate / rolled back model
        all_models = self.registry.list_models()
        latest_candidate = None
        for m in reversed(all_models):
            if m.get("model_id") != prod_id:
                latest_candidate = m
                break

        # Load benchmark summary if available
        summary_data = self._load_benchmark_summary()
        exp_c = summary_data.get("C", {})
        exp_f = summary_data.get("F", {})

        return LifecycleStateResponse(
            production_model_id=prod_id,
            production_model_version=prod_ver,
            production_algorithm=prod_alg,
            current_lifecycle_state="PRODUCTION",
            latest_candidate_id=latest_candidate.get("model_id") if latest_candidate else None,
            latest_candidate_version=latest_candidate.get("version") if latest_candidate else None,
            latest_candidate_status=latest_candidate.get("status") if latest_candidate else None,
            latest_retraining_trigger_reason=exp_c.get("drift", {}).get("trigger_reason"),
            latest_validation_result=exp_c.get("gate_a"),
            shadow_status="PASSED" if exp_c.get("shadow_validation", {}).get("passed") else "IDLE",
            canary_status="COMPLETED" if exp_c.get("canary_50", {}).get("passed") else "IDLE",
            rollback_status="ROLLED_BACK" if exp_f.get("status") == "PASS" else "STABLE",
            recent_lifecycle_events=self._get_recent_lifecycle_events(summary_data),
        )

    def get_retraining_status(self) -> RetrainingStatusResponse:
        """Returns current retraining pipeline status."""
        prod = self.registry.get_production_model()
        prod_id = prod.get("model_id", "model_lightgbm_v1") if prod else "model_lightgbm_v1"
        summary_data = self._load_benchmark_summary()

        return RetrainingStatusResponse(
            is_retraining_in_progress=False,
            active_production_model_id=prod_id,
            last_trigger_decision=summary_data.get("C", {}).get("drift"),
            last_retraining_result=summary_data.get("C", {}).get("candidate"),
        )

    def get_retraining_history(self) -> Dict[str, Any]:
        """Returns historical benchmark records."""
        return self._load_benchmark_summary()

    def get_rollback_status(self) -> RollbackStatusResponse:
        """Returns read-only rollback status and watchdog policies."""
        prod = self.registry.get_production_model()
        prod_id = prod.get("model_id", "model_lightgbm_v1") if prod else "model_lightgbm_v1"
        prod_ver = prod.get("version", "v1.0.0") if prod else "v1.0.0"
        summary_data = self._load_benchmark_summary()
        exp_f = summary_data.get("F", {})

        history = []
        if exp_f:
            history.append(exp_f)

        return RollbackStatusResponse(
            production_model_id=prod_id,
            production_model_version=prod_ver,
            watchdog_window_size=500,
            max_degradation_ratio=1.25,
            max_latency_ms=20.0,
            max_allowed_exceptions=0,
            pre_promotion_baseline_mae_sec=38.77,
            rollback_history=history,
        )

    def _load_benchmark_summary(self) -> Dict[str, Any]:
        if self.benchmark_summary_path.exists():
            try:
                with open(self.benchmark_summary_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading benchmark summary: {e}")
        return {}

    def _get_recent_lifecycle_events(self, summary_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        events = []
        if "C" in summary_data:
            events.append({
                "phase": "EXPERIMENT_C",
                "name": "Candidate Retraining & Promotion",
                "result": summary_data["C"].get("status"),
                "promoted_model": "model_lightgbm_v1.1.0",
            })
        if "D" in summary_data:
            events.append({
                "phase": "EXPERIMENT_D",
                "name": "Bad Candidate Rejection Gate",
                "result": summary_data["D"].get("status"),
                "rejected_model": "model_lightgbm_v1.1.0_bad_candidate",
            })
        if "E" in summary_data:
            events.append({
                "phase": "EXPERIMENT_E",
                "name": "State Machine Transition Safety Audit",
                "result": summary_data["E"].get("status"),
            })
        if "F" in summary_data:
            events.append({
                "phase": "EXPERIMENT_F",
                "name": "Automated Watchdog Rollback to Champion",
                "result": summary_data["F"].get("status"),
                "restored_model": "model_lightgbm_v1 (v1.0.0)",
            })
        return events
