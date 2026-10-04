"""TransitVision AI - Lifecycle Model Registry.
Manages candidate models, formal lifecycle state machine, transitions,
metadata audit trails, and production champion tracking.
"""
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import METADATA_DIR, MODELS_DIR

logger = logging.getLogger("TransitVision.ModelRegistry")


class ModelStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    SHADOW = "SHADOW"
    CANARY_10 = "CANARY_10"
    CANARY_50 = "CANARY_50"
    PROMOTION_CANDIDATE = "PROMOTION_CANDIDATE"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    ARCHIVED = "ARCHIVED"


# Define strictly permitted forward, failure, and rollback transitions
LEGAL_TRANSITIONS = {
    ModelStatus.CANDIDATE: {ModelStatus.VALIDATED, ModelStatus.REJECTED},
    ModelStatus.VALIDATED: {ModelStatus.SHADOW, ModelStatus.REJECTED},
    ModelStatus.SHADOW: {ModelStatus.CANARY_10, ModelStatus.REJECTED},
    ModelStatus.CANARY_10: {ModelStatus.CANARY_50, ModelStatus.REJECTED},
    ModelStatus.CANARY_50: {ModelStatus.PROMOTION_CANDIDATE, ModelStatus.REJECTED},
    ModelStatus.PROMOTION_CANDIDATE: {ModelStatus.PRODUCTION, ModelStatus.REJECTED},
    ModelStatus.PRODUCTION: {ModelStatus.ARCHIVED, ModelStatus.ROLLED_BACK},
    ModelStatus.REJECTED: set(),
    ModelStatus.ROLLED_BACK: set(),
    ModelStatus.ARCHIVED: {ModelStatus.PRODUCTION},  # May be restored on rollback
}


class LifecycleModelRegistry:
    """
    State machine and persistence registry for all model lifecycle states.
    """

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = Path(registry_file) if registry_file else (METADATA_DIR / "model_registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading registry {self.registry_file}: {e}. Creating default structure.")
                self.data = {"models": [], "production_model_id": "model_lightgbm_v1"}
        else:
            self.data = {"models": [], "production_model_id": "model_lightgbm_v1"}

    def _save(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"Model registry persisted to {self.registry_file}")

    def list_models(self) -> List[Dict[str, Any]]:
        """Retrieves all model entries in the registry."""
        return list(self.data.get("models", []))

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a model entry by ID."""
        for m in self.data.get("models", []):
            if m.get("model_id") == model_id:
                return m
        return None

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """Retrieves the active PRODUCTION model entry."""
        prod_id = self.data.get("production_model_id")
        if prod_id:
            m = self.get_model(prod_id)
            if m and m.get("status") == ModelStatus.PRODUCTION.value:
                return m
        # Fallback to searching status == PRODUCTION
        for m in self.data.get("models", []):
            if m.get("status") == ModelStatus.PRODUCTION.value:
                return m
        return None

    def register_candidate(
        self,
        model_id: str,
        version: str,
        algorithm: str,
        training_rows: int,
        validation_rows: int,
        hyperparameters: Dict[str, Any],
        parent_model_id: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        artifact_hashes: Optional[Dict[str, str]] = None,
        dataset_provenance_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Registers a new model in CANDIDATE state."""
        now_iso = datetime.now(timezone.utc).isoformat()
        entry = {
            "model_id": model_id,
            "version": version,
            "algorithm": algorithm,
            "parent_model_id": parent_model_id,
            "target": "eta_to_next_stop_sec",
            "status": ModelStatus.CANDIDATE.value,
            "registered_at": now_iso,
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "hyperparameters": hyperparameters,
            "metrics": {},
            "artifact_dir": artifact_dir,
            "artifact_hashes": artifact_hashes or {},
            "dataset_provenance_hash": dataset_provenance_hash,
            "lifecycle_history": [
                {
                    "from_status": None,
                    "to_status": ModelStatus.CANDIDATE.value,
                    "timestamp": now_iso,
                    "reason": "Initial candidate registration after training.",
                }
            ],
        }

        # Update existing or append
        existing_idx = next((i for i, m in enumerate(self.data["models"]) if m["model_id"] == model_id), -1)
        if existing_idx >= 0:
            self.data["models"][existing_idx] = entry
        else:
            self.data["models"].append(entry)

        self._save()
        logger.info(f"Registered candidate model {model_id} ({version})")
        return entry

    def transition_status(
        self,
        model_id: str,
        target_status: ModelStatus,
        reason: str,
        metrics_update: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a formal lifecycle state transition with validation.
        """
        entry = self.get_model(model_id)
        if not entry:
            raise KeyError(f"Model ID '{model_id}' not found in registry.")

        current_status_str = entry.get("status", ModelStatus.CANDIDATE.value)
        try:
            current_status = ModelStatus(current_status_str)
        except ValueError:
            current_status = ModelStatus.CANDIDATE

        # Verify legal transition
        allowed = LEGAL_TRANSITIONS.get(current_status, set())
        if target_status not in allowed:
            raise ValueError(
                f"Illegal lifecycle transition for {model_id}: Cannot transition from "
                f"'{current_status.value}' to '{target_status.value}'. Allowed: {[s.value for s in allowed]}"
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        entry["status"] = target_status.value
        if metrics_update:
            entry.setdefault("metrics", {}).update(metrics_update)

        entry.setdefault("lifecycle_history", []).append({
            "from_status": current_status.value,
            "to_status": target_status.value,
            "timestamp": now_iso,
            "reason": reason,
        })

        if target_status == ModelStatus.REJECTED:
            entry["rejected_at"] = now_iso
            entry["rejection_reason"] = reason
        elif target_status == ModelStatus.ROLLED_BACK:
            entry["rolled_back_at"] = now_iso
            entry["rollback_reason"] = reason

        self._save()
        logger.info(f"Transitioned {model_id} from {current_status.value} -> {target_status.value}: {reason}")
        return entry

    def promote_to_production(self, model_id: str, reason: str = "Promotion Gate and Canary Passed") -> Dict[str, Any]:
        """
        Promotes a PROMOTION_CANDIDATE model to active PRODUCTION.
        Archives previous champion model.
        """
        entry = self.get_model(model_id)
        if not entry:
            raise KeyError(f"Model ID '{model_id}' not found in registry.")

        current_prod = self.get_production_model()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Archive previous production champion if different
        if current_prod and current_prod["model_id"] != model_id:
            old_id = current_prod["model_id"]
            current_prod["status"] = ModelStatus.ARCHIVED.value
            current_prod["archived_at"] = now_iso
            current_prod.setdefault("lifecycle_history", []).append({
                "from_status": ModelStatus.PRODUCTION.value,
                "to_status": ModelStatus.ARCHIVED.value,
                "timestamp": now_iso,
                "reason": f"Superseded by promoted model {model_id}.",
            })
            logger.info(f"Archived former production model: {old_id}")

        # Transition candidate to PRODUCTION
        # If currently in PROMOTION_CANDIDATE, validate transition
        self.transition_status(model_id, ModelStatus.PRODUCTION, reason)
        entry["promoted_to_production_at"] = now_iso
        self.data["production_model_id"] = model_id
        self._save()

        logger.info(f"SUCCESS: Model {model_id} is now the active PRODUCTION champion.")
        return entry

    def rollback_production(self, rollback_reason: str) -> Dict[str, Any]:
        """
        Executes atomic rollback of current production model to the previous champion.
        """
        current_prod = self.get_production_model()
        if not current_prod:
            raise RuntimeError("No active production model found to rollback.")

        current_prod_id = current_prod["model_id"]
        parent_id = current_prod.get("parent_model_id")

        if not parent_id:
            # Fallback to model_lightgbm_v1 if no parent specified
            parent_id = "model_lightgbm_v1"

        parent_entry = self.get_model(parent_id)
        if not parent_entry:
            raise RuntimeError(f"Previous champion model {parent_id} not found in registry for rollback.")

        now_iso = datetime.now(timezone.utc).isoformat()

        # Mark current prod as ROLLED_BACK
        self.transition_status(current_prod_id, ModelStatus.ROLLED_BACK, rollback_reason)

        # Restore previous champion to PRODUCTION
        parent_entry["status"] = ModelStatus.PRODUCTION.value
        parent_entry["restored_to_production_at"] = now_iso
        parent_entry.setdefault("lifecycle_history", []).append({
            "from_status": parent_entry.get("status", "ARCHIVED"),
            "to_status": ModelStatus.PRODUCTION.value,
            "timestamp": now_iso,
            "reason": f"Restored to production following rollback of {current_prod_id}.",
        })
        self.data["production_model_id"] = parent_id
        self._save()

        logger.warning(f"ROLLBACK EXECUTED: Restored {parent_id} to PRODUCTION. Failed model {current_prod_id} marked ROLLED_BACK.")
        return parent_entry
