"""TransitVision AI - Model Registry & Artifact Inspection Service."""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from config.settings import METADATA_DIR, MODELS_DIR
from registry.model_registry import LifecycleModelRegistry, ModelStatus
from backend.schemas import ModelDetail, ModelListResponse

logger = logging.getLogger("TransitVision.ModelService")


class ModelService:
    """
    Read-only service exposing model registry records,
    lineage, hyperparameters, and production champion status.
    """

    def __init__(self, registry: Optional[LifecycleModelRegistry] = None):
        self.registry = registry or LifecycleModelRegistry()

    def list_models(self) -> ModelListResponse:
        """Lists all models in the registry."""
        raw_models = self.registry.list_models()
        prod_model = self.registry.get_production_model()
        prod_id = prod_model.get("model_id", "model_lightgbm_v1") if prod_model else "model_lightgbm_v1"

        models_list = [self._convert_to_model_detail(m) for m in raw_models]
        return ModelListResponse(
            production_model_id=prod_id,
            total_models=len(models_list),
            models=models_list,
        )

    def get_production_model(self) -> Optional[ModelDetail]:
        """Returns the active PRODUCTION model."""
        m = self.registry.get_production_model()
        if not m:
            return None
        return self._convert_to_model_detail(m)

    def get_model_by_id(self, model_id: str) -> Optional[ModelDetail]:
        """Finds model by model_id."""
        m = self.registry.get_model(model_id)
        if not m:
            return None
        return self._convert_to_model_detail(m)

    def _convert_to_model_detail(self, m: Dict[str, Any]) -> ModelDetail:
        return ModelDetail(
            model_id=m["model_id"],
            version=m.get("version", "v1.0.0"),
            algorithm=m.get("algorithm", "Unknown"),
            status=m.get("status", "CANDIDATE"),
            parent_model_id=m.get("parent_model_id"),
            created_at_utc=m.get("created_at_utc"),
            promoted_at_utc=m.get("promoted_at_utc"),
            training_rows=m.get("training_rows"),
            validation_metrics=m.get("validation_metrics"),
            hyperparameters=m.get("hyperparameters"),
            notes=m.get("notes"),
        )
