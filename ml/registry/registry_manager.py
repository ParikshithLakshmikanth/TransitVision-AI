"""TransitVision AI - Model Registry and Lifecycle Manager.
Tracks candidate models, executes automated performance gating, manages versioning,
and manages promotion to PRODUCTION.
"""
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib

logger = logging.getLogger("TransitVision.ModelRegistry")


class ModelRegistryManager:
    """Manages model artifacts, versioning, metadata, and production status transitions."""

    def __init__(self, registry_file: Path, models_dir: Path):
        self.registry_file = Path(registry_file)
        self.models_dir = Path(models_dir)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.registry_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing registry: {e}. Initializing clean registry.")
                self.registry_data = {"models": [], "production_model_id": None}
        else:
            self.registry_data = {"models": [], "production_model_id": None}

    def _save_registry(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry_data, f, indent=2)
        logger.info(f"Model registry updated: {self.registry_file}")

    def register_model(
        self,
        model_id: str,
        version: str,
        algorithm: str,
        metrics: Dict[str, Any],
        training_rows: int,
        validation_rows: int,
        hyperparameters: Dict[str, Any],
        feature_version: str = "v1.0.0",
        target: str = "eta_to_next_stop_sec",
        model_artifact_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registers a new model candidate in the registry."""
        entry = {
            "model_id": model_id,
            "version": version,
            "algorithm": algorithm,
            "target": target,
            "status": "CANDIDATE",
            "registered_at": pd_now_iso(),
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "feature_version": feature_version,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "artifact_path": model_artifact_path
        }
        
        # Replace if model_id already exists, else append
        existing_idx = next((i for i, m in enumerate(self.registry_data["models"]) if m["model_id"] == model_id), -1)
        if existing_idx >= 0:
            self.registry_data["models"][existing_idx] = entry
        else:
            self.registry_data["models"].append(entry)
            
        self._save_registry()
        logger.info(f"Registered model {model_id} ({algorithm}) as CANDIDATE.")
        return entry

    def evaluate_performance_gate(
        self,
        model_id: str,
        max_mae_sec: float = 48.0,
        min_r2: float = 0.65,
        max_p95_sec: float = 160.0
    ) -> Tuple[bool, str]:
        """
        Validates model against production performance gate thresholds.
        """
        model_entry = next((m for m in self.registry_data["models"] if m["model_id"] == model_id), None)
        if not model_entry:
            return False, f"Model {model_id} not found in registry."

        metrics = model_entry.get("metrics", {})
        mae = metrics.get("mae_sec", float("inf"))
        r2 = metrics.get("r2", -float("inf"))
        p95 = metrics.get("p95_error_sec", float("inf"))

        checks = []
        if mae > max_mae_sec:
            checks.append(f"MAE ({mae:.2f}s) exceeded limit ({max_mae_sec}s)")
        if r2 < min_r2:
            checks.append(f"R² ({r2:.4f}) below limit ({min_r2})")
        if p95 > max_p95_sec:
            checks.append(f"P95 error ({p95:.2f}s) exceeded limit ({max_p95_sec}s)")

        if checks:
            reason = "Performance gate failed: " + "; ".join(checks)
            logger.warning(f"Gate check failed for {model_id}: {reason}")
            return False, reason

        logger.info(f"Model {model_id} PASSED performance gate (MAE: {mae:.2f}s, R²: {r2:.4f}, P95: {p95:.2f}s).")
        return True, "Passed all performance gate criteria."

    def promote_to_production(
        self,
        model_id: str,
        model_obj: Any,
        preprocessor_obj: Any,
        feature_config: Dict[str, Any]
    ) -> Path:
        """
        Promotes a validated model to PRODUCTION status, archives any existing PRODUCTION model,
        and saves complete versioned artifacts.
        """
        model_entry = next((m for m in self.registry_data["models"] if m["model_id"] == model_id), None)
        if not model_entry:
            raise ValueError(f"Model {model_id} not found in registry.")

        # Demote previous PRODUCTION model to ARCHIVED
        for m in self.registry_data["models"]:
            if m["status"] == "PRODUCTION" and m["model_id"] != model_id:
                m["status"] = "ARCHIVED"
                m["archived_at"] = pd_now_iso()

        # Update target model
        model_entry["status"] = "PRODUCTION"
        model_entry["promoted_to_production_at"] = pd_now_iso()
        self.registry_data["production_model_id"] = model_id

        # Save artifacts under versioned model directory
        version_dir = self.models_dir / f"eta_model_{model_entry['version']}"
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.joblib"
        preproc_path = version_dir / "preprocessor.joblib"
        config_path = version_dir / "feature_config.json"
        meta_path = version_dir / "metadata.json"

        joblib.dump(model_obj, model_path)
        joblib.dump(preprocessor_obj, preproc_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(feature_config, f, indent=2)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(model_entry, f, indent=2)

        model_entry["artifact_dir"] = str(version_dir)
        self._save_registry()

        logger.info(f"Model {model_id} successfully promoted to PRODUCTION at {version_dir}")
        return version_dir

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """Returns metadata of currently active production model."""
        prod_id = self.registry_data.get("production_model_id")
        if not prod_id:
            return None
        return next((m for m in self.registry_data["models"] if m["model_id"] == prod_id), None)


def pd_now_iso() -> str:
    import pandas as pd
    return pd.Timestamp.now(tz="UTC").isoformat()


TupleBoolReason = tuple[bool, str]
