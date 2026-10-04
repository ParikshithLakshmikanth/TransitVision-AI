"""TransitVision AI - Candidate Model Trainer.
Trains candidate challenger models using the verified production feature pipeline
and stores candidate artifacts strictly in models/candidates/<version>/.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from config.settings import MODELS_DIR
from ml.preprocessing.pipeline_preprocessor import (
    TabularPreprocessor,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
)
from ml.training.leak_guard import LeakageGuard
from retraining.dataset_builder import DatasetProvenanceManifest

logger = logging.getLogger("TransitVision.CandidateTrainer")


def hash_artifact_file(path: Path) -> str:
    """Computes SHA-256 of an artifact file."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class CandidateTrainer:
    """
    Trains candidate challenger models with full reproducibility, feature schema isolation,
    and provenance tracking.
    """

    def __init__(self, candidates_root_dir: Optional[Path] = None):
        self.candidates_root_dir = candidates_root_dir or (MODELS_DIR / "candidates")
        self.candidates_root_dir.mkdir(parents=True, exist_ok=True)
        self.leak_guard = LeakageGuard()

    def train_candidate(
        self,
        candidate_train_df,
        candidate_version: str,
        parent_model_id: str,
        provenance_manifest: DatasetProvenanceManifest,
        algorithm: str = "LightGBMRegressor",
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Trains and persists candidate model artifacts into models/candidates/<version>/.
        
        Args:
            candidate_train_df: Combined historical anchor + stream training records.
            candidate_version: Version identifier (e.g., 'v1.1.0').
            parent_model_id: Model ID of current production champion.
            provenance_manifest: Dataset provenance record.
            algorithm: Algorithm class name.
            hyperparameters: Optional dictionary of hyperparameters.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with candidate metadata, artifact directory, and artifact hashes.
        """
        # Ensure candidate directory is inside models/candidates/ and not models/eta_model_v1.0.0
        candidate_dir_name = f"eta_model_{candidate_version}"
        candidate_dir = self.candidates_root_dir / candidate_dir_name
        candidate_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initiating candidate training for {candidate_version} ({algorithm}) in {candidate_dir}")

        # 1. Feature / Target Separation & Leakage Guard
        feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in candidate_train_df.columns]
        self.leak_guard.validate_schema(feature_cols, dataset_name=f"Candidate {candidate_version} Training Features")

        X_train = candidate_train_df[feature_cols].copy()
        y_train = candidate_train_df[TARGET_COLUMN].copy()

        # 2. Determine model type and instantiate
        is_tree = "ridge" not in algorithm.lower() and "linear" not in algorithm.lower()

        if algorithm == "LightGBMRegressor":
            params = {
                "n_estimators": 180,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": random_seed,
                "verbose": -1,
            }
            if hyperparameters:
                params.update(hyperparameters)
            model = lgb.LGBMRegressor(**params)
        elif algorithm == "RandomForestRegressor":
            params = {
                "n_estimators": 100,
                "max_depth": 16,
                "min_samples_split": 10,
                "random_state": random_seed,
            }
            if hyperparameters:
                params.update(hyperparameters)
            model = RandomForestRegressor(**params)
        elif algorithm == "Ridge":
            params = {
                "alpha": 10.0,
                "random_state": random_seed,
            }
            if hyperparameters:
                params.update(hyperparameters)
            model = Ridge(**params)
        else:
            raise ValueError(f"Unsupported candidate algorithm: {algorithm}")

        # 3. Fit Preprocessor and Model
        preprocessor = TabularPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train, is_tree=is_tree)
        model.fit(X_train_proc, y_train)

        # 4. Save Artifacts
        model_path = candidate_dir / "model.joblib"
        preproc_path = candidate_dir / "preprocessor.joblib"
        config_path = candidate_dir / "feature_config.json"
        prov_path = candidate_dir / "dataset_provenance.json"
        meta_path = candidate_dir / "metadata.json"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preproc_path)

        feature_config = {
            "all_features": feature_cols,
            "categorical_features": [c for c in CATEGORICAL_COLUMNS if c in feature_cols],
            "numerical_features": [c for c in NUMERICAL_COLUMNS if c in feature_cols],
            "target_column": TARGET_COLUMN,
            "feature_version": "v1.0.0",
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(feature_config, f, indent=2)

        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance_manifest.to_dict(), f, indent=2)

        metadata = {
            "model_id": f"model_lightgbm_{candidate_version}" if "LightGBM" in algorithm else f"model_{algorithm.lower()}_{candidate_version}",
            "version": candidate_version,
            "algorithm": algorithm,
            "parent_model_id": parent_model_id,
            "target": TARGET_COLUMN,
            "status": "CANDIDATE",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(candidate_train_df),
            "validation_rows": provenance_manifest.original_validation_rows,
            "hyperparameters": getattr(model, "get_params", lambda: params)(),
            "artifact_dir": str(candidate_dir),
            "dataset_provenance_hash": provenance_manifest.candidate_train_data_hash,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5. Compute Artifact SHA-256 Hashes
        artifact_hashes = {
            "model.joblib": hash_artifact_file(model_path),
            "preprocessor.joblib": hash_artifact_file(preproc_path),
            "feature_config.json": hash_artifact_file(config_path),
            "dataset_provenance.json": hash_artifact_file(prov_path),
            "metadata.json": hash_artifact_file(meta_path),
        }
        metadata["artifact_hashes"] = artifact_hashes

        logger.info(f"Candidate {candidate_version} successfully trained and serialized. Artifacts stored in {candidate_dir}")
        return {
            "model_id": metadata["model_id"],
            "version": candidate_version,
            "algorithm": algorithm,
            "parent_model_id": parent_model_id,
            "artifact_dir": str(candidate_dir),
            "artifact_hashes": artifact_hashes,
            "metadata": metadata,
            "model": model,
            "preprocessor": preprocessor,
        }
