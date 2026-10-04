"""TransitVision AI - Production ETA Inference Interface.
Exposes clean, schema-validated ETA prediction service for real-time bus telemetry
with zero-downtime atomic hot-swapping and model lifecycle reload capabilities.
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import joblib
import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, METADATA_DIR
from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS
from ml.training.leak_guard import LeakageGuard

logger = logging.getLogger("TransitVision.Inference")


class ETAPredictor:
    """Production segment ETA inference service with thread-safe atomic hot swap."""

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self._swap_lock = threading.Lock()
        self.leak_guard = LeakageGuard()
        self.model_dir = Path(model_dir) if model_dir else self._find_production_model_dir()
        self.model = None
        self.preprocessor = None
        self.feature_config = None
        self.metadata = None
        self.is_tree_model = True
        self.expected_features: List[str] = ALL_FEATURE_COLUMNS
        self._load_artifacts()

    def _find_production_model_dir(self) -> Path:
        """Finds production model directory from metadata registry or fallback."""
        reg_path = METADATA_DIR / "model_registry.json"
        if reg_path.exists():
            try:
                with open(reg_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                prod_id = reg.get("production_model_id")
                for m in reg.get("models", []):
                    if m.get("model_id") == prod_id and "artifact_dir" in m and m["artifact_dir"]:
                        p = Path(m["artifact_dir"])
                        if p.exists():
                            return p
            except Exception as e:
                logger.warning(f"Could not read registry for production model: {e}")

        # Fallback to default eta_model_v1.0.0 or eta_model_v1
        for candidate in ["eta_model_v1.0.0", "eta_model_v1"]:
            default_dir = MODELS_DIR / candidate
            if default_dir.exists():
                return default_dir
        return MODELS_DIR / "eta_model_v1.0.0"

    def _load_artifacts(self) -> None:
        """Loads serialized model, preprocessor, and feature configuration once at startup."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        model_path = self.model_dir / "model.joblib"
        preproc_path = self.model_dir / "preprocessor.joblib"
        config_path = self.model_dir / "feature_config.json"
        meta_path = self.model_dir / "metadata.json"

        if not model_path.exists() or not preproc_path.exists():
            raise FileNotFoundError(f"Missing model or preprocessor artifacts in {self.model_dir}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preproc_path)

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.feature_config = json.load(f)
                if "all_features" in self.feature_config:
                    self.expected_features = self.feature_config["all_features"]

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        algo = self.metadata.get("algorithm", "").lower() if self.metadata else ""
        self.is_tree_model = "ridge" not in algo and "linear" not in algo

        # Schema-level zero leakage validation executed ONCE at initialization
        self.leak_guard.validate_schema(self.expected_features, dataset_name="Production Predictor Schema")
        logger.info(f"Loaded ETA model: {self.metadata.get('model_id', 'unknown')} (algo: {algo}) from {self.model_dir}")

    def hot_swap_model(self, new_model_dir: Union[str, Path]) -> bool:
        """
        Executes zero-downtime atomic hot swap of the in-memory model.
        
        Sequence:
        1. Load new artifacts separately into temporary containers.
        2. Validate artifact completeness, schema compatibility, and leakage.
        3. Preprocess and perform warmup prediction on synthetic sample.
        4. Atomically swap in-memory references under thread lock.
        5. If ANY step fails, current champion remains untouched.
        """
        target_dir = Path(new_model_dir)
        logger.info(f"Attempting atomic hot swap to candidate at {target_dir}")

        if not target_dir.exists():
            logger.error(f"Hot swap aborted: Target model directory does not exist: {target_dir}")
            return False

        model_path = target_dir / "model.joblib"
        preproc_path = target_dir / "preprocessor.joblib"
        config_path = target_dir / "feature_config.json"
        meta_path = target_dir / "metadata.json"

        if not model_path.exists() or not preproc_path.exists():
            logger.error(f"Hot swap aborted: Missing required model.joblib or preprocessor.joblib in {target_dir}")
            return False

        try:
            # 1. Load candidate artifacts separately
            new_model = joblib.load(model_path)
            new_preproc = joblib.load(preproc_path)
            new_config = None
            new_features = ALL_FEATURE_COLUMNS
            new_metadata = None

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    new_config = json.load(f)
                    if "all_features" in new_config:
                        new_features = new_config["all_features"]

            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    new_metadata = json.load(f)

            # 2. Schema and leakage validation
            self.leak_guard.validate_schema(new_features, dataset_name=f"HotSwap Candidate {target_dir.name}")

            algo = new_metadata.get("algorithm", "").lower() if new_metadata else ""
            is_tree = "ridge" not in algo and "linear" not in algo

            # 3. Warmup inference
            sample_dict = {feat: 1.0 for feat in new_features}
            sample_df = pd.DataFrame([sample_dict])
            if is_tree:
                X_warmup = new_preproc.transform_for_trees(sample_df)
            else:
                X_warmup = new_preproc.transform_for_linear(sample_df)
            warmup_pred = new_model.predict(X_warmup)
            assert warmup_pred is not None, "Warmup prediction returned None."

            # 4. Atomic Reference Swap under Lock
            with self._swap_lock:
                self.model = new_model
                self.preprocessor = new_preproc
                self.feature_config = new_config
                self.metadata = new_metadata
                self.expected_features = new_features
                self.is_tree_model = is_tree
                self.model_dir = target_dir

            logger.info(
                f"HOT SWAP SUCCESS: In-memory model swapped to {self.metadata.get('model_id', 'unknown')} "
                f"(version: {self.metadata.get('version', 'unknown')}) from {target_dir}"
            )
            return True

        except Exception as e:
            logger.error(f"HOT SWAP FAILED: {e}. Active champion retained without downtime.", exc_info=True)
            return False

    def reload_model(self, new_model_dir: Optional[Union[str, Path]] = None) -> bool:
        """Reloads active production model from registry or specified directory."""
        target_dir = Path(new_model_dir) if new_model_dir else self._find_production_model_dir()
        return self.hot_swap_model(target_dir)

    def predict(self, features: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Executes inference on one or multiple bus telemetry records under swap lock safety.
        """
        start_time = time.perf_counter()

        if isinstance(features, dict):
            df = pd.DataFrame([features])
            single_item = True
        elif isinstance(features, list):
            df = pd.DataFrame(features)
            single_item = False
        elif isinstance(features, pd.DataFrame):
            df = features
            single_item = len(df) == 1
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

        # 1. Fast per-request leakage & missing features validation (silent unless error)
        self.leak_guard.validate_features(
            df,
            dataset_name="Inference Request",
            log_success=False,
            required_features=self.expected_features,
        )

        with self._swap_lock:
            active_model = self.model
            active_preprocessor = self.preprocessor
            is_tree = self.is_tree_model
            meta = self.metadata

        # 2. Preprocess
        if is_tree:
            X_trans = active_preprocessor.transform_for_trees(df)
        else:
            X_trans = active_preprocessor.transform_for_linear(df)

        # 3. Model Inference
        raw_preds = active_model.predict(X_trans)

        # Enforce minimum physical travel time of 3 seconds
        clipped_preds = np.clip(raw_preds, 3.0, 3600.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if single_item:
            res_pred = round(float(clipped_preds[0]), 2)
            raw_pred_val = round(float(raw_preds[0]), 2)
        else:
            res_pred = [round(float(p), 2) for p in clipped_preds]
            raw_pred_val = [round(float(p), 2) for p in raw_preds]

        return {
            "predicted_eta_sec": res_pred,
            "raw_prediction_sec": raw_pred_val,
            "model_id": meta.get("model_id", "eta_model_v1") if meta else "eta_model_v1",
            "model_version": meta.get("version", "v1.0.0") if meta else "v1.0.0",
            "algorithm": meta.get("algorithm", "LightGBMRegressor") if meta else "LightGBMRegressor",
            "latency_ms": round(elapsed_ms, 3),
            "records_processed": len(df),
            "status": "SUCCESS",
        }

    def health_check(self) -> Dict[str, Any]:
        """Validates model operational readiness."""
        with self._swap_lock:
            return {
                "status": "HEALTHY" if self.model is not None else "UNINITIALIZED",
                "model_dir": str(self.model_dir),
                "model_id": self.metadata.get("model_id") if self.metadata else None,
                "model_version": self.metadata.get("version") if self.metadata else None,
                "is_tree_model": self.is_tree_model,
            }


def predict_eta(features: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Top-level functional inference interface."""
    predictor = ETAPredictor()
    return predictor.predict(features)
