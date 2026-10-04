"""TransitVision AI - Dual-Holdout Evaluator & Metrics Computation.
Evaluates champion and challenger candidate models on independent Original Holdout
and Recent Holdout datasets.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS, TARGET_COLUMN
from ml.training.leak_guard import LeakageGuard

logger = logging.getLogger("TransitVision.Validator")


@dataclass
class EvaluationMetrics:
    """Standardized performance and safety metrics for an evaluation dataset."""
    dataset_name: str
    samples: int
    mae_sec: float
    rmse_sec: float
    median_ae_sec: float
    p90_error_sec: float
    p95_error_sec: float
    p99_error_sec: float
    r2: float
    mape_pct: float
    smape_pct: float
    non_positive_predictions_count: int
    non_positive_predictions_pct: float
    excessive_predictions_count: int
    excessive_predictions_pct: float
    inference_latency_ms: float
    leakage_guard_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "samples": self.samples,
            "mae_sec": round(self.mae_sec, 2),
            "rmse_sec": round(self.rmse_sec, 2),
            "median_ae_sec": round(self.median_ae_sec, 2),
            "p90_error_sec": round(self.p90_error_sec, 2),
            "p95_error_sec": round(self.p95_error_sec, 2),
            "p99_error_sec": round(self.p99_error_sec, 2),
            "r2": round(self.r2, 4),
            "mape_pct": round(self.mape_pct, 2),
            "smape_pct": round(self.smape_pct, 2),
            "non_positive_predictions_count": self.non_positive_predictions_count,
            "non_positive_predictions_pct": round(self.non_positive_predictions_pct, 3),
            "excessive_predictions_count": self.excessive_predictions_count,
            "excessive_predictions_pct": round(self.excessive_predictions_pct, 3),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "leakage_guard_passed": self.leakage_guard_passed,
        }


class DualHoldoutValidator:
    """
    Evaluates models across both independent holdout partitions.
    """

    def __init__(self):
        self.leak_guard = LeakageGuard()

    def evaluate_model(
        self,
        model,
        preprocessor,
        eval_df: pd.DataFrame,
        dataset_name: str = "Evaluation Set",
    ) -> Tuple[EvaluationMetrics, np.ndarray]:
        """
        Executes complete evaluation on a dataset with safety and sanity checks.
        """
        if eval_df.empty:
            raise ValueError(f"Cannot evaluate on empty dataset: {dataset_name}")

        feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in eval_df.columns]
        self.leak_guard.validate_schema(feature_cols, dataset_name=f"{dataset_name} Evaluation")

        X = eval_df[feature_cols].copy()
        
        # Determine actual ground truth column
        if TARGET_COLUMN in eval_df.columns:
            y_true = eval_df[TARGET_COLUMN].values
        elif "actual_eta_sec" in eval_df.columns:
            y_true = eval_df["actual_eta_sec"].values
        else:
            raise KeyError(f"Target column '{TARGET_COLUMN}' or 'actual_eta_sec' not found in evaluation dataset.")

        model_name = type(model).__name__.lower()
        is_tree = "ridge" not in model_name and "linear" not in model_name

        start_time = time.perf_counter()
        if hasattr(preprocessor, "transform"):
            X_proc = preprocessor.transform(X, is_tree=is_tree)
        elif is_tree and hasattr(preprocessor, "transform_for_trees"):
            X_proc = preprocessor.transform_for_trees(X)
        else:
            X_proc = preprocessor.transform_for_linear(X)

        y_pred = model.predict(X_proc)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        errors = np.abs(y_true - y_pred)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        median_ae = float(np.median(errors))
        p90 = float(np.percentile(errors, 90))
        p95 = float(np.percentile(errors, 95))
        p99 = float(np.percentile(errors, 99))

        # MAPE and SMAPE on targets >= 10.0s
        valid_mask = y_true >= 10.0
        if np.any(valid_mask):
            mape = float(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100.0)
            denom = np.abs(y_true[valid_mask]) + np.abs(y_pred[valid_mask])
            smape = float(np.mean(2.0 * np.abs(y_pred[valid_mask] - y_true[valid_mask]) / np.where(denom == 0, 1.0, denom)) * 100.0)
        else:
            mape, smape = 0.0, 0.0

        non_positive_count = int(np.sum(y_pred <= 0.0))
        excessive_count = int(np.sum(y_pred > 3600.0))
        total = len(y_pred)

        metrics = EvaluationMetrics(
            dataset_name=dataset_name,
            samples=total,
            mae_sec=mae,
            rmse_sec=rmse,
            median_ae_sec=median_ae,
            p90_error_sec=p90,
            p95_error_sec=p95,
            p99_error_sec=p99,
            r2=r2,
            mape_pct=mape,
            smape_pct=smape,
            non_positive_predictions_count=non_positive_count,
            non_positive_predictions_pct=(non_positive_count / total) * 100.0 if total > 0 else 0.0,
            excessive_predictions_count=excessive_count,
            excessive_predictions_pct=(excessive_count / total) * 100.0 if total > 0 else 0.0,
            inference_latency_ms=latency_ms,
            leakage_guard_passed=True,
        )
        return metrics, y_pred
