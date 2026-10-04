"""TransitVision AI - Model Evaluator.
Calculates rigorous regression error metrics in seconds, percentile errors,
and sanity distribution checks on validation sets.
"""
import logging
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

logger = logging.getLogger("TransitVision.Evaluator")


class ModelEvaluator:
    """Computes comprehensive regression evaluation metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        min_mape_target_sec: float = 10.0
    ) -> Dict[str, Any]:
        """
        Calculates all standard and robust error metrics for segment ETA regression.
        
        Args:
            y_true: Ground truth observed segment travel times in seconds.
            y_pred: Model predicted segment travel times in seconds.
            min_mape_target_sec: Threshold for stable MAPE calculation.
            
        Returns:
            Dictionary containing metrics in seconds and percentage forms.
        """
        y_true_arr = np.asarray(y_true, dtype=np.float64)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)

        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(f"Shape mismatch: y_true ({len(y_true_arr)}) vs y_pred ({len(y_pred_arr)})")

        # Absolute errors
        abs_errors = np.abs(y_true_arr - y_pred_arr)
        
        # 1. Standard Regression Metrics
        mae_sec = float(mean_absolute_error(y_true_arr, y_pred_arr))
        rmse_sec = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        r2 = float(r2_score(y_true_arr, y_pred_arr))
        median_ae_sec = float(median_absolute_error(y_true_arr, y_pred_arr))

        # 2. Percentile Absolute Errors
        p90_error_sec = float(np.percentile(abs_errors, 90))
        p95_error_sec = float(np.percentile(abs_errors, 95))
        p99_error_sec = float(np.percentile(abs_errors, 99))

        # 3. MAPE calculation (Safe: targets >= min_mape_target_sec to prevent division by near-zero)
        valid_mape_mask = y_true_arr >= min_mape_target_sec
        if np.any(valid_mape_mask):
            mape = float(np.mean(abs_errors[valid_mape_mask] / y_true_arr[valid_mape_mask])) * 100.0
        else:
            mape = float(np.nan)

        # 4. Symmetric MAPE (sMAPE): 200 * |y - y_hat| / (|y| + |y_hat|)
        denominator = np.abs(y_true_arr) + np.abs(y_pred_arr)
        nonzero_mask = denominator > 1e-6
        if np.any(nonzero_mask):
            smape = float(np.mean(2.0 * abs_errors[nonzero_mask] / denominator[nonzero_mask])) * 100.0
        else:
            smape = 0.0

        # 5. Sanity Checks on Physical Plausibility
        total_samples = len(y_pred_arr)
        non_positive_count = int(np.sum(y_pred_arr <= 0))
        excessive_count = int(np.sum(y_pred_arr >= 3600.0))  # Segment ETA > 1 hour is implausible

        metrics = {
            "samples": total_samples,
            "mae_sec": round(mae_sec, 2),
            "rmse_sec": round(rmse_sec, 2),
            "median_ae_sec": round(median_ae_sec, 2),
            "p90_error_sec": round(p90_error_sec, 2),
            "p95_error_sec": round(p95_error_sec, 2),
            "p99_error_sec": round(p99_error_sec, 2),
            "r2": round(r2, 4),
            "mape_pct": round(mape, 2),
            "smape_pct": round(smape, 2),
            "mape_definition": f"mean(|y_true - y_pred| / y_true) * 100 on targets >= {min_mape_target_sec}s",
            "sanity_checks": {
                "non_positive_predictions_count": non_positive_count,
                "non_positive_predictions_pct": round(non_positive_count / total_samples * 100, 3),
                "excessive_predictions_count": excessive_count,
                "excessive_predictions_pct": round(excessive_count / total_samples * 100, 3)
            }
        }

        return metrics
