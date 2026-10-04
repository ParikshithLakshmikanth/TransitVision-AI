"""TransitVision AI - Rolling Performance Drift Monitor.
Tracks online regression error metrics over sliding windows and flags
statistically meaningful degradation against production reference baselines.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from monitoring.drift_models import (
    DriftType,
    DriftSeverity,
    PerformanceDriftResult,
    DriftConfig
)

logger = logging.getLogger("TransitVision.PerformanceMonitor")


class RollingPerformanceMonitor:
    """Monitors online accuracy metrics (MAE, RMSE, MedAE, P90, P95, Bias) over sliding windows."""

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        reference_mae: float = 38.77,
        reference_rmse: float = 60.83
    ):
        self.config = config or DriftConfig()
        self.reference_mae = float(reference_mae)
        self.reference_rmse = float(reference_rmse)
        
        # Buffer of (y_true, y_pred, record_idx)
        self.history_true: List[float] = []
        self.history_pred: List[float] = []
        self.history_idx: List[int] = []
        
        self.window_count = 0
        self.last_evaluated_idx = 0

    def add_observation(self, y_true: float, y_pred: float, record_idx: int) -> None:
        """Appends resolved ground-truth observation strictly post-prediction."""
        self.history_true.append(float(y_true))
        self.history_pred.append(float(y_pred))
        self.history_idx.append(int(record_idx))

    def evaluate_current_window(self) -> Optional[PerformanceDriftResult]:
        """
        Evaluates the most recent window of predictions if sufficient samples exist.
        """
        n_samples = len(self.history_true)
        if n_samples < self.config.min_window_size:
            return None

        # Take last window_size samples
        win_size = min(self.config.window_size, n_samples)
        y_t = np.array(self.history_true[-win_size:])
        y_p = np.array(self.history_pred[-win_size:])
        start_idx = self.history_idx[-win_size]
        end_idx = self.history_idx[-1]

        signed_errors = y_t - y_p
        abs_errors = np.abs(signed_errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(signed_errors ** 2)))
        med_ae = float(np.median(abs_errors))
        p90 = float(np.percentile(abs_errors, 90))
        p95 = float(np.percentile(abs_errors, 95))
        bias = float(np.mean(signed_errors))

        mae_ratio = mae / max(self.reference_mae, 1e-3)
        rmse_ratio = rmse / max(self.reference_rmse, 1e-3)

        self.window_count += 1
        is_drift = False
        severity = DriftSeverity.NONE

        if mae_ratio >= self.config.perf_critical_multiplier or rmse_ratio >= self.config.perf_critical_multiplier:
            is_drift = True
            severity = DriftSeverity.CRITICAL
        elif mae_ratio >= self.config.perf_drift_multiplier or rmse_ratio >= self.config.perf_drift_multiplier:
            is_drift = True
            severity = DriftSeverity.HIGH
        elif mae_ratio >= self.config.perf_warning_multiplier:
            is_drift = True
            severity = DriftSeverity.MEDIUM

        result = PerformanceDriftResult(
            window_id=self.window_count,
            window_start_idx=start_idx,
            window_end_idx=end_idx,
            sample_count=win_size,
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            median_ae=round(med_ae, 2),
            p90_error=round(p90, 2),
            p95_error=round(p95, 2),
            mean_signed_error=round(bias, 2),
            reference_mae=round(self.reference_mae, 2),
            reference_rmse=round(self.reference_rmse, 2),
            mae_degradation_ratio=round(mae_ratio, 3),
            rmse_degradation_ratio=round(rmse_ratio, 3),
            is_drift=is_drift,
            severity=severity,
            metadata={
                "error_variance": round(float(np.var(signed_errors)), 2),
                "underestimation_rate": round(float(np.mean(signed_errors > 0)), 3)
            }
        )
        return result

    def reset(self) -> None:
        """Clears monitoring history."""
        self.history_true.clear()
        self.history_pred.clear()
        self.history_idx.clear()
        self.window_count = 0
