"""TransitVision AI - Sequential Residual & Concept Drift Monitor.
Implements Page-Hinkley cumulative sum and ADWIN adaptive windowing
detectors to identify structural shifts in the P(Y|X) conditional relationship.
"""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from monitoring.drift_models import (
    DriftType,
    DriftSeverity,
    ResidualDriftResult,
    DriftConfig
)

logger = logging.getLogger("TransitVision.ResidualMonitor")


class PageHinkleyDetector:
    """
    Standardized Page-Hinkley sequential change-point detector for residual shift.
    
    Mathematical Formulation:
        z_t = (|e_t| - reference_mae) / reference_std
        U_t = alpha * U_{t-1} + (z_t - delta)
        m_t = min_{1 <= i <= t} U_i
        PH_t = U_t - m_t
        Alarm if PH_t > lambda (resets cumulative sum upon alarm to identify distinct shift points)
    """

    def __init__(
        self,
        reference_mae: float = 38.77,
        reference_std: float = 46.85,
        delta: float = 0.25,
        threshold_lambda: float = 80.0,
        alpha: float = 0.999
    ):
        self.ref_mae = float(reference_mae)
        self.ref_std = float(max(reference_std, 5.0))
        self.delta = float(delta)
        self.threshold_lambda = float(threshold_lambda)
        self.alpha = float(alpha)
        
        self.n_samples = 0
        self.sum_deviation = 0.0
        self.min_sum_deviation = 0.0
        self.ph_statistic = 0.0
        self.drift_detected = False
        self.last_drift_index: Optional[int] = None

    def update(self, residual_error: float, sample_idx: int) -> Tuple[bool, float, DriftSeverity]:
        """
        Processes single incoming error residual and updates detector state.
        
        Returns:
            (is_drift, ph_statistic, severity)
        """
        abs_err = float(abs(residual_error))
        # Standardized deviation from reference expected error
        z_t = (abs_err - self.ref_mae) / self.ref_std
        self.n_samples += 1

        # Update cumulative sum with slight dampening
        self.sum_deviation = (self.alpha * self.sum_deviation) + (z_t - self.delta)
        if self.sum_deviation < self.min_sum_deviation:
            self.min_sum_deviation = self.sum_deviation

        self.ph_statistic = max(0.0, self.sum_deviation - self.min_sum_deviation)

        is_drift = False
        severity = DriftSeverity.NONE

        if self.ph_statistic >= (self.threshold_lambda * 1.5):
            is_drift = True
            severity = DriftSeverity.CRITICAL
            self.drift_detected = True
            self.last_drift_index = sample_idx
            self.sum_deviation = 0.0
            self.min_sum_deviation = 0.0
        elif self.ph_statistic >= self.threshold_lambda:
            is_drift = True
            severity = DriftSeverity.HIGH
            self.drift_detected = True
            self.last_drift_index = sample_idx
            self.sum_deviation = 0.0
            self.min_sum_deviation = 0.0
        elif self.ph_statistic >= (self.threshold_lambda * 0.6):
            severity = DriftSeverity.MEDIUM

        return is_drift, round(self.ph_statistic, 2), severity

    def reset(self) -> None:
        """Resets sequential state after change-point remediation."""
        self.n_samples = 0
        self.sum_deviation = 0.0
        self.min_sum_deviation = 0.0
        self.ph_statistic = 0.0
        self.drift_detected = False


class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) detector for mean shifts in sequential data stream.
    Splits the dynamic window into sub-windows and tests for statistical difference.
    """

    def __init__(self, delta: float = 0.001, max_window: int = 1000):
        self.delta = float(delta)
        self.max_window = max_window
        self.window: List[float] = []
        self.n_samples = 0
        self.drift_detected = False
        self.last_drift_index: Optional[int] = None
        self.cooldown = 0

    def update(self, val: float, sample_idx: int) -> Tuple[bool, float, DriftSeverity]:
        """Adds observation and checks for sub-window distribution split."""
        self.window.append(float(abs(val)))
        self.n_samples += 1

        if len(self.window) > self.max_window:
            self.window.pop(0)

        is_drift = False
        stat = 0.0
        severity = DriftSeverity.NONE

        if self.cooldown > 0:
            self.cooldown -= 1
            return False, 0.0, DriftSeverity.NONE

        if len(self.window) >= 200:
            mid = len(self.window) // 2
            w0 = np.array(self.window[:mid])
            w1 = np.array(self.window[mid:])

            n0 = len(w0)
            n1 = len(w1)
            mean0 = np.mean(w0)
            mean1 = np.mean(w1)
            diff = abs(mean0 - mean1)

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            var = np.var(self.window)
            eps_cut = math.sqrt((2.0 / m) * math.log(2.0 / self.delta)) * (math.sqrt(var) + 1.0)

            stat = float(diff)
            # Require statistical significance AND meaningful error increase (>15s)
            if diff > eps_cut and mean1 > mean0 and diff >= 15.0:
                is_drift = True
                severity = DriftSeverity.HIGH if diff > (1.5 * eps_cut) else DriftSeverity.MEDIUM
                self.drift_detected = True
                self.last_drift_index = sample_idx
                self.window = self.window[mid:]
                self.cooldown = 100

        return is_drift, round(stat, 2), severity

    def reset(self) -> None:
        self.window.clear()
        self.n_samples = 0
        self.drift_detected = False
        self.cooldown = 0


class ResidualConceptMonitor:
    """Orchestrates sequential concept drift monitoring over prediction residuals."""

    def __init__(self, config: Optional[DriftConfig] = None, reference_mae: float = 38.77, reference_rmse: float = 60.83):
        self.config = config or DriftConfig()
        self.ph_detector = PageHinkleyDetector(
            reference_mae=reference_mae,
            reference_std=np.sqrt(max(reference_rmse**2 - reference_mae**2, 25.0)),
            delta=self.config.ph_delta,
            threshold_lambda=self.config.ph_lambda
        )
        self.adwin_detector = ADWINDetector(delta=self.config.adwin_delta)
        self.sample_count = 0
        self.residuals_buffer: List[float] = []

    def update(self, residual_error: float, sample_idx: int) -> List[ResidualDriftResult]:
        """Updates both sequential detectors on the newly resolved residual."""
        self.sample_count += 1
        self.residuals_buffer.append(float(residual_error))
        if len(self.residuals_buffer) > 500:
            self.residuals_buffer.pop(0)

        ph_drift, ph_stat, ph_sev = self.ph_detector.update(residual_error, sample_idx)
        ad_drift, ad_stat, ad_sev = self.adwin_detector.update(residual_error, sample_idx)

        res_list = []
        mean_res = float(np.mean(self.residuals_buffer))
        var_res = float(np.var(self.residuals_buffer))

        # Page-Hinkley Result
        res_list.append(ResidualDriftResult(
            detector_name="PageHinkley",
            statistic=ph_stat,
            threshold=self.config.ph_lambda,
            in_warning=(ph_sev in [DriftSeverity.LOW, DriftSeverity.MEDIUM]),
            in_drift=ph_drift,
            severity=ph_sev,
            sample_count=self.sample_count,
            mean_residual=round(mean_res, 2),
            variance_residual=round(var_res, 2),
            detection_index=self.ph_detector.last_drift_index if ph_drift else None,
            metadata={"delta": self.config.ph_delta}
        ))

        # ADWIN Result
        res_list.append(ResidualDriftResult(
            detector_name="ADWIN",
            statistic=ad_stat,
            threshold=round(self.config.adwin_delta, 4),
            in_warning=(ad_sev in [DriftSeverity.LOW, DriftSeverity.MEDIUM]),
            in_drift=ad_drift,
            severity=ad_sev,
            sample_count=self.sample_count,
            mean_residual=round(mean_res, 2),
            variance_residual=round(var_res, 2),
            detection_index=self.adwin_detector.last_drift_index if ad_drift else None
        ))

        return res_list

    def reset(self) -> None:
        self.ph_detector.reset()
        self.adwin_detector.reset()
        self.sample_count = 0
        self.residuals_buffer.clear()
