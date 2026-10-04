"""TransitVision AI - Post-Promotion Watchdog & Automated Rollback Manager.
Monitors initial post-promotion serving window and executes atomic rollback
to the previous champion if degradation, excessive latency, or runtime exceptions occur.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from ml.inference.predictor import ETAPredictor
from registry.model_registry import LifecycleModelRegistry, ModelStatus

logger = logging.getLogger("TransitVision.RollbackManager")

DEFAULT_WATCHDOG_WINDOW_SIZE = 500
DEFAULT_MAX_DEGRADATION_RATIO = 1.25    # MAE > 1.25x pre-promotion baseline triggers rollback
DEFAULT_MAX_LATENCY_MS = 20.0           # Inference latency > 20ms triggers rollback
DEFAULT_MAX_ALLOWED_EXCEPTIONS = 0      # Any runtime exception triggers rollback


@dataclass
class WatchdogEvaluationResult:
    """Outcome of the post-promotion watchdog monitoring window."""
    rollback_triggered: bool
    samples_monitored: int
    observed_mae_sec: float
    baseline_mae_sec: float
    mae_ratio: float
    max_allowed_mae_ratio: float
    avg_latency_ms: float
    max_allowed_latency_ms: float
    exceptions_count: int
    active_production_model_id: str
    rollback_reason: Optional[str] = None
    applied_thresholds: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_triggered": self.rollback_triggered,
            "samples_monitored": self.samples_monitored,
            "observed_mae_sec": round(self.observed_mae_sec, 2),
            "baseline_mae_sec": round(self.baseline_mae_sec, 2),
            "mae_ratio": round(self.mae_ratio, 4),
            "max_allowed_mae_ratio": self.max_allowed_mae_ratio,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "max_allowed_latency_ms": self.max_allowed_latency_ms,
            "exceptions_count": self.exceptions_count,
            "active_production_model_id": self.active_production_model_id,
            "rollback_reason": self.rollback_reason,
            "applied_thresholds": self.applied_thresholds,
            "details": self.details,
        }


class RollbackManager:
    """
    Post-promotion watchdog and automated emergency rollback executor.
    """

    def __init__(
        self,
        registry: Optional[LifecycleModelRegistry] = None,
        predictor: Optional[ETAPredictor] = None,
        watchdog_window_size: int = DEFAULT_WATCHDOG_WINDOW_SIZE,
        max_degradation_ratio: float = DEFAULT_MAX_DEGRADATION_RATIO,
        max_latency_ms: float = DEFAULT_MAX_LATENCY_MS,
        max_allowed_exceptions: int = DEFAULT_MAX_ALLOWED_EXCEPTIONS,
    ):
        self.registry = registry or LifecycleModelRegistry()
        self.predictor = predictor or ETAPredictor()
        self.watchdog_window_size = watchdog_window_size
        self.max_degradation_ratio = max_degradation_ratio
        self.max_latency_ms = max_latency_ms
        self.max_allowed_exceptions = max_allowed_exceptions

    def get_configured_thresholds(self) -> Dict[str, Any]:
        """Returns dictionary of currently configured rollback thresholds."""
        return {
            "watchdog_window_size": self.watchdog_window_size,
            "max_degradation_ratio": self.max_degradation_ratio,
            "max_latency_ms": self.max_latency_ms,
            "max_allowed_exceptions": self.max_allowed_exceptions,
        }

    def monitor_and_enforce(
        self,
        evaluation_events: List[Any],
        baseline_mae: float = 38.77,
    ) -> WatchdogEvaluationResult:
        """
        Evaluates post-promotion evaluation events and triggers rollback if degraded.
        """
        current_prod = self.registry.get_production_model()
        if not current_prod:
            raise RuntimeError("No active production model found in registry.")

        current_prod_id = current_prod["model_id"]
        errors = []
        latencies = []
        exceptions_count = 0
        thresholds = self.get_configured_thresholds()

        for event in evaluation_events[:self.watchdog_window_size]:
            if hasattr(event, "error_sec"):
                errors.append(abs(event.error_sec))
            elif hasattr(event, "absolute_error_sec"):
                errors.append(abs(event.absolute_error_sec))
            elif isinstance(event, dict) and "error_sec" in event:
                errors.append(abs(event["error_sec"]))
            elif isinstance(event, dict) and "absolute_error_sec" in event:
                errors.append(abs(event["absolute_error_sec"]))

            if hasattr(event, "latency_ms"):
                latencies.append(event.latency_ms)
            elif isinstance(event, dict) and "latency_ms" in event:
                latencies.append(event["latency_ms"])

            if isinstance(event, dict) and event.get("exception"):
                exceptions_count += 1

        samples = len(errors)
        avg_lat = float(np.mean(latencies)) if latencies else 0.0

        if samples == 0:
            return WatchdogEvaluationResult(
                rollback_triggered=False,
                samples_monitored=0,
                observed_mae_sec=0.0,
                baseline_mae_sec=baseline_mae,
                mae_ratio=0.0,
                max_allowed_mae_ratio=self.max_degradation_ratio,
                avg_latency_ms=avg_lat,
                max_allowed_latency_ms=self.max_latency_ms,
                exceptions_count=0,
                active_production_model_id=current_prod_id,
                applied_thresholds=thresholds,
            )

        observed_mae = float(np.mean(errors))
        mae_ratio = observed_mae / baseline_mae if baseline_mae > 0 else 1.0

        mae_degraded = mae_ratio > self.max_degradation_ratio
        latency_exceeded = avg_lat > self.max_latency_ms
        exceptions_occurred = exceptions_count > self.max_allowed_exceptions

        degraded = mae_degraded or latency_exceeded or exceptions_occurred

        if degraded:
            reasons = []
            if mae_degraded:
                reasons.append(f"MAE ({observed_mae:.2f}s) reached {mae_ratio:.2f}x of baseline ({baseline_mae:.2f}s, threshold={self.max_degradation_ratio:.2f}x)")
            if latency_exceeded:
                reasons.append(f"Latency ({avg_lat:.2f}ms) exceeded SLA ({self.max_latency_ms:.2f}ms)")
            if exceptions_occurred:
                reasons.append(f"{exceptions_count} runtime exceptions observed")

            rollback_reason = "Watchdog Emergency Rollback Triggered: " + "; ".join(reasons)
            logger.warning(f"WATCHDOG TRIGGERED ROLLBACK for {current_prod_id}: {rollback_reason}")

            # Execute rollback in registry
            restored_champion = self.registry.rollback_production(rollback_reason=rollback_reason)
            restored_id = restored_champion["model_id"]
            restored_dir = restored_champion.get("artifact_dir")

            # Execute atomic hot swap back to restored champion
            if restored_dir and Path(restored_dir).exists():
                self.predictor.hot_swap_model(restored_dir)
            else:
                self.predictor.reload_model()

            return WatchdogEvaluationResult(
                rollback_triggered=True,
                samples_monitored=samples,
                observed_mae_sec=observed_mae,
                baseline_mae_sec=baseline_mae,
                mae_ratio=mae_ratio,
                max_allowed_mae_ratio=self.max_degradation_ratio,
                avg_latency_ms=avg_lat,
                max_allowed_latency_ms=self.max_latency_ms,
                exceptions_count=exceptions_count,
                active_production_model_id=restored_id,
                rollback_reason=rollback_reason,
                applied_thresholds=thresholds,
                details={
                    "previous_failed_model_id": current_prod_id,
                    "restored_model_id": restored_id,
                }
            )

        return WatchdogEvaluationResult(
            rollback_triggered=False,
            samples_monitored=samples,
            observed_mae_sec=observed_mae,
            baseline_mae_sec=baseline_mae,
            mae_ratio=mae_ratio,
            max_allowed_mae_ratio=self.max_degradation_ratio,
            avg_latency_ms=avg_lat,
            max_allowed_latency_ms=self.max_latency_ms,
            exceptions_count=exceptions_count,
            active_production_model_id=current_prod_id,
            applied_thresholds=thresholds,
        )
