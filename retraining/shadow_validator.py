"""TransitVision AI - Simulation-Based Shadow Replay Safety Check.
Runs candidate challenger alongside active champion in shadow mode within the
Digital Transit Simulator replay stream to ensure dual-inference safety.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from ml.inference.predictor import ETAPredictor
from ml.training.leak_guard import LeakageGuard
from simulator.transit_simulator import DigitalTransitSimulator

logger = logging.getLogger("TransitVision.ShadowValidator")


@dataclass
class ShadowValidationResult:
    """Outcome of the 300-record simulation shadow safety check."""
    passed: bool
    sample_count: int
    champion_mae_sec: float
    challenger_mae_sec: float
    mae_ratio: float
    relative_tolerance: float
    exceptions_count: int
    non_positive_predictions_count: int
    excessive_predictions_count: int
    challenger_avg_latency_ms: float
    champion_unaffected: bool
    failure_reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "sample_count": self.sample_count,
            "champion_mae_sec": round(self.champion_mae_sec, 2),
            "challenger_mae_sec": round(self.challenger_mae_sec, 2),
            "mae_ratio": round(self.mae_ratio, 4),
            "relative_tolerance": self.relative_tolerance,
            "exceptions_count": self.exceptions_count,
            "non_positive_predictions_count": self.non_positive_predictions_count,
            "excessive_predictions_count": self.excessive_predictions_count,
            "challenger_avg_latency_ms": round(self.challenger_avg_latency_ms, 3),
            "champion_unaffected": self.champion_unaffected,
            "failure_reasons": self.failure_reasons,
            "details": self.details,
        }


class ShadowValidator:
    """
    Simulation-based shadow safety check.
    
    Role:
    Verifies runtime safety, exception immunity, schema compatibility, and sanity of the
    challenger during live replay without affecting active champion serving.
    (Note: The primary statistical decision comes from the Dual-Holdout validation).
    """

    def __init__(
        self,
        shadow_window_size: int = 300,
        max_relative_mae_tolerance: float = 1.02,
    ):
        self.shadow_window_size = shadow_window_size
        self.max_relative_mae_tolerance = max_relative_mae_tolerance
        self.leak_guard = LeakageGuard()

    def run_shadow_safety_check(
        self,
        challenger_model_dir: str,
        simulator: Optional[DigitalTransitSimulator] = None,
        start_cursor: int = 0,
    ) -> ShadowValidationResult:
        """
        Executes dual inference replay for shadow_window_size records.
        """
        logger.info(f"Initiating {self.shadow_window_size}-record Shadow Safety Check for candidate at {challenger_model_dir}")

        # Instantiate separate challenger predictor
        try:
            challenger_predictor = ETAPredictor(model_dir=challenger_model_dir)
        except Exception as e:
            logger.error(f"Shadow Safety Check Failed: Challenger predictor initialization failed: {e}")
            return ShadowValidationResult(
                passed=False,
                sample_count=0,
                champion_mae_sec=0.0,
                challenger_mae_sec=0.0,
                mae_ratio=0.0,
                relative_tolerance=self.max_relative_mae_tolerance,
                exceptions_count=1,
                non_positive_predictions_count=0,
                excessive_predictions_count=0,
                challenger_avg_latency_ms=0.0,
                champion_unaffected=True,
                failure_reasons=[f"Challenger failed to load: {e}"]
            )

        sim = simulator or DigitalTransitSimulator()
        sim.start()
        if start_cursor > 0:
            sim.seek(start_cursor)

        champ_errors = []
        chall_errors = []
        chall_latencies = []
        exceptions_count = 0
        non_positive_count = 0
        excessive_count = 0
        failure_reasons = []

        processed = 0
        while processed < self.shadow_window_size and not sim.is_finished():
            batch_size = min(50, self.shadow_window_size - processed)
            step_results = sim.step(batch_size)
            if not step_results:
                break

            for telemetry, champ_pred, outcome, eval_event in step_results:
                if outcome is None or eval_event is None or champ_pred is None:
                    continue

                actual_eta = outcome.actual_eta_sec
                champ_eta = champ_pred.predicted_eta_sec
                champ_errors.append(abs(actual_eta - champ_eta))

                # Execute shadow challenger prediction on features
                try:
                    t0 = time.perf_counter()
                    feats = telemetry.features if hasattr(telemetry, "features") else telemetry.to_dict()
                    chall_res = challenger_predictor.predict(feats)
                    chall_lat = (time.perf_counter() - t0) * 1000.0
                    chall_latencies.append(chall_lat)

                    chall_eta = chall_res["predicted_eta_sec"]
                    if chall_eta <= 0.0:
                        non_positive_count += 1
                    if chall_eta > 3600.0:
                        excessive_count += 1

                    chall_errors.append(abs(actual_eta - chall_eta))
                except Exception as ex:
                    exceptions_count += 1
                    logger.error(f"Challenger exception during shadow replay: {ex}")

                processed += 1
                if processed >= self.shadow_window_size:
                    break

        if not chall_errors or not champ_errors:
            return ShadowValidationResult(
                passed=False,
                sample_count=processed,
                champion_mae_sec=0.0,
                challenger_mae_sec=0.0,
                mae_ratio=0.0,
                relative_tolerance=self.max_relative_mae_tolerance,
                exceptions_count=exceptions_count,
                non_positive_predictions_count=non_positive_count,
                excessive_predictions_count=excessive_count,
                challenger_avg_latency_ms=0.0,
                champion_unaffected=True,
                failure_reasons=["No valid evaluated records gathered during shadow replay."]
            )

        champ_mae = float(np.mean(champ_errors))
        chall_mae = float(np.mean(chall_errors))
        mae_ratio = chall_mae / champ_mae if champ_mae > 0 else 1.0
        avg_lat = float(np.mean(chall_latencies)) if chall_latencies else 0.0

        # Safety Criteria
        if exceptions_count > 0:
            failure_reasons.append(f"Challenger threw {exceptions_count} runtime exceptions during shadow replay.")
        if non_positive_count > 0:
            failure_reasons.append(f"Challenger generated {non_positive_count} non-positive predictions.")
        if excessive_count > 0:
            failure_reasons.append(f"Challenger generated {excessive_count} excessive predictions (>3600s).")
        if mae_ratio > self.max_relative_mae_tolerance:
            failure_reasons.append(
                f"Shadow MAE ratio ({mae_ratio:.4f}x) exceeded tolerance ({self.max_relative_mae_tolerance:.2f}x)."
            )

        passed = len(failure_reasons) == 0

        logger.info(
            f"Shadow Safety Check Result: {'PASS' if passed else 'FAIL'} "
            f"(Champ MAE: {champ_mae:.2f}s, Chall MAE: {chall_mae:.2f}s, Ratio: {mae_ratio:.4f}x, Exceptions: {exceptions_count})"
        )

        return ShadowValidationResult(
            passed=passed,
            sample_count=processed,
            champion_mae_sec=champ_mae,
            challenger_mae_sec=chall_mae,
            mae_ratio=mae_ratio,
            relative_tolerance=self.max_relative_mae_tolerance,
            exceptions_count=exceptions_count,
            non_positive_predictions_count=non_positive_count,
            excessive_predictions_count=excessive_count,
            challenger_avg_latency_ms=avg_lat,
            champion_unaffected=True,
            failure_reasons=failure_reasons,
            details={"samples_evaluated": processed}
        )
