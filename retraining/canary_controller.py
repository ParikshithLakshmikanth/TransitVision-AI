"""TransitVision AI - Deterministic Simulation Canary Deployment Controller.
Executes multi-stage canary routing (10% -> 50% -> PROMOTION_CANDIDATE) using
deterministic trip-hash partitioning in the Digital Transit Simulator, with
fair, identical-record comparative evaluation between Champion and Challenger.
"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import numpy as np

from ml.inference.predictor import ETAPredictor
from simulator.transit_simulator import DigitalTransitSimulator

logger = logging.getLogger("TransitVision.CanaryController")


def deterministic_canary_route(trip_id: Any, canary_percentage: int) -> bool:
    """
    Deterministically routes traffic to canary evaluation based on hashed trip_id.
    
    Returns True if trip is selected for canary evaluation, False otherwise.
    """
    if canary_percentage <= 0:
        return False
    if canary_percentage >= 100:
        return True
    
    h = int(hashlib.md5(str(trip_id).encode("utf-8")).hexdigest(), 16)
    return (h % 100) < canary_percentage


@dataclass
class CanaryStageResult:
    """Outcome of a single canary deployment stage with identical-record comparison."""
    stage_name: str
    canary_percentage: int
    passed: bool
    total_records_processed: int
    selected_canary_records: int
    champion_evaluation_count: int
    challenger_evaluation_count: int
    unique_trips_count: int
    time_range_start: Optional[str]
    time_range_end: Optional[str]
    champion_mae_sec: float
    challenger_mae_sec: float
    champion_rmse_sec: float
    challenger_rmse_sec: float
    champion_p95_sec: float
    challenger_p95_sec: float
    mae_ratio: float
    max_allowed_mae_ratio: float
    exceptions_count: int
    record_ids_match: bool
    failure_reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "canary_percentage": self.canary_percentage,
            "passed": self.passed,
            "total_records_processed": self.total_records_processed,
            "selected_canary_records": self.selected_canary_records,
            "champion_evaluation_count": self.champion_evaluation_count,
            "challenger_evaluation_count": self.challenger_evaluation_count,
            "unique_trips_count": self.unique_trips_count,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "champion_mae_sec": round(self.champion_mae_sec, 2),
            "challenger_mae_sec": round(self.challenger_mae_sec, 2),
            "champion_rmse_sec": round(self.champion_rmse_sec, 2),
            "challenger_rmse_sec": round(self.challenger_rmse_sec, 2),
            "champion_p95_sec": round(self.champion_p95_sec, 2),
            "challenger_p95_sec": round(self.challenger_p95_sec, 2),
            "mae_ratio": round(self.mae_ratio, 4),
            "max_allowed_mae_ratio": self.max_allowed_mae_ratio,
            "exceptions_count": self.exceptions_count,
            "record_ids_match": self.record_ids_match,
            "failure_reasons": self.failure_reasons,
            "details": self.details,
        }


class CanaryController:
    """
    Manages multi-stage canary rollout in simulation replay with strictly fair comparison.
    
    Guarantee:
    Whenever Champion and Challenger are compared, BOTH models evaluate the EXACT SAME records.
    """

    def __init__(
        self,
        canary_stages: Optional[List[int]] = None,
        records_per_stage: int = 300,
        min_canary_samples: int = 15,
        max_mae_degradation_ratio: float = 1.05,
    ):
        self.canary_stages = canary_stages or [10, 50]
        self.records_per_stage = records_per_stage
        self.min_canary_samples = min_canary_samples
        self.max_mae_degradation_ratio = max_mae_degradation_ratio

    def run_canary_stage(
        self,
        challenger_model_dir: str,
        canary_percentage: int,
        simulator: Optional[DigitalTransitSimulator] = None,
        start_cursor: int = 0,
    ) -> CanaryStageResult:
        """
        Executes a single canary stage replay with identical-record dual evaluation.
        """
        stage_name = f"CANARY_{canary_percentage}"
        logger.info(f"Initiating {stage_name} ({canary_percentage}% traffic) for {self.records_per_stage} records.")

        try:
            challenger_predictor = ETAPredictor(model_dir=challenger_model_dir)
        except Exception as e:
            return CanaryStageResult(
                stage_name=stage_name,
                canary_percentage=canary_percentage,
                passed=False,
                total_records_processed=0,
                selected_canary_records=0,
                champion_evaluation_count=0,
                challenger_evaluation_count=0,
                unique_trips_count=0,
                time_range_start=None,
                time_range_end=None,
                champion_mae_sec=0.0,
                challenger_mae_sec=0.0,
                champion_rmse_sec=0.0,
                challenger_rmse_sec=0.0,
                champion_p95_sec=0.0,
                challenger_p95_sec=0.0,
                mae_ratio=0.0,
                max_allowed_mae_ratio=self.max_mae_degradation_ratio,
                exceptions_count=1,
                record_ids_match=False,
                failure_reasons=[f"Challenger failed to load: {e}"]
            )

        sim = simulator or DigitalTransitSimulator()
        sim.start()
        if start_cursor > 0:
            sim.seek(start_cursor)

        champ_errors: List[float] = []
        chall_errors: List[float] = []
        champ_record_ids: List[str] = []
        chall_record_ids: List[str] = []
        unique_trips: Set[str] = set()
        timestamps: List[str] = []
        exceptions_count = 0
        failure_reasons: List[str] = []

        processed_total = 0
        while processed_total < self.records_per_stage and not sim.is_finished():
            batch_size = min(50, self.records_per_stage - processed_total)
            step_results = sim.step(batch_size)
            if not step_results:
                break

            for telemetry, champ_pred, outcome, eval_event in step_results:
                if outcome is None or eval_event is None or champ_pred is None:
                    continue

                processed_total += 1
                actual_eta = outcome.actual_eta_sec
                trip_id = str(telemetry.trip_id)
                event_id = getattr(telemetry, "event_id", f"EVT_{processed_total}")
                event_ts = getattr(telemetry, "event_timestamp_utc", None)

                # Check if this trip is selected under canary routing percentage
                is_canary_traffic = deterministic_canary_route(trip_id, canary_percentage)
                if not is_canary_traffic:
                    continue

                # Record belongs to canary evaluation: BOTH MODELS EVALUATE THIS EXACT RECORD
                unique_trips.add(trip_id)
                if event_ts:
                    timestamps.append(event_ts)

                champ_eta = champ_pred.predicted_eta_sec
                champ_err = abs(actual_eta - champ_eta)
                champ_errors.append(champ_err)
                champ_record_ids.append(event_id)

                # Challenger prediction on identical input features
                try:
                    feats = telemetry.features if hasattr(telemetry, "features") else telemetry.to_dict()
                    chall_res = challenger_predictor.predict(feats)
                    chall_eta = chall_res["predicted_eta_sec"]
                    chall_err = abs(actual_eta - chall_eta)
                    chall_errors.append(chall_err)
                    chall_record_ids.append(event_id)
                except Exception as ex:
                    exceptions_count += 1
                    logger.error(f"Challenger exception during {stage_name}: {ex}")

                if processed_total >= self.records_per_stage:
                    break

        # Verification of Identical Evaluation Records
        record_ids_match = (champ_record_ids == chall_record_ids) and (len(champ_record_ids) > 0)
        selected_count = len(champ_errors)

        if not record_ids_match:
            failure_reasons.append("Fatal: Champion and Challenger were not evaluated on identical record IDs.")

        if selected_count < self.min_canary_samples:
            failure_reasons.append(
                f"Fail-Safe: Insufficient canary samples gathered ({selected_count} < min required {self.min_canary_samples})."
            )

        if exceptions_count > 0:
            failure_reasons.append(f"Challenger threw {exceptions_count} runtime exceptions in {stage_name}.")

        if selected_count > 0 and len(chall_errors) > 0:
            champ_mae = float(np.mean(champ_errors))
            chall_mae = float(np.mean(chall_errors))
            champ_rmse = float(np.sqrt(np.mean(np.array(champ_errors) ** 2)))
            chall_rmse = float(np.sqrt(np.mean(np.array(chall_errors) ** 2)))
            champ_p95 = float(np.percentile(champ_errors, 95))
            chall_p95 = float(np.percentile(chall_errors, 95))
            mae_ratio = chall_mae / champ_mae if champ_mae > 0 else 1.0

            if mae_ratio > self.max_mae_degradation_ratio:
                failure_reasons.append(
                    f"{stage_name} MAE ratio ({mae_ratio:.4f}x) exceeded tolerance ({self.max_mae_degradation_ratio:.2f}x)."
                )
        else:
            champ_mae = 0.0
            chall_mae = 0.0
            champ_rmse = 0.0
            chall_rmse = 0.0
            champ_p95 = 0.0
            chall_p95 = 0.0
            mae_ratio = 0.0

        passed = len(failure_reasons) == 0

        logger.info(
            f"{stage_name} Result: {'PASS' if passed else 'FAIL'} on {selected_count} identical records "
            f"(Challenger MAE: {chall_mae:.2f}s vs Champion MAE: {champ_mae:.2f}s, Ratio: {mae_ratio:.4f}x, Record Match: {record_ids_match})"
        )

        return CanaryStageResult(
            stage_name=stage_name,
            canary_percentage=canary_percentage,
            passed=passed,
            total_records_processed=processed_total,
            selected_canary_records=selected_count,
            champion_evaluation_count=len(champ_errors),
            challenger_evaluation_count=len(chall_errors),
            unique_trips_count=len(unique_trips),
            time_range_start=timestamps[0] if timestamps else None,
            time_range_end=timestamps[-1] if timestamps else None,
            champion_mae_sec=champ_mae,
            challenger_mae_sec=chall_mae,
            champion_rmse_sec=champ_rmse,
            challenger_rmse_sec=chall_rmse,
            champion_p95_sec=champ_p95,
            challenger_p95_sec=chall_p95,
            mae_ratio=mae_ratio,
            max_allowed_mae_ratio=self.max_mae_degradation_ratio,
            exceptions_count=exceptions_count,
            record_ids_match=record_ids_match,
            failure_reasons=failure_reasons,
            details={
                "routing_percentage": canary_percentage,
                "min_samples_required": self.min_canary_samples,
            }
        )
