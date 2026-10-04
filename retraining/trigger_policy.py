"""TransitVision AI - Autonomous Retraining Trigger Policy.
Evaluates statistical drift events and determines whether retraining is warranted,
advisory, or suppressed due to cooldown / minimum data / temporal boundary constraints.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger("TransitVision.RetrainingPolicy")


@dataclass
class TriggerDecision:
    """Decision object produced by RetrainingTriggerPolicy."""
    should_retrain: bool
    trigger_reason: str
    severity: str
    window_indices: List[int] = field(default_factory=list)
    resolved_record_count: int = 0
    cooldown_active: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class RetrainingTriggerPolicy:
    """
    Deterministic drift-to-retraining policy.
    
    Decoupled Decision Triad:
    DRIFT DETECTED != RETRAINING TRIGGERED != MODEL PROMOTED
    
    Policy Rules:
    1. Data Drift (PSI / KS / JS): Advisory telemetry alert only. Retraining NOT triggered.
    2. Isolated Single-Window Performance Spike: Warning only. Retraining NOT triggered.
    3. Sustained Performance Drift: >= 2 consecutive evaluation windows (N >= 500)
       with MAE ratio >= 1.60x baseline.
    4. Temporally Bounded Emergency Concept Drift: >= 3 Page-Hinkley residual alarm points
       occurring within a recent temporal window of `recent_concept_window_records` (2,500 records).
       Alarms spread sparsely across long-running baseline history do NOT trigger retraining.
    5. Cooldown Guardrail: Retraining suppressed if fewer than cooldown_records (1,500)
       have elapsed since the last retraining execution.
    6. Minimum Data Guardrail: Retraining requires >= min_new_records (1,000)
       newly resolved stream records in the buffer.
    """

    def __init__(
        self,
        sustained_window_count: int = 2,
        min_window_samples: int = 500,
        mae_degradation_ratio: float = 1.60,
        concept_drift_alarm_threshold: int = 3,
        recent_concept_window_records: int = 2500,
        cooldown_records: int = 1500,
        min_new_records: int = 1000,
    ):
        self.sustained_window_count = sustained_window_count
        self.min_window_samples = min_window_samples
        self.mae_degradation_ratio = mae_degradation_ratio
        self.concept_drift_alarm_threshold = concept_drift_alarm_threshold
        self.recent_concept_window_records = recent_concept_window_records
        self.cooldown_records = cooldown_records
        self.min_new_records = min_new_records
        self.last_retrain_stream_index: Optional[int] = None
        self.last_triggered_event_ids: Set[str] = set()

    def record_retraining_execution(self, stream_index: int) -> None:
        """Records the stream index when a retraining cycle was executed."""
        self.last_retrain_stream_index = stream_index
        logger.info(f"Retraining execution recorded at stream index {stream_index}.")

    def evaluate(
        self,
        drift_events: List[Dict[str, Any]],
        current_stream_index: int,
        resolved_stream_buffer_size: int,
        rolling_metrics_history: Optional[List[Dict[str, Any]]] = None,
        baseline_mae: float = 38.77,
    ) -> TriggerDecision:
        """
        Evaluates drift events and rolling metrics to produce a deterministic trigger decision.
        """
        rolling_metrics_history = rolling_metrics_history or []
        
        # 1. Check Cooldown Guardrail
        if self.last_retrain_stream_index is not None:
            records_since_last = current_stream_index - self.last_retrain_stream_index
            if records_since_last < self.cooldown_records:
                return TriggerDecision(
                    should_retrain=False,
                    trigger_reason=f"Retraining suppressed: Cooldown active ({records_since_last}/{self.cooldown_records} records elapsed).",
                    severity="INFO",
                    cooldown_active=True,
                    resolved_record_count=resolved_stream_buffer_size,
                    details={"records_since_last_retrain": records_since_last, "cooldown_required": self.cooldown_records}
                )

        # 2. Check Minimum Data Buffer Guardrail
        if resolved_stream_buffer_size < self.min_new_records:
            return TriggerDecision(
                should_retrain=False,
                trigger_reason=f"Retraining suppressed: Insufficient resolved stream buffer ({resolved_stream_buffer_size}/{self.min_new_records} records).",
                severity="INFO",
                resolved_record_count=resolved_stream_buffer_size,
                details={"buffer_size": resolved_stream_buffer_size, "min_required": self.min_new_records}
            )

        # 3. Analyze Sustained Performance Drift
        # Check consecutive windows in rolling_metrics_history
        consecutive_degraded = 0
        degraded_windows = []
        for idx, win in enumerate(rolling_metrics_history):
            samples = win.get("samples", 0)
            mae = win.get("mae_sec", 0.0)
            ratio = (mae / baseline_mae) if baseline_mae > 0 else 1.0
            if samples >= self.min_window_samples and ratio >= self.mae_degradation_ratio:
                consecutive_degraded += 1
                degraded_windows.append(win.get("window_index", idx))
            else:
                consecutive_degraded = 0
                degraded_windows = []

            if consecutive_degraded >= self.sustained_window_count:
                return TriggerDecision(
                    should_retrain=True,
                    trigger_reason=f"Sustained Performance Drift: {consecutive_degraded} consecutive windows with MAE >= {self.mae_degradation_ratio:.2f}x baseline.",
                    severity="CRITICAL",
                    window_indices=degraded_windows,
                    resolved_record_count=resolved_stream_buffer_size,
                    details={"consecutive_windows": consecutive_degraded, "degraded_window_indices": degraded_windows}
                )

        # 4. Analyze Temporally Bounded Emergency Concept Drift from Drift Events
        recent_concept_drift_alarms = []
        for e in drift_events:
            is_concept = (e.get("drift_type") == "CONCEPT_DRIFT" or "Page-Hinkley" in e.get("message", ""))
            if not is_concept:
                continue
            
            # Extract alarm stream record position (default to current_stream_index if omitted in tests)
            alarm_pos = e.get("window_end_idx")
            if alarm_pos is None:
                alarm_pos = e.get("record_index")
            if alarm_pos is None:
                alarm_pos = current_stream_index
            
            # Bounded temporal window check
            if (current_stream_index - alarm_pos) <= self.recent_concept_window_records and alarm_pos <= current_stream_index:
                recent_concept_drift_alarms.append(e)

        if len(recent_concept_drift_alarms) >= self.concept_drift_alarm_threshold:
            # Check for de-duplication: ensure there are new alarms not already triggering
            current_event_ids = {e.get("event_id", str(idx)) for idx, e in enumerate(recent_concept_drift_alarms)}
            if current_event_ids and current_event_ids == self.last_triggered_event_ids:
                return TriggerDecision(
                    should_retrain=False,
                    trigger_reason=f"Concept Drift acknowledgment: {len(recent_concept_drift_alarms)} alarms already evaluated; awaiting outcome.",
                    severity="INFO",
                    resolved_record_count=resolved_stream_buffer_size,
                    details={"recent_alarms_count": len(recent_concept_drift_alarms)}
                )
            
            self.last_triggered_event_ids = current_event_ids
            return TriggerDecision(
                should_retrain=True,
                trigger_reason=f"Emergency Concept Drift: {len(recent_concept_drift_alarms)} Page-Hinkley alarms detected within recent {self.recent_concept_window_records} records.",
                severity="CRITICAL",
                resolved_record_count=resolved_stream_buffer_size,
                details={
                    "recent_concept_alarms_count": len(recent_concept_drift_alarms),
                    "window_records": self.recent_concept_window_records,
                }
            )

        # 5. Check if only Data Drift or isolated warning events occurred
        data_drift_events = [e for e in drift_events if e.get("drift_type") == "DATA_DRIFT"]
        if data_drift_events:
            return TriggerDecision(
                should_retrain=False,
                trigger_reason="Data / Covariate drift detected: Advisory telemetry only. Retraining not triggered.",
                severity="WARNING",
                resolved_record_count=resolved_stream_buffer_size,
                details={"data_drift_events_count": len(data_drift_events)}
            )

        return TriggerDecision(
            should_retrain=False,
            trigger_reason="System operating within nominal performance thresholds. No retraining required.",
            severity="NORMAL",
            resolved_record_count=resolved_stream_buffer_size
        )
