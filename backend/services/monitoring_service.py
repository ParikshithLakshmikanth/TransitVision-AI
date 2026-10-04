"""TransitVision AI - Monitoring & Drift Service."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from config.settings import METADATA_DIR, PROCESSED_DATA_DIR
from monitoring.drift_engine import DriftEngine
from monitoring.drift_models import DriftConfig, DriftType
from backend.schemas import DriftStatusResponse, DriftSummaryResponse, DriftEventResponse

logger = logging.getLogger("TransitVision.MonitoringService")


class MonitoringService:
    """
    Coordinates DriftEngine state, drift event persistence,
    and statistical summary generation for API consumers.
    """

    def __init__(self, drift_engine: Optional[DriftEngine] = None):
        self.reference_profile_path = METADATA_DIR / "drift_reference_profile.json"
        ref_profile = self._load_reference_profile()
        
        self.drift_engine = drift_engine or DriftEngine(
            config=DriftConfig(window_size=500, step_size=250),
            reference_profile=ref_profile,
        )

    def _load_reference_profile(self) -> Optional[Dict[str, Any]]:
        if self.reference_profile_path.exists():
            try:
                with open(self.reference_profile_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading drift reference profile: {e}")
        return None

    def process_observation(
        self,
        record_idx: int,
        features: Dict[str, Any],
        predicted_eta: float,
        actual_eta: float,
        timestamp_utc: str,
        model_version: str = "v1.0.0",
    ) -> List[DriftEventResponse]:
        """Processes a single telemetry/outcome record through the DriftEngine."""
        alerts = self.drift_engine.process_record(
            record_idx=record_idx,
            features=features,
            predicted_eta=predicted_eta,
            actual_eta=actual_eta,
            timestamp_utc=timestamp_utc,
            model_version=model_version,
        )
        return [self._convert_to_response(a) for a in alerts]

    def set_scenario_context(
        self,
        scenario_id: str,
        scenario_name: str,
        synthetic_disturbance: bool = False,
        intensity: float = 0.0,
    ) -> None:
        """Propagates scenario context to DriftEngine."""
        self.drift_engine.set_scenario_context(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            synthetic_disturbance=synthetic_disturbance,
            intensity=intensity,
        )

    def get_summary(self) -> DriftSummaryResponse:
        """Returns statistical summary of drift engine."""
        s = self.drift_engine.get_summary()
        return DriftSummaryResponse(
            scenario_id=s["scenario_id"],
            scenario_name=s["scenario_name"],
            total_records_monitored=s["total_records_monitored"],
            windows_evaluated=s["windows_evaluated"],
            total_alerts=s["total_alerts"],
            data_drift_alerts=s["data_drift_alerts"],
            performance_drift_alerts=s["performance_drift_alerts"],
            concept_drift_alerts=s["concept_drift_alerts"],
            first_detection_index=s.get("first_detection_index"),
            first_detection_delay_records=s.get("first_detection_delay_records"),
            first_detection_timestamp=s.get("first_detection_timestamp"),
            first_detection_type=s.get("first_detection_type"),
            first_detector_name=s.get("first_detector_name"),
            max_severity=s["max_severity"],
            frequently_drifted_features=s["frequently_drifted_features"],
        )

    def get_recent_events(self, limit: int = 50) -> List[DriftEventResponse]:
        """Returns recent drift events."""
        events = self.drift_engine.events_history[-limit:]
        return [self._convert_to_response(e) for e in events]

    def get_drift_status(self) -> DriftStatusResponse:
        """Aggregates real-time drift status."""
        summary = self.get_summary()
        recent = self.get_recent_events(limit=20)
        
        is_data = summary.data_drift_alerts > 0
        is_perf = summary.performance_drift_alerts > 0
        is_concept = summary.concept_drift_alerts > 0

        return DriftStatusResponse(
            is_data_drift_active=is_data,
            is_performance_drift_active=is_perf,
            is_concept_drift_active=is_concept,
            latest_severity=summary.max_severity,
            active_scenario=summary.scenario_name,
            summary=summary,
            recent_events=recent,
        )

    def reset(self) -> None:
        """Resets drift monitor state."""
        self.drift_engine.reset()

    def _convert_to_response(self, evt: Any) -> DriftEventResponse:
        """Converts internal DriftEvent to DriftEventResponse schema."""
        d = evt.to_dict()
        return DriftEventResponse(
            event_id=d["event_id"],
            timestamp_utc=d["timestamp_utc"],
            drift_type=d["drift_type"],
            severity=d["severity"],
            detector=d["detector"],
            scenario_id=d["scenario_id"],
            scenario_name=d["scenario_name"],
            synthetic_disturbance=d["synthetic_disturbance"],
            disturbance_intensity=float(d["disturbance_intensity"]),
            source_type=d["source_type"],
            model_id=d["model_id"],
            model_version=d["model_version"],
            window_start_idx=int(d["window_start_idx"]),
            window_end_idx=int(d["window_end_idx"]),
            sample_count=int(d["sample_count"]),
            statistic=float(d["statistic"]),
            threshold=float(d["threshold"]),
            p_value=float(d["p_value"]) if d.get("p_value") is not None else None,
            baseline_metric=float(d["baseline_metric"]),
            current_metric=float(d["current_metric"]),
            affected_features=list(d["affected_features"]),
            evidence=dict(d.get("evidence", {})),
        )
