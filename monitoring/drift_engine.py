"""TransitVision AI - Unified Drift Detection & Statistical Monitoring Engine.
Orchestrates Covariate Drift, Performance Drift, and Concept Drift detection
across sequential streaming windows, maintaining detection latencies and emitting DriftEvents.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config.settings import METADATA_DIR, PROCESSED_DATA_DIR
from monitoring.drift_models import (
    DriftType,
    DriftSeverity,
    DriftEvent,
    DriftConfig,
    FeatureDriftResult,
    PerformanceDriftResult,
    ResidualDriftResult
)
from monitoring.drift_detectors import FeatureDriftDetector, ReferenceProfiler
from monitoring.performance_monitor import RollingPerformanceMonitor
from monitoring.residual_monitor import ResidualConceptMonitor

logger = logging.getLogger("TransitVision.DriftEngine")


class DriftEngine:
    """Unified engine coordinating 3-dimensional drift detection."""

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        reference_profile: Optional[Dict[str, Any]] = None,
        output_events_path: Optional[Path] = None
    ):
        self.config = config or DriftConfig()
        self.events_output_path = Path(output_events_path or (METADATA_DIR / "drift_events.json"))

        # Initialize sub-monitors
        self.feature_detector = FeatureDriftDetector(
            config=self.config,
            reference_profile=reference_profile
        )
        self.performance_monitor = RollingPerformanceMonitor(config=self.config)
        self.residual_monitor = ResidualConceptMonitor(config=self.config)

        # Internal state & buffers
        self.buffered_features: List[Dict[str, Any]] = []
        self.buffered_indices: List[int] = []
        self.buffered_timestamps: List[str] = []

        self.events_history: List[DriftEvent] = []
        self.total_processed_records = 0
        self.windows_evaluated = 0

        # Scenario provenance state
        self.current_scenario_id = "BASELINE"
        self.current_scenario_name = "Baseline Control"
        self.current_synthetic = False
        self.current_intensity = 0.0

        # First detection latency tracking
        self.first_detection_idx: Optional[int] = None
        self.first_detection_timestamp: Optional[str] = None
        self.first_detection_type: Optional[DriftType] = None
        self.first_detector_name: Optional[str] = None

    def set_scenario_context(
        self,
        scenario_id: str,
        scenario_name: str,
        synthetic_disturbance: bool = False,
        intensity: float = 0.0
    ) -> None:
        """Sets metadata context for streaming scenario events."""
        self.current_scenario_id = scenario_id
        self.current_scenario_name = scenario_name
        self.current_synthetic = synthetic_disturbance
        self.current_intensity = intensity

    def process_record(
        self,
        record_idx: int,
        features: Dict[str, Any],
        predicted_eta: float,
        actual_eta: float,
        timestamp_utc: str,
        model_version: str = "v1.0.0"
    ) -> List[DriftEvent]:
        """
        Processes a single streaming observation strictly post-outcome resolution.
        Updates residual detector, rolling performance buffer, and sliding feature window.
        """
        self.total_processed_records += 1
        new_alerts: List[DriftEvent] = []
        sim_ts = datetime.now(timezone.utc).isoformat()

        # 1. Update Sequential Residual Monitor
        residual = actual_eta - predicted_eta
        res_results = self.residual_monitor.update(residual, record_idx)
        for res in res_results:
            if res.in_drift:
                evt = DriftEvent(
                    event_id=f"DRIFT_RES_{self.current_scenario_id}_{record_idx:06d}_{res.detector_name}",
                    timestamp_utc=timestamp_utc,
                    drift_type=DriftType.CONCEPT_DRIFT,
                    severity=res.severity,
                    detector=res.detector_name,
                    scenario_id=self.current_scenario_id,
                    scenario_name=self.current_scenario_name,
                    synthetic_disturbance=self.current_synthetic,
                    disturbance_intensity=self.current_intensity,
                    source_type="SYNTHETIC_SIMULATION" if self.current_synthetic else "REAL_REPLAY",
                    model_id="eta_model_v1",
                    model_version=model_version,
                    window_start_idx=max(1, record_idx - 100),
                    window_end_idx=record_idx,
                    sample_count=res.sample_count,
                    statistic=res.statistic,
                    threshold=res.threshold,
                    p_value=None,
                    baseline_metric=0.0,
                    current_metric=res.mean_residual,
                    affected_features=["eta_to_next_stop_sec_residual"],
                    evidence={"mean_residual": res.mean_residual, "variance": res.variance_residual}
                )
                self._record_alert(evt)
                new_alerts.append(evt)

        # 2. Update Performance Monitor Buffer
        self.performance_monitor.add_observation(actual_eta, predicted_eta, record_idx)

        # 3. Buffer features for sliding window evaluation
        self.buffered_features.append(features)
        self.buffered_indices.append(record_idx)
        self.buffered_timestamps.append(timestamp_utc)

        # 4. Trigger Window Evaluation when window_size reached
        if len(self.buffered_features) >= self.config.window_size:
            window_alerts = self._evaluate_window(model_version=model_version)
            new_alerts.extend(window_alerts)

            # Slide window by step_size
            slide_n = min(self.config.step_size, len(self.buffered_features))
            self.buffered_features = self.buffered_features[slide_n:]
            self.buffered_indices = self.buffered_indices[slide_n:]
            self.buffered_timestamps = self.buffered_timestamps[slide_n:]

        return new_alerts

    def _evaluate_window(self, model_version: str = "v1.0.0") -> List[DriftEvent]:
        """Evaluates data drift and performance drift over current window buffer."""
        self.windows_evaluated += 1
        alerts: List[DriftEvent] = []
        window_df = pd.DataFrame(self.buffered_features)
        w_start = self.buffered_indices[0]
        w_end = self.buffered_indices[-1]
        w_ts = self.buffered_timestamps[-1]
        w_len = len(window_df)

        # A. Evaluate Feature / Data Drift
        feat_results = self.feature_detector.evaluate_window(window_df)
        drifted_features = [f for f in feat_results if f.is_drift]

        if drifted_features:
            max_stat = max(f.statistic for f in drifted_features)
            highest_sev = max((f.severity for f in drifted_features), key=lambda s: ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s.value))
            
            evt = DriftEvent(
                event_id=f"DRIFT_DATA_{self.current_scenario_id}_W{self.windows_evaluated:04d}",
                timestamp_utc=w_ts,
                drift_type=DriftType.DATA_DRIFT,
                severity=highest_sev,
                detector="PSI_KS_HYBRID",
                scenario_id=self.current_scenario_id,
                scenario_name=self.current_scenario_name,
                synthetic_disturbance=self.current_synthetic,
                disturbance_intensity=self.current_intensity,
                source_type="SYNTHETIC_SIMULATION" if self.current_synthetic else "REAL_REPLAY",
                model_id="eta_model_v1",
                model_version=model_version,
                window_start_idx=w_start,
                window_end_idx=w_end,
                sample_count=w_len,
                statistic=round(max_stat, 4),
                threshold=self.config.psi_drift_threshold,
                p_value=min([f.p_value for f in drifted_features if f.p_value is not None] or [1.0]),
                baseline_metric=0.0,
                current_metric=round(max_stat, 4),
                affected_features=[f.feature_name for f in drifted_features],
                evidence={f.feature_name: {"statistic": f.statistic, "severity": f.severity.value} for f in drifted_features}
            )
            self._record_alert(evt)
            alerts.append(evt)

        # B. Evaluate Performance Drift
        perf_res = self.performance_monitor.evaluate_current_window()
        if perf_res and perf_res.is_drift:
            evt = DriftEvent(
                event_id=f"DRIFT_PERF_{self.current_scenario_id}_W{self.windows_evaluated:04d}",
                timestamp_utc=w_ts,
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=perf_res.severity,
                detector="RollingPerformanceMonitor",
                scenario_id=self.current_scenario_id,
                scenario_name=self.current_scenario_name,
                synthetic_disturbance=self.current_synthetic,
                disturbance_intensity=self.current_intensity,
                source_type="SYNTHETIC_SIMULATION" if self.current_synthetic else "REAL_REPLAY",
                model_id="eta_model_v1",
                model_version=model_version,
                window_start_idx=w_start,
                window_end_idx=w_end,
                sample_count=w_len,
                statistic=perf_res.mae_degradation_ratio,
                threshold=self.config.perf_drift_multiplier,
                p_value=None,
                baseline_metric=perf_res.reference_mae,
                current_metric=perf_res.mae,
                affected_features=["mae_sec", "rmse_sec"],
                evidence={
                    "mae": perf_res.mae,
                    "rmse": perf_res.rmse,
                    "median_ae": perf_res.median_ae,
                    "p90_error": perf_res.p90_error,
                    "mae_degradation_ratio": perf_res.mae_degradation_ratio
                }
            )
            self._record_alert(evt)
            alerts.append(evt)

        return alerts

    def flush(self, model_version: str = "v1.0.0") -> List[DriftEvent]:
        """Flushes remaining samples in the buffer for final window evaluation."""
        if len(self.buffered_features) >= self.config.min_window_size:
            return self._evaluate_window(model_version=model_version)
        return []

    def _record_alert(self, evt: DriftEvent) -> None:
        """Records alert and updates first detection latency state."""
        self.events_history.append(evt)
        if self.first_detection_idx is None:
            self.first_detection_idx = evt.window_end_idx
            self.first_detection_timestamp = evt.timestamp_utc
            self.first_detection_type = evt.drift_type
            self.first_detector_name = evt.detector

    def get_summary(self) -> Dict[str, Any]:
        """Generates comprehensive monitoring summary."""
        data_alerts = [e for e in self.events_history if e.drift_type == DriftType.DATA_DRIFT]
        perf_alerts = [e for e in self.events_history if e.drift_type == DriftType.PERFORMANCE_DRIFT]
        concept_alerts = [e for e in self.events_history if e.drift_type == DriftType.CONCEPT_DRIFT]

        severities = [e.severity.value for e in self.events_history]
        max_sev = "NONE"
        for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if s in severities:
                max_sev = s
                break

        # Affected features frequency
        feat_freq: Dict[str, int] = {}
        for a in data_alerts:
            for f in a.affected_features:
                feat_freq[f] = feat_freq.get(f, 0) + 1

        return {
            "scenario_id": self.current_scenario_id,
            "scenario_name": self.current_scenario_name,
            "total_records_monitored": self.total_processed_records,
            "windows_evaluated": self.windows_evaluated,
            "total_alerts": len(self.events_history),
            "data_drift_alerts": len(data_alerts),
            "performance_drift_alerts": len(perf_alerts),
            "concept_drift_alerts": len(concept_alerts),
            "first_detection_index": self.first_detection_idx,
            "first_detection_delay_records": self.first_detection_idx if self.first_detection_idx else None,
            "first_detection_timestamp": self.first_detection_timestamp,
            "first_detection_type": self.first_detection_type.value if self.first_detection_type else None,
            "first_detector_name": self.first_detector_name,
            "max_severity": max_sev,
            "frequently_drifted_features": sorted(feat_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def save_events(self, filepath: Optional[Path] = None) -> None:
        """Persists recorded DriftEvent history to JSON file."""
        target = Path(filepath or self.events_output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        events_dicts = [e.to_dict() for e in self.events_history]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(events_dicts, f, indent=2)
        logger.info(f"Persisted {len(events_dicts)} drift events to {target}")

    def reset(self) -> None:
        """Resets all detector states."""
        self.buffered_features.clear()
        self.buffered_indices.clear()
        self.buffered_timestamps.clear()
        self.events_history.clear()
        self.total_processed_records = 0
        self.windows_evaluated = 0
        self.first_detection_idx = None
        self.first_detection_timestamp = None
        self.first_detection_type = None
        self.first_detector_name = None
        self.performance_monitor.reset()
        self.residual_monitor.reset()
