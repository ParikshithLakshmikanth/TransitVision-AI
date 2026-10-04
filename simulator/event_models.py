"""TransitVision AI - Simulation Event and State Models.
Defines schemas for replayed telemetry events, production predictions,
resolved actual outcomes, online evaluations, and scenario metadata.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import pandas as pd


@dataclass
class TelemetryEvent:
    """Represents an incoming real bus telemetry event from the chronological replay stream."""
    event_id: str
    event_timestamp_utc: str
    simulation_timestamp_utc: str
    trip_id: str
    deviceid: str
    direction: int
    segment: int
    features: Dict[str, Any]
    source_type: str = "REAL_REPLAY"
    source_dataset: str = "kandy_eta_stream.parquet"
    simulation_generated: bool = True
    synthetic_disturbance: bool = False
    scenario_id: str = "BASELINE"
    scenario_name: str = "Baseline Control Replay"
    disturbance_intensity: float = 0.0
    disturbance_applied: bool = False
    affected_scope: str = "GLOBAL"
    # Ground truth stored internally by the simulator for eventual outcome resolution
    # NEVER passed to the ML inference model
    _ground_truth_eta_sec: Optional[float] = None
    _baseline_ground_truth_eta_sec: Optional[float] = None

    def to_dict(self, include_ground_truth: bool = False) -> Dict[str, Any]:
        """Serializes telemetry event to dict, optionally suppressing ground truth."""
        d = {
            "event_id": self.event_id,
            "event_timestamp_utc": self.event_timestamp_utc,
            "simulation_timestamp_utc": self.simulation_timestamp_utc,
            "trip_id": self.trip_id,
            "deviceid": self.deviceid,
            "direction": self.direction,
            "segment": self.segment,
            "features": self.features,
            "source_type": self.source_type,
            "source_dataset": self.source_dataset,
            "simulation_generated": self.simulation_generated,
            "synthetic_disturbance": self.synthetic_disturbance,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "disturbance_intensity": self.disturbance_intensity,
            "disturbance_applied": self.disturbance_applied,
            "affected_scope": self.affected_scope
        }
        if include_ground_truth:
            d["_ground_truth_eta_sec"] = self._ground_truth_eta_sec
            d["_baseline_ground_truth_eta_sec"] = self._baseline_ground_truth_eta_sec
        return d


@dataclass
class PredictionEvent:
    """Represents a production model ETA prediction generated for a telemetry event."""
    prediction_id: str
    event_id: str
    trip_id: str
    deviceid: str
    direction: int
    segment: int
    predicted_eta_sec: float
    raw_prediction_sec: float
    prediction_timestamp_utc: str
    model_id: str
    model_version: str
    algorithm: str
    latency_ms: float
    scenario_id: str = "BASELINE"
    synthetic_disturbance: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutcomeEvent:
    """Represents the arrival / conclusion of a segment with observed ground truth."""
    outcome_id: str
    prediction_id: str
    event_id: str
    trip_id: str
    deviceid: str
    direction: int
    segment: int
    actual_eta_sec: float
    baseline_actual_eta_sec: float
    outcome_timestamp_utc: str
    scenario_id: str = "BASELINE"
    synthetic_disturbance: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationEvent:
    """Represents the online comparison between predicted ETA and actual observed ETA."""
    evaluation_id: str
    prediction_id: str
    event_id: str
    trip_id: str
    deviceid: str
    direction: int
    segment: int
    predicted_eta_sec: float
    actual_eta_sec: float
    baseline_actual_eta_sec: float
    signed_error_sec: float
    absolute_error_sec: float
    percentage_error: float
    evaluation_timestamp_utc: str
    model_version: str
    scenario_id: str = "BASELINE"
    synthetic_disturbance: bool = False
    disturbance_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BusState:
    """Maintains the live spatiotemporal operational state of an active bus device."""
    deviceid: str
    trip_id: str
    direction: int
    current_segment: int
    last_event_timestamp_utc: str
    last_simulation_timestamp_utc: str
    last_predicted_eta_sec: Optional[float] = None
    last_actual_eta_sec: Optional[float] = None
    last_error_sec: Optional[float] = None
    status: str = "ACTIVE"  # "ACTIVE", "COMPLETED", "IDLE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationConfig:
    """Configuration options for a digital transit simulation replay session."""
    replay_speed: float = 100.0  # Multiplier factor, 0.0=paused, "MAX" handled as <= 0.0 or special
    filter_trip_id: Optional[str] = None
    filter_deviceid: Optional[str] = None
    filter_direction: Optional[int] = None
    filter_segment: Optional[int] = None
    auto_predict: bool = True
    scenario_id: str = "BASELINE"
    scenario_name: str = "Baseline Control Replay"
    scenario_intensity: float = 0.0
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    synthetic_disturbance: bool = False
    source_dataset_path: str = "data/processed/kandy_eta_stream.parquet"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
