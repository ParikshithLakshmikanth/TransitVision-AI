"""TransitVision AI - Controlled Disturbance Scenarios.
Defines deterministic, physically consistent operational disruptions
applied to streaming transit telemetry and simulated ground truth.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Set
import pandas as pd
import numpy as np


class BaseScenario(ABC):
    """Abstract base class for simulation scenarios."""

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str,
        intensity: float = 0.0,
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None,
        affected_segments: Optional[List[int]] = None
    ):
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"Intensity must be in range [0.0, 1.0], got {intensity}")
        
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.intensity = float(intensity)
        self.start_time_utc = pd.to_datetime(start_time_utc, utc=True) if start_time_utc else None
        self.end_time_utc = pd.to_datetime(end_time_utc, utc=True) if end_time_utc else None
        self.affected_segments: Optional[Set[int]] = set(affected_segments) if affected_segments else None

    @property
    def is_synthetic(self) -> bool:
        return self.scenario_id != "BASELINE" and self.intensity > 0.0

    def is_active(self, event_timestamp_utc: str, segment: int) -> bool:
        """Determines if scenario conditions apply to a specific event."""
        if self.intensity <= 0.0:
            return False

        # Spatial localization check
        if self.affected_segments is not None and segment not in self.affected_segments:
            return False

        # Temporal localization check
        if self.start_time_utc or self.end_time_utc:
            ts = pd.to_datetime(event_timestamp_utc, utc=True)
            if self.start_time_utc and ts < self.start_time_utc:
                return False
            if self.end_time_utc and ts > self.end_time_utc:
                return False

        return True

    @abstractmethod
    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        Applies disturbance to observable features and simulated ground truth.
        
        Returns:
            Tuple of (modified_features, modified_ground_truth_eta, disturbance_applied_bool)
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Returns scenario metadata dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "intensity": self.intensity,
            "synthetic_disturbance": self.is_synthetic,
            "start_time_utc": str(self.start_time_utc) if self.start_time_utc else None,
            "end_time_utc": str(self.end_time_utc) if self.end_time_utc else None,
            "affected_segments": sorted(list(self.affected_segments)) if self.affected_segments else "GLOBAL"
        }


# =====================================================================
# Concrete Scenarios
# =====================================================================

class BaselineScenario(BaseScenario):
    """Control Group: Pure unmodified historical replay."""

    def __init__(self):
        super().__init__(
            scenario_id="BASELINE",
            name="Baseline Replay",
            description="Pure real historical telemetry replay without disturbance (Control Group).",
            intensity=0.0
        )

    def is_active(self, event_timestamp_utc: str, segment: int) -> bool:
        return False

    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        # Return exact identical copies without modification
        return dict(features), float(ground_truth_eta), False


class RushHourScenario(BaseScenario):
    """Simulates rush hour peak traffic surges during morning/evening windows."""

    def __init__(
        self,
        intensity: float = 0.75,
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None,
        affected_segments: Optional[List[int]] = None
    ):
        super().__init__(
            scenario_id="RUSH_HOUR",
            name="Rush Hour Congestion Surge",
            description="Simulates peak diurnal traffic volume with elevated delay ratios and segment running times.",
            intensity=intensity,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            affected_segments=affected_segments
        )

    def is_active(self, event_timestamp_utc: str, segment: int) -> bool:
        if not super().is_active(event_timestamp_utc, segment):
            return False
        # If no explicit date window given, apply based on diurnal peak hours (07:00-09:00 or 16:30-18:30)
        if not self.start_time_utc and not self.end_time_utc:
            ts = pd.to_datetime(event_timestamp_utc, utc=True)
            mins = ts.hour * 60 + ts.minute
            is_peak = (7 * 60 <= mins <= 9 * 60) or (16 * 60 + 30 <= mins <= 18 * 60 + 30)
            return is_peak
        return True

    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        if not self.is_active(event_timestamp_utc, segment):
            return dict(features), float(ground_truth_eta), False

        mod_feat = dict(features)
        mod_feat["is_peak_period"] = 1
        mod_feat["congestion_proxy"] = 1
        
        delay_scale = 1.0 + (0.60 * self.intensity)
        mod_feat["segment_delay_ratio"] = round(float(mod_feat.get("segment_delay_ratio", 1.0)) * delay_scale, 4)
        mod_feat["previous_segment_run_time"] = round(float(mod_feat.get("previous_segment_run_time", 120.0)) * (1.0 + 0.35 * self.intensity), 2)
        mod_feat["rolling_prev_segment_mean"] = round(float(mod_feat.get("rolling_prev_segment_mean", 120.0)) * (1.0 + 0.35 * self.intensity), 2)

        # Ground truth simulated segment running time increases
        sim_gt = round(ground_truth_eta * (1.0 + 0.70 * self.intensity), 2)
        return mod_feat, sim_gt, True


class HeavyRainScenario(BaseScenario):
    """Simulates monsoon precipitation and wet asphalt vehicle deceleration."""

    def __init__(
        self,
        intensity: float = 0.80,
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None,
        affected_segments: Optional[List[int]] = None
    ):
        super().__init__(
            scenario_id="HEAVY_RAIN",
            name="Heavy Monsoon Rain",
            description="Simulates torrential precipitation, saturated humidity, and roadway traction reduction.",
            intensity=intensity,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            affected_segments=affected_segments
        )

    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        if not self.is_active(event_timestamp_utc, segment):
            return dict(features), float(ground_truth_eta), False

        mod_feat = dict(features)
        mod_feat["precipitation"] = round(max(float(mod_feat.get("precipitation", 0.0)), 16.0 * self.intensity), 2)
        mod_feat["rain"] = round(max(float(mod_feat.get("rain", 0.0)), 14.0 * self.intensity), 2)
        mod_feat["relative_humidity_2m"] = round(min(100.0, max(float(mod_feat.get("relative_humidity_2m", 70.0)), 88.0 + 10.0 * self.intensity)), 1)
        mod_feat["weather_code"] = 65 if self.intensity > 0.5 else 63  # Heavy/Moderate rain code

        mod_feat["segment_delay_ratio"] = round(float(mod_feat.get("segment_delay_ratio", 1.0)) * (1.0 + 0.25 * self.intensity), 4)

        sim_gt = round(ground_truth_eta * (1.0 + 0.35 * self.intensity), 2)
        return mod_feat, sim_gt, True


class CongestionSurgeScenario(BaseScenario):
    """Simulates sudden unexpected non-recurring traffic corridor bottleneck."""

    def __init__(
        self,
        intensity: float = 0.70,
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None,
        affected_segments: Optional[List[int]] = None
    ):
        super().__init__(
            scenario_id="CONGESTION_SURGE",
            name="Corridor Congestion Surge",
            description="Simulates sudden severe traffic saturation, elevating historical delay ratios.",
            intensity=intensity,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            affected_segments=affected_segments
        )

    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        if not self.is_active(event_timestamp_utc, segment):
            return dict(features), float(ground_truth_eta), False

        mod_feat = dict(features)
        mod_feat["congestion_proxy"] = 1
        mod_feat["segment_delay_ratio"] = round(float(mod_feat.get("segment_delay_ratio", 1.0)) * (1.0 + 1.10 * self.intensity), 4)
        mod_feat["previous_segment_run_time"] = round(float(mod_feat.get("previous_segment_run_time", 120.0)) * (1.0 + 0.75 * self.intensity), 2)
        mod_feat["rolling_prev_segment_mean"] = round(float(mod_feat.get("rolling_prev_segment_mean", 120.0)) * (1.0 + 0.75 * self.intensity), 2)

        sim_gt = round(ground_truth_eta * (1.0 + 1.05 * self.intensity), 2)
        return mod_feat, sim_gt, True


class RoadIncidentScenario(BaseScenario):
    """Simulates localized lane blockage or collision on designated segments."""

    def __init__(
        self,
        intensity: float = 0.85,
        affected_segments: Optional[List[int]] = None,
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None
    ):
        # Default incident locus on Kandy urban bottleneck segments (e.g. 8, 9, 10)
        target_segs = affected_segments if affected_segments is not None else [8, 9, 10]
        super().__init__(
            scenario_id="ROAD_INCIDENT",
            name="Localized Road Incident",
            description="Simulates road closure / incident causing localized queueing on specific route segments.",
            intensity=intensity,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            affected_segments=target_segs
        )

    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        if not self.is_active(event_timestamp_utc, segment):
            return dict(features), float(ground_truth_eta), False

        mod_feat = dict(features)
        mod_feat["congestion_proxy"] = 1
        mod_feat["segment_delay_ratio"] = round(float(mod_feat.get("segment_delay_ratio", 1.0)) * (1.0 + 1.50 * self.intensity), 4)
        
        # Additive blockage queue delay + scaling factor
        sim_gt = round(ground_truth_eta * (1.0 + 1.40 * self.intensity) + (80.0 * self.intensity), 2)
        return mod_feat, sim_gt, True


class DwellTimeScenario(BaseScenario):
    """Simulates passenger boarding/alighting surges causing extended dwell times."""

    def __init__(
        self,
        intensity: float = 0.75,
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None,
        affected_segments: Optional[List[int]] = None
    ):
        super().__init__(
            scenario_id="DWELL_SURGE",
            name="Passenger Dwell-Time Surge",
            description="Simulates heavy boarding/alighting queues at stops, adding dwell burden.",
            intensity=intensity,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            affected_segments=affected_segments
        )

    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        if not self.is_active(event_timestamp_utc, segment):
            return dict(features), float(ground_truth_eta), False

        mod_feat = dict(features)
        mod_feat["cumulative_trip_time_sec"] = round(float(mod_feat.get("cumulative_trip_time_sec", 0.0)) + (90.0 * self.intensity), 2)
        mod_feat["previous_segment_run_time"] = round(float(mod_feat.get("previous_segment_run_time", 120.0)) + (40.0 * self.intensity), 2)

        # Additive stop dwell burden
        sim_gt = round(ground_truth_eta + (65.0 * self.intensity), 2)
        return mod_feat, sim_gt, True


class CombinedDisruptionScenario(BaseScenario):
    """Compound multi-factor distribution shift (Monsoon Rain + Congestion Surge + Dwell Surge)."""

    def __init__(
        self,
        intensity: float = 0.85,
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None,
        affected_segments: Optional[List[int]] = None
    ):
        super().__init__(
            scenario_id="COMBINED_DISRUPTION",
            name="Compound Severe Disruption",
            description="Composes monsoon downpour, corridor congestion surge, and boarding queue spikes.",
            intensity=intensity,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            affected_segments=affected_segments
        )
        self.rain_sub = HeavyRainScenario(intensity=intensity, start_time_utc=start_time_utc, end_time_utc=end_time_utc, affected_segments=affected_segments)
        self.cong_sub = CongestionSurgeScenario(intensity=intensity, start_time_utc=start_time_utc, end_time_utc=end_time_utc, affected_segments=affected_segments)
        self.dwell_sub = DwellTimeScenario(intensity=intensity, start_time_utc=start_time_utc, end_time_utc=end_time_utc, affected_segments=affected_segments)

    def apply(
        self,
        features: Dict[str, Any],
        ground_truth_eta: float,
        event_timestamp_utc: str,
        segment: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        if not self.is_active(event_timestamp_utc, segment):
            return dict(features), float(ground_truth_eta), False

        # Apply composed transformations
        f1, gt1, _ = self.rain_sub.apply(features, ground_truth_eta, event_timestamp_utc, segment)
        f2, gt2, _ = self.cong_sub.apply(f1, gt1, event_timestamp_utc, segment)
        f3, gt3, _ = self.dwell_sub.apply(f2, gt2, event_timestamp_utc, segment)

        return f3, gt3, True
