"""TransitVision AI - Scenario Engine Coordinator.
Manages scenario registration, instantiation, validation, and lifecycle metadata.
"""
import logging
from typing import Dict, Any, Type, Optional, List
from simulator.scenarios import (
    BaseScenario,
    BaselineScenario,
    RushHourScenario,
    HeavyRainScenario,
    CongestionSurgeScenario,
    RoadIncidentScenario,
    DwellTimeScenario,
    CombinedDisruptionScenario
)

logger = logging.getLogger("TransitVision.ScenarioEngine")

SCENARIO_REGISTRY: Dict[str, Type[BaseScenario]] = {
    "BASELINE": BaselineScenario,
    "RUSH_HOUR": RushHourScenario,
    "HEAVY_RAIN": HeavyRainScenario,
    "CONGESTION_SURGE": CongestionSurgeScenario,
    "ROAD_INCIDENT": RoadIncidentScenario,
    "DWELL_SURGE": DwellTimeScenario,
    "COMBINED_DISRUPTION": CombinedDisruptionScenario
}

SCENARIO_ALIASES: Dict[str, str] = {
    "baseline": "BASELINE",
    "rush_hour": "RUSH_HOUR",
    "rushhour": "RUSH_HOUR",
    "rain": "HEAVY_RAIN",
    "heavy_rain": "HEAVY_RAIN",
    "congestion": "CONGESTION_SURGE",
    "congestion_surge": "CONGESTION_SURGE",
    "incident": "ROAD_INCIDENT",
    "road_incident": "ROAD_INCIDENT",
    "dwell": "DWELL_SURGE",
    "dwell_surge": "DWELL_SURGE",
    "dwell_increase": "DWELL_SURGE",
    "combined": "COMBINED_DISRUPTION",
    "combined_disruption": "COMBINED_DISRUPTION"
}


class ScenarioEngine:
    """Factory and manager for transit disturbance scenarios."""

    @staticmethod
    def get_scenario(
        scenario_name_or_id: str = "BASELINE",
        intensity: float = 0.0,
        **kwargs
    ) -> BaseScenario:
        """
        Instantiates a scenario by ID or alias with validated intensity and parameters.
        """
        canonical_id = SCENARIO_ALIASES.get(scenario_name_or_id.lower().strip(), scenario_name_or_id.upper().strip())
        
        if canonical_id not in SCENARIO_REGISTRY:
            valid_keys = list(SCENARIO_REGISTRY.keys()) + list(SCENARIO_ALIASES.keys())
            raise ValueError(f"Unknown scenario '{scenario_name_or_id}'. Valid options: {valid_keys}")

        scenario_cls = SCENARIO_REGISTRY[canonical_id]
        
        if canonical_id == "BASELINE":
            return BaselineScenario()
        
        return scenario_cls(intensity=intensity, **kwargs)

    @staticmethod
    def list_scenarios() -> List[Dict[str, Any]]:
        """Returns metadata for all available scenario classes."""
        return [
            {
                "scenario_id": k,
                "class_name": cls.__name__,
                "doc": cls.__doc__
            }
            for k, cls in SCENARIO_REGISTRY.items()
        ]
