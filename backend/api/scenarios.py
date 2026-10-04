"""TransitVision AI - Scenario Engine API."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_simulator_service
from backend.services.simulator_service import SimulatorService
from backend.schemas import (
    ScenarioListResponse,
    ScenarioInfo,
    ScenarioActivateRequest,
    ScenarioActivateResponse,
)
from simulator.scenario_engine import ScenarioEngine, SCENARIO_REGISTRY

router = APIRouter()

SCENARIO_METADATA = {
    "BASELINE": {
        "name": "Baseline Control Replay",
        "description": "Nominal corridor operation without disturbances or synthetic alterations.",
        "default_intensity": 0.0,
        "is_synthetic": False,
        "affected_segments": None,
    },
    "RUSH_HOUR": {
        "name": "Rush Hour Corridor Delay",
        "description": "Peak-hour traffic congestion and elevated passenger boarding volumes.",
        "default_intensity": 0.75,
        "is_synthetic": True,
        "affected_segments": None,
    },
    "HEAVY_RAIN": {
        "name": "Severe Heavy Rain Weather Disruption",
        "description": "Tropical torrential downpour impairing visibility and corridor road speeds.",
        "default_intensity": 0.85,
        "is_synthetic": True,
        "affected_segments": None,
    },
    "CONGESTION_SURGE": {
        "name": "Corridor Congestion Surge",
        "description": "Severe bottleneck gridlock across key corridor transit nodes.",
        "default_intensity": 0.80,
        "is_synthetic": True,
        "affected_segments": None,
    },
    "ROAD_INCIDENT": {
        "name": "Road Obstruction & Spatial Incident",
        "description": "Spurred localized delays and bottlenecks across specific route segments.",
        "default_intensity": 0.80,
        "is_synthetic": True,
        "affected_segments": [3, 4, 5],
    },
    "DWELL_SURGE": {
        "name": "Passenger Dwell Time Surge",
        "description": "Elevated bus stop boarding surges multiplying station dwell durations.",
        "default_intensity": 0.75,
        "is_synthetic": True,
        "affected_segments": None,
    },
    "COMBINED_DISRUPTION": {
        "name": "Multi-Factor Combined Corridor Disruption",
        "description": "Simultaneous compounding of heavy rain, road incident, and peak dwell surges.",
        "default_intensity": 0.90,
        "is_synthetic": True,
        "affected_segments": [3, 4, 5],
    },
}


@router.get("", response_model=ScenarioListResponse, summary="List available transit disturbance scenarios")
async def list_scenarios(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Returns full catalog of validated Phase 7 disturbance scenarios and current active scenario.
    """
    scenarios_list = []
    for sc_id in SCENARIO_REGISTRY.keys():
        meta = SCENARIO_METADATA.get(sc_id, {
            "name": sc_id.replace("_", " ").title(),
            "description": "Disturbance scenario",
            "default_intensity": 0.75,
            "is_synthetic": sc_id != "BASELINE",
            "affected_segments": None,
        })
        scenarios_list.append(
            ScenarioInfo(
                scenario_id=sc_id,
                name=meta["name"],
                description=meta["description"],
                default_intensity=meta["default_intensity"],
                is_synthetic=meta["is_synthetic"],
                affected_segments=meta["affected_segments"],
            )
        )

    st = simulator_service.simulator
    return ScenarioListResponse(
        active_scenario_id=st.scenario.scenario_id,
        active_scenario_intensity=st.scenario.intensity,
        available_scenarios=scenarios_list,
    )


@router.post("/activate", response_model=ScenarioActivateResponse, summary="Activate disturbance scenario")
async def activate_scenario(
    req: ScenarioActivateRequest,
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Activates an existing Phase 7 scenario with the specified intensity factor.
    """
    try:
        res = simulator_service.activate_scenario(
            scenario_id=req.scenario_id,
            intensity=req.intensity,
        )
        return ScenarioActivateResponse(
            success=True,
            message=f"Scenario '{res['scenario_name']}' successfully activated.",
            scenario_id=res["scenario_id"],
            scenario_name=res["scenario_name"],
            intensity=res["intensity"],
            is_synthetic=res["is_synthetic"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
