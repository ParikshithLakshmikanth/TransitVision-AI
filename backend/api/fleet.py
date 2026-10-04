"""TransitVision AI - Bus Fleet Telemetry & Live State API."""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_simulator_service
from backend.services.simulator_service import SimulatorService
from backend.schemas import FleetResponse, BusStateResponse

router = APIRouter()


@router.get("", response_model=FleetResponse, summary="Get active bus fleet state")
async def get_fleet(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Returns current live snapshot of all active bus devices along the Route 654 corridor,
    including current segment, latest predictions, resolved outcomes, and errors.
    """
    buses = simulator_service.get_fleet()
    return FleetResponse(
        total_active_buses=len(buses),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        buses=buses,
    )


@router.get("/{bus_id}", response_model=BusStateResponse, summary="Get specific bus telemetry and state")
async def get_bus_state(
    bus_id: str,
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Returns spatiotemporal state and latest prediction evaluation for an individual bus device.
    """
    bus = simulator_service.get_bus_state(bus_id)
    if not bus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bus with ID '{bus_id}' is not currently active in the simulator fleet.",
        )
    return bus
