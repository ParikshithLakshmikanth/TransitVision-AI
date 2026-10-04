"""TransitVision AI - Digital Transit Simulator Control API."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_simulator_service
from backend.services.simulator_service import SimulatorService
from backend.schemas import (
    SimulationStatusResponse,
    SimulationControlResponse,
    SimulationStepRequest,
    SimulationSeekRequest,
    SimulationSpeedRequest,
    PredictionItem,
)

router = APIRouter()


@router.get("/status", response_model=SimulationStatusResponse, summary="Get simulator operational state")
async def get_simulation_status(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Returns current simulator cursor position, total records, active status,
    replay speed factor, and active scenario metadata.
    """
    return simulator_service.get_status()


@router.post("/start", response_model=SimulationControlResponse, summary="Start simulation replay")
async def start_simulation(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Starts chronological telemetry stream replay.
    """
    st = simulator_service.start()
    return SimulationControlResponse(
        success=True,
        message="Simulation replay started.",
        simulator_state=st,
    )


@router.post("/pause", response_model=SimulationControlResponse, summary="Pause simulation replay")
async def pause_simulation(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Pauses streaming playback at current cursor.
    """
    st = simulator_service.pause()
    return SimulationControlResponse(
        success=True,
        message="Simulation replay paused.",
        simulator_state=st,
    )


@router.post("/resume", response_model=SimulationControlResponse, summary="Resume simulation replay")
async def resume_simulation(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Resumes playback from current pause point.
    """
    st = simulator_service.resume()
    return SimulationControlResponse(
        success=True,
        message="Simulation replay resumed.",
        simulator_state=st,
    )


@router.post("/stop", response_model=SimulationControlResponse, summary="Stop simulation replay")
async def stop_simulation(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Terminates active simulation playback.
    """
    st = simulator_service.stop()
    return SimulationControlResponse(
        success=True,
        message="Simulation replay stopped.",
        simulator_state=st,
    )


@router.post("/reset", response_model=SimulationControlResponse, summary="Reset simulation replay to beginning")
async def reset_simulation(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Resets cursor back to record index 0 and flushes online metrics.
    """
    st = simulator_service.reset()
    return SimulationControlResponse(
        success=True,
        message="Simulation reset to cursor 0.",
        simulator_state=st,
    )


@router.post("/step", response_model=List[PredictionItem], summary="Manually advance simulation by N steps")
async def step_simulation(
    req: SimulationStepRequest,
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Steps the simulator deterministically by n records and returns generated predictions.
    """
    return simulator_service.step(n=req.n)


@router.post("/seek", response_model=SimulationControlResponse, summary="Seek simulation to specific record index")
async def seek_simulation(
    req: SimulationSeekRequest,
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Seeks cursor directly to record index.
    """
    try:
        st = simulator_service.seek(index=req.index)
        return SimulationControlResponse(
            success=True,
            message=f"Simulation sought to index {req.index}.",
            simulator_state=st,
        )
    except IndexError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/speed", response_model=SimulationControlResponse, summary="Set replay speed factor")
async def set_simulation_speed(
    req: SimulationSpeedRequest,
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Sets replay speed multiplier (0.1x to 100.0x).
    """
    st = simulator_service.set_speed(speed=req.speed)
    return SimulationControlResponse(
        success=True,
        message=f"Simulation replay speed set to {req.speed}x.",
        simulator_state=st,
    )
