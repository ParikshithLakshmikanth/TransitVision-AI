"""TransitVision AI - Online Error Metrics & Evaluation API."""
from fastapi import APIRouter, Depends

from backend.dependencies import get_simulator_service
from backend.services.simulator_service import SimulatorService
from backend.schemas import MetricsResponse, MetricsHistoryResponse

router = APIRouter()


@router.get("", response_model=MetricsResponse, summary="Get current online performance metrics")
async def get_metrics(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Returns online streaming evaluation metrics including MAE, RMSE,
    Median Absolute Error, P90/P95 tail errors, mean signed bias, and average latency.
    """
    return simulator_service.get_metrics()


@router.get("/history", response_model=MetricsHistoryResponse, summary="Get error metrics history over time")
async def get_metrics_history(
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Returns historical evaluation metrics snapshots grouped by rolling streaming windows.
    """
    return simulator_service.get_metrics_history()
