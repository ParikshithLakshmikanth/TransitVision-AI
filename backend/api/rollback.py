"""TransitVision AI - Rollback Manager & Watchdog Status API."""
from fastapi import APIRouter, Depends

from backend.dependencies import get_lifecycle_service
from backend.services.lifecycle_service import LifecycleService
from backend.schemas import RollbackStatusResponse

router = APIRouter()


@router.get("/status", response_model=RollbackStatusResponse, summary="Get rollback status and watchdog policies")
async def get_rollback_status(
    lifecycle_service: LifecycleService = Depends(get_lifecycle_service),
):
    """
    Returns read-only status of the post-promotion watchdog sentinel,
    configured safety thresholds (1.25x MAE, 20ms latency, 0 exceptions),
    and historical rollback records.
    """
    return lifecycle_service.get_rollback_status()
