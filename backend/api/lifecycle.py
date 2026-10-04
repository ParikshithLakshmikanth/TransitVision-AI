"""TransitVision AI - MLOps Lifecycle Pipeline Graph API."""
from fastapi import APIRouter, Depends

from backend.dependencies import get_lifecycle_service
from backend.services.lifecycle_service import LifecycleService
from backend.schemas import LifecycleStateResponse

router = APIRouter()


@router.get("", response_model=LifecycleStateResponse, summary="Get full end-to-end MLOps lifecycle state")
async def get_lifecycle_state(
    lifecycle_service: LifecycleService = Depends(get_lifecycle_service),
):
    """
    Returns unified MLOps pipeline graph state across:
    DRIFT -> TRIGGER -> RETRAIN -> VALIDATE -> SHADOW -> CANARY -> PROMOTE -> MONITOR -> ROLLBACK
    """
    return lifecycle_service.get_lifecycle_state()
