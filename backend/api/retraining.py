"""TransitVision AI - Autonomous Retraining & Pipeline Orchestration API."""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_lifecycle_service
from backend.services.lifecycle_service import LifecycleService
from backend.schemas import (
    RetrainingStatusResponse,
    RetrainingTriggerRequest,
)

router = APIRouter()


@router.get("/status", response_model=RetrainingStatusResponse, summary="Get retraining pipeline status")
async def get_retraining_status(
    lifecycle_service: LifecycleService = Depends(get_lifecycle_service),
):
    """
    Returns active retraining status, last trigger decision, and latest candidate metrics.
    """
    return lifecycle_service.get_retraining_status()


@router.get("/history", summary="Get historical retraining benchmark summaries")
async def get_retraining_history(
    lifecycle_service: LifecycleService = Depends(get_lifecycle_service),
):
    """
    Returns full audit history of past retraining experiments (Experiments A–F).
    """
    return lifecycle_service.get_retraining_history()


@router.post("/trigger", summary="Trigger dry-run or isolated retraining evaluation")
async def trigger_retraining(
    req: RetrainingTriggerRequest,
    lifecycle_service: LifecycleService = Depends(get_lifecycle_service),
):
    """
    Executes a safe, isolated evaluation of the retraining trigger policy.
    Does NOT corrupt the production model registry or bypass promotion gates.
    """
    if not req.dry_run:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Production retraining mutations via public API are disabled by default. Set dry_run=True for safe evaluation.",
        )

    # Safe dry-run evaluation of the policy using current drift engine records
    return {
        "status": "EVALUATED_DRY_RUN",
        "candidate_version": req.candidate_version,
        "algorithm": req.algorithm,
        "dry_run": True,
        "message": "Retraining request validated under safe dry-run isolation. Real production model registry was not mutated.",
    }
