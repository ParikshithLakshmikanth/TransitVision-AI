"""TransitVision AI - Statistical Drift Detection & Monitoring API."""
from typing import List
from fastapi import APIRouter, Depends, Query

from backend.dependencies import get_monitoring_service
from backend.services.monitoring_service import MonitoringService
from backend.schemas import (
    DriftStatusResponse,
    DriftSummaryResponse,
    DriftEventResponse,
)

router = APIRouter()


@router.get("", response_model=DriftStatusResponse, summary="Get comprehensive drift monitoring status")
async def get_drift_status(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """
    Returns current statistical drift state across Covariate (Data), Performance (MAE/RMSE),
    and Concept (Residual Page-Hinkley) detection dimensions.
    """
    return monitoring_service.get_drift_status()


@router.get("/summary", response_model=DriftSummaryResponse, summary="Get drift summary metrics and frequencies")
async def get_drift_summary(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """
    Returns alert counters, detection latencies, and most frequently drifted corridor features.
    """
    return monitoring_service.get_summary()


@router.get("/events", response_model=List[DriftEventResponse], summary="Get recent drift events")
async def get_drift_events(
    limit: int = Query(default=50, ge=1, le=500, description="Max number of recent events to return"),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """
    Returns sequential log of detected drift alarms and statistical evidence.
    """
    return monitoring_service.get_recent_events(limit=limit)
