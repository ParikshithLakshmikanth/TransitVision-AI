"""TransitVision AI - ETA Predictions & Error Evaluation API."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.dependencies import get_simulator_service
from backend.services.simulator_service import SimulatorService
from backend.schemas import PredictionsListResponse, PredictionItem

router = APIRouter()


@router.get("", response_model=PredictionsListResponse, summary="Get recent production predictions")
async def get_predictions(
    limit: int = Query(default=50, ge=1, le=500, description="Max number of recent predictions to return"),
    trip_id: Optional[str] = Query(default=None, description="Filter by trip ID"),
    deviceid: Optional[str] = Query(default=None, description="Filter by bus device ID"),
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Returns recent model predictions with corresponding ground truth outcomes,
    signed errors, percentage errors, and inference latency.
    """
    preds = simulator_service.get_predictions(limit=limit, trip_id=trip_id, deviceid=deviceid)
    return PredictionsListResponse(
        total_predictions=len(preds),
        limit=limit,
        predictions=preds,
    )


@router.get("/{prediction_id}", response_model=PredictionItem, summary="Get prediction details by ID")
async def get_prediction_by_id(
    prediction_id: str,
    simulator_service: SimulatorService = Depends(get_simulator_service),
):
    """
    Retrieves full details and evaluation metrics for a specific prediction ID.
    """
    pred = simulator_service.get_prediction_by_id(prediction_id)
    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID '{prediction_id}' not found in recent history buffer.",
        )
    return pred
