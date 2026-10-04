"""TransitVision AI - Model Registry & Artifact Metadata API."""
from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_model_service
from backend.services.model_service import ModelService
from backend.schemas import ModelListResponse, ModelDetail

router = APIRouter()


@router.get("", response_model=ModelListResponse, summary="List all models in the registry")
async def list_models(
    model_service: ModelService = Depends(get_model_service),
):
    """
    Returns full catalog of all registered models across candidate, validated,
    shadow, canary, production, rejected, and rolled-back stages.
    """
    return model_service.list_models()


@router.get("/production", response_model=ModelDetail, summary="Get active PRODUCTION model details")
async def get_production_model(
    model_service: ModelService = Depends(get_model_service),
):
    """
    Returns metadata, hyperparameters, and lineage of the actively serving champion model.
    """
    prod = model_service.get_production_model()
    if not prod:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active PRODUCTION model currently designated in the registry.",
        )
    return prod


@router.get("/{model_id}", response_model=ModelDetail, summary="Get model details by model ID")
async def get_model_by_id(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Retrieves full record, metrics, and hyperparameters for a specific model ID.
    """
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID '{model_id}' was not found in the registry.",
        )
    return model
