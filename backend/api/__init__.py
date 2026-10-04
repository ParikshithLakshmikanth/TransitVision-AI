"""TransitVision AI - API Routers Package."""
from fastapi import APIRouter

from backend.api.health import router as health_router
from backend.api.fleet import router as fleet_router
from backend.api.predictions import router as predictions_router
from backend.api.metrics import router as metrics_router
from backend.api.drift import router as drift_router
from backend.api.models import router as models_router
from backend.api.simulation import router as simulation_router
from backend.api.scenarios import router as scenarios_router
from backend.api.retraining import router as retraining_router
from backend.api.rollback import router as rollback_router
from backend.api.lifecycle import router as lifecycle_router
from backend.api.events import router as events_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["Health"])
api_router.include_router(fleet_router, prefix="/fleet", tags=["Fleet"])
api_router.include_router(predictions_router, prefix="/predictions", tags=["Predictions"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
api_router.include_router(drift_router, prefix="/drift", tags=["Drift Detection"])
api_router.include_router(models_router, prefix="/models", tags=["Model Registry"])
api_router.include_router(simulation_router, prefix="/simulation", tags=["Simulation Control"])
api_router.include_router(scenarios_router, prefix="/scenarios", tags=["Scenario Engine"])
api_router.include_router(retraining_router, prefix="/retraining", tags=["Autonomous Retraining"])
api_router.include_router(rollback_router, prefix="/rollback", tags=["Rollback Manager"])
api_router.include_router(lifecycle_router, prefix="/lifecycle", tags=["MLOps Lifecycle"])
api_router.include_router(events_router, prefix="/events", tags=["System Events"])
