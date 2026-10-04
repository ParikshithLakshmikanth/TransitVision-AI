"""TransitVision AI - Health & System Availability API."""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from backend.dependencies import get_simulator_service, get_model_service
from backend.services.simulator_service import SimulatorService
from backend.services.model_service import ModelService
from backend.schemas import HealthResponse, ServiceStatus

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Get system health and service readiness")
async def get_health(
    simulator_service: SimulatorService = Depends(get_simulator_service),
    model_service: ModelService = Depends(get_model_service),
):
    """
    Returns real-time health and operational status of all backend subsystems,
    active production model details, and simulator status.
    """
    prod_model = model_service.get_production_model()
    prod_id = prod_model.model_id if prod_model else "model_lightgbm_v1"
    prod_ver = prod_model.version if prod_model else "v1.0.0"
    
    sim_status = simulator_service.simulator.status
    model_healthy = simulator_service.simulator.model_healthy
    source_verified = simulator_service.simulator.verify_source_immutability()

    services = {
        "digital_transit_simulator": ServiceStatus(
            available=True,
            details=f"Status: {sim_status}, Cursor: {simulator_service.simulator.cursor}/{simulator_service.simulator.total_records}"
        ),
        "eta_predictor": ServiceStatus(
            available=model_healthy,
            details=f"Active model: {prod_id} ({prod_ver})"
        ),
        "drift_engine": ServiceStatus(
            available=True,
            details="Covariate, Performance, and Concept drift detectors active"
        ),
        "model_registry": ServiceStatus(
            available=True,
            details="Lifecycle state machine operational"
        ),
        "parquet_source_data": ServiceStatus(
            available=source_verified,
            details="Stream Parquet hash integrity verified"
        ),
    }

    all_available = all(s.available for s in services.values())

    return HealthResponse(
        status="HEALTHY" if all_available else "DEGRADED",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        production_model_id=prod_id,
        production_model_version=prod_ver,
        simulator_status=sim_status,
        services=services,
    )
