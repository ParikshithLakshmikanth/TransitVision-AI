"""TransitVision AI - Production FastAPI Backend & Real-Time Streaming Gateway."""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.api import api_router
from backend.dependencies import get_event_service, get_simulator_service

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("TransitVision.Backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and graceful shutdown."""
    logger.info("Initializing TransitVision AI Backend Service...")
    sim_svc = get_simulator_service()
    logger.info(
        f"Simulator Initialized: {sim_svc.simulator.simulation_id} "
        f"(Records: {sim_svc.simulator.total_records}, Model: {sim_svc.simulator.predictor.metadata.get('model_id') if sim_svc.simulator.predictor else 'None'})"
    )
    yield
    logger.info("Shutting down TransitVision AI Backend Service...")
    sim_svc.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# -------------------------------------------------------------
# CORS Middleware
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# -------------------------------------------------------------
# Global Exception Handlers
# -------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation Error", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "message": str(exc)},
    )


# -------------------------------------------------------------
# Mount REST API Routes
# -------------------------------------------------------------
app.include_router(api_router, prefix=settings.api_v1_prefix)


# -------------------------------------------------------------
# WebSocket Live Telemetry & Event Streaming Endpoint
# -------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """
    Real-time WebSocket endpoint streaming authoritative simulator events,
    predictions, ground-truth outcomes, drift alarms, and lifecycle state changes.
    """
    event_svc = get_event_service()
    await event_svc.connect(websocket)
    try:
        # Initial greeting / connection acknowledgment
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "message": "Connected to TransitVision AI Live Telemetry Gateway",
            "server_time_utc": str(asyncio.get_event_loop().time()),
        })
        
        while True:
            # Handle client heartbeats/messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        event_svc.disconnect(websocket)
    except asyncio.CancelledError:
        event_svc.disconnect(websocket)
    except Exception as e:
        logger.debug(f"WebSocket client error: {e}")
        event_svc.disconnect(websocket)


@app.get("/", summary="Root Health Check")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "ONLINE",
        "docs": "/docs",
        "api": "/api/health",
        "websocket": "/ws/live",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
