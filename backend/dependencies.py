"""TransitVision AI - FastAPI Dependencies & Singleton Service Providers."""
from functools import lru_cache
from typing import Generator

from backend.services.event_service import EventService
from backend.services.monitoring_service import MonitoringService
from backend.services.simulator_service import SimulatorService
from backend.services.model_service import ModelService
from backend.services.lifecycle_service import LifecycleService

# Singletons shared across route requests
_event_service = EventService()
_monitoring_service = MonitoringService()
_simulator_service = SimulatorService(
    event_service=_event_service,
    monitoring_service=_monitoring_service,
)
_model_service = ModelService()
_lifecycle_service = LifecycleService()


def get_event_service() -> EventService:
    return _event_service


def get_monitoring_service() -> MonitoringService:
    return _monitoring_service


def get_simulator_service() -> SimulatorService:
    return _simulator_service


def get_model_service() -> ModelService:
    return _model_service


def get_lifecycle_service() -> LifecycleService:
    return _lifecycle_service
