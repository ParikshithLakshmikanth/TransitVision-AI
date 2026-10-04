"""TransitVision AI - Normalized System Events API."""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from backend.dependencies import get_event_service
from backend.services.event_service import EventService
from backend.schemas import SystemEvent, SystemEventType

router = APIRouter()


@router.get("", response_model=List[SystemEvent], summary="Get recent normalized system events")
async def get_recent_events(
    limit: int = Query(default=50, ge=1, le=500, description="Number of recent events to retrieve"),
    event_type: Optional[SystemEventType] = Query(default=None, description="Filter by event type"),
    event_service: EventService = Depends(get_event_service),
):
    """
    Returns rolling chronological buffer of normalized events emitted across all system layers.
    """
    return event_service.get_recent_events(limit=limit, event_type=event_type)
