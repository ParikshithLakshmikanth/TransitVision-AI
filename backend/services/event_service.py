"""TransitVision AI - Unified Event System & WebSocket Broadcast Manager."""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Optional
from fastapi import WebSocket

from backend.schemas import SystemEvent, SystemEventType

logger = logging.getLogger("TransitVision.EventService")


class EventService:
    """
    Centralized event manager maintaining rolling event logs
    and broadcasting real-time system events to WebSocket clients.
    """

    def __init__(self, max_buffer_size: int = 500):
        self.max_buffer_size = max_buffer_size
        self._events_buffer: List[SystemEvent] = []
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Registers a new WebSocket client connection."""
        await websocket.accept()
        self._active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self._active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregisters a WebSocket client connection."""
        self._active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self._active_connections)}")

    async def emit_event(
        self,
        event_type: SystemEventType,
        component: str,
        payload: Dict[str, Any],
        model_version: str = "v1.0.0",
        scenario_id: str = "BASELINE",
    ) -> SystemEvent:
        """
        Emits a structured SystemEvent, records it to the rolling buffer,
        and broadcasts it asynchronously to all active WebSocket clients.
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        event_id = f"EVT_{event_type.value}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
        
        event = SystemEvent(
            event_id=event_id,
            timestamp_utc=now_ts,
            event_type=event_type,
            component=component,
            model_version=model_version,
            scenario_id=scenario_id,
            payload=payload,
        )

        # Buffer event
        self._events_buffer.append(event)
        if len(self._events_buffer) > self.max_buffer_size:
            self._events_buffer.pop(0)

        # Broadcast to WebSocket clients
        await self._broadcast(event)
        return event

    def emit_event_sync(
        self,
        event_type: SystemEventType,
        component: str,
        payload: Dict[str, Any],
        model_version: str = "v1.0.0",
        scenario_id: str = "BASELINE",
    ) -> SystemEvent:
        """Synchronous helper for emitting events from worker threads."""
        now_ts = datetime.now(timezone.utc).isoformat()
        event_id = f"EVT_{event_type.value}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
        
        event = SystemEvent(
            event_id=event_id,
            timestamp_utc=now_ts,
            event_type=event_type,
            component=component,
            model_version=model_version,
            scenario_id=scenario_id,
            payload=payload,
        )

        self._events_buffer.append(event)
        if len(self._events_buffer) > self.max_buffer_size:
            self._events_buffer.pop(0)

        # Schedule async broadcast if an event loop is running
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast(event))
        except RuntimeError:
            pass  # No running event loop in current thread

        return event

    async def _broadcast(self, event: SystemEvent) -> None:
        """Broadcasts event payload to all connected clients."""
        if not self._active_connections:
            return

        message_str = event.model_dump_json()
        disconnected_clients = set()

        for connection in list(self._active_connections):
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.debug(f"Error sending WebSocket message: {e}")
                disconnected_clients.add(connection)

        for conn in disconnected_clients:
            self._active_connections.discard(conn)

    def get_recent_events(self, limit: int = 50, event_type: Optional[SystemEventType] = None) -> List[SystemEvent]:
        """Returns recent system events with optional filtering."""
        if event_type:
            filtered = [e for e in self._events_buffer if e.event_type == event_type]
            return filtered[-limit:]
        return self._events_buffer[-limit:]

    def clear(self) -> None:
        """Clears the event buffer."""
        self._events_buffer.clear()
