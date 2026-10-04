import { useEffect, useRef, useState, useCallback } from 'react';
import { SystemEvent, SystemEventType } from '../types/api';

export type WebSocketStatus = 'CONNECTING' | 'OPEN' | 'CLOSED' | 'RECONNECTING';

export interface UseWebSocketReturn {
  status: WebSocketStatus;
  lastEvent: SystemEvent | null;
  eventsBuffer: SystemEvent[];
  sendPing: () => void;
  reconnect: () => void;
  clearEvents: () => void;
}

export function useWebSocket(maxBuffer = 200): UseWebSocketReturn {
  const [status, setStatus] = useState<WebSocketStatus>('CONNECTING');
  const [lastEvent, setLastEvent] = useState<SystemEvent | null>(null);
  const [eventsBuffer, setEventsBuffer] = useState<SystemEvent[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);
  const retryCountRef = useRef(0);

  const getWsUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    // If running via vite dev proxy on 5173, /ws/live routes to backend
    return `${protocol}//${host}/ws/live`;
  };

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      setStatus(retryCountRef.current > 0 ? 'RECONNECTING' : 'CONNECTING');
      const wsUrl = getWsUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('OPEN');
        retryCountRef.current = 0;
        
        // Start ping heartbeat
        if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = window.setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
          }
        }, 5000);
      };

      ws.onmessage = (event) => {
        try {
          if (event.data === 'pong') return;
          const parsed = JSON.parse(event.data);
          
          if (parsed.event_type) {
            const sysEvent = parsed as SystemEvent;
            setLastEvent(sysEvent);
            setEventsBuffer((prev) => {
              const updated = [...prev, sysEvent];
              return updated.length > maxBuffer ? updated.slice(updated.length - maxBuffer) : updated;
            });
          }
        } catch (e) {
          // Non-JSON or handshake message
        }
      };

      ws.onclose = () => {
        setStatus('CLOSED');
        if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
        
        // Schedule auto-reconnect with exponential backoff (max 10s)
        const delay = Math.min(1000 * Math.pow(1.5, retryCountRef.current), 10000);
        retryCountRef.current += 1;
        
        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch (err) {
      setStatus('CLOSED');
    }
  }, [maxBuffer]);

  const sendPing = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send('ping');
    }
  }, []);

  const reconnect = useCallback(() => {
    retryCountRef.current = 0;
    if (wsRef.current) {
      wsRef.current.close();
    }
    connect();
  }, [connect]);

  const clearEvents = useCallback(() => {
    setEventsBuffer([]);
    setLastEvent(null);
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return {
    status,
    lastEvent,
    eventsBuffer,
    sendPing,
    reconnect,
    clearEvents,
  };
}
