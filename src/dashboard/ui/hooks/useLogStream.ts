import { useState, useEffect, useRef, useCallback } from 'react';
import type { LogMessage, LogEvent } from '../../server/types';

interface UseLogStreamOptions {
  containerId: string | null;
  apiBase: string;
  tail?: number;
  maxLines?: number;
}

interface LogStreamState {
  lines: LogMessage[];
  connected: boolean;
  error: string | null;
  droppedCount: number;
}

export function useLogStream(options: UseLogStreamOptions): LogStreamState & { clear: () => void } {
  const { containerId, apiBase, tail = 200, maxLines = 2000 } = options;
  const [lines, setLines] = useState<LogMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [droppedCount, setDroppedCount] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  const clear = useCallback(() => {
    setLines([]);
    setDroppedCount(0);
  }, []);

  useEffect(() => {
    if (!containerId) return;

    const wsUrl = apiBase.replace(/^http/, 'ws');
    const ws = new WebSocket(`${wsUrl}/api/logs/${containerId}?tail=${tail}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      const data: LogEvent = JSON.parse(event.data);

      if (data.type === 'log') {
        setLines((prev) => {
          const next = [...prev, data];
          return next.length > maxLines ? next.slice(-maxLines) : next;
        });
      } else if (data.type === 'dropped') {
        setDroppedCount((prev) => prev + (data.count ?? 0));
      } else if (data.type === 'closed') {
        setError(data.reason ?? 'Stream closed');
        setConnected(false);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
      setConnected(false);
    };

    ws.onclose = () => {
      setConnected(false);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [containerId, apiBase, tail, maxLines]);

  return { lines, connected, error, droppedCount, clear };
}
