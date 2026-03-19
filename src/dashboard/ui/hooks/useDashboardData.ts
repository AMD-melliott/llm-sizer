import { useState, useEffect, useCallback, useRef } from 'react';
import type { DashboardSnapshot, VllmInstance, GpuDevice, GpuMetrics } from '../../server/types';

interface DashboardStatus {
  timestamp: number;
  pollIntervalMs: number;
  summary: DashboardSnapshot['summary'];
  warnings: string[];
}

interface DashboardData {
  status: DashboardStatus | null;
  instances: VllmInstance[];
  gpus: { devices: GpuDevice[]; metrics: GpuMetrics[] } | null;
  loading: boolean;
  error: string | null;
}

export function parseDashboardStatus(raw: any): DashboardStatus {
  return {
    timestamp: raw.timestamp,
    pollIntervalMs: raw.pollIntervalMs,
    summary: raw.summary,
    warnings: raw.warnings ?? [],
  };
}

export function shouldRefreshData(
  previousTimestamp: number | null,
  newTimestamp: number
): boolean {
  return previousTimestamp === null || previousTimestamp !== newTimestamp;
}

export function useDashboardData(apiBase: string): DashboardData {
  const [status, setStatus] = useState<DashboardStatus | null>(null);
  const [instances, setInstances] = useState<VllmInstance[]>([]);
  const [gpus, setGpus] = useState<DashboardData['gpus']>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const lastTimestamp = useRef<number | null>(null);
  const pollIntervalRef = useRef(5000);

  const fetchFullData = useCallback(async () => {
    try {
      const [instancesRes, gpusRes] = await Promise.all([
        fetch(`${apiBase}/api/instances`),
        fetch(`${apiBase}/api/gpus`),
      ]);
      if (instancesRes.ok) setInstances(await instancesRes.json());
      if (gpusRes.ok) setGpus(await gpusRes.json());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch data');
    }
  }, [apiBase]);

  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/status`);
      if (!res.ok) {
        setError(`Server returned ${res.status}`);
        return;
      }
      const raw = await res.json();
      const parsed = parseDashboardStatus(raw);
      setStatus(parsed);
      pollIntervalRef.current = parsed.pollIntervalMs;

      if (shouldRefreshData(lastTimestamp.current, parsed.timestamp)) {
        lastTimestamp.current = parsed.timestamp;
        await fetchFullData();
      }
      setLoading(false);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection failed');
      setLoading(false);
    }
  }, [apiBase, fetchFullData]);

  useEffect(() => {
    pollStatus();
    const interval = setInterval(() => pollStatus(), pollIntervalRef.current);
    return () => clearInterval(interval);
  }, [pollStatus]);

  return { status, instances, gpus, loading, error };
}
