// --- GPU Provider types ---

export interface GpuDevice {
  id: string;
  physicalId: number;
  logicalId?: number;
  name: string;
  vramTotalMb: number;
  partitionMode?: 'SPX' | 'DPX' | 'CPX';
  partitionIndex?: number;
}

export interface GpuMetrics {
  deviceId: string;
  vramUsedMb: number;
  vramTotalMb: number;
  utilizationPercent: number;
  temperatureC: number;
  powerW: number;
}

export interface GpuProcess {
  deviceId: string;
  pid: number;
  vramUsedMb: number;
  processName: string;
}

export interface GpuTopology {
  partitionMode: 'SPX' | 'DPX' | 'CPX' | 'unknown';
  physicalGpus: Array<{
    physicalId: number;
    name: string;
    partitions: Array<{
      logicalId: number;
      vramTotalMb: number;
    }>;
  }>;
}

export interface GpuMetricsProvider {
  detect(): Promise<boolean>;
  getDevices(): Promise<GpuDevice[]>;
  getMetrics(): Promise<GpuMetrics[]>;
  getTopology(): Promise<GpuTopology>;
  getProcesses(): Promise<GpuProcess[]>;
}

// --- vLLM Instance types ---

export interface VllmInstance {
  containerId: string;
  containerName: string;
  status: 'running' | 'starting' | 'stopped' | 'error';
  modelName: string;
  gpuIds: string[];
  partitionInfo?: string;
  quantization?: string;
  tensorParallelSize: number;
  port?: number;
  vramUsedMb: number;
  vramTotalMb: number;
  kvCachePercent?: number;
  runningRequests?: number;
  waitingRequests?: number;
  launchArgs: string[];
  envVars: Record<string, string>;
}

// --- Dashboard Snapshot ---

export interface DashboardSnapshot {
  timestamp: number;
  pollIntervalMs: number;
  summary: {
    instanceCount: number;
    totalVramMb: number;
    usedVramMb: number;
    totalActiveRequests: number;
  };
  instances: VllmInstance[];
  gpus: GpuDevice[];
  gpuMetrics: GpuMetrics[];
  warnings: string[];
}

// --- Log Streaming types ---

export interface LogMessage {
  type: 'log';
  timestamp: string;
  stream: 'stdout' | 'stderr';
  line: string;
}

export interface LogControl {
  type: 'closed' | 'dropped';
  reason?: string;
  count?: number;
}

export type LogEvent = LogMessage | LogControl;
