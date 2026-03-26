import type { DashboardSnapshot, VllmInstance } from '../types.js';
import type { DockerService } from './DockerService.js';
import type { VllmMetricsService } from './VllmMetricsService.js';
import type { GpuMetricsProvider } from '../types.js';
import { readFileSync } from 'fs';

export interface MetricsCollectorOptions {
  pollIntervalMs: number;
}

export class MetricsCollector {
  private dockerService: DockerService;
  private gpuProvider: GpuMetricsProvider;
  private vllmService: VllmMetricsService;
  private options: MetricsCollectorOptions;
  private snapshot: DashboardSnapshot | null = null;
  private intervalHandle: ReturnType<typeof setInterval> | null = null;
  private gpuAvailable: boolean | null = null;
  private dockerAvailable: boolean | null = null;

  constructor(
    dockerService: DockerService,
    gpuProvider: GpuMetricsProvider,
    vllmService: VllmMetricsService,
    options: MetricsCollectorOptions
  ) {
    this.dockerService = dockerService;
    this.gpuProvider = gpuProvider;
    this.vllmService = vllmService;
    this.options = options;
  }

  async collect(): Promise<DashboardSnapshot> {
    const warnings: string[] = [];

    if (this.gpuAvailable === null) {
      try {
        this.gpuAvailable = await this.gpuProvider.detect();
      } catch {
        this.gpuAvailable = false;
      }
    }
    if (!this.gpuAvailable) {
      warnings.push('GPU metrics unavailable: amd-smi not found or failed');
    }

    let containers: any[] = [];
    let containerPids: Record<string, number> = {};
    try {
      [containers, containerPids] = await Promise.all([
        this.dockerService.discoverContainers(),
        this.dockerService.getContainerPids(),
      ]);
      this.dockerAvailable = true;
    } catch {
      this.dockerAvailable = false;
      warnings.push('Docker unavailable: could not connect to Docker socket');
    }

    let gpuDevices: any[] = [];
    let gpuMetrics: any[] = [];
    let gpuProcesses: any[] = [];
    if (this.gpuAvailable) {
      try {
        [gpuDevices, gpuMetrics, gpuProcesses] = await Promise.all([
          this.gpuProvider.getDevices(),
          this.gpuProvider.getMetrics(),
          this.gpuProvider.getProcesses(),
        ]);
      } catch (err) {
        warnings.push(`GPU metrics collection failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    const instances: VllmInstance[] = [];
    for (const container of containers) {
      const instance: VllmInstance = {
        containerId: container.containerId,
        containerName: container.containerName,
        status: container.status,
        modelName: container.modelName,
        gpuIds: container.gpuIds,
        partitionInfo: container.partitionInfo,
        quantization: container.quantization,
        tensorParallelSize: container.tensorParallelSize,
        port: container.port,
        vramUsedMb: 0,
        vramTotalMb: 0,
        launchArgs: container.launchArgs,
        envVars: container.envVars,
      };

      if (this.gpuAvailable && containerPids[container.containerId]) {
        const topPid = containerPids[container.containerId];
        const childPids = this.getChildPids(topPid);
        const allPids = new Set([topPid, ...childPids]);
        const matchedProcesses = gpuProcesses.filter((p: any) => allPids.has(p.pid));
        instance.vramUsedMb = matchedProcesses.reduce((sum: number, p: any) => sum + p.vramUsedMb, 0);

        for (const gpuId of container.gpuIds) {
          const device = gpuDevices.find((d: any) => d.id === `gpu-${gpuId}` || String(d.physicalId) === gpuId);
          if (device) {
            instance.vramTotalMb += device.vramTotalMb;
          }
        }
      }

      if (container.port && container.status === 'running') {
        const vllmMetrics = await this.vllmService.fetchMetrics('localhost', container.port);
        if (vllmMetrics) {
          instance.kvCachePercent = vllmMetrics.kvCachePercent;
          instance.runningRequests = vllmMetrics.runningRequests;
          instance.waitingRequests = vllmMetrics.waitingRequests;
        }
      }

      instances.push(instance);
    }

    const totalActiveRequests = instances.reduce(
      (sum, i) => sum + (i.runningRequests ?? 0),
      0
    );
    const totalVramMb = gpuMetrics.reduce((sum: number, m: any) => sum + m.vramTotalMb, 0);
    const usedVramMb = gpuMetrics.reduce((sum: number, m: any) => sum + m.vramUsedMb, 0);

    this.snapshot = {
      timestamp: Date.now(),
      pollIntervalMs: this.options.pollIntervalMs,
      summary: {
        instanceCount: instances.length,
        totalVramMb,
        usedVramMb,
        totalActiveRequests,
      },
      instances,
      gpus: gpuDevices,
      gpuMetrics,
      warnings,
    };

    return this.snapshot;
  }

  getSnapshot(): DashboardSnapshot | null {
    return this.snapshot;
  }

  start(): void {
    this.collect().catch(() => {});
    this.intervalHandle = setInterval(() => this.collect().catch(() => {}), this.options.pollIntervalMs);
  }

  stop(): void {
    if (this.intervalHandle) {
      clearInterval(this.intervalHandle);
      this.intervalHandle = null;
    }
  }

  private getChildPids(parentPid: number): number[] {
    try {
      const children = readFileSync(`/proc/${parentPid}/task/${parentPid}/children`, 'utf-8');
      const directChildren = children.trim().split(/\s+/).filter(Boolean).map(Number);
      const allChildren: number[] = [];
      for (const child of directChildren) {
        allChildren.push(child, ...this.getChildPids(child));
      }
      return allChildren;
    } catch {
      return [];
    }
  }
}
