import { MetricsCollector } from '../../../src/dashboard/server/services/MetricsCollector';
import type { GpuMetricsProvider } from '../../../src/dashboard/server/types';

function createMockDockerService() {
  return {
    discoverContainers: jest.fn().mockResolvedValue([
      {
        containerId: 'abc123',
        containerName: 'vllm-test',
        status: 'running' as const,
        modelName: 'deepseek-r1-70b',
        gpuIds: ['0'],
        tensorParallelSize: 1,
        port: 8000,
        launchArgs: ['--model', 'deepseek-r1-70b'],
        envVars: { AMD_VISIBLE_DEVICES: '0' },
        pid: 12345,
      },
    ]),
    getContainerPids: jest.fn().mockResolvedValue({ abc123: 12345 }),
    getDocker: jest.fn(),
  };
}

function createMockGpuProvider(): GpuMetricsProvider {
  return {
    detect: jest.fn().mockResolvedValue(true),
    getDevices: jest.fn().mockResolvedValue([
      { id: 'gpu-0', physicalId: 0, name: 'MI325X', vramTotalMb: 196608 },
    ]),
    getMetrics: jest.fn().mockResolvedValue([
      {
        deviceId: 'gpu-0',
        vramUsedMb: 50000,
        vramTotalMb: 196608,
        utilizationPercent: 45,
        temperatureC: 65,
        powerW: 300,
      },
    ]),
    getTopology: jest.fn().mockResolvedValue({
      partitionMode: 'SPX',
      physicalGpus: [],
    }),
    getProcesses: jest.fn().mockResolvedValue([
      { deviceId: 'gpu-0', pid: 12345, vramUsedMb: 50000, processName: 'python' },
    ]),
  };
}

function createMockVllmService() {
  return {
    fetchMetrics: jest.fn().mockResolvedValue({
      runningRequests: 3,
      waitingRequests: 1,
      kvCachePercent: 42,
    }),
    fetchModels: jest.fn().mockResolvedValue(['deepseek-r1-70b']),
  };
}

describe('MetricsCollector', () => {
  test('collect() merges Docker, GPU, and vLLM data into a snapshot', async () => {
    const collector = new MetricsCollector(
      createMockDockerService() as any,
      createMockGpuProvider(),
      createMockVllmService() as any,
      { pollIntervalMs: 5000 }
    );

    const snapshot = await collector.collect();

    expect(snapshot.summary.instanceCount).toBe(1);
    expect(snapshot.instances).toHaveLength(1);
    expect(snapshot.instances[0].containerId).toBe('abc123');
    expect(snapshot.instances[0].vramUsedMb).toBe(50000);
    expect(snapshot.instances[0].kvCachePercent).toBe(42);
    expect(snapshot.instances[0].runningRequests).toBe(3);
    expect(snapshot.gpus).toHaveLength(1);
    expect(snapshot.gpuMetrics).toHaveLength(1);
    expect(snapshot.pollIntervalMs).toBe(5000);
    expect(snapshot.warnings).toEqual([]);
  });

  test('degrades gracefully when GPU provider is unavailable', async () => {
    const gpuProvider = createMockGpuProvider();
    (gpuProvider.detect as jest.Mock).mockResolvedValue(false);

    const collector = new MetricsCollector(
      createMockDockerService() as any,
      gpuProvider,
      createMockVllmService() as any,
      { pollIntervalMs: 5000 }
    );

    const snapshot = await collector.collect();

    expect(snapshot.instances).toHaveLength(1);
    expect(snapshot.gpus).toEqual([]);
    expect(snapshot.gpuMetrics).toEqual([]);
    expect(snapshot.warnings.some((w) => w.includes('GPU'))).toBe(true);
  });

  test('degrades gracefully when vLLM metrics fail', async () => {
    const vllmService = createMockVllmService();
    vllmService.fetchMetrics.mockResolvedValue(null);

    const collector = new MetricsCollector(
      createMockDockerService() as any,
      createMockGpuProvider(),
      vllmService as any,
      { pollIntervalMs: 5000 }
    );

    const snapshot = await collector.collect();

    expect(snapshot.instances[0].kvCachePercent).toBeUndefined();
    expect(snapshot.instances[0].runningRequests).toBeUndefined();
  });

  test('getSnapshot() returns last collected snapshot', async () => {
    const collector = new MetricsCollector(
      createMockDockerService() as any,
      createMockGpuProvider(),
      createMockVllmService() as any,
      { pollIntervalMs: 5000 }
    );

    expect(collector.getSnapshot()).toBeNull();
    await collector.collect();
    expect(collector.getSnapshot()).not.toBeNull();
    expect(collector.getSnapshot()!.summary.instanceCount).toBe(1);
  });

  test('summary totals are computed correctly', async () => {
    const gpuProvider = createMockGpuProvider();
    (gpuProvider.getMetrics as jest.Mock).mockResolvedValue([
      { deviceId: 'gpu-0', vramUsedMb: 50000, vramTotalMb: 196608, utilizationPercent: 45, temperatureC: 65, powerW: 300 },
      { deviceId: 'gpu-1', vramUsedMb: 80000, vramTotalMb: 196608, utilizationPercent: 60, temperatureC: 70, powerW: 350 },
    ]);
    (gpuProvider.getDevices as jest.Mock).mockResolvedValue([
      { id: 'gpu-0', physicalId: 0, name: 'MI325X', vramTotalMb: 196608 },
      { id: 'gpu-1', physicalId: 1, name: 'MI325X', vramTotalMb: 196608 },
    ]);

    const collector = new MetricsCollector(
      createMockDockerService() as any,
      gpuProvider,
      createMockVllmService() as any,
      { pollIntervalMs: 5000 }
    );

    const snapshot = await collector.collect();
    expect(snapshot.summary.totalVramMb).toBe(196608 * 2);
    expect(snapshot.summary.usedVramMb).toBe(130000);
    expect(snapshot.summary.totalActiveRequests).toBe(3);
  });
});
