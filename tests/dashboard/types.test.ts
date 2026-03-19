import type {
  GpuDevice,
  GpuMetrics,
  GpuProcess,
  GpuTopology,
  GpuMetricsProvider,
  DashboardSnapshot,
  VllmInstance,
  LogMessage,
  LogControl,
} from '../../src/dashboard/server/types';

describe('Dashboard types', () => {
  test('VllmInstance has required fields', () => {
    const instance: VllmInstance = {
      containerId: 'abc123',
      containerName: 'vllm-deepseek',
      status: 'running',
      modelName: 'DeepSeek-R1-70B',
      gpuIds: ['0'],
      tensorParallelSize: 1,
      vramUsedMb: 50000,
      vramTotalMb: 76000,
      launchArgs: ['--model', 'deepseek-r1-70b'],
      envVars: { AMD_VISIBLE_DEVICES: '0' },
    };
    expect(instance.status).toBe('running');
    expect(instance.containerId).toBe('abc123');
  });

  test('DashboardSnapshot has required structure', () => {
    const snapshot: DashboardSnapshot = {
      timestamp: Date.now(),
      pollIntervalMs: 5000,
      summary: {
        instanceCount: 1,
        totalVramMb: 192000,
        usedVramMb: 50000,
        totalActiveRequests: 3,
      },
      instances: [],
      gpus: [],
      gpuMetrics: [],
      warnings: [],
    };
    expect(snapshot.summary.instanceCount).toBe(1);
    expect(snapshot.warnings).toEqual([]);
  });

  test('GpuDevice supports partition info', () => {
    const device: GpuDevice = {
      id: 'gpu-0-p0',
      physicalId: 0,
      logicalId: 0,
      name: 'AMD Instinct MI325X',
      vramTotalMb: 38000,
      partitionMode: 'CPX',
      partitionIndex: 0,
    };
    expect(device.partitionMode).toBe('CPX');
    expect(device.partitionIndex).toBe(0);
  });

  test('LogMessage types cover stdout and stderr', () => {
    const stdout: LogMessage = {
      type: 'log',
      timestamp: '2026-03-13T10:00:00Z',
      stream: 'stdout',
      line: 'INFO: Server started',
    };
    const stderr: LogMessage = {
      type: 'log',
      timestamp: '2026-03-13T10:00:01Z',
      stream: 'stderr',
      line: 'WARNING: High VRAM usage',
    };
    expect(stdout.stream).toBe('stdout');
    expect(stderr.stream).toBe('stderr');
  });

  test('LogControl types cover close and drop events', () => {
    const closed: LogControl = { type: 'closed', reason: 'container_stopped' };
    const dropped: LogControl = { type: 'dropped', count: 15 };
    expect(closed.type).toBe('closed');
    expect(dropped.type).toBe('dropped');
  });
});
