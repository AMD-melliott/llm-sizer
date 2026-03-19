let mockStdout = '';
let mockError: Error | null = null;

jest.mock('util', () => {
  const actual = jest.requireActual<typeof import('util')>('util');
  return {
    ...actual,
    promisify: () => {
      return (..._args: unknown[]) => {
        if (mockError) {
          return Promise.reject(mockError);
        }
        return Promise.resolve({ stdout: mockStdout, stderr: '' });
      };
    },
  };
});

import { AmdSmiProvider } from '../../../src/dashboard/server/providers/AmdSmiProvider';

function mockExecFileResult(stdout: string) {
  mockError = null;
  mockStdout = stdout;
}

function mockExecFileError(error: Error) {
  mockError = error;
  mockStdout = '';
}

describe('AmdSmiProvider', () => {
  let provider: AmdSmiProvider;

  beforeEach(() => {
    mockError = null;
    mockStdout = '';
    provider = new AmdSmiProvider();
  });

  test('detect() returns true when amd-smi is available', async () => {
    mockExecFileResult('amdsmi version\n');
    const result = await provider.detect();
    expect(result).toBe(true);
  });

  test('detect() returns false when amd-smi is not found', async () => {
    mockExecFileError(new Error('command not found'));
    const result = await provider.detect();
    expect(result).toBe(false);
  });

  test('getDevices() parses amd-smi metric output', async () => {
    const smiOutput = JSON.stringify([
      {
        gpu: 0,
        gpu_id: '0x74a1',
        name: 'AMD Instinct MI325X',
        vram_total: 196608,
        vram_used: 50000,
        gpu_use_percent: 45,
        temperature: 65,
        power: 300,
      },
    ]);
    mockExecFileResult(smiOutput);

    const devices = await provider.getDevices();
    expect(devices).toHaveLength(1);
    expect(devices[0].name).toBe('AMD Instinct MI325X');
    expect(devices[0].vramTotalMb).toBe(196608);
    expect(devices[0].physicalId).toBe(0);
  });

  test('getMetrics() parses utilization data', async () => {
    const smiOutput = JSON.stringify([
      {
        gpu: 0,
        gpu_id: '0x74a1',
        name: 'AMD Instinct MI325X',
        vram_total: 196608,
        vram_used: 50000,
        gpu_use_percent: 45,
        temperature: 65,
        power: 300,
      },
    ]);
    mockExecFileResult(smiOutput);

    const metrics = await provider.getMetrics();
    expect(metrics).toHaveLength(1);
    expect(metrics[0].vramUsedMb).toBe(50000);
    expect(metrics[0].utilizationPercent).toBe(45);
    expect(metrics[0].temperatureC).toBe(65);
    expect(metrics[0].powerW).toBe(300);
  });

  test('getProcesses() parses process list', async () => {
    const processOutput = JSON.stringify([
      {
        gpu: 0,
        pid: 12345,
        vram_usage: 32000,
        process_name: 'python',
      },
      {
        gpu: 0,
        pid: 12346,
        vram_usage: 18000,
        process_name: 'python',
      },
    ]);
    mockExecFileResult(processOutput);

    const processes = await provider.getProcesses();
    expect(processes).toHaveLength(2);
    expect(processes[0].pid).toBe(12345);
    expect(processes[0].vramUsedMb).toBe(32000);
    expect(processes[0].deviceId).toBe('gpu-0');
  });

  test('getTopology() detects partition mode', async () => {
    const topoOutput = JSON.stringify({
      partition_mode: 'CPX',
      gpus: [
        {
          gpu: 0,
          name: 'AMD Instinct MI325X',
          partitions: [
            { logical_id: 0, vram_total: 24576 },
            { logical_id: 1, vram_total: 24576 },
            { logical_id: 2, vram_total: 24576 },
            { logical_id: 3, vram_total: 24576 },
          ],
        },
      ],
    });
    mockExecFileResult(topoOutput);

    const topology = await provider.getTopology();
    expect(topology.partitionMode).toBe('CPX');
    expect(topology.physicalGpus).toHaveLength(1);
    expect(topology.physicalGpus[0].partitions).toHaveLength(4);
  });
});
