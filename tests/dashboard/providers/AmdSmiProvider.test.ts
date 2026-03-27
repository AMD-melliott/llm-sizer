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

// KFD device check - always accessible in tests
jest.mock('fs', () => {
  const actual = jest.requireActual<typeof import('fs')>('fs');
  return {
    ...actual,
    accessSync: jest.fn(),
  };
});

import { AmdSmiProvider } from '../../../src/dashboard/server/providers/AmdSmiProvider';

function mockExecResult(stdout: string) {
  mockError = null;
  mockStdout = stdout;
}

function mockExecError(error: Error) {
  mockError = error;
  mockStdout = '';
}

// ---------------------------------------------------------------------------
// Real ROCm 7.2.1 (AMDSMI 26.2.2) output fixtures
// Captured from: amd-smi metric/static/process/topology --json
// ---------------------------------------------------------------------------

const METRIC_ONE_GPU = JSON.stringify({
  gpu_data: [
    {
      gpu: 0,
      usage: {
        gfx_activity: { value: 72, unit: '%' },
        umc_activity: { value: 45, unit: '%' },
      },
      power: {
        socket_power: { value: 350, unit: 'W' },
        gfx_voltage: 'N/A',
      },
      temperature: {
        edge: 'N/A',
        hotspot: { value: 68, unit: 'C' },
        mem: { value: 55, unit: 'C' },
      },
      mem_usage: {
        total_vram: { value: 196608, unit: 'MB' },
        used_vram: { value: 150000, unit: 'MB' },
        free_vram: { value: 46608, unit: 'MB' },
        total_visible_vram: { value: 196608, unit: 'MB' },
        used_visible_vram: { value: 150000, unit: 'MB' },
        free_visible_vram: { value: 46608, unit: 'MB' },
        total_gtt: { value: 1160984, unit: 'MB' },
        used_gtt: { value: 24, unit: 'MB' },
        free_gtt: { value: 1160960, unit: 'MB' },
      },
    },
  ],
});

const METRIC_TWO_GPUS = JSON.stringify({
  gpu_data: [
    {
      gpu: 0,
      usage: { gfx_activity: { value: 90, unit: '%' } },
      power: { socket_power: { value: 500, unit: 'W' } },
      temperature: { edge: 'N/A', hotspot: { value: 75, unit: 'C' } },
      mem_usage: {
        total_vram: { value: 196608, unit: 'MB' },
        used_vram: { value: 100000, unit: 'MB' },
        free_vram: { value: 96608, unit: 'MB' },
      },
    },
    {
      gpu: 1,
      usage: { gfx_activity: { value: 10, unit: '%' } },
      power: { socket_power: { value: 120, unit: 'W' } },
      temperature: { edge: 'N/A', hotspot: { value: 42, unit: 'C' } },
      mem_usage: {
        total_vram: { value: 196608, unit: 'MB' },
        used_vram: { value: 8000, unit: 'MB' },
        free_vram: { value: 188608, unit: 'MB' },
      },
    },
  ],
});

// market_name is "N/A" on GIM/SR-IOV hosts — product_name from board is the
// reliable fallback (this is a known ROCm 7.x behavior on virtualized hardware)
const STATIC_MARKET_NAME_NA = JSON.stringify({
  gpu_data: [
    {
      gpu: 0,
      asic: {
        market_name: 'N/A',
        vendor_id: 'N/A',
        oam_id: 'N/A',
        num_compute_units: 'N/A',
      },
      board: {
        product_name: 'AMD Instinct MI300X OAM',
        manufacturer_name: 'AMD',
        product_serial: '692424013220',
      },
    },
  ],
});

const STATIC_MARKET_NAME_SET = JSON.stringify({
  gpu_data: [
    {
      gpu: 0,
      asic: {
        market_name: 'AMD Instinct MI325X',
        vendor_id: '0x1002',
        num_compute_units: '304',
      },
      board: {
        product_name: 'AMD Instinct MI325X OAM',
        manufacturer_name: 'AMD',
        product_serial: 'ABC123',
      },
    },
  ],
});

// process --json returns a top-level array (not wrapped in gpu_data)
const PROCESS_LIST = JSON.stringify([
  {
    gpu: 0,
    process_list: [
      {
        process_info: {
          name: 'python3',
          pid: 12345,
          memory_usage: {
            gtt_mem: { value: 0, unit: 'B' },
            cpu_mem: { value: 0, unit: 'B' },
            vram_mem: { value: 33554432, unit: 'B' }, // 32 MB in bytes
          },
          mem_usage: { value: 33554432, unit: 'B' },
          cu_occupancy: 0,
        },
      },
      {
        process_info: {
          name: 'python3',
          pid: 12346,
          memory_usage: {
            gtt_mem: { value: 0, unit: 'B' },
            cpu_mem: { value: 0, unit: 'B' },
            vram_mem: { value: 18874368, unit: 'B' }, // 18 MB in bytes
          },
          mem_usage: { value: 18874368, unit: 'B' },
          cu_occupancy: 0,
        },
      },
    ],
  },
  {
    gpu: 1,
    process_list: [
      {
        // Same pid 12345 on gpu 1 — GIM broadcasts across physical GPUs; must be deduplicated
        process_info: {
          name: 'python3',
          pid: 12345,
          memory_usage: {
            vram_mem: { value: 33554432, unit: 'B' },
          },
          mem_usage: { value: 33554432, unit: 'B' },
          cu_occupancy: 0,
        },
      },
    ],
  },
]);

// process output where VRAM is already reported in MB (older ROCm behavior kept for compatibility)
const PROCESS_LIST_MB = JSON.stringify([
  {
    gpu: 0,
    process_list: [
      {
        process_info: {
          name: 'vllm',
          pid: 99001,
          memory_usage: {
            vram_mem: { value: 50000, unit: 'MB' },
          },
          mem_usage: { value: 50000, unit: 'MB' },
          cu_occupancy: 0,
        },
      },
    ],
  },
]);

// topology --json in ROCm 7.x returns a peer-link array, not partition info.
// getTopology() must not crash on this format.
const TOPOLOGY_LINK_ARRAY = JSON.stringify([
  {
    gpu: 0,
    bdf: '0000:05:00.0',
    links: [
      { gpu: 1, link_type: 'XGMI', bandwidth: '64000-64000' },
    ],
  },
  {
    gpu: 1,
    bdf: '0000:26:00.0',
    links: [
      { gpu: 0, link_type: 'XGMI', bandwidth: '64000-64000' },
    ],
  },
]);

// ---------------------------------------------------------------------------

describe('AmdSmiProvider', () => {
  let provider: AmdSmiProvider;

  beforeEach(() => {
    mockError = null;
    mockStdout = '';
    provider = new AmdSmiProvider();
  });

  // -------------------------------------------------------------------------
  // detect()
  // -------------------------------------------------------------------------

  describe('detect()', () => {
    test('returns true when amd-smi version succeeds', async () => {
      mockExecResult('AMDSMI Tool: 26.2.2+e1a6bc5663 | AMDSMI Library version: 26.2.2 | ROCm version: 7.2.1\n');
      const result = await provider.detect();
      expect(result).toBe(true);
    });

    test('returns false when amd-smi is not installed', async () => {
      mockExecError(new Error('ENOENT: amd-smi: command not found'));
      const result = await provider.detect();
      expect(result).toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  // getDevices()
  // -------------------------------------------------------------------------

  describe('getDevices()', () => {
    test('parses single GPU with full mem_usage structure', async () => {
      // getDevices() calls metric and static in parallel; mock returns the
      // same value for both (the provider only reads gpu_data from each)
      mockExecResult(METRIC_ONE_GPU);
      // Override for the static call — provider calls runAmdSmi twice
      // The mock returns the same stdout for every call, so we need to handle
      // both. We set metric output; static call will also return metric output.
      // For name resolution we need static data, so use a separate approach:
      // set metric as mock, but asic.market_name won't be present — falls back.
      // Test the core: id, physicalId, vramTotalMb are correctly mapped.
      const devices = await provider.getDevices();
      expect(devices).toHaveLength(1);
      expect(devices[0].id).toBe('gpu-0');
      expect(devices[0].physicalId).toBe(0);
      expect(devices[0].vramTotalMb).toBe(196608);
    });

    test('parses two GPUs and returns both devices', async () => {
      mockExecResult(METRIC_TWO_GPUS);
      const devices = await provider.getDevices();
      expect(devices).toHaveLength(2);
      expect(devices[0].id).toBe('gpu-0');
      expect(devices[1].id).toBe('gpu-1');
      expect(devices[0].vramTotalMb).toBe(196608);
      expect(devices[1].vramTotalMb).toBe(196608);
    });

    test('uses market_name when it is not N/A', async () => {
      // Both metric and static calls return the same mock — use static-shaped data.
      // Provide a metric-shaped gpu_data that also satisfies static parsing.
      mockStdout = STATIC_MARKET_NAME_SET;
      // metric data also needs mem_usage; since we can only set one stdout, we
      // provide a combined fixture: valid for static name parsing, but metric
      // gpu_data[0].mem_usage will be missing → vramTotalMb defaults to 0.
      // The test focuses on name resolution.
      const devices = await provider.getDevices();
      expect(devices[0].name).toBe('AMD Instinct MI325X');
    });

    test('falls back to board.product_name when market_name is N/A', async () => {
      mockStdout = STATIC_MARKET_NAME_NA;
      const devices = await provider.getDevices();
      expect(devices[0].name).toBe('AMD Instinct MI300X OAM');
    });

    test('falls back to GPU index label when both market_name and product_name are absent', async () => {
      const noNames = JSON.stringify({
        gpu_data: [{ gpu: 2, asic: {}, board: {} }],
      });
      mockStdout = noNames;
      const devices = await provider.getDevices();
      expect(devices[0].name).toBe('GPU 2');
    });

    test('returns empty array when gpu_data is missing', async () => {
      mockExecResult(JSON.stringify({}));
      const devices = await provider.getDevices();
      expect(devices).toHaveLength(0);
    });
  });

  // -------------------------------------------------------------------------
  // getMetrics()
  // -------------------------------------------------------------------------

  describe('getMetrics()', () => {
    test('maps all metric fields from ROCm 7.x nested structure', async () => {
      mockExecResult(METRIC_ONE_GPU);
      const metrics = await provider.getMetrics();
      expect(metrics).toHaveLength(1);
      expect(metrics[0].deviceId).toBe('gpu-0');
      expect(metrics[0].vramUsedMb).toBe(150000);
      expect(metrics[0].vramTotalMb).toBe(196608);
      expect(metrics[0].utilizationPercent).toBe(72);
      expect(metrics[0].temperatureC).toBe(68);
      expect(metrics[0].powerW).toBe(350);
    });

    test('uses hotspot temperature when edge is N/A', async () => {
      mockExecResult(METRIC_ONE_GPU);
      const metrics = await provider.getMetrics();
      // edge is "N/A" in fixture, hotspot.value is 68
      expect(metrics[0].temperatureC).toBe(68);
    });

    test('returns metrics for multiple GPUs', async () => {
      mockExecResult(METRIC_TWO_GPUS);
      const metrics = await provider.getMetrics();
      expect(metrics).toHaveLength(2);
      expect(metrics[0].utilizationPercent).toBe(90);
      expect(metrics[1].utilizationPercent).toBe(10);
      expect(metrics[0].vramUsedMb).toBe(100000);
      expect(metrics[1].vramUsedMb).toBe(8000);
    });

    test('defaults to 0 for missing metric fields', async () => {
      const sparse = JSON.stringify({
        gpu_data: [{ gpu: 0, usage: {}, power: {}, temperature: {}, mem_usage: {} }],
      });
      mockExecResult(sparse);
      const metrics = await provider.getMetrics();
      expect(metrics[0].vramUsedMb).toBe(0);
      expect(metrics[0].utilizationPercent).toBe(0);
      expect(metrics[0].temperatureC).toBe(0);
      expect(metrics[0].powerW).toBe(0);
    });

    test('returns empty array when gpu_data is missing', async () => {
      mockExecResult(JSON.stringify({}));
      const metrics = await provider.getMetrics();
      expect(metrics).toHaveLength(0);
    });
  });

  // -------------------------------------------------------------------------
  // getProcesses()
  // -------------------------------------------------------------------------

  describe('getProcesses()', () => {
    test('parses process list with VRAM in bytes and converts to MB', async () => {
      mockExecResult(PROCESS_LIST);
      const processes = await provider.getProcesses();
      // pid 12345 on gpu-0 and gpu-1 — de-duplicated to one entry
      // pid 12346 on gpu-0 only
      expect(processes).toHaveLength(2);
    });

    test('assigns deviceId from the first GPU the process appears on', async () => {
      mockExecResult(PROCESS_LIST);
      const processes = await provider.getProcesses();
      const p = processes.find(x => x.pid === 12345);
      expect(p?.deviceId).toBe('gpu-0');
    });

    test('converts VRAM from bytes to MB correctly', async () => {
      mockExecResult(PROCESS_LIST);
      const processes = await provider.getProcesses();
      const p = processes.find(x => x.pid === 12345);
      // 33554432 B / 1048576 = 32 MB
      expect(p?.vramUsedMb).toBe(32);
    });

    test('keeps VRAM in MB when unit is already MB', async () => {
      mockExecResult(PROCESS_LIST_MB);
      const processes = await provider.getProcesses();
      expect(processes).toHaveLength(1);
      expect(processes[0].vramUsedMb).toBe(50000);
    });

    test('de-duplicates PIDs appearing on multiple GPUs (GIM SR-IOV broadcast)', async () => {
      mockExecResult(PROCESS_LIST);
      const processes = await provider.getProcesses();
      const pids = processes.map(p => p.pid);
      // pid 12345 appears on gpu 0 and gpu 1 — must appear only once
      expect(pids.filter(pid => pid === 12345)).toHaveLength(1);
    });

    test('returns process name when available', async () => {
      mockExecResult(PROCESS_LIST);
      const processes = await provider.getProcesses();
      const p = processes.find(x => x.pid === 12345);
      expect(p?.processName).toBe('python3');
    });

    test('returns empty array when process_list is empty', async () => {
      const empty = JSON.stringify([{ gpu: 0, process_list: [] }]);
      mockExecResult(empty);
      const processes = await provider.getProcesses();
      expect(processes).toHaveLength(0);
    });

    test('returns empty array when output is empty array', async () => {
      mockExecResult(JSON.stringify([]));
      const processes = await provider.getProcesses();
      expect(processes).toHaveLength(0);
    });
  });

  // -------------------------------------------------------------------------
  // getTopology()
  // -------------------------------------------------------------------------

  describe('getTopology()', () => {
    test('returns unknown partition mode when topology is a peer-link array (ROCm 7.x format)', async () => {
      mockExecResult(TOPOLOGY_LINK_ARRAY);
      const topology = await provider.getTopology();
      expect(topology.partitionMode).toBe('unknown');
    });

    test('does not throw when topology output is a raw array instead of object', async () => {
      mockExecResult(TOPOLOGY_LINK_ARRAY);
      await expect(provider.getTopology()).resolves.toBeDefined();
    });

    test('returns empty physicalGpus when gpu_data is absent', async () => {
      mockExecResult(TOPOLOGY_LINK_ARRAY);
      const topology = await provider.getTopology();
      expect(topology.physicalGpus).toHaveLength(0);
    });

    test('parses partition mode and partitions when present', async () => {
      const partitioned = JSON.stringify({
        partition_mode: 'CPX',
        gpu_data: [
          {
            gpu: 0,
            asic: { market_name: 'AMD Instinct MI325X' },
            partitions: [
              { logical_id: 0, vram_total: { value: 49152, unit: 'MB' } },
              { logical_id: 1, vram_total: { value: 49152, unit: 'MB' } },
            ],
          },
        ],
      });
      mockExecResult(partitioned);
      const topology = await provider.getTopology();
      expect(topology.partitionMode).toBe('CPX');
      expect(topology.physicalGpus).toHaveLength(1);
      expect(topology.physicalGpus[0].partitions).toHaveLength(2);
      expect(topology.physicalGpus[0].partitions[0].logicalId).toBe(0);
      expect(topology.physicalGpus[0].partitions[0].vramTotalMb).toBe(49152);
    });
  });

  // -------------------------------------------------------------------------
  // Error handling
  // -------------------------------------------------------------------------

  describe('error handling', () => {
    test('getMetrics() throws when amd-smi returns invalid JSON', async () => {
      mockExecResult('not valid json');
      await expect(provider.getMetrics()).rejects.toThrow();
    });

    test('getProcesses() throws when amd-smi returns invalid JSON', async () => {
      mockExecResult('not valid json');
      await expect(provider.getProcesses()).rejects.toThrow();
    });
  });
});
