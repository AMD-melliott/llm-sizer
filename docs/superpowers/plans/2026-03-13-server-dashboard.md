# Server Dashboard Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a monitoring dashboard backend + frontend that discovers running vLLM Docker containers, shows GPU/VRAM metrics via amd-smi, displays inference stats from vLLM APIs, and streams container logs over WebSocket.

**Architecture:** Separate Fastify backend service within the llm-sizer repo. Backend polls Docker, amd-smi, and vLLM endpoints on a configurable interval, serves a merged snapshot via REST API. Frontend is React + Tailwind (matching existing llm-sizer stack), fetches snapshots and streams logs via WebSocket. GPU metrics abstracted behind a provider interface for future NVIDIA support.

**Tech Stack:** Fastify, @fastify/websocket, @fastify/static, dockerode, tsup, React, Tailwind CSS

**Spec:** `docs/superpowers/specs/2026-03-13-server-dashboard-design.md`

---

## Chunk 1: Project Setup, Types, and Backend Foundation

### Task 1: Install Dependencies and Configure Build

**Files:**
- Modify: `package.json`
- Create: `tsconfig.dashboard.json`

- [ ] **Step 1: Install backend dependencies**

```bash
npm install fastify @fastify/websocket @fastify/static dockerode
npm install -D @types/dockerode
```

- [ ] **Step 2: Create tsconfig.dashboard.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "skipLibCheck": true,
    "outDir": "dist",
    "declaration": true,
    "isolatedModules": true
  },
  "include": ["src/dashboard/**/*", "src/types/**/*"]
}
```

- [ ] **Step 3: Add dashboard scripts to package.json**

Add to `"scripts"`:
```json
"dashboard": "tsx src/dashboard/server/index.ts",
"build:dashboard": "tsup --entry.dashboard src/dashboard/server/index.ts --format esm --target node20 --platform node --tsconfig tsconfig.dashboard.json"
```

- [ ] **Step 4: Add moduleNameMapper for dockerode in jest.config.js**

Dockerode uses native modules that Jest can't import. Add to `moduleNameMapper`:
```js
'^dockerode$': '<rootDir>/__mocks__/dockerode.js',
```

Create `__mocks__/dockerode.js`:
```js
module.exports = class Docker {
  constructor() {}
  listContainers() { return Promise.resolve([]); }
  getContainer() { return { inspect: () => Promise.resolve({}) }; }
};
```

- [ ] **Step 5: Verify existing tests still pass**

Run: `npm test`
Expected: All existing tests pass (no regressions from dependency additions).

- [ ] **Step 6: Commit**

```bash
git add package.json package-lock.json tsconfig.dashboard.json jest.config.js __mocks__/dockerode.js
git commit -m "chore: add dashboard dependencies and build config"
```

---

### Task 2: Define Dashboard Types

**Files:**
- Create: `src/dashboard/server/types.ts`
- Test: `tests/dashboard/types.test.ts`

- [ ] **Step 1: Write type validation test**

```typescript
// tests/dashboard/types.test.ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/dashboard/types.test.ts`
Expected: FAIL — cannot resolve module `../../src/dashboard/server/types`

- [ ] **Step 3: Write the types file**

```typescript
// src/dashboard/server/types.ts

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/dashboard/types.test.ts`
Expected: PASS — all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/server/types.ts tests/dashboard/types.test.ts
git commit -m "feat(dashboard): add type definitions for dashboard backend"
```

---

### Task 3: DockerService — Container Discovery

**Files:**
- Create: `src/dashboard/server/services/DockerService.ts`
- Test: `tests/dashboard/services/DockerService.test.ts`

- [ ] **Step 1: Write failing tests for DockerService**

```typescript
// tests/dashboard/services/DockerService.test.ts
import { DockerService } from '../../../src/dashboard/server/services/DockerService';

// Mock dockerode
const mockContainerInspect = jest.fn();
const mockListContainers = jest.fn();
const mockContainerTop = jest.fn();

jest.mock('dockerode', () => {
  return jest.fn().mockImplementation(() => ({
    listContainers: mockListContainers,
    getContainer: jest.fn().mockReturnValue({
      inspect: mockContainerInspect,
      top: mockContainerTop,
    }),
  }));
});

describe('DockerService', () => {
  let service: DockerService;

  beforeEach(() => {
    jest.clearAllMocks();
    service = new DockerService();
  });

  test('discovers vLLM containers by image name', async () => {
    mockListContainers.mockResolvedValue([
      {
        Id: 'abc123',
        Names: ['/vllm-deepseek'],
        Image: 'vllm/vllm-openai:latest',
        State: 'running',
        Ports: [{ PublicPort: 8000, PrivatePort: 8000 }],
      },
      {
        Id: 'def456',
        Names: ['/nginx-proxy'],
        Image: 'nginx:latest',
        State: 'running',
        Ports: [],
      },
    ]);

    mockContainerInspect.mockResolvedValue({
      Id: 'abc123',
      Name: '/vllm-deepseek',
      State: { Status: 'running', Pid: 12345 },
      Config: {
        Image: 'vllm/vllm-openai:latest',
        Env: [
          'AMD_VISIBLE_DEVICES=0',
          'MODEL_NAME=deepseek-r1-70b',
        ],
        Cmd: ['--model', 'deepseek-r1-70b', '--dtype', 'float16'],
      },
      HostConfig: {},
      NetworkSettings: {
        Ports: { '8000/tcp': [{ HostPort: '8000' }] },
      },
      Created: '2026-03-13T10:00:00Z',
    });

    const containers = await service.discoverContainers();
    expect(containers).toHaveLength(1);
    expect(containers[0].containerId).toBe('abc123');
    expect(containers[0].containerName).toBe('vllm-deepseek');
  });

  test('extracts GPU assignment from AMD_VISIBLE_DEVICES', async () => {
    mockListContainers.mockResolvedValue([
      {
        Id: 'abc123',
        Names: ['/vllm-llama'],
        Image: 'rocm/vllm:latest',
        State: 'running',
        Ports: [{ PublicPort: 8001, PrivatePort: 8000 }],
      },
    ]);

    mockContainerInspect.mockResolvedValue({
      Id: 'abc123',
      Name: '/vllm-llama',
      State: { Status: 'running', Pid: 12345 },
      Config: {
        Image: 'rocm/vllm:latest',
        Env: [
          'AMD_VISIBLE_DEVICES=1,2,3,4',
        ],
        Cmd: ['--model', 'llama-405b', '--tensor-parallel-size', '4', '--quantization', 'fp8'],
      },
      HostConfig: {},
      NetworkSettings: {
        Ports: { '8000/tcp': [{ HostPort: '8001' }] },
      },
      Created: '2026-03-13T10:00:00Z',
    });

    const containers = await service.discoverContainers();
    expect(containers[0].gpuIds).toEqual(['1', '2', '3', '4']);
    expect(containers[0].tensorParallelSize).toBe(4);
    expect(containers[0].quantization).toBe('fp8');
    expect(containers[0].port).toBe(8001);
  });

  test('falls back to ROCR_VISIBLE_DEVICES', async () => {
    mockListContainers.mockResolvedValue([
      {
        Id: 'abc123',
        Names: ['/vllm-old'],
        Image: 'vllm/vllm-openai:v0.4',
        State: 'running',
        Ports: [],
      },
    ]);

    mockContainerInspect.mockResolvedValue({
      Id: 'abc123',
      Name: '/vllm-old',
      State: { Status: 'running', Pid: 12345 },
      Config: {
        Image: 'vllm/vllm-openai:v0.4',
        Env: ['ROCR_VISIBLE_DEVICES=0'],
        Cmd: ['--model', 'qwen-7b'],
      },
      HostConfig: {},
      NetworkSettings: { Ports: {} },
      Created: '2026-03-13T10:00:00Z',
    });

    const containers = await service.discoverContainers();
    expect(containers[0].gpuIds).toEqual(['0']);
  });

  test('uses default port 8000 when no ports mapped', async () => {
    mockListContainers.mockResolvedValue([
      {
        Id: 'abc123',
        Names: ['/vllm-noport'],
        Image: 'vllm/vllm-openai:latest',
        State: 'running',
        Ports: [],
      },
    ]);

    mockContainerInspect.mockResolvedValue({
      Id: 'abc123',
      Name: '/vllm-noport',
      State: { Status: 'running', Pid: 12345 },
      Config: {
        Image: 'vllm/vllm-openai:latest',
        Env: [],
        Cmd: ['--model', 'test-model'],
      },
      HostConfig: {},
      NetworkSettings: { Ports: {} },
      Created: '2026-03-13T10:00:00Z',
    });

    const containers = await service.discoverContainers();
    expect(containers[0].port).toBe(8000);
  });

  test('handles custom image patterns', async () => {
    const customService = new DockerService({
      imagePatterns: ['my-registry/vllm*'],
    });

    mockListContainers.mockResolvedValue([
      {
        Id: 'abc123',
        Names: ['/custom-vllm'],
        Image: 'my-registry/vllm-custom:latest',
        State: 'running',
        Ports: [],
      },
    ]);

    mockContainerInspect.mockResolvedValue({
      Id: 'abc123',
      Name: '/custom-vllm',
      State: { Status: 'running', Pid: 12345 },
      Config: {
        Image: 'my-registry/vllm-custom:latest',
        Env: [],
        Cmd: ['--model', 'test'],
      },
      HostConfig: {},
      NetworkSettings: { Ports: {} },
      Created: '2026-03-13T10:00:00Z',
    });

    const containers = await customService.discoverContainers();
    expect(containers).toHaveLength(1);
  });

  test('returns container PIDs for GPU process mapping', async () => {
    mockListContainers.mockResolvedValue([
      {
        Id: 'abc123',
        Names: ['/vllm-test'],
        Image: 'vllm/vllm-openai:latest',
        State: 'running',
        Ports: [],
      },
    ]);

    mockContainerInspect.mockResolvedValue({
      Id: 'abc123',
      Name: '/vllm-test',
      State: { Status: 'running', Pid: 54321 },
      Config: {
        Image: 'vllm/vllm-openai:latest',
        Env: [],
        Cmd: ['--model', 'test'],
      },
      HostConfig: {},
      NetworkSettings: { Ports: {} },
      Created: '2026-03-13T10:00:00Z',
    });

    const pids = await service.getContainerPids();
    expect(pids).toEqual({ abc123: 54321 });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/dashboard/services/DockerService.test.ts`
Expected: FAIL — cannot resolve `DockerService`

- [ ] **Step 3: Implement DockerService**

```typescript
// src/dashboard/server/services/DockerService.ts
import Docker from 'dockerode';
import type { VllmInstance } from '../types.js';

export interface DockerServiceOptions {
  socketPath?: string;
  imagePatterns?: string[];
}

interface DiscoveredContainer extends Omit<VllmInstance, 'vramUsedMb' | 'vramTotalMb'> {
  pid: number;
}

const DEFAULT_IMAGE_PATTERNS = ['vllm/*', 'rocm/vllm*'];
const DEFAULT_VLLM_PORT = 8000;

export class DockerService {
  private docker: Docker;
  private imagePatterns: string[];

  constructor(options: DockerServiceOptions = {}) {
    this.docker = new Docker({
      socketPath: options.socketPath ?? '/var/run/docker.sock',
    });
    this.imagePatterns = options.imagePatterns ?? DEFAULT_IMAGE_PATTERNS;
  }

  async discoverContainers(): Promise<DiscoveredContainer[]> {
    const allContainers = await this.docker.listContainers({ all: true });
    const vllmContainers = allContainers.filter((c) => this.matchesImagePattern(c.Image));

    const results: DiscoveredContainer[] = [];
    for (const container of vllmContainers) {
      const detail = await this.docker.getContainer(container.Id).inspect();
      results.push(this.parseContainer(detail));
    }
    return results;
  }

  async getContainerPids(): Promise<Record<string, number>> {
    const containers = await this.discoverContainers();
    const pids: Record<string, number> = {};
    for (const c of containers) {
      if (c.pid > 0) {
        pids[c.containerId] = c.pid;
      }
    }
    return pids;
  }

  getDocker(): Docker {
    return this.docker;
  }

  private matchesImagePattern(image: string): boolean {
    return this.imagePatterns.some((pattern) => {
      const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '(:|$)');
      return regex.test(image);
    });
  }

  private parseContainer(detail: Docker.ContainerInspectInfo): DiscoveredContainer {
    const env = this.parseEnvVars(detail.Config.Env ?? []);
    const cmd = detail.Config.Cmd ?? [];
    const status = this.mapStatus(detail.State.Status);
    const port = this.extractPort(detail);
    const gpuIds = this.extractGpuIds(env);
    const tensorParallelSize = this.extractTensorParallel(cmd);
    const quantization = this.extractQuantization(cmd);
    const modelName = this.extractModelName(cmd, env);

    return {
      containerId: detail.Id,
      containerName: detail.Name.replace(/^\//, ''),
      status,
      modelName,
      gpuIds,
      tensorParallelSize,
      quantization: quantization ?? undefined,
      port,
      launchArgs: cmd,
      envVars: env,
      pid: detail.State.Pid,
    };
  }

  private parseEnvVars(envList: string[]): Record<string, string> {
    const env: Record<string, string> = {};
    for (const entry of envList) {
      const eqIdx = entry.indexOf('=');
      if (eqIdx > 0) {
        env[entry.substring(0, eqIdx)] = entry.substring(eqIdx + 1);
      }
    }
    return env;
  }

  private mapStatus(dockerStatus: string): VllmInstance['status'] {
    switch (dockerStatus) {
      case 'running': return 'running';
      case 'created':
      case 'restarting': return 'starting';
      case 'exited':
      case 'dead':
      case 'removing': return 'stopped';
      default: return 'error';
    }
  }

  private extractPort(detail: Docker.ContainerInspectInfo): number {
    const ports = detail.NetworkSettings?.Ports ?? {};
    for (const [, bindings] of Object.entries(ports)) {
      if (bindings && bindings.length > 0 && bindings[0].HostPort) {
        return parseInt(bindings[0].HostPort, 10);
      }
    }
    return DEFAULT_VLLM_PORT;
  }

  private extractGpuIds(env: Record<string, string>): string[] {
    const deviceVar =
      env['AMD_VISIBLE_DEVICES'] ??
      env['ROCR_VISIBLE_DEVICES'] ??
      env['CUDA_VISIBLE_DEVICES'];
    if (!deviceVar) return [];
    return deviceVar.split(',').map((s) => s.trim());
  }

  private extractTensorParallel(cmd: string[]): number {
    const idx = cmd.indexOf('--tensor-parallel-size');
    if (idx >= 0 && idx + 1 < cmd.length) {
      return parseInt(cmd[idx + 1], 10) || 1;
    }
    const tpIdx = cmd.indexOf('-tp');
    if (tpIdx >= 0 && tpIdx + 1 < cmd.length) {
      return parseInt(cmd[tpIdx + 1], 10) || 1;
    }
    return 1;
  }

  private extractQuantization(cmd: string[]): string | null {
    const idx = cmd.indexOf('--quantization');
    if (idx >= 0 && idx + 1 < cmd.length) {
      return cmd[idx + 1];
    }
    return null;
  }

  private extractModelName(cmd: string[], env: Record<string, string>): string {
    const idx = cmd.indexOf('--model');
    if (idx >= 0 && idx + 1 < cmd.length) {
      return cmd[idx + 1];
    }
    return env['MODEL_NAME'] ?? 'unknown';
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest tests/dashboard/services/DockerService.test.ts`
Expected: PASS — all 6 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/server/services/DockerService.ts tests/dashboard/services/DockerService.test.ts
git commit -m "feat(dashboard): add DockerService for vLLM container discovery"
```

---

### Task 4: AmdSmiProvider — GPU Metrics

**Files:**
- Create: `src/dashboard/server/providers/GpuMetricsProvider.ts`
- Create: `src/dashboard/server/providers/AmdSmiProvider.ts`
- Test: `tests/dashboard/providers/AmdSmiProvider.test.ts`

- [ ] **Step 1: Write failing tests for AmdSmiProvider**

```typescript
// tests/dashboard/providers/AmdSmiProvider.test.ts
import { AmdSmiProvider } from '../../../src/dashboard/server/providers/AmdSmiProvider';
import { execFile } from 'child_process';

jest.mock('child_process', () => ({
  execFile: jest.fn(),
}));

const mockExecFile = execFile as unknown as jest.Mock;

// Helper to make execFile resolve with stdout
// Handle both 3-arg (cmd, args, cb) and 4-arg (cmd, args, opts, cb) calls
function mockExecFileResult(stdout: string) {
  mockExecFile.mockImplementation((...args: any[]) => {
    const callback = args[args.length - 1];
    callback(null, stdout, '');
  });
}

function mockExecFileError(error: Error) {
  mockExecFile.mockImplementation((...args: any[]) => {
    const callback = args[args.length - 1];
    callback(error, '', '');
  });
}

describe('AmdSmiProvider', () => {
  let provider: AmdSmiProvider;

  beforeEach(() => {
    jest.clearAllMocks();
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/dashboard/providers/AmdSmiProvider.test.ts`
Expected: FAIL — cannot resolve modules

- [ ] **Step 3: Create GpuMetricsProvider interface file**

```typescript
// src/dashboard/server/providers/GpuMetricsProvider.ts
export type { GpuMetricsProvider, GpuDevice, GpuMetrics, GpuProcess, GpuTopology } from '../types.js';
```

- [ ] **Step 4: Implement AmdSmiProvider**

```typescript
// src/dashboard/server/providers/AmdSmiProvider.ts
import { execFile as execFileCb } from 'child_process';
import { promisify } from 'util';
import type {
  GpuMetricsProvider,
  GpuDevice,
  GpuMetrics,
  GpuProcess,
  GpuTopology,
} from '../types.js';

const execFile = promisify(execFileCb);

export class AmdSmiProvider implements GpuMetricsProvider {
  async detect(): Promise<boolean> {
    try {
      await execFile('amd-smi', ['version']);
      return true;
    } catch {
      return false;
    }
  }

  async getDevices(): Promise<GpuDevice[]> {
    const raw = await this.runAmdSmi(['metric', '--json']);
    const gpus = JSON.parse(raw);
    return gpus.map((g: any) => ({
      id: `gpu-${g.gpu}`,
      physicalId: g.gpu,
      name: g.name,
      vramTotalMb: g.vram_total,
    }));
  }

  async getMetrics(): Promise<GpuMetrics[]> {
    const raw = await this.runAmdSmi(['metric', '--json']);
    const gpus = JSON.parse(raw);
    return gpus.map((g: any) => ({
      deviceId: `gpu-${g.gpu}`,
      vramUsedMb: g.vram_used,
      vramTotalMb: g.vram_total,
      utilizationPercent: g.gpu_use_percent,
      temperatureC: g.temperature,
      powerW: g.power,
    }));
  }

  async getProcesses(): Promise<GpuProcess[]> {
    const raw = await this.runAmdSmi(['process', '--json']);
    const procs = JSON.parse(raw);
    return procs.map((p: any) => ({
      deviceId: `gpu-${p.gpu}`,
      pid: p.pid,
      vramUsedMb: p.vram_usage,
      processName: p.process_name,
    }));
  }

  async getTopology(): Promise<GpuTopology> {
    const raw = await this.runAmdSmi(['topology', '--json']);
    const topo = JSON.parse(raw);
    return {
      partitionMode: topo.partition_mode ?? 'unknown',
      physicalGpus: (topo.gpus ?? []).map((g: any) => ({
        physicalId: g.gpu,
        name: g.name,
        partitions: (g.partitions ?? []).map((p: any) => ({
          logicalId: p.logical_id,
          vramTotalMb: p.vram_total,
        })),
      })),
    };
  }

  private async runAmdSmi(args: string[]): Promise<string> {
    const { stdout } = await execFile('amd-smi', args, { timeout: 10000 });
    return stdout;
  }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx jest tests/dashboard/providers/AmdSmiProvider.test.ts`
Expected: PASS — all 5 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/server/providers/GpuMetricsProvider.ts src/dashboard/server/providers/AmdSmiProvider.ts tests/dashboard/providers/AmdSmiProvider.test.ts
git commit -m "feat(dashboard): add GpuMetricsProvider interface and AmdSmiProvider"
```

---

## Chunk 2: VllmMetricsService, MetricsCollector, API Routes, and Log Streaming

### Task 5: VllmMetricsService — Inference Metrics

**Files:**
- Create: `src/dashboard/server/services/VllmMetricsService.ts`
- Test: `tests/dashboard/services/VllmMetricsService.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/dashboard/services/VllmMetricsService.test.ts
import { VllmMetricsService } from '../../../src/dashboard/server/services/VllmMetricsService';

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch as any;

describe('VllmMetricsService', () => {
  let service: VllmMetricsService;

  beforeEach(() => {
    jest.clearAllMocks();
    service = new VllmMetricsService();
  });

  test('fetches metrics from vLLM /metrics endpoint', async () => {
    const prometheusText = [
      '# HELP vllm:num_requests_running Number of running requests',
      '# TYPE vllm:num_requests_running gauge',
      'vllm:num_requests_running{model_name="deepseek-r1-70b"} 3',
      '# HELP vllm:num_requests_waiting Number of waiting requests',
      '# TYPE vllm:num_requests_waiting gauge',
      'vllm:num_requests_waiting{model_name="deepseek-r1-70b"} 1',
      '# HELP vllm:gpu_cache_usage_perc GPU cache usage percentage',
      '# TYPE vllm:gpu_cache_usage_perc gauge',
      'vllm:gpu_cache_usage_perc{model_name="deepseek-r1-70b"} 0.42',
    ].join('\n');

    mockFetch.mockResolvedValue({
      ok: true,
      text: () => Promise.resolve(prometheusText),
    });

    const metrics = await service.fetchMetrics('localhost', 8000);
    expect(metrics.runningRequests).toBe(3);
    expect(metrics.waitingRequests).toBe(1);
    expect(metrics.kvCachePercent).toBeCloseTo(42);
  });

  test('returns null metrics when container is not ready', async () => {
    mockFetch.mockRejectedValue(new Error('ECONNREFUSED'));

    const metrics = await service.fetchMetrics('localhost', 8000);
    expect(metrics).toBeNull();
  });

  test('returns null metrics when /metrics returns non-200', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 503,
      text: () => Promise.resolve('Service Unavailable'),
    });

    const metrics = await service.fetchMetrics('localhost', 8000);
    expect(metrics).toBeNull();
  });

  test('fetches model info from /v1/models', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        data: [{ id: 'deepseek-r1-70b', object: 'model' }],
      }),
    });

    const models = await service.fetchModels('localhost', 8000);
    expect(models).toEqual(['deepseek-r1-70b']);
  });

  test('returns empty models list on error', async () => {
    mockFetch.mockRejectedValue(new Error('ECONNREFUSED'));

    const models = await service.fetchModels('localhost', 8000);
    expect(models).toEqual([]);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/dashboard/services/VllmMetricsService.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Implement VllmMetricsService**

```typescript
// src/dashboard/server/services/VllmMetricsService.ts

export interface VllmMetricsResult {
  runningRequests: number;
  waitingRequests: number;
  kvCachePercent: number;
}

export class VllmMetricsService {
  private timeoutMs: number;

  constructor(timeoutMs = 3000) {
    this.timeoutMs = timeoutMs;
  }

  async fetchMetrics(host: string, port: number): Promise<VllmMetricsResult | null> {
    try {
      const response = await fetch(`http://${host}:${port}/metrics`, {
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!response.ok) return null;
      const text = await response.text();
      return this.parsePrometheusMetrics(text);
    } catch {
      return null;
    }
  }

  async fetchModels(host: string, port: number): Promise<string[]> {
    try {
      const response = await fetch(`http://${host}:${port}/v1/models`, {
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!response.ok) return [];
      const data = await response.json() as { data: Array<{ id: string }> };
      return data.data.map((m) => m.id);
    } catch {
      return [];
    }
  }

  private parsePrometheusMetrics(text: string): VllmMetricsResult {
    const running = this.extractMetricValue(text, 'vllm:num_requests_running');
    const waiting = this.extractMetricValue(text, 'vllm:num_requests_waiting');
    const kvCache = this.extractMetricValue(text, 'vllm:gpu_cache_usage_perc');

    return {
      runningRequests: running ?? 0,
      waitingRequests: waiting ?? 0,
      kvCachePercent: (kvCache ?? 0) * 100,
    };
  }

  private extractMetricValue(text: string, metricName: string): number | null {
    const regex = new RegExp(`^${metricName.replace(/:/g, ':')}(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)`, 'm');
    const match = text.match(regex);
    if (!match) return null;
    return parseFloat(match[1]);
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest tests/dashboard/services/VllmMetricsService.test.ts`
Expected: PASS — all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/server/services/VllmMetricsService.ts tests/dashboard/services/VllmMetricsService.test.ts
git commit -m "feat(dashboard): add VllmMetricsService for inference metrics"
```

---

### Task 6: MetricsCollector — Orchestrator

**Files:**
- Create: `src/dashboard/server/services/MetricsCollector.ts`
- Test: `tests/dashboard/services/MetricsCollector.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/dashboard/services/MetricsCollector.test.ts
import { MetricsCollector } from '../../../src/dashboard/server/services/MetricsCollector';
import type { GpuMetricsProvider, DashboardSnapshot } from '../../../src/dashboard/server/types';

// Create mock services
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
    expect(snapshot.warnings).toContain(expect.stringContaining('GPU'));
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/dashboard/services/MetricsCollector.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Implement MetricsCollector**

```typescript
// src/dashboard/server/services/MetricsCollector.ts
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
  private gpuAvailable: boolean | null = null; // null = not yet detected

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

    // Detect GPU availability once on first collect
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

    // Fetch Docker containers
    const containers = await this.dockerService.discoverContainers();
    const containerPids = await this.dockerService.getContainerPids();

    // Fetch GPU data (if available)
    let gpuDevices = [];
    let gpuMetrics = [];
    let gpuProcesses = [];
    if (this.gpuAvailable) {
      [gpuDevices, gpuMetrics, gpuProcesses] = await Promise.all([
        this.gpuProvider.getDevices(),
        this.gpuProvider.getMetrics(),
        this.gpuProvider.getProcesses(),
      ]);
    }

    // Fetch vLLM metrics for each running container
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

      // Map VRAM from GPU processes (match top-level PID + child PIDs)
      if (this.gpuAvailable && containerPids[container.containerId]) {
        const topPid = containerPids[container.containerId];
        const childPids = this.getChildPids(topPid);
        const allPids = new Set([topPid, ...childPids]);
        const matchedProcesses = gpuProcesses.filter((p) => allPids.has(p.pid));
        instance.vramUsedMb = matchedProcesses.reduce((sum, p) => sum + p.vramUsedMb, 0);

        // Get total VRAM for assigned GPUs
        for (const gpuId of container.gpuIds) {
          const device = gpuDevices.find((d) => d.id === `gpu-${gpuId}` || String(d.physicalId) === gpuId);
          if (device) {
            instance.vramTotalMb += device.vramTotalMb;
          }
        }
      }

      // Fetch vLLM metrics if port is available
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

    // Compute summary
    const totalActiveRequests = instances.reduce(
      (sum, i) => sum + (i.runningRequests ?? 0),
      0
    );
    const totalVramMb = gpuMetrics.reduce((sum, m) => sum + m.vramTotalMb, 0);
    const usedVramMb = gpuMetrics.reduce((sum, m) => sum + m.vramUsedMb, 0);

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

  async start(): Promise<void> {
    await this.collect();
    this.intervalHandle = setInterval(() => this.collect(), this.options.pollIntervalMs);
  }

  stop(): void {
    if (this.intervalHandle) {
      clearInterval(this.intervalHandle);
      this.intervalHandle = null;
    }
  }

  /** Read child PIDs from /proc on Linux. Returns empty array on failure. */
  private getChildPids(parentPid: number): number[] {
    try {
      const children = readFileSync(`/proc/${parentPid}/task/${parentPid}/children`, 'utf-8');
      const directChildren = children.trim().split(/\s+/).filter(Boolean).map(Number);
      // Recurse to get grandchildren (vLLM worker processes)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest tests/dashboard/services/MetricsCollector.test.ts`
Expected: PASS — all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/server/services/MetricsCollector.ts tests/dashboard/services/MetricsCollector.test.ts
git commit -m "feat(dashboard): add MetricsCollector to orchestrate polling"
```

---

### Task 7: API Routes — Status, Instances, GPUs

**Files:**
- Create: `src/dashboard/server/routes/status.ts`
- Create: `src/dashboard/server/routes/instances.ts`
- Create: `src/dashboard/server/routes/gpus.ts`
- Test: `tests/dashboard/routes/api.test.ts`

- [ ] **Step 1: Write failing tests for all three REST routes**

```typescript
// tests/dashboard/routes/api.test.ts
import { registerStatusRoute } from '../../../src/dashboard/server/routes/status';
import { registerInstancesRoute } from '../../../src/dashboard/server/routes/instances';
import { registerGpusRoute } from '../../../src/dashboard/server/routes/gpus';
import type { DashboardSnapshot } from '../../../src/dashboard/server/types';

// Minimal mock Fastify for route testing
function createMockFastify() {
  const routes: Record<string, Function> = {};
  return {
    get: jest.fn((path: string, handler: Function) => {
      routes[path] = handler;
    }),
    routes,
  };
}

const MOCK_SNAPSHOT: DashboardSnapshot = {
  timestamp: 1710300000000,
  pollIntervalMs: 5000,
  summary: {
    instanceCount: 2,
    totalVramMb: 393216,
    usedVramMb: 130000,
    totalActiveRequests: 5,
  },
  instances: [
    {
      containerId: 'abc123',
      containerName: 'vllm-deepseek',
      status: 'running',
      modelName: 'deepseek-r1-70b',
      gpuIds: ['0'],
      tensorParallelSize: 1,
      port: 8000,
      vramUsedMb: 50000,
      vramTotalMb: 196608,
      kvCachePercent: 42,
      runningRequests: 3,
      waitingRequests: 1,
      launchArgs: ['--model', 'deepseek-r1-70b'],
      envVars: { AMD_VISIBLE_DEVICES: '0' },
    },
    {
      containerId: 'def456',
      containerName: 'vllm-llama',
      status: 'running',
      modelName: 'llama-405b',
      gpuIds: ['1', '2', '3', '4'],
      tensorParallelSize: 4,
      port: 8001,
      vramUsedMb: 80000,
      vramTotalMb: 196608,
      kvCachePercent: 71,
      runningRequests: 2,
      waitingRequests: 0,
      launchArgs: ['--model', 'llama-405b', '--tensor-parallel-size', '4'],
      envVars: { AMD_VISIBLE_DEVICES: '1,2,3,4' },
    },
  ],
  gpus: [
    { id: 'gpu-0', physicalId: 0, name: 'MI325X', vramTotalMb: 196608 },
    { id: 'gpu-1', physicalId: 1, name: 'MI325X', vramTotalMb: 196608 },
  ],
  gpuMetrics: [
    { deviceId: 'gpu-0', vramUsedMb: 50000, vramTotalMb: 196608, utilizationPercent: 45, temperatureC: 65, powerW: 300 },
    { deviceId: 'gpu-1', vramUsedMb: 80000, vramTotalMb: 196608, utilizationPercent: 60, temperatureC: 70, powerW: 350 },
  ],
  warnings: [],
};

function createMockCollector(snapshot: DashboardSnapshot | null = MOCK_SNAPSHOT) {
  return { getSnapshot: jest.fn().mockReturnValue(snapshot) };
}

describe('API Routes', () => {
  describe('GET /api/status', () => {
    test('returns summary and timestamp', async () => {
      const fastify = createMockFastify();
      const collector = createMockCollector();
      registerStatusRoute(fastify as any, collector as any);

      const reply = { code: jest.fn().mockReturnThis(), send: jest.fn() };
      await fastify.routes['/api/status']({}, reply);

      expect(reply.send).toHaveBeenCalledWith({
        timestamp: 1710300000000,
        pollIntervalMs: 5000,
        summary: MOCK_SNAPSHOT.summary,
        warnings: [],
      });
    });

    test('returns 503 when no snapshot available', async () => {
      const fastify = createMockFastify();
      const collector = createMockCollector(null);
      registerStatusRoute(fastify as any, collector as any);

      const reply = { code: jest.fn().mockReturnThis(), send: jest.fn() };
      await fastify.routes['/api/status']({}, reply);

      expect(reply.code).toHaveBeenCalledWith(503);
    });
  });

  describe('GET /api/instances', () => {
    test('returns all instances', async () => {
      const fastify = createMockFastify();
      const collector = createMockCollector();
      registerInstancesRoute(fastify as any, collector as any);

      const reply = { code: jest.fn().mockReturnThis(), send: jest.fn() };
      await fastify.routes['/api/instances']({}, reply);

      expect(reply.send).toHaveBeenCalledWith(MOCK_SNAPSHOT.instances);
    });

    test('returns single instance by id', async () => {
      const fastify = createMockFastify();
      const collector = createMockCollector();
      registerInstancesRoute(fastify as any, collector as any);

      const reply = { code: jest.fn().mockReturnThis(), send: jest.fn() };
      await fastify.routes['/api/instances/:id'](
        { params: { id: 'abc123' } },
        reply
      );

      expect(reply.send).toHaveBeenCalledWith(MOCK_SNAPSHOT.instances[0]);
    });

    test('returns 404 for unknown instance', async () => {
      const fastify = createMockFastify();
      const collector = createMockCollector();
      registerInstancesRoute(fastify as any, collector as any);

      const reply = { code: jest.fn().mockReturnThis(), send: jest.fn() };
      await fastify.routes['/api/instances/:id'](
        { params: { id: 'unknown' } },
        reply
      );

      expect(reply.code).toHaveBeenCalledWith(404);
    });
  });

  describe('GET /api/gpus', () => {
    test('returns devices and metrics', async () => {
      const fastify = createMockFastify();
      const collector = createMockCollector();
      registerGpusRoute(fastify as any, collector as any);

      const reply = { code: jest.fn().mockReturnThis(), send: jest.fn() };
      await fastify.routes['/api/gpus']({}, reply);

      expect(reply.send).toHaveBeenCalledWith({
        devices: MOCK_SNAPSHOT.gpus,
        metrics: MOCK_SNAPSHOT.gpuMetrics,
      });
    });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/dashboard/routes/api.test.ts`
Expected: FAIL — cannot resolve modules

- [ ] **Step 3: Implement status route**

```typescript
// src/dashboard/server/routes/status.ts
import type { FastifyInstance } from 'fastify';
import type { MetricsCollector } from '../services/MetricsCollector.js';

export function registerStatusRoute(
  fastify: FastifyInstance,
  collector: MetricsCollector
) {
  fastify.get('/api/status', async (_request, reply) => {
    const snapshot = collector.getSnapshot();
    if (!snapshot) {
      return reply.code(503).send({ error: 'Dashboard not ready' });
    }
    return reply.send({
      timestamp: snapshot.timestamp,
      pollIntervalMs: snapshot.pollIntervalMs,
      summary: snapshot.summary,
      warnings: snapshot.warnings,
    });
  });
}
```

- [ ] **Step 4: Implement instances route**

```typescript
// src/dashboard/server/routes/instances.ts
import type { FastifyInstance } from 'fastify';
import type { MetricsCollector } from '../services/MetricsCollector.js';

export function registerInstancesRoute(
  fastify: FastifyInstance,
  collector: MetricsCollector
) {
  fastify.get('/api/instances', async (_request, reply) => {
    const snapshot = collector.getSnapshot();
    if (!snapshot) {
      return reply.code(503).send({ error: 'Dashboard not ready' });
    }
    return reply.send(snapshot.instances);
  });

  fastify.get('/api/instances/:id', async (request, reply) => {
    const snapshot = collector.getSnapshot();
    if (!snapshot) {
      return reply.code(503).send({ error: 'Dashboard not ready' });
    }
    const { id } = request.params as { id: string };
    const instance = snapshot.instances.find((i) => i.containerId === id);
    if (!instance) {
      return reply.code(404).send({ error: 'Instance not found' });
    }
    return reply.send(instance);
  });
}
```

- [ ] **Step 5: Implement gpus route**

```typescript
// src/dashboard/server/routes/gpus.ts
import type { FastifyInstance } from 'fastify';
import type { MetricsCollector } from '../services/MetricsCollector.js';

export function registerGpusRoute(
  fastify: FastifyInstance,
  collector: MetricsCollector
) {
  fastify.get('/api/gpus', async (_request, reply) => {
    const snapshot = collector.getSnapshot();
    if (!snapshot) {
      return reply.code(503).send({ error: 'Dashboard not ready' });
    }
    return reply.send({
      devices: snapshot.gpus,
      metrics: snapshot.gpuMetrics,
    });
  });
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npx jest tests/dashboard/routes/api.test.ts`
Expected: PASS — all 6 tests pass

- [ ] **Step 7: Commit**

```bash
git add src/dashboard/server/routes/status.ts src/dashboard/server/routes/instances.ts src/dashboard/server/routes/gpus.ts tests/dashboard/routes/api.test.ts
git commit -m "feat(dashboard): add REST API routes for status, instances, gpus"
```

> **Note:** `GET /api/gpus/:id/metrics` (time-series data) is deferred — it requires Prometheus integration which is out of scope for v1. The route is listed in the spec as optional.

---

### Task 8: WebSocket Log Streaming Route

**Files:**
- Create: `src/dashboard/server/routes/logs.ts`
- Test: `tests/dashboard/routes/logs.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/dashboard/routes/logs.test.ts
import { LogStreamManager } from '../../../src/dashboard/server/routes/logs';
import { EventEmitter, Readable } from 'stream';

describe('LogStreamManager', () => {
  test('parses Docker log stream into LogMessage events', async () => {
    const manager = new LogStreamManager();
    const messages: any[] = [];

    // Simulate a Docker log stream (multiplexed format)
    const mockStream = new Readable({ read() {} });

    const sendFn = jest.fn((data: string) => {
      messages.push(JSON.parse(data));
    });

    manager.attachStream(mockStream, sendFn);

    // Push a stdout line
    mockStream.push('2026-03-13T10:00:00.000Z INFO: Server started\n');
    // Allow microtask to process
    await new Promise((r) => setTimeout(r, 10));

    expect(messages.length).toBeGreaterThanOrEqual(1);
    expect(messages[0].type).toBe('log');
    expect(messages[0].line).toContain('Server started');
  });

  test('sends closed event when stream ends', async () => {
    const manager = new LogStreamManager();
    const messages: any[] = [];

    const mockStream = new Readable({ read() {} });
    const sendFn = jest.fn((data: string) => {
      messages.push(JSON.parse(data));
    });

    manager.attachStream(mockStream, sendFn);
    mockStream.push(null); // end of stream

    await new Promise((r) => setTimeout(r, 10));

    const closedMsg = messages.find((m) => m.type === 'closed');
    expect(closedMsg).toBeDefined();
    expect(closedMsg.reason).toBe('container_stopped');
  });

  test('respects buffer limit and sends dropped message', async () => {
    const manager = new LogStreamManager({ maxBuffer: 5 });
    const messages: any[] = [];

    const mockStream = new Readable({ read() {} });
    let sendBlocked = true;
    const sendFn = jest.fn((data: string) => {
      if (!sendBlocked) {
        messages.push(JSON.parse(data));
      }
    });

    manager.attachStream(mockStream, sendFn);

    // Push more lines than the buffer can hold while send is blocked
    for (let i = 0; i < 10; i++) {
      mockStream.push(`line ${i}\n`);
    }

    await new Promise((r) => setTimeout(r, 50));

    // Unblock and flush
    sendBlocked = false;
    manager.flush(sendFn);

    const dropped = messages.find((m) => m.type === 'dropped');
    expect(dropped).toBeDefined();
    expect(dropped.count).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/dashboard/routes/logs.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Implement LogStreamManager and log route**

```typescript
// src/dashboard/server/routes/logs.ts
import type { FastifyInstance } from 'fastify';
import type { Readable } from 'stream';
import Docker from 'dockerode';
import type { LogMessage, LogControl } from '../types.js';

export interface LogStreamOptions {
  maxBuffer?: number;
}

export class LogStreamManager {
  private maxBuffer: number;
  private buffer: LogMessage[] = [];
  private droppedCount = 0;

  constructor(options: LogStreamOptions = {}) {
    this.maxBuffer = options.maxBuffer ?? 1000;
  }

  attachStream(stream: Readable, send: (data: string) => void): void {
    let partial = '';

    stream.on('data', (chunk: Buffer | string) => {
      const text = partial + chunk.toString();
      const lines = text.split('\n');
      partial = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.trim()) continue;
        const msg: LogMessage = {
          type: 'log',
          timestamp: new Date().toISOString(),
          stream: 'stdout',
          line: line.replace(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*/, ''),
        };

        // Parse timestamp from line if present
        const tsMatch = line.match(/^(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s*/);
        if (tsMatch) {
          msg.timestamp = tsMatch[1];
          msg.line = line.substring(tsMatch[0].length);
        }

        // Detect stderr by common patterns
        if (/^(ERROR|WARN|WARNING|CRITICAL|FATAL)/i.test(msg.line)) {
          msg.stream = 'stderr';
        }

        this.enqueue(msg, send);
      }
    });

    stream.on('end', () => {
      const control: LogControl = { type: 'closed', reason: 'container_stopped' };
      send(JSON.stringify(control));
    });

    stream.on('error', () => {
      const control: LogControl = { type: 'closed', reason: 'stream_error' };
      send(JSON.stringify(control));
    });
  }

  flush(send: (data: string) => void): void {
    if (this.droppedCount > 0) {
      const dropped: LogControl = { type: 'dropped', count: this.droppedCount };
      send(JSON.stringify(dropped));
      this.droppedCount = 0;
    }
    for (const msg of this.buffer) {
      send(JSON.stringify(msg));
    }
    this.buffer = [];
  }

  private enqueue(msg: LogMessage, send: (data: string) => void): void {
    try {
      send(JSON.stringify(msg));
    } catch {
      // Send failed — buffer the message
      if (this.buffer.length >= this.maxBuffer) {
        this.buffer.shift();
        this.droppedCount++;
      }
      this.buffer.push(msg);
    }
  }
}

export function registerLogsRoute(
  fastify: FastifyInstance,
  docker: Docker
) {
  fastify.get(
    '/api/logs/:containerId',
    { websocket: true },
    async (socket, request) => {
      const { containerId } = request.params as { containerId: string };
      const url = new URL(request.url, 'http://localhost');
      const tail = parseInt(url.searchParams.get('tail') ?? '200', 10);

      const container = docker.getContainer(containerId);
      let logStream: NodeJS.ReadableStream;

      try {
        logStream = await container.logs({
          follow: true,
          stdout: true,
          stderr: true,
          tail,
          timestamps: true,
        });
      } catch {
        const err: LogControl = { type: 'closed', reason: 'container_not_found' };
        socket.send(JSON.stringify(err));
        socket.close();
        return;
      }

      const manager = new LogStreamManager();
      const sendFn = (data: string) => {
        if (socket.readyState !== 1) {
          throw new Error('socket not open'); // triggers buffering in enqueue()
        }
        socket.send(data);
      };
      manager.attachStream(logStream as any, sendFn);

      socket.on('close', () => {
        (logStream as any).destroy?.();
      });
    }
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest tests/dashboard/routes/logs.test.ts`
Expected: PASS — all 3 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/server/routes/logs.ts tests/dashboard/routes/logs.test.ts
git commit -m "feat(dashboard): add WebSocket log streaming route"
```

---

### Task 9: Fastify Server Entry Point

**Files:**
- Create: `src/dashboard/server/index.ts`

- [ ] **Step 1: Implement server entry point**

```typescript
// src/dashboard/server/index.ts
import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import fastifyStatic from '@fastify/static';
import { resolve } from 'path';
import { DockerService } from './services/DockerService.js';
import { AmdSmiProvider } from './providers/AmdSmiProvider.js';
import { VllmMetricsService } from './services/VllmMetricsService.js';
import { MetricsCollector } from './services/MetricsCollector.js';
import { registerStatusRoute } from './routes/status.js';
import { registerInstancesRoute } from './routes/instances.js';
import { registerGpusRoute } from './routes/gpus.js';
import { registerLogsRoute } from './routes/logs.js';

interface ServerOptions {
  port?: number;
  host?: string;
  pollInterval?: number;
  socketPath?: string;
  imagePatterns?: string[];
}

async function startServer(options: ServerOptions = {}) {
  const port = options.port ?? 3001;
  const host = options.host ?? '0.0.0.0';
  const pollInterval = options.pollInterval ?? 5000;

  const fastify = Fastify({ logger: true });

  // Register plugins
  await fastify.register(fastifyWebsocket);
  await fastify.register(fastifyStatic, {
    root: resolve(import.meta.dirname, '../../..', 'dist/dashboard'),
    prefix: '/',
    wildcard: false,
  });

  // Initialize services
  const dockerService = new DockerService({
    socketPath: options.socketPath,
    imagePatterns: options.imagePatterns,
  });
  const gpuProvider = new AmdSmiProvider();
  const vllmService = new VllmMetricsService();
  const collector = new MetricsCollector(
    dockerService,
    gpuProvider,
    vllmService,
    { pollIntervalMs: pollInterval }
  );

  // Register routes
  registerStatusRoute(fastify, collector);
  registerInstancesRoute(fastify, collector);
  registerGpusRoute(fastify, collector);
  registerLogsRoute(fastify, dockerService.getDocker());

  // Start polling and server
  await collector.start();

  try {
    await fastify.listen({ port, host });
    console.log(`Dashboard running at http://${host}:${port}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }

  // Graceful shutdown
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.on(signal, async () => {
      collector.stop();
      await fastify.close();
      process.exit(0);
    });
  }
}

// Parse CLI args
const args = process.argv.slice(2);
const portIdx = args.indexOf('--port');
const pollIdx = args.indexOf('--poll-interval');

startServer({
  port: portIdx >= 0 ? parseInt(args[portIdx + 1], 10) : undefined,
  pollInterval: pollIdx >= 0 ? parseInt(args[pollIdx + 1], 10) : undefined,
});
```

- [ ] **Step 2: Verify the server compiles**

Run: `npx tsc --noEmit --project tsconfig.dashboard.json`
Expected: No type errors. If `tsconfig.dashboard.json` fails to resolve dashboard imports, ensure its `include` array covers `src/dashboard/**/*`.

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/server/index.ts
git commit -m "feat(dashboard): add Fastify server entry point with CLI args"
```

---

## Chunk 3: Frontend Dashboard UI

### Task 10: Dashboard Data Hook

**Files:**
- Create: `src/dashboard/ui/hooks/useDashboardData.ts`
- Test: `tests/dashboard/ui/useDashboardData.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// tests/dashboard/ui/useDashboardData.test.ts
import { parseDashboardStatus, shouldRefreshData } from '../../src/dashboard/ui/hooks/useDashboardData';

describe('useDashboardData helpers', () => {
  test('parseDashboardStatus extracts fields correctly', () => {
    const raw = {
      timestamp: 1710300000000,
      pollIntervalMs: 5000,
      summary: { instanceCount: 2, totalVramMb: 393216, usedVramMb: 130000, totalActiveRequests: 5 },
      warnings: [],
    };
    const result = parseDashboardStatus(raw);
    expect(result.timestamp).toBe(1710300000000);
    expect(result.summary.instanceCount).toBe(2);
  });

  test('shouldRefreshData returns true when timestamp changes', () => {
    expect(shouldRefreshData(1000, 2000)).toBe(true);
  });

  test('shouldRefreshData returns false when timestamp is same', () => {
    expect(shouldRefreshData(1000, 1000)).toBe(false);
  });

  test('shouldRefreshData returns true on first load (null previous)', () => {
    expect(shouldRefreshData(null, 1000)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/dashboard/ui/useDashboardData.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Implement useDashboardData hook**

```typescript
// src/dashboard/ui/hooks/useDashboardData.ts
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

// Exported for testing
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/dashboard/ui/useDashboardData.test.ts`
Expected: PASS — all 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/ui/hooks/useDashboardData.ts tests/dashboard/ui/useDashboardData.test.ts
git commit -m "feat(dashboard): add useDashboardData hook for REST polling"
```

---

### Task 11: Log Stream Hook

**Files:**
- Create: `src/dashboard/ui/hooks/useLogStream.ts`

> **No unit test for this hook.** It's a thin wrapper around the browser WebSocket API. Testing it properly requires jsdom + a WebSocket mock server, which adds disproportionate complexity for a stateless bridge. The WebSocket protocol and message parsing are tested server-side in Task 8.

- [ ] **Step 1: Implement useLogStream hook**

```typescript
// src/dashboard/ui/hooks/useLogStream.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import type { LogMessage, LogControl, LogEvent } from '../../server/types';

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
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/ui/hooks/useLogStream.ts
git commit -m "feat(dashboard): add useLogStream WebSocket hook"
```

---

### Task 12: SummaryBar Component

**Files:**
- Create: `src/dashboard/ui/SummaryBar.tsx`

- [ ] **Step 1: Implement SummaryBar**

```tsx
// src/dashboard/ui/SummaryBar.tsx
import type { DashboardSnapshot } from '../server/types';

interface SummaryBarProps {
  summary: DashboardSnapshot['summary'] | null;
  warnings: string[];
}

function formatVram(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(0)} GB`;
  return `${mb} MB`;
}

export function SummaryBar({ summary, warnings }: SummaryBarProps) {
  if (!summary) {
    return (
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-center animate-pulse">
            <div className="h-3 bg-gray-700 rounded w-20 mx-auto mb-2" />
            <div className="h-6 bg-gray-700 rounded w-12 mx-auto" />
          </div>
        ))}
      </div>
    );
  }

  const stats = [
    { label: 'Instances', value: summary.instanceCount, color: 'text-blue-400' },
    { label: 'Total VRAM', value: formatVram(summary.totalVramMb), color: 'text-green-400' },
    { label: 'VRAM Used', value: formatVram(summary.usedVramMb), color: 'text-orange-400' },
    { label: 'Active Requests', value: summary.totalActiveRequests, color: 'text-purple-400' },
  ];

  return (
    <div>
      <div className="grid grid-cols-4 gap-3 mb-6">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-center"
          >
            <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">
              {stat.label}
            </div>
            <div className={`text-2xl font-bold ${stat.color}`}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>
      {warnings.length > 0 && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 mb-4 text-yellow-300 text-sm">
          {warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/ui/SummaryBar.tsx
git commit -m "feat(dashboard): add SummaryBar component"
```

---

### Task 13: InstanceCard Component

**Files:**
- Create: `src/dashboard/ui/InstanceCard.tsx`

- [ ] **Step 1: Implement InstanceCard**

```tsx
// src/dashboard/ui/InstanceCard.tsx
import { useState } from 'react';
import type { VllmInstance, GpuDevice } from '../server/types';
import { LogViewer } from './LogViewer';

interface InstanceCardProps {
  instance: VllmInstance;
  gpuDevices: GpuDevice[];
  apiBase: string;
}

const STATUS_COLORS: Record<VllmInstance['status'], { bg: string; text: string }> = {
  running: { bg: 'bg-green-900/50', text: 'text-green-400' },
  starting: { bg: 'bg-yellow-900/50', text: 'text-yellow-400' },
  stopped: { bg: 'bg-gray-700/50', text: 'text-gray-400' },
  error: { bg: 'bg-red-900/50', text: 'text-red-400' },
};

function vramBarColor(percent: number): string {
  if (percent >= 90) return 'bg-red-500';
  if (percent >= 70) return 'bg-orange-500';
  return 'bg-blue-500';
}

function formatVram(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(0)} GB`;
  return `${mb} MB`;
}

export function InstanceCard({ instance, gpuDevices, apiBase }: InstanceCardProps) {
  const [showLaunchArgs, setShowLaunchArgs] = useState(false);
  const [showLogs, setShowLogs] = useState(false);

  const vramPercent =
    instance.vramTotalMb > 0
      ? (instance.vramUsedMb / instance.vramTotalMb) * 100
      : 0;

  const statusStyle = STATUS_COLORS[instance.status];
  // Resolve GPU names from device list
  const gpuLabel = instance.gpuIds.length > 0
    ? instance.gpuIds.map((id) => {
        const device = gpuDevices.find((d) => d.id === `gpu-${id}` || String(d.physicalId) === id);
        return device ? `GPU ${id} (${device.name})` : `GPU ${id}`;
      }).join(', ')
    : 'No GPU assigned';

  const infoItems = [
    gpuLabel,
    instance.partitionInfo,
    instance.quantization ? instance.quantization.toUpperCase() : null,
    instance.tensorParallelSize > 1 ? `TP=${instance.tensorParallelSize}` : null,
  ].filter(Boolean);

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-2">
        <div>
          <span className="text-white font-semibold text-sm">{instance.modelName}</span>
          <span className="text-gray-500 text-xs ml-2">{instance.containerName}</span>
        </div>
        <span className={`${statusStyle.bg} ${statusStyle.text} px-2 py-0.5 rounded text-xs font-medium uppercase`}>
          {instance.status}
        </span>
      </div>

      {/* Info line */}
      <div className="text-gray-400 text-xs mb-3">
        {infoItems.join(' · ')}
      </div>

      {/* VRAM bar */}
      {instance.vramTotalMb > 0 && (
        <div className="mb-3">
          <div className="bg-gray-900 rounded-full h-2 mb-1">
            <div
              className={`h-full rounded-full transition-all ${vramBarColor(vramPercent)}`}
              style={{ width: `${Math.min(vramPercent, 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-gray-400 text-xs">
            <span>VRAM: {formatVram(instance.vramUsedMb)} / {formatVram(instance.vramTotalMb)}</span>
            <span>{vramPercent.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Metrics row */}
      {instance.kvCachePercent !== undefined && (
        <div className="flex gap-4 text-xs text-gray-400 mb-3">
          <span>KV Cache: <span className="text-white">{instance.kvCachePercent.toFixed(0)}%</span></span>
          <span>Running: <span className="text-white">{instance.runningRequests ?? 0}</span></span>
          <span>Waiting: <span className="text-white">{instance.waitingRequests ?? 0}</span></span>
        </div>
      )}

      {/* Expandable sections */}
      <div className="flex gap-2 text-xs">
        <button
          onClick={() => setShowLaunchArgs(!showLaunchArgs)}
          className="text-blue-400 hover:text-blue-300"
        >
          {showLaunchArgs ? 'Hide' : 'Show'} launch args
        </button>
        <button
          onClick={() => setShowLogs(!showLogs)}
          className="text-blue-400 hover:text-blue-300"
        >
          {showLogs ? 'Hide' : 'Show'} logs
        </button>
      </div>

      {/* Launch args panel */}
      {showLaunchArgs && (
        <div className="mt-3 bg-gray-900 rounded p-3 text-xs font-mono text-gray-300 overflow-x-auto">
          <div className="text-gray-500 mb-1">Command:</div>
          <div>{instance.launchArgs.join(' ')}</div>
          {Object.keys(instance.envVars).length > 0 && (
            <>
              <div className="text-gray-500 mt-2 mb-1">Environment:</div>
              {Object.entries(instance.envVars).map(([k, v]) => (
                <div key={k}>{k}={v}</div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Log viewer panel */}
      {showLogs && (
        <div className="mt-3">
          <LogViewer
            containerId={instance.containerId}
            apiBase={apiBase}
          />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/ui/InstanceCard.tsx
git commit -m "feat(dashboard): add InstanceCard component"
```

---

### Task 14: LogViewer Component

**Files:**
- Create: `src/dashboard/ui/LogViewer.tsx`

- [ ] **Step 1: Implement LogViewer**

```tsx
// src/dashboard/ui/LogViewer.tsx
import { useRef, useEffect, useState } from 'react';
import { useLogStream } from './hooks/useLogStream';

interface LogViewerProps {
  containerId: string;
  apiBase: string;
}

export function LogViewer({ containerId, apiBase }: LogViewerProps) {
  const { lines, connected, error, droppedCount, clear } = useLogStream({
    containerId,
    apiBase,
  });
  const [paused, setPaused] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!paused && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, paused]);

  return (
    <div className="bg-gray-950 border border-gray-700 rounded">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-700 text-xs">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-gray-400">{connected ? 'Connected' : 'Disconnected'}</span>
          {droppedCount > 0 && (
            <span className="text-yellow-400">{droppedCount} lines dropped</span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setPaused(!paused)}
            className="text-gray-400 hover:text-white"
          >
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button
            onClick={clear}
            className="text-gray-400 hover:text-white"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Log content */}
      <div
        ref={scrollRef}
        className="p-2 font-mono text-xs max-h-64 overflow-y-auto"
      >
        {error && <div className="text-red-400 mb-1">{error}</div>}
        {lines.length === 0 && !error && (
          <div className="text-gray-600">Waiting for logs...</div>
        )}
        {lines.map((line, i) => (
          <div
            key={i}
            className={line.stream === 'stderr' ? 'text-red-400' : 'text-green-300'}
          >
            <span className="text-gray-600 mr-2 select-none">
              {new Date(line.timestamp).toLocaleTimeString()}
            </span>
            {line.line}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard/ui/LogViewer.tsx
git commit -m "feat(dashboard): add LogViewer component with streaming support"
```

---

### Task 15: InstanceGrid and DashboardPage

**Files:**
- Create: `src/dashboard/ui/InstanceGrid.tsx`
- Create: `src/dashboard/ui/DashboardPage.tsx`

- [ ] **Step 1: Implement InstanceGrid**

```tsx
// src/dashboard/ui/InstanceGrid.tsx
import type { VllmInstance, GpuDevice } from '../server/types';
import { InstanceCard } from './InstanceCard';

interface InstanceGridProps {
  instances: VllmInstance[];
  gpuDevices: GpuDevice[];
  apiBase: string;
}

export function InstanceGrid({ instances, gpuDevices, apiBase }: InstanceGridProps) {
  if (instances.length === 0) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-8 text-center">
        <div className="text-gray-400 text-lg mb-2">No vLLM instances found</div>
        <div className="text-gray-500 text-sm">
          Make sure vLLM containers are running and use a recognized image
          (vllm/*, rocm/vllm*).
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {instances.map((instance) => (
        <InstanceCard
          key={instance.containerId}
          instance={instance}
          gpuDevices={gpuDevices}
          apiBase={apiBase}
        />
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Implement DashboardPage**

```tsx
// src/dashboard/ui/DashboardPage.tsx
import { SummaryBar } from './SummaryBar';
import { InstanceGrid } from './InstanceGrid';
import { useDashboardData } from './hooks/useDashboardData';

interface DashboardPageProps {
  apiBase?: string;
}

export function DashboardPage({ apiBase = 'http://localhost:3001' }: DashboardPageProps) {
  const { status, instances, gpus, loading, error } = useDashboardData(apiBase);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">vLLM Dashboard</h1>
          {status && (
            <div className="text-gray-500 text-xs">
              Last updated: {new Date(status.timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>

        {/* Connection error */}
        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Loading state */}
        {loading && !error && (
          <div className="text-gray-400 text-center py-12">
            Connecting to dashboard backend...
          </div>
        )}

        {/* Dashboard content */}
        {!loading && (
          <>
            <SummaryBar
              summary={status?.summary ?? null}
              warnings={status?.warnings ?? []}
            />
            <InstanceGrid instances={instances} gpuDevices={gpus?.devices ?? []} apiBase={apiBase} />
          </>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/ui/InstanceGrid.tsx src/dashboard/ui/DashboardPage.tsx
git commit -m "feat(dashboard): add InstanceGrid and DashboardPage components"
```

---

### Task 16: Dashboard HTML Entry Point

**Files:**
- Create: `src/dashboard/ui/index.html`
- Create: `src/dashboard/ui/main.tsx`

- [ ] **Step 1: Create dashboard HTML entry point**

```html
<!-- src/dashboard/ui/index.html -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>vLLM Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="./main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 2: Create dashboard React entry point**

```tsx
// src/dashboard/ui/main.tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { DashboardPage } from './DashboardPage';
import '../../index.css';

const apiBase = window.location.origin;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DashboardPage apiBase={apiBase} />
  </StrictMode>
);
```

- [ ] **Step 3: Add Vite config for dashboard build**

Add to `package.json` scripts:
```json
"build:dashboard-ui": "vite build --config vite.dashboard.config.ts"
```

Create `vite.dashboard.config.ts`:
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  root: 'src/dashboard/ui',
  build: {
    outDir: '../../../dist/dashboard',
    emptyOutDir: true,
  },
});
```

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/ui/index.html src/dashboard/ui/main.tsx vite.dashboard.config.ts package.json
git commit -m "feat(dashboard): add frontend entry point and Vite dashboard build"
```

---

### Task 17: End-to-End Verification

- [ ] **Step 1: Run all tests**

Run: `npm test`
Expected: All tests pass (existing + new dashboard tests)

- [ ] **Step 2: Verify TypeScript compilation**

Run: `npx tsc --noEmit --project tsconfig.dashboard.json`
Expected: No type errors

- [ ] **Step 3: Build the dashboard UI**

Run: `npm run build:dashboard-ui`
Expected: Build completes, output in `dist/dashboard/`

- [ ] **Step 4: Build the dashboard server**

Run: `npm run build:dashboard`
Expected: Build completes, output in `dist/dashboard.js`

- [ ] **Step 5: Commit any remaining fixes**

```bash
git add -u
git commit -m "chore(dashboard): fix build issues from integration"
```

- [ ] **Step 6: Final commit message summarizing the feature**

Only if there were integration fixes in step 5. Otherwise this task is done.
