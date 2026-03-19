import { DockerService } from '../../../src/dashboard/server/services/DockerService';

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
