import { registerStatusRoute } from '../../../src/dashboard/server/routes/status';
import { registerInstancesRoute } from '../../../src/dashboard/server/routes/instances';
import { registerGpusRoute } from '../../../src/dashboard/server/routes/gpus';
import type { DashboardSnapshot } from '../../../src/dashboard/server/types';

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
