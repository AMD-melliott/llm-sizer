import { VllmMetricsService } from '../../../src/dashboard/server/services/VllmMetricsService';

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
    expect(metrics!.runningRequests).toBe(3);
    expect(metrics!.waitingRequests).toBe(1);
    expect(metrics!.kvCachePercent).toBeCloseTo(42);
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
