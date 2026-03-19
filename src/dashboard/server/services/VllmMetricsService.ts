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
