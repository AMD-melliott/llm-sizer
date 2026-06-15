import { buildCalculateResult } from '../../../src/cli/commands/calculate';

describe('Calculate Command', () => {
  test('calculates memory for llama-3-70b on mi300x (fp16)', () => {
    const result = buildCalculateResult({
      model: 'llama-3-70b',
      gpu: 'mi300x',
      gpus: 1,
      quant: 'fp16',
      kvQuant: 'fp16_bf16',
      type: 'generation',
      batchSize: 1,
      seqLength: 4096,
      users: 1,
    });

    // 70B fp16 on 1x MI300X (192GB): usedVRAM ~174GB, vramPercentage ~90.8% → 'warning'
    expect(result.status).not.toBe('okay');
    expect(result.usedVRAM).toBeGreaterThan(100);
    expect(result.memoryBreakdown.baseWeights).toBeCloseTo(140, 0);
    expect(result.inputs.model).toBe('llama-3-70b');
    expect(result.inputs.gpu).toBe('mi300x');
  });

  test('calculates memory for llama-3-70b on 2x mi300x (fp8)', () => {
    const result = buildCalculateResult({
      model: 'llama-3-70b',
      gpu: 'mi300x',
      gpus: 2,
      quant: 'fp8',
      kvQuant: 'fp8_bf16',
      type: 'generation',
      batchSize: 1,
      seqLength: 4096,
      users: 1,
    });

    expect(result.status).not.toBe('error');
    expect(result.usedVRAM).toBeGreaterThan(0);
    expect(result.totalVRAM).toBe(384);
    expect(result.memoryBreakdown.baseWeights).toBeCloseTo(70, 0);
  });

  test('includes performance metrics for generation models', () => {
    const result = buildCalculateResult({
      model: 'llama-3-70b',
      gpu: 'mi300x',
      gpus: 2,
      quant: 'fp8',
      kvQuant: 'fp8_bf16',
      type: 'generation',
      batchSize: 1,
      seqLength: 4096,
      users: 1,
    });

    expect(result.performance).toBeDefined();
    expect(result.performance.totalThroughput).toBeGreaterThan(0);
  });

  test('throws for unknown model ID', () => {
    expect(() => buildCalculateResult({
      model: 'nonexistent',
      gpu: 'mi300x',
      gpus: 1,
      quant: 'fp16',
      kvQuant: 'fp16_bf16',
      type: 'generation',
      batchSize: 1,
      seqLength: 4096,
      users: 1,
    })).toThrow(/model not found/i);
  });

  test('throws for unknown GPU ID', () => {
    expect(() => buildCalculateResult({
      model: 'llama-3-70b',
      gpu: 'nonexistent',
      gpus: 1,
      quant: 'fp16',
      kvQuant: 'fp16_bf16',
      type: 'generation',
      batchSize: 1,
      seqLength: 4096,
      users: 1,
    })).toThrow(/gpu not found/i);
  });

  test('echoes back input parameters in result', () => {
    const opts = {
      model: 'llama-3-70b',
      gpu: 'mi300x',
      gpus: 2,
      quant: 'fp8' as const,
      kvQuant: 'fp8_bf16' as const,
      type: 'generation' as const,
      batchSize: 4,
      seqLength: 2048,
      users: 2,
    };
    const result = buildCalculateResult(opts);

    expect(result.inputs.model).toBe('llama-3-70b');
    expect(result.inputs.gpu).toBe('mi300x');
    expect(result.inputs.gpus).toBe(2);
    expect(result.inputs.quant).toBe('fp8');
    expect(result.inputs.batchSize).toBe(4);
    expect(result.inputs.seqLength).toBe(2048);
    expect(result.inputs.users).toBe(2);
  });
});
