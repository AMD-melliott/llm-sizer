import {
  resolveModel,
  resolveGPU,
  resolveModelByType,
  resolveHFModelId,
} from '../../../src/cli/utils/model-resolver';

describe('Model Resolver', () => {
  test('resolveModel finds model by ID', () => {
    const model = resolveModel('llama-3-70b');
    expect(model).not.toBeNull();
    expect(model!.id).toBe('llama-3-70b');
    expect(model!.parameters_billions).toBe(70);
  });

  test('resolveModel returns null for unknown ID', () => {
    const model = resolveModel('nonexistent-model');
    expect(model).toBeNull();
  });

  test('resolveGPU finds GPU by ID', () => {
    const gpu = resolveGPU('mi300x');
    expect(gpu).not.toBeNull();
    expect(gpu!.id).toBe('mi300x');
    expect(gpu!.vram_gb).toBe(192);
  });

  test('resolveGPU returns null for unknown ID', () => {
    const gpu = resolveGPU('nonexistent-gpu');
    expect(gpu).toBeNull();
  });

  test('resolveModelByType resolves generation model', () => {
    const model = resolveModelByType('llama-3-70b', 'generation');
    expect(model).not.toBeNull();
    expect(model!.id).toBe('llama-3-70b');
  });

  test('resolveModelByType resolves embedding model', () => {
    const model = resolveModelByType('bge-large-en-v1.5', 'embedding');
    expect(model).not.toBeNull();
    expect(model!.id).toBe('bge-large-en-v1.5');
  });

  test('resolveModelByType resolves reranking model', () => {
    const model = resolveModelByType('bge-reranker-large', 'reranking');
    expect(model).not.toBeNull();
    expect(model!.id).toBe('bge-reranker-large');
  });

  test('resolveModelByType returns null for type mismatch', () => {
    const model = resolveModelByType('llama-3-70b', 'embedding');
    expect(model).toBeNull();
  });

  test('resolveHFModelId returns hf_model_id for known model', () => {
    const hfId = resolveHFModelId('llama-3-70b');
    expect(hfId).not.toBeNull();
    expect(hfId).toContain('Llama');
  });

  test('resolveHFModelId returns null for unknown model', () => {
    const hfId = resolveHFModelId('nonexistent');
    expect(hfId).toBeNull();
  });
});
