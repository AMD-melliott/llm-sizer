import { loadModels, loadGPUs, loadEmbeddingModels, loadRerankingModels } from '../../../src/cli/utils/data-loader';

describe('Data Loader', () => {
  test('loadModels returns array of generation models', () => {
    const models = loadModels();
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);
    expect(models[0]).toHaveProperty('id');
    expect(models[0]).toHaveProperty('parameters_billions');
    expect(models[0]).toHaveProperty('hidden_size');
    expect(models[0]).toHaveProperty('num_layers');
    expect(models[0]).toHaveProperty('num_heads');
  });

  test('loadGPUs returns array of GPUs', () => {
    const gpus = loadGPUs();
    expect(Array.isArray(gpus)).toBe(true);
    expect(gpus.length).toBeGreaterThan(0);
    expect(gpus[0]).toHaveProperty('id');
    expect(gpus[0]).toHaveProperty('vram_gb');
    expect(gpus[0]).toHaveProperty('memory_bandwidth_gbps');
  });

  test('loadEmbeddingModels returns array of embedding models', () => {
    const models = loadEmbeddingModels();
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);
    expect(models[0]).toHaveProperty('dimensions');
  });

  test('loadRerankingModels returns array of reranking models', () => {
    const models = loadRerankingModels();
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);
    expect(models[0]).toHaveProperty('max_query_length');
  });
});
