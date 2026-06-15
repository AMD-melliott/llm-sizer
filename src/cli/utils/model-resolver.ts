import type { Model, GPU, EmbeddingModel, RerankingModel, ModelType } from '../../types';
import { loadModels, loadGPUs, loadEmbeddingModels, loadRerankingModels } from './data-loader';

export function resolveModel(id: string): Model | null {
  const models = loadModels();
  return models.find(m => m.id === id) ?? null;
}

export function resolveGPU(id: string): GPU | null {
  const gpus = loadGPUs();
  return gpus.find(g => g.id === id) ?? null;
}

export function resolveModelByType(
  id: string,
  type: ModelType
): Model | EmbeddingModel | RerankingModel | null {
  switch (type) {
    case 'generation':
      return resolveModel(id);
    case 'embedding': {
      const models = loadEmbeddingModels();
      return models.find(m => m.id === id) ?? null;
    }
    case 'reranking': {
      const models = loadRerankingModels();
      return models.find(m => m.id === id) ?? null;
    }
    default:
      return null;
  }
}

export function resolveHFModelId(id: string): string | null {
  const model = resolveModel(id);
  return model?.hf_model_id ?? null;
}
