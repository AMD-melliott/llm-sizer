import modelsData from '../../data/models.json';
import gpusData from '../../data/gpus.json';
import embeddingModelsData from '../../data/embedding-models.json';
import rerankingModelsData from '../../data/reranking-models.json';
import type { Model, GPU, EmbeddingModel, RerankingModel } from '../../types';

export function loadModels(): Model[] {
  return modelsData.models as Model[];
}

export function loadGPUs(): GPU[] {
  return gpusData.gpus as GPU[];
}

export function loadEmbeddingModels(): EmbeddingModel[] {
  return (embeddingModelsData as any).models as EmbeddingModel[];
}

export function loadRerankingModels(): RerankingModel[] {
  return (rerankingModelsData as any).models as RerankingModel[];
}
