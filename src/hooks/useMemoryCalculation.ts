import { useState, useEffect, useMemo } from 'react';
import {
  Model,
  EmbeddingModel,
  RerankingModel,
  GPU,
  InferenceQuantization,
  KVCacheQuantization,
  CalculationResults,
  ModelType
} from '../types';
import { calculateMemoryRequirements } from '../utils/memoryCalculator';
import { estimatePerformance } from '../utils/performanceEstimator';
import { calculateEmbeddingMemory } from '../utils/embeddingCalculator';
import { calculateRerankingMemory } from '../utils/rerankingCalculator';
import modelsData from '../data/models.json';
import embeddingModelsData from '../data/embedding-models.json';
import rerankingModelsData from '../data/reranking-models.json';
import gpusData from '../data/gpus.json';

interface UseMemoryCalculationProps {
  modelType: ModelType;
  selectedModelId: string;
  selectedGPUId: string;
  inferenceQuantization: InferenceQuantization;
  kvCacheQuantization: KVCacheQuantization;
  batchSize: number;
  sequenceLength: number;
  concurrentUsers: number;
  numGPUs: number;
  numImages: number;
  imageResolution: number;
  embeddingBatchSize: number;
  documentsPerBatch: number;
  avgDocumentSize: number;
  chunkSize: number;
  chunkOverlap: number;
  rerankingBatchSize: number;
  numQueries: number;
  docsPerQuery: number;
  maxQueryLength: number;
  maxDocLength: number;
  customModelParams?: number;
  customHiddenSize?: number;
  customNumLayers?: number;
  customNumHeads?: number;
  customVRAM?: number;
}

export function useMemoryCalculation({
  modelType,
  selectedModelId,
  selectedGPUId,
  inferenceQuantization,
  kvCacheQuantization,
  batchSize,
  sequenceLength,
  concurrentUsers,
  numGPUs,
  numImages,
  imageResolution,
  embeddingBatchSize,
  documentsPerBatch,
  avgDocumentSize,
  chunkSize,
  chunkOverlap,
  rerankingBatchSize,
  numQueries,
  docsPerQuery,
  maxQueryLength,
  maxDocLength,
  customModelParams,
  customHiddenSize,
  customNumLayers,
  customNumHeads,
  customVRAM,
}: UseMemoryCalculationProps) {
  const [results, setResults] = useState<CalculationResults | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);

  // Get model and GPU data
  const models = useMemo(() => modelsData.models as Model[], []);
  const embeddingModels = useMemo(() => (embeddingModelsData as any).models as EmbeddingModel[], []);
  const rerankingModels = useMemo(() => (rerankingModelsData as any).models as RerankingModel[], []);
  const gpus = useMemo(() => gpusData.gpus as GPU[], []);

  const selectedModel = useMemo(() => {
    const baseModel = models.find(m => m.id === selectedModelId);
    if (!baseModel) return null;

    // Override with custom values if provided
    if (selectedModelId === 'custom') {
      return {
        ...baseModel,
        parameters_billions: customModelParams || baseModel.parameters_billions,
        hidden_size: customHiddenSize || baseModel.hidden_size,
        num_layers: customNumLayers || baseModel.num_layers,
        num_heads: customNumHeads || baseModel.num_heads,
      };
    }

    return baseModel;
  }, [
    selectedModelId,
    models,
    customModelParams,
    customHiddenSize,
    customNumLayers,
    customNumHeads
  ]);

  const selectedGPU = useMemo(() => {
    const baseGPU = gpus.find(g => g.id === selectedGPUId);
    if (!baseGPU) return null;

    // Override VRAM if custom value provided
    if (selectedGPUId === 'custom' && customVRAM) {
      return {
        ...baseGPU,
        vram_gb: customVRAM,
      };
    }

    return baseGPU;
  }, [selectedGPUId, gpus, customVRAM]);

  // Calculate memory requirements whenever inputs change
  useEffect(() => {
    if (!selectedGPU) {
      setResults(null);
      return;
    }

    setIsCalculating(true);

    // Use setTimeout to prevent blocking UI during calculation
    const timer = setTimeout(() => {
      try {
        let finalResults: CalculationResults;

        if (modelType === 'embedding') {
          const embModel = embeddingModels.find(m => m.id === selectedModelId);
          if (!embModel) {
            setResults(null);
            setIsCalculating(false);
            return;
          }

          finalResults = calculateEmbeddingMemory(
            embModel,
            selectedGPU,
            inferenceQuantization,
            embeddingBatchSize,
            documentsPerBatch,
            avgDocumentSize,
            numGPUs
          );
        } else if (modelType === 'reranking') {
          const rerankModel = rerankingModels.find(m => m.id === selectedModelId);
          if (!rerankModel) {
            setResults(null);
            setIsCalculating(false);
            return;
          }

          finalResults = calculateRerankingMemory(
            rerankModel,
            selectedGPU,
            inferenceQuantization,
            numQueries,
            docsPerQuery,
            maxQueryLength,
            maxDocLength,
            numGPUs,
            rerankingBatchSize
          );
        } else {
          // Generation model
          if (!selectedModel) {
            setResults(null);
            setIsCalculating(false);
            return;
          }

          const memoryResults = calculateMemoryRequirements(
            selectedModel,
            selectedGPU,
            inferenceQuantization,
            kvCacheQuantization,
            batchSize,
            sequenceLength,
            concurrentUsers,
            numGPUs,
            numImages,
            imageResolution
          );

          const performanceMetrics = estimatePerformance(
            selectedModel,
            selectedGPU,
            inferenceQuantization,
            batchSize,
            sequenceLength,
            concurrentUsers,
            numGPUs,
            memoryResults.vramPercentage
          );

          finalResults = {
            ...memoryResults,
            performance: performanceMetrics,
          };
        }

        setResults(finalResults);
      } catch (error) {
        console.error('Error calculating memory requirements:', error);
        setResults(null);
      } finally {
        setIsCalculating(false);
      }
    }, 100); // Small delay to debounce rapid changes

    return () => clearTimeout(timer);
  }, [
    modelType,
    selectedModel,
    selectedModelId,
    selectedGPU,
    inferenceQuantization,
    kvCacheQuantization,
    batchSize,
    sequenceLength,
    concurrentUsers,
    numGPUs,
    numImages,
    imageResolution,
    embeddingBatchSize,
    documentsPerBatch,
    avgDocumentSize,
    chunkSize,
    chunkOverlap,
    embeddingModels,
    rerankingBatchSize,
    numQueries,
    docsPerQuery,
    maxQueryLength,
    maxDocLength,
    rerankingModels,
  ]);

  return {
    results,
    isCalculating,
    models,
    embeddingModels,
    rerankingModels,
    gpus,
    selectedModel,
    selectedGPU,
  };
}