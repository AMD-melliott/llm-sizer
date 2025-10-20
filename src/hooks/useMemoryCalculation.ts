import { useState, useEffect, useMemo } from 'react';
import {
  Model,
  GPU,
  InferenceQuantization,
  KVCacheQuantization,
  CalculationResults
} from '../types';
import { calculateMemoryRequirements } from '../utils/memoryCalculator';
import { estimatePerformance } from '../utils/performanceEstimator';
import modelsData from '../data/models.json';
import gpusData from '../data/gpus.json';

interface UseMemoryCalculationProps {
  selectedModelId: string;
  selectedGPUId: string;
  inferenceQuantization: InferenceQuantization;
  kvCacheQuantization: KVCacheQuantization;
  batchSize: number;
  sequenceLength: number;
  concurrentUsers: number;
  numGPUs: number;
  enableOffloading: boolean;
  customModelParams?: number;
  customHiddenSize?: number;
  customNumLayers?: number;
  customNumHeads?: number;
  customVRAM?: number;
}

export function useMemoryCalculation({
  selectedModelId,
  selectedGPUId,
  inferenceQuantization,
  kvCacheQuantization,
  batchSize,
  sequenceLength,
  concurrentUsers,
  numGPUs,
  enableOffloading,
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
    if (!selectedModel || !selectedGPU) {
      setResults(null);
      return;
    }

    setIsCalculating(true);

    // Use setTimeout to prevent blocking UI during calculation
    const timer = setTimeout(() => {
      try {
        // Calculate memory requirements
        const memoryResults = calculateMemoryRequirements(
          selectedModel,
          selectedGPU,
          inferenceQuantization,
          kvCacheQuantization,
          batchSize,
          sequenceLength,
          concurrentUsers,
          numGPUs,
          enableOffloading
        );

        // Estimate performance
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

        // Combine results
        const finalResults: CalculationResults = {
          ...memoryResults,
          performance: performanceMetrics,
        };

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
    selectedModel,
    selectedGPU,
    inferenceQuantization,
    kvCacheQuantization,
    batchSize,
    sequenceLength,
    concurrentUsers,
    numGPUs,
    enableOffloading,
  ]);

  return {
    results,
    isCalculating,
    models,
    gpus,
    selectedModel,
    selectedGPU,
  };
}