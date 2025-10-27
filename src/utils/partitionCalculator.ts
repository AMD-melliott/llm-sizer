import {
  Model,
  EmbeddingModel,
  RerankingModel,
  GPU,
  PartitionMode,
  PartitionConfiguration,
  PartitionMemoryResult,
  PartitionAnalysisResults,
  InferenceQuantization,
  KVCacheQuantization,
  ModelType,
  PartitionModelSelection,
  MemoryBreakdown,
} from '../types';
import { calculateMemoryRequirements } from './memoryCalculator';
import { calculateEmbeddingMemory } from './embeddingCalculator';
import { calculateRerankingMemory } from './rerankingCalculator';

/**
 * Calculate memory requirements for a single partition with a specific model
 */
export function calculatePartitionMemory(
  model: Model | EmbeddingModel | RerankingModel,
  partitionMode: PartitionMode,
  sourceGPU: GPU,
  inferenceQuantization: InferenceQuantization,
  kvCacheQuantization: KVCacheQuantization,
  batchSize: number,
  sequenceLength: number,
  concurrentUsers: number,
  modelType: ModelType
): PartitionMemoryResult {
  // Create a temporary GPU object for the partition
  // Inherit properties from source GPU rather than using hard-coded fallbacks
  const partitionGPU: GPU = {
    id: 'partition',
    vendor: sourceGPU.vendor,
    name: 'Partition',
    category: sourceGPU.category,
    tier: sourceGPU.tier,
    vram_gb: partitionMode.vramPerPartition,
    memory_type: sourceGPU.memory_type,
    memory_bandwidth_gbps: partitionMode.bandwidthPerPartition,
    compute_tflops_fp16: partitionMode.computeFP16PerPartition,
    compute_tflops_fp8: partitionMode.computeFP8PerPartition,
    pcie_gen: sourceGPU.pcie_gen,
    tdp_watts: 0, // Not relevant for partition
    multi_gpu_capable: false,
    release_year: sourceGPU.release_year,
  };

  let memoryUsed = 0;
  let memoryBreakdown: MemoryBreakdown;
  let recommendedQuantization: InferenceQuantization | undefined;

  if (modelType === 'generation' && 'parameters_billions' in model) {
    // Text generation model
    const results = calculateMemoryRequirements(
      model as Model,
      partitionGPU,
      inferenceQuantization,
      kvCacheQuantization,
      batchSize,
      sequenceLength,
      concurrentUsers,
      1, // Single partition acts as single GPU
      1, // No images for now (can be extended)
      336 // Default image resolution
    );
    memoryUsed = results.usedVRAM;
    memoryBreakdown = results.memoryBreakdown;
    // Determine recommended quantization (lowest precision that fits). Order from highest to lowest quality.
    const quantizationOrder: InferenceQuantization[] = ['fp16', 'fp8', 'int8', 'int4'];
    for (const q of quantizationOrder) {
      const alt = calculateMemoryRequirements(
        model as Model,
        partitionGPU,
        q,
        kvCacheQuantization,
        batchSize,
        sequenceLength,
        concurrentUsers,
        1,
        1,
        336
      );
      if (alt.usedVRAM <= partitionMode.vramPerPartition) {
        recommendedQuantization = q;
        break;
      }
    }
  } else if (modelType === 'embedding' && 'dimensions' in model) {
    // Embedding model
    const embeddingModel = model as EmbeddingModel;
    const results = calculateEmbeddingMemory(
      embeddingModel,
      partitionGPU,
      inferenceQuantization,
      batchSize,
      10, // Default documents per batch
      512, // Default avg document size
      1 // Single partition
    );
    memoryUsed = results.usedVRAM;
    memoryBreakdown = results.memoryBreakdown;
    const quantizationOrder: InferenceQuantization[] = ['fp16', 'fp8', 'int8', 'int4'];
    for (const q of quantizationOrder) {
      const alt = calculateEmbeddingMemory(
        embeddingModel,
        partitionGPU,
        q,
        batchSize,
        10,
        512,
        1
      );
      if (alt.usedVRAM <= partitionMode.vramPerPartition) {
        recommendedQuantization = q;
        break;
      }
    }
  } else if (modelType === 'reranking' && 'max_query_length' in model) {
    // Reranking model
    const rerankingModel = model as RerankingModel;
    const results = calculateRerankingMemory(
      rerankingModel,
      partitionGPU,
      inferenceQuantization,
      concurrentUsers, // Use as numQueries
      10, // Default docs per query
      Math.min(sequenceLength / 2, rerankingModel.max_query_length), // Default query length
      Math.min(sequenceLength, rerankingModel.max_doc_length), // Default doc length
      1, // Single partition
      batchSize
    );
    memoryUsed = results.usedVRAM;
    memoryBreakdown = results.memoryBreakdown;
    const quantizationOrder: InferenceQuantization[] = ['fp16', 'fp8', 'int8', 'int4'];
    for (const q of quantizationOrder) {
      const alt = calculateRerankingMemory(
        rerankingModel,
        partitionGPU,
        q,
        concurrentUsers,
        10,
        Math.min(sequenceLength / 2, rerankingModel.max_query_length),
        Math.min(sequenceLength, rerankingModel.max_doc_length),
        1,
        batchSize
      );
      if (alt.usedVRAM <= partitionMode.vramPerPartition) {
        recommendedQuantization = q;
        break;
      }
    }
  } else {
    // Unknown model type
    memoryBreakdown = {
      baseWeights: 0,
      activations: 0,
      kvCache: 0,
      frameworkOverhead: 0,
      multiGPUOverhead: 0,
    };
  }

  const memoryAvailable = partitionMode.vramPerPartition;
  const percentUsed = (memoryUsed / memoryAvailable) * 100;
  const fits = memoryUsed <= memoryAvailable;

  // Determine status
  let status: 'fits' | 'tight' | 'no-fit';
  if (!fits) {
    status = 'no-fit';
  } else if (percentUsed > 80) {
    status = 'tight';
  } else {
    status = 'fits';
  }

  return {
    partitionIndex: 0,
    model,
    memoryUsed,
    memoryAvailable,
    percentUsed,
    fits,
    status,
    memoryBreakdown,
    recommendedQuantization,
  };
}

/**
 * Analyze all models for a given partition configuration
 */
export function analyzePartitionConfiguration(
  configuration: PartitionConfiguration,
  modelType: PartitionModelSelection,
  models: (Model | EmbeddingModel | RerankingModel)[]
): PartitionAnalysisResults {
  const compatibleModels: PartitionMemoryResult[] = [];

  for (const model of models) {
    // Determine the actual model type from the model structure
    let actualModelType: ModelType;
    if ('parameters_billions' in model) {
      actualModelType = 'generation';
    } else if ('dimensions' in model) {
      actualModelType = 'embedding';
    } else {
      actualModelType = 'reranking';
    }

    const result = calculatePartitionMemory(
      model,
      configuration.mode,
      configuration.gpu,
      configuration.inferenceQuantization,
      configuration.kvCacheQuantization,
      configuration.batchSize,
      configuration.sequenceLength,
      configuration.concurrentUsers,
      actualModelType
    );
    compatibleModels.push(result);
  }

  // Sort by memory used (ascending)
  compatibleModels.sort((a, b) => a.memoryUsed - b.memoryUsed);

  return {
    configuration,
    compatibleModels,
    modelType, // Preserve the original selection including 'all'
  };
}

/**
 * Get partition-capable GPUs
 */
export function getPartitionCapableGPUs(gpus: GPU[]): GPU[] {
  return gpus.filter(gpu => gpu.partitioning?.supported);
}

/**
 * Get partition mode for a GPU
 */
export function getPartitionMode(gpu: GPU, modeType: string): PartitionMode | undefined {
  return gpu.partitioning?.modes.find(mode => mode.mode === modeType);
}

/**
 * Format bandwidth for display with automatic unit scaling
 */
export function formatBandwidth(gbps: number): string {
  if (gbps >= 1000) {
    return `${(gbps / 1000).toFixed(2)} TB/s`;
  }
  return `${gbps.toFixed(1)} GB/s`;
}

/**
 * Format memory size for display
 */
export function formatMemorySize(gb: number): string {
  if (gb < 1) {
    return `${(gb * 1024).toFixed(1)} MB`;
  }
  return `${gb.toFixed(1)} GB`;
}

/**
 * Format compute for display
 */
export function formatCompute(tflops: number): string {
  return `${tflops.toFixed(1)} TFLOPS`;
}

/**
 * Get model name for display
 */
export function getModelDisplayName(model: Model | EmbeddingModel | RerankingModel): string {
  if ('parameters_billions' in model) {
    const genModel = model as Model;
    return `${genModel.name} (${genModel.parameters_billions}B)`;
  } else if ('dimensions' in model) {
    const embModel = model as EmbeddingModel;
    const params = embModel.parameters_millions;
    if (params >= 1000) {
      return `${embModel.name} (${(params / 1000).toFixed(1)}B)`;
    }
    return `${embModel.name} (${params}M)`;
  } else if ('max_query_length' in model) {
    const rerankModel = model as RerankingModel;
    const params = rerankModel.parameters_millions;
    if (params >= 1000) {
      return `${rerankModel.name} (${(params / 1000).toFixed(1)}B)`;
    }
    return `${rerankModel.name} (${params}M)`;
  }
  return '';
}
