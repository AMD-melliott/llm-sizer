import {
  Model,
  GPU,
  InferenceQuantization,
  KVCacheQuantization,
  MemoryBreakdown,
  CalculationResults
} from '../types';

// Quantization bit sizes
const QUANTIZATION_BITS: Record<InferenceQuantization, number> = {
  'fp16': 16,
  'fp8': 8,
  'int8': 8,
  'int4': 4,
};

const KV_CACHE_BITS: Record<KVCacheQuantization, number> = {
  'fp16_bf16': 16,
  'fp8_bf16': 8,
  'int8': 8,
};

export function calculateMemoryRequirements(
  model: Model,
  gpu: GPU,
  inferenceQuantization: InferenceQuantization,
  kvCacheQuantization: KVCacheQuantization,
  batchSize: number,
  sequenceLength: number,
  concurrentUsers: number,
  numGPUs: number,
  enableOffloading: boolean
): CalculationResults {
  // Calculate base model weights in GB
  const bitsPerParam = QUANTIZATION_BITS[inferenceQuantization];
  const baseWeights = (model.parameters_billions * 1e9 * bitsPerParam) / 8 / 1e9;

  // Calculate KV cache requirements
  // KV cache per token = 2 (K+V) * num_layers * hidden_size * kv_bits / 8
  const kvBitsPerParam = KV_CACHE_BITS[kvCacheQuantization];
  const kvCachePerToken = 2 * model.num_layers * model.hidden_size * kvBitsPerParam / 8;

  // Total KV cache for all users and sequences
  const totalKVCache = (kvCachePerToken * batchSize * sequenceLength * concurrentUsers) / 1e9;

  // Calculate activation memory (rough estimate based on model size and batch)
  // Activations scale with batch size and model dimensions
  const activationsPerBatch = (
    batchSize * sequenceLength * model.hidden_size * 4 * // 4 bytes for fp32 intermediate
    model.num_layers * 0.1 // Rough factor for activation checkpointing
  ) / 1e9;

  const activations = activationsPerBatch * concurrentUsers;

  // Framework overhead (5-10% of base memory usage)
  const frameworkOverhead = (baseWeights + totalKVCache + activations) * 0.08;

  // Multi-GPU overhead (increases with GPU count due to communication)
  const multiGPUOverhead = numGPUs > 1
    ? (baseWeights + totalKVCache + activations + frameworkOverhead) * 0.02 * (numGPUs - 1)
    : 0;

  // Calculate total memory usage
  const usedVRAM = baseWeights + totalKVCache + activations + frameworkOverhead + multiGPUOverhead;
  const totalVRAM = gpu.vram_gb * numGPUs;
  const vramPercentage = (usedVRAM / totalVRAM) * 100;

  // Determine status based on usage
  let status: 'okay' | 'warning' | 'error' = 'okay';
  let message = '';

  if (vramPercentage > 100) {
    status = 'error';
    message = `Memory requirement exceeds available VRAM by ${(usedVRAM - totalVRAM).toFixed(1)} GB. Consider: reducing batch size, using more aggressive quantization, adding more GPUs, or enabling offloading.`;
  } else if (vramPercentage > 90) {
    status = 'warning';
    message = 'Very high VRAM usage (>90%). Performance may degrade due to limited memory for caching and buffers.';
  } else if (vramPercentage > 80) {
    status = 'warning';
    message = 'High VRAM usage (>80%). Consider leaving more headroom for optimal performance.';
  }

  // If offloading is enabled and we're over capacity, adjust the message
  if (enableOffloading && vramPercentage > 100) {
    status = 'warning';
    message = `Offloading enabled: ${(usedVRAM - totalVRAM).toFixed(1)} GB will be offloaded to system RAM/NVMe. Performance will be reduced.`;
  }

  const memoryBreakdown: MemoryBreakdown = {
    baseWeights,
    activations,
    kvCache: totalKVCache,
    frameworkOverhead,
    multiGPUOverhead,
  };

  // Calculate performance metrics (will be done in performanceEstimator.ts)
  const performance = {
    generationSpeed: 0, // Placeholder
    totalThroughput: 0, // Placeholder
    perUserSpeed: 0, // Placeholder
  };

  return {
    totalVRAM,
    usedVRAM,
    vramPercentage,
    memoryBreakdown,
    performance,
    status,
    message,
  };
}

export function formatMemorySize(sizeInGB: number): string {
  if (sizeInGB < 1) {
    return `${(sizeInGB * 1024).toFixed(1)} MB`;
  }
  return `${sizeInGB.toFixed(2)} GB`;
}

export function getQuantizationInfo(quantization: InferenceQuantization | KVCacheQuantization): string {
  const info: Record<string, string> = {
    'fp16': 'Full precision (16-bit) - Best quality, highest memory usage',
    'fp8': '8-bit floating point - Good balance of quality and memory',
    'int8': '8-bit integer - Reduced memory, slight quality loss',
    'int4': '4-bit integer - Minimal memory, noticeable quality loss',
    'fp16_bf16': '16-bit KV cache - Best quality for attention',
    'fp8_bf16': '8-bit KV cache - Reduced memory with minimal quality loss',
  };

  return info[quantization] || 'Unknown quantization';
}