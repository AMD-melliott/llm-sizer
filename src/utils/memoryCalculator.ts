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
  enableOffloading: boolean,
  numImages: number = 1,
  imageResolution: number = 336
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

  // Multimodal memory components (only for multimodal models)
  // NOTE: These calculations provide reasonable estimates for vision-language models
  // but actual memory usage may vary based on:
  // - Specific vision encoder architecture (ViT, CLIP, DaViT, etc.)
  // - Image preprocessing pipeline and tensor formats
  // - Framework-specific optimizations (gradient checkpointing, fused kernels)
  // - Projector/adapter architecture (linear, MLP, resampler, etc.)
  let visionWeights = 0;
  let visionActivations = 0;
  let projectorWeights = 0;
  let imagePreprocessing = 0;
  let imageTokensKV = 0;

  if (model.modality === 'multimodal' && model.vision_config && model.multimodal_config) {
    // Vision encoder weights (quantized at same level as base model)
    // Assumes vision encoder uses same quantization as language model
    visionWeights = (model.vision_config.parameters_millions * 1e6 * bitsPerParam) / 8 / 1e9;

    // Vision encoder activations (per image, per batch)
    // Factor of 4 accounts for intermediate activations in attention and FFN layers
    const patchSize = Array.isArray(model.vision_config.patch_size)
      ? model.vision_config.patch_size[0]  // Use first patch size for hierarchical models
      : model.vision_config.patch_size;
    const numPatches = (imageResolution / patchSize) ** 2;
    visionActivations = (
      numImages * batchSize * numPatches *
      model.vision_config.hidden_size *
      model.vision_config.num_layers * 4 / 1e9  // 4 bytes per fp32 activation
    ) * concurrentUsers;

    // Projector/adapter weights (maps vision features to language model space)
    projectorWeights = ((model.multimodal_config.projector_params_millions || 0) * 1e6 * bitsPerParam) / 8 / 1e9;

    // Image preprocessing buffer (RGB images stored as fp32 tensors)
    // 3 channels (RGB) * 4 bytes per float32
    imagePreprocessing = (batchSize * numImages * imageResolution * imageResolution * 3 * 4) / 1e9;

    // Additional KV cache for image tokens in language model
    // Image tokens are processed through language model's attention layers
    const imageTokenCount = model.multimodal_config.image_token_count || 576;
    const imageKVCachePerToken = 2 * model.num_layers * model.hidden_size * kvBitsPerParam / 8;
    imageTokensKV = (imageKVCachePerToken * imageTokenCount * numImages * batchSize * concurrentUsers) / 1e9;
  }

  const totalMultimodalMemory = visionWeights + visionActivations + projectorWeights + imagePreprocessing + imageTokensKV;

  // Framework overhead (5-10% of base memory usage)
  const frameworkOverhead = (baseWeights + totalKVCache + activations + totalMultimodalMemory) * 0.08;

  // Multi-GPU overhead (increases with GPU count due to communication)
  const multiGPUOverhead = numGPUs > 1
    ? (baseWeights + totalKVCache + activations + totalMultimodalMemory + frameworkOverhead) * 0.02 * (numGPUs - 1)
    : 0;

  // Calculate total memory usage
  const usedVRAM = baseWeights + totalKVCache + activations + totalMultimodalMemory + frameworkOverhead + multiGPUOverhead;
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
    // Include multimodal components if applicable
    ...(model.modality === 'multimodal' && {
      visionWeights,
      visionActivations,
      projectorWeights,
      imagePreprocessing,
      imageTokensKV,
    }),
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