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
  numImages: number = 1,
  imageResolution: number = 336
): CalculationResults {
  // Calculate base model weights in GB
  const bitsPerParam = QUANTIZATION_BITS[inferenceQuantization];
  const baseWeights = (model.parameters_billions * 1e9 * bitsPerParam) / 8 / 1e9;

  // Calculate KV cache requirements
  // KV cache per token = 2 (K+V) * num_layers * num_kv_heads * head_size * kv_bytes
  // For GQA models, num_kv_heads < num_heads, reducing KV cache proportionally.
  // For MHA models (num_kv_heads == num_heads), this equals the hidden_size formula.
  // Matches vLLM's formula: 2 * block_size * num_kv_heads * head_size * dtype_size
  const kvBitsPerParam = KV_CACHE_BITS[kvCacheQuantization];
  const numKVHeads = model.num_kv_heads ?? model.num_heads;
  const headSize = model.hidden_size / model.num_heads;
  const kvBytesPerElement = kvBitsPerParam / 8;
  const kvCachePerToken = 2 * model.num_layers * numKVHeads * headSize * kvBytesPerElement;

  // Total KV cache for all users and sequences
  const totalKVCache = (kvCachePerToken * batchSize * sequenceLength * concurrentUsers) / 1e9;

  // Activation memory (peak, ~one layer's intermediates)
  // Layers are processed sequentially so activations are reused between layers.
  // With FlashAttention (used by vLLM), attention scores aren't fully materialized.
  // The dominant term is the FFN intermediate buffer (intermediate_size, typically 4x hidden_size).
  // Factor of 2 accounts for input + intermediate tensors coexisting during FFN computation.
  // Distributed across GPUs in tensor parallelism.
  const intermediateSize = model.intermediate_size ?? model.hidden_size * 4;
  const activations = (batchSize * sequenceLength * intermediateSize * 2) / numGPUs / 1e9;

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
    // Uses same GQA-aware formula as text KV cache
    const imageTokenCount = model.multimodal_config.image_token_count || 576;
    const imageKVCachePerToken = 2 * model.num_layers * numKVHeads * headSize * kvBytesPerElement;
    imageTokensKV = (imageKVCachePerToken * imageTokenCount * numImages * batchSize * concurrentUsers) / 1e9;
  }

  const totalMultimodalMemory = visionWeights + visionActivations + projectorWeights + imagePreprocessing + imageTokensKV;

  // Framework overhead (CUDA context, PyTorch runtime, inference engine, memory pools)
  // Uses a baseline + proportional model instead of a flat percentage:
  // - Fixed: ~1.5 GB for CUDA context + PyTorch runtime + engine initialization
  // - Proportional: ~5% of model memory for internal buffers and compilation artifacts
  // - Multi-GPU: ~0.5 GB per additional GPU for NCCL communication buffers
  // This matches vLLM's profiled non_torch_memory behavior where small models see
  // high relative overhead (fixed costs dominate) while large models see lower relative overhead.
  const fixedOverhead = 1.5;
  const proportionalOverhead = (baseWeights + totalKVCache + activations) * 0.05;
  const ncclOverhead = numGPUs > 1 ? 0.5 * (numGPUs - 1) : 0;
  const frameworkOverhead = fixedOverhead + proportionalOverhead + ncclOverhead;

  // Multi-GPU overhead (tensor parallelism communication buffers)
  // NCCL fixed costs are already included in frameworkOverhead above.
  // This remaining term covers all-reduce buffers and activation synchronization,
  // which scale with model weights (not KV cache or activations).
  const multiGPUOverhead = numGPUs > 1 ? baseWeights * 0.01 * (numGPUs - 1) : 0;

  // Calculate total memory usage
  const usedVRAM = baseWeights + totalKVCache + activations + totalMultimodalMemory + frameworkOverhead + multiGPUOverhead;
  const totalVRAM = gpu.vram_gb * numGPUs;
  const vramPercentage = (usedVRAM / totalVRAM) * 100;

  // Determine status based on usage
  let status: 'okay' | 'warning' | 'error' = 'okay';
  let message = '';

  if (vramPercentage > 100) {
    status = 'error';
    const gpusNeeded = Math.ceil(usedVRAM / gpu.vram_gb);
    message = `Memory requirement exceeds available VRAM by ${(usedVRAM - totalVRAM).toFixed(1)} GB. Recommended options: ${gpusNeeded}x ${gpu.name}, reduce batch size, or use more aggressive quantization.`;
  } else if (vramPercentage > 90) {
    status = 'warning';
    message = 'Very high VRAM usage (>90%). Performance may degrade due to limited memory for caching and buffers.';
  } else if (vramPercentage > 80) {
    status = 'warning';
    message = 'High VRAM usage (>80%). Consider leaving more headroom for optimal performance.';
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