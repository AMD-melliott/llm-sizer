// Test for generation model (LLM) calculator
// Run with: node test-generation.js

// Quantization bit sizes
const QUANTIZATION_BITS = {
  'fp16': 16,
  'fp8': 8,
  'int8': 8,
  'int4': 4,
};

const KV_CACHE_BITS = {
  'fp16_bf16': 16,
  'fp8_bf16': 8,
  'int8': 8,
};

function calculateMemoryRequirements(
  model,
  gpu,
  inferenceQuantization,
  kvCacheQuantization,
  batchSize,
  sequenceLength,
  concurrentUsers,
  numGPUs,
  enableOffloading,
  numImages = 1,
  imageResolution = 336
) {
  // Calculate base model weights in GB
  const bitsPerParam = QUANTIZATION_BITS[inferenceQuantization];
  const baseWeights = (model.parameters_billions * 1e9 * bitsPerParam) / 8 / 1e9;

  // Calculate KV cache requirements
  const kvBitsPerParam = KV_CACHE_BITS[kvCacheQuantization];
  const kvCachePerToken = 2 * model.num_layers * model.hidden_size * kvBitsPerParam / 8;
  const totalKVCache = (kvCachePerToken * batchSize * sequenceLength * concurrentUsers) / 1e9;

  // Calculate activation memory
  const activationsPerBatch = (
    batchSize * sequenceLength * model.hidden_size * 4 * 
    model.num_layers * 0.1
  ) / 1e9;
  const activations = activationsPerBatch * concurrentUsers;

  // Multimodal memory components
  let visionWeights = 0;
  let visionActivations = 0;
  let projectorWeights = 0;
  let imagePreprocessing = 0;
  let imageTokensKV = 0;

  if (model.modality === 'multimodal' && model.vision_config && model.multimodal_config) {
    visionWeights = (model.vision_config.parameters_millions * 1e6 * bitsPerParam) / 8 / 1e9;

    const patchSize = Array.isArray(model.vision_config.patch_size)
      ? model.vision_config.patch_size[0]
      : model.vision_config.patch_size;
    const numPatches = (imageResolution / patchSize) ** 2;
    visionActivations = (
      numImages * batchSize * numPatches *
      model.vision_config.hidden_size *
      model.vision_config.num_layers * 4 / 1e9
    ) * concurrentUsers;

    projectorWeights = ((model.multimodal_config.projector_params_millions || 0) * 1e6 * bitsPerParam) / 8 / 1e9;
    imagePreprocessing = (batchSize * numImages * imageResolution * imageResolution * 3 * 4) / 1e9;

    const imageTokenCount = model.multimodal_config.image_token_count || 576;
    const imageKVCachePerToken = 2 * model.num_layers * model.hidden_size * kvBitsPerParam / 8;
    imageTokensKV = (imageKVCachePerToken * imageTokenCount * numImages * batchSize * concurrentUsers) / 1e9;
  }

  const totalMultimodalMemory = visionWeights + visionActivations + projectorWeights + imagePreprocessing + imageTokensKV;

  // Framework overhead
  const frameworkOverhead = (baseWeights + totalKVCache + activations + totalMultimodalMemory) * 0.08;

  // Multi-GPU overhead with tensor parallelism
  // With tensor parallelism, model weights and KV cache are SPLIT across GPUs (not duplicated)
  // There's minimal memory overhead (~2-5% total) for communication buffers and gradients
  // This is a small constant overhead, not per-GPU multiplicative
  const multiGPUOverhead = numGPUs > 1
    ? (baseWeights + totalKVCache + activations + totalMultimodalMemory + frameworkOverhead) * 0.03
    : 0;

  const usedVRAM = baseWeights + totalKVCache + activations + totalMultimodalMemory + frameworkOverhead + multiGPUOverhead;
  const totalVRAM = gpu.vram_gb * numGPUs;
  const vramPercentage = (usedVRAM / totalVRAM) * 100;

  return {
    totalVRAM,
    usedVRAM,
    vramPercentage,
    breakdown: {
      baseWeights,
      totalKVCache,
      activations,
      frameworkOverhead,
      multiGPUOverhead,
      visionWeights,
      visionActivations,
      projectorWeights,
      imagePreprocessing,
      imageTokensKV,
    },
  };
}

// Test cases
const llama370b = {
  id: "llama-3-70b",
  name: "Meta Llama 3 70B",
  parameters_billions: 70,
  hidden_size: 8192,
  num_layers: 80,
  num_heads: 64,
  modality: "text"
};

const llama38b = {
  id: "llama-3-8b",
  name: "Meta Llama 3 8B",
  parameters_billions: 8,
  hidden_size: 4096,
  num_layers: 32,
  num_heads: 32,
  modality: "text"
};

const gpu = {
  id: "mi300x",
  name: "AMD Instinct MI300X",
  vram_gb: 192,
};

console.log("=".repeat(80));
console.log("GENERATION MODEL (LLM) MEMORY CALCULATION TEST");
console.log("=".repeat(80));
console.log();

// Test 1: Llama 3 70B - Single user, FP16
console.log("Test 1: Llama 3 70B - Single User, FP16");
console.log("-".repeat(80));
console.log(`Model: ${llama370b.name} (${llama370b.parameters_billions}B parameters)`);
console.log(`GPU: ${gpu.name} (${gpu.vram_gb}GB VRAM)`);
let result = calculateMemoryRequirements(
  llama370b, gpu, 'fp16', 'fp16_bf16',
  1, // batchSize
  4096, // sequenceLength
  1, // concurrentUsers
  1, // numGPUs
  false // enableOffloading
);
console.log(`Config: Batch=1, SeqLen=4096, Users=1, Quantization=FP16`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  KV Cache: ${result.breakdown.totalKVCache.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 2: Llama 3 70B - INT4 quantization
console.log("Test 2: Llama 3 70B - INT4 Quantization");
console.log("-".repeat(80));
console.log(`Model: ${llama370b.name} (${llama370b.parameters_billions}B parameters)`);
console.log(`GPU: ${gpu.name} (${gpu.vram_gb}GB VRAM)`);
result = calculateMemoryRequirements(
  llama370b, gpu, 'int4', 'int8',
  1, // batchSize
  4096, // sequenceLength
  1, // concurrentUsers
  1, // numGPUs
  false // enableOffloading
);
console.log(`Config: Batch=1, SeqLen=4096, Users=1, Quantization=INT4, KV Cache=INT8`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  KV Cache: ${result.breakdown.totalKVCache.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 3: Llama 3 70B - Multi-user scenario
console.log("Test 3: Llama 3 70B - Multi-User Scenario");
console.log("-".repeat(80));
console.log(`Model: ${llama370b.name} (${llama370b.parameters_billions}B parameters)`);
console.log(`GPU: ${gpu.name} (${gpu.vram_gb}GB VRAM)`);
result = calculateMemoryRequirements(
  llama370b, gpu, 'fp16', 'fp8_bf16',
  8, // batchSize
  2048, // sequenceLength
  4, // concurrentUsers
  1, // numGPUs
  false // enableOffloading
);
console.log(`Config: Batch=8, SeqLen=2048, Users=4, Quantization=FP16, KV=FP8`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  KV Cache: ${result.breakdown.totalKVCache.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 4: Llama 3 8B - High throughput
console.log("Test 4: Llama 3 8B - High Throughput");
console.log("-".repeat(80));
console.log(`Model: ${llama38b.name} (${llama38b.parameters_billions}B parameters)`);
console.log(`GPU: ${gpu.name} (${gpu.vram_gb}GB VRAM)`);
result = calculateMemoryRequirements(
  llama38b, gpu, 'fp16', 'fp16_bf16',
  32, // batchSize
  2048, // sequenceLength
  1, // concurrentUsers
  1, // numGPUs
  false // enableOffloading
);
console.log(`Config: Batch=32, SeqLen=2048, Users=1, Quantization=FP16`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  KV Cache: ${result.breakdown.totalKVCache.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 5: Multi-GPU deployment
console.log("Test 5: Llama 3 70B - Multi-GPU Deployment (4x MI300X)");
console.log("-".repeat(80));
console.log(`Model: ${llama370b.name} (${llama370b.parameters_billions}B parameters)`);
console.log(`GPU: 4x ${gpu.name} (${gpu.vram_gb * 4}GB total VRAM)`);
result = calculateMemoryRequirements(
  llama370b, gpu, 'fp16', 'fp16_bf16',
  16, // batchSize
  4096, // sequenceLength
  8, // concurrentUsers
  4, // numGPUs
  false // enableOffloading
);
console.log(`Config: Batch=16, SeqLen=4096, Users=8, Quantization=FP16, 4 GPUs`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  KV Cache: ${result.breakdown.totalKVCache.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`  Multi-GPU: ${result.breakdown.multiGPUOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`Total VRAM: ${result.totalVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();
