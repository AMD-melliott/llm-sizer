// Test for embedding calculator
// Run with: node test-embedding.js

// Simulate the calculation
const QUANTIZATION_BITS = {
  'fp16': 16,
  'fp8': 8,
  'int8': 8,
  'int4': 4,
};

function calculateEmbeddingMemory(
  model,
  gpu,
  quantization,
  batchSize,
  documentsPerBatch,
  avgDocumentSize,
  numGPUs = 1
) {
  // Calculate model weights in GB
  const bitsPerParam = QUANTIZATION_BITS[quantization];
  const baseWeights = (model.parameters_millions * 1e6 * bitsPerParam) / 8 / 1e9;

  // Calculate batch input memory
  const totalTokens = batchSize * avgDocumentSize;
  const batchInputMemory = (totalTokens * model.hidden_size * 4) / 1e9; // 4 bytes for FP32

  // Calculate attention memory for transformer-based embedders
  const seqLength = Math.min(avgDocumentSize, model.max_tokens);
  const attentionMemory = (
    batchSize * 
    seqLength * seqLength * 
    model.num_heads * 
    model.num_layers * 
    4 // FP32 attention scores
  ) / 1e9;

  // Output embeddings storage (batch_size x embedding_dimension)
  const embeddingStorage = (batchSize * model.dimensions * 4) / 1e9; // FP32 embeddings

  // Intermediate activations (FFN layers, layer norms, etc.)
  const activations = (
    batchSize * 
    seqLength * 
    model.hidden_size * 
    model.num_layers * 
    2 * 
    4 // FP32
  ) / 1e9;

  // Framework overhead (10% of model weights + activations)
  const frameworkOverhead = (baseWeights + activations) * 0.1;

  // Multi-GPU overhead (2% per additional GPU for communication)
  const multiGPUOverhead = numGPUs > 1
    ? (baseWeights + attentionMemory + activations + frameworkOverhead) * 0.02 * (numGPUs - 1)
    : 0;

  const usedVRAM = baseWeights + batchInputMemory + attentionMemory + 
                   embeddingStorage + activations + frameworkOverhead + multiGPUOverhead;
  const totalVRAM = gpu.vram_gb * numGPUs;
  const vramPercentage = (usedVRAM / totalVRAM) * 100;

  return {
    totalVRAM,
    usedVRAM,
    vramPercentage,
    breakdown: {
      baseWeights,
      batchInputMemory,
      attentionMemory,
      embeddingStorage,
      activations,
      frameworkOverhead,
      multiGPUOverhead,
    },
  };
}

// Test cases
const model = {
  id: "bge-large-en-v1.5",
  name: "BAAI/bge-large-en-v1.5",
  parameters_millions: 335,
  dimensions: 1024,
  max_tokens: 512,
  hidden_size: 1024,
  num_layers: 24,
  num_heads: 16
};

const gpu = {
  id: "mi300x",
  name: "AMD Instinct MI300X",
  vram_gb: 192,
  compute_tflops_fp16: 1300,
};

console.log("=".repeat(80));
console.log("EMBEDDING MEMORY CALCULATION TEST");
console.log("=".repeat(80));
console.log(`Model: ${model.name} (${model.parameters_millions}M parameters)`);
console.log(`GPU: ${gpu.name} (${gpu.vram_gb}GB VRAM)`);
console.log();

// Test 1: Small batch
console.log("Test 1: Small Batch Processing");
console.log("-".repeat(80));
let result = calculateEmbeddingMemory(
  model, gpu, 'fp16',
  32, // batchSize
  64, // documentsPerBatch
  512, // avgDocumentSize
  1 // numGPUs
);
console.log(`Batch Size: 32, Docs/Batch: 64, Avg Doc Size: 512 tokens`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Batch Input: ${result.breakdown.batchInputMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Embedding Storage: ${result.breakdown.embeddingStorage.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 2: Default configuration
console.log("Test 2: Default Configuration");
console.log("-".repeat(80));
result = calculateEmbeddingMemory(
  model, gpu, 'fp16',
  256, // batchSize
  64, // documentsPerBatch
  512, // avgDocumentSize
  1 // numGPUs
);
console.log(`Batch Size: 256, Docs/Batch: 64, Avg Doc Size: 512 tokens`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Batch Input: ${result.breakdown.batchInputMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Embedding Storage: ${result.breakdown.embeddingStorage.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 3: Large documents
console.log("Test 3: Large Documents");
console.log("-".repeat(80));
result = calculateEmbeddingMemory(
  model, gpu, 'fp16',
  128, // batchSize
  32, // documentsPerBatch
  1024, // avgDocumentSize (exceeds max_tokens)
  1 // numGPUs
);
console.log(`Batch Size: 128, Docs/Batch: 32, Avg Doc Size: 1024 tokens (capped at ${model.max_tokens})`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Batch Input: ${result.breakdown.batchInputMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Embedding Storage: ${result.breakdown.embeddingStorage.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 4: Quantized (int8)
console.log("Test 4: Quantized Model (INT8)");
console.log("-".repeat(80));
result = calculateEmbeddingMemory(
  model, gpu, 'int8',
  256, // batchSize
  64, // documentsPerBatch
  512, // avgDocumentSize
  1 // numGPUs
);
console.log(`Batch Size: 256, Docs/Batch: 64, Avg Doc Size: 512 tokens, INT8 quantization`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Batch Input: ${result.breakdown.batchInputMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Embedding Storage: ${result.breakdown.embeddingStorage.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 5: Multi-GPU setup
console.log("Test 5: Multi-GPU Setup (4x MI300X)");
console.log("-".repeat(80));
result = calculateEmbeddingMemory(
  model, gpu, 'fp16',
  1024, // batchSize (large batch across GPUs)
  128, // documentsPerBatch
  512, // avgDocumentSize
  4 // numGPUs
);
console.log(`Batch Size: 1024, Docs/Batch: 128, Avg Doc Size: 512 tokens, 4 GPUs`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Batch Input: ${result.breakdown.batchInputMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Embedding Storage: ${result.breakdown.embeddingStorage.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`  Multi-GPU: ${result.breakdown.multiGPUOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`Total VRAM: ${result.totalVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();