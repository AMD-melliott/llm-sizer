// Quick test to verify reranking calculations
// Run with: node test-reranking.js

// Simulate the calculation
const QUANTIZATION_BITS = {
  'fp16': 16,
  'fp8': 8,
  'int8': 8,
  'int4': 4,
};

function calculateRerankingMemory(
  model,
  gpu,
  quantization,
  numQueries,
  docsPerQuery,
  maxQueryLength,
  maxDocLength,
  numGPUs = 1,
  batchSize = 32
) {
  const bitsPerParam = QUANTIZATION_BITS[quantization];
  const baseWeights = (model.parameters_millions * 1e6 * bitsPerParam) / 8 / 1e9;

  const pairLength = Math.min(
    maxQueryLength + maxDocLength,
    Math.min(model.max_query_length, model.max_doc_length)
  );
  
  const totalPairs = numQueries * docsPerQuery;
  const effectiveBatchSize = Math.min(batchSize, totalPairs);

  const pairBatchMemory = (
    effectiveBatchSize * 
    pairLength * 
    model.hidden_size * 
    4
  ) / 1e9;

  // Only one layer's attention is stored at a time
  const attentionMemory = (
    effectiveBatchSize * 
    pairLength * pairLength * 
    model.num_heads * 
    4
  ) / 1e9;

  const scoringMemory = (effectiveBatchSize * 4) / 1e9;

  // Activations: attn output, FFN up, FFN down, residual (4x multiplier)
  const activations = (
    effectiveBatchSize * 
    pairLength * 
    model.hidden_size * 
    4 * 
    4
  ) / 1e9;

  const frameworkOverhead = (baseWeights + activations) * 0.1;
  const multiGPUOverhead = numGPUs > 1
    ? (baseWeights + attentionMemory + activations + frameworkOverhead) * 0.02 * (numGPUs - 1)
    : 0;

  const usedVRAM = baseWeights + pairBatchMemory + attentionMemory + 
                   scoringMemory + activations + frameworkOverhead + multiGPUOverhead;
  const totalVRAM = gpu.vram_gb * numGPUs;
  const vramPercentage = (usedVRAM / totalVRAM) * 100;

  return {
    totalVRAM,
    usedVRAM,
    vramPercentage,
    breakdown: {
      baseWeights,
      pairBatchMemory,
      attentionMemory,
      scoringMemory,
      activations,
      frameworkOverhead,
      multiGPUOverhead,
    },
    effectiveBatchSize,
    totalPairs,
  };
}

// Test cases
const model = {
  id: "bge-reranker-large",
  name: "BAAI/bge-reranker-large",
  parameters_millions: 560,
  max_query_length: 512,
  max_doc_length: 512,
  hidden_size: 1024,
  num_layers: 24,
  num_heads: 16
};

const gpu = {
  id: "h100-sxm",
  name: "H100 SXM",
  vram_gb: 80,
};

console.log("=" .repeat(80));
console.log("RERANKING MEMORY CALCULATION TEST");
console.log("=" .repeat(80));
console.log(`Model: ${model.name} (${model.parameters_millions}M parameters)`);
console.log(`GPU: ${gpu.name} (${gpu.vram_gb}GB VRAM)`);
console.log();

// Test 1: Original problematic case
console.log("Test 1: Original Case (Before Fix - would have used totalPairs)");
console.log("-" .repeat(80));
let result = calculateRerankingMemory(
  model, gpu, 'fp16',
  10, // numQueries
  100, // docsPerQuery  
  512, // maxQueryLength
  512, // maxDocLength
  1, // numGPUs
  32 // batchSize
);
console.log(`Queries: 10, Docs/Query: 100, Batch Size: 32`);
console.log(`Total Pairs: ${result.totalPairs}, Effective Batch: ${result.effectiveBatchSize}`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Pair Batch: ${result.breakdown.pairBatchMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 2: New defaults
console.log("Test 2: New Defaults");
console.log("-" .repeat(80));
result = calculateRerankingMemory(
  model, gpu, 'fp16',
  1, // numQueries
  50, // docsPerQuery  
  256, // maxQueryLength
  512, // maxDocLength
  1, // numGPUs
  32 // batchSize
);
console.log(`Queries: 1, Docs/Query: 50, Batch Size: 32`);
console.log(`Total Pairs: ${result.totalPairs}, Effective Batch: ${result.effectiveBatchSize}`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Pair Batch: ${result.breakdown.pairBatchMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 3: Small batch (typical use case)
console.log("Test 3: Typical RAG Use Case");
console.log("-" .repeat(80));
result = calculateRerankingMemory(
  model, gpu, 'fp16',
  1, // numQueries
  20, // docsPerQuery  
  128, // maxQueryLength
  512, // maxDocLength
  1, // numGPUs
  20 // batchSize (process all at once)
);
console.log(`Queries: 1, Docs/Query: 20, Batch Size: 20`);
console.log(`Total Pairs: ${result.totalPairs}, Effective Batch: ${result.effectiveBatchSize}`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Pair Batch: ${result.breakdown.pairBatchMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 4: Large batch size
console.log("Test 4: Large Batch Processing");
console.log("-" .repeat(80));
result = calculateRerankingMemory(
  model, gpu, 'fp16',
  5, // numQueries
  100, // docsPerQuery  
  256, // maxQueryLength
  512, // maxDocLength
  1, // numGPUs
  128 // batchSize
);
console.log(`Queries: 5, Docs/Query: 100, Batch Size: 128`);
console.log(`Total Pairs: ${result.totalPairs}, Effective Batch: ${result.effectiveBatchSize}`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Pair Batch: ${result.breakdown.pairBatchMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();

// Test 5: With 8x H100 
console.log("Test 5: Multi-GPU Setup (8x H100)");
console.log("-" .repeat(80));
result = calculateRerankingMemory(
  model, gpu, 'fp16',
  100, // numQueries
  100, // docsPerQuery  
  256, // maxQueryLength
  512, // maxDocLength
  8, // numGPUs
  256 // batchSize
);
console.log(`Queries: 100, Docs/Query: 100, Batch Size: 256, GPUs: 8`);
console.log(`Total Pairs: ${result.totalPairs}, Effective Batch: ${result.effectiveBatchSize}`);
console.log(`\nMemory Breakdown:`);
console.log(`  Model Weights: ${result.breakdown.baseWeights.toFixed(2)} GB`);
console.log(`  Pair Batch: ${result.breakdown.pairBatchMemory.toFixed(2)} GB`);
console.log(`  Attention: ${result.breakdown.attentionMemory.toFixed(2)} GB`);
console.log(`  Activations: ${result.breakdown.activations.toFixed(2)} GB`);
console.log(`  Framework: ${result.breakdown.frameworkOverhead.toFixed(2)} GB`);
console.log(`  Multi-GPU: ${result.breakdown.multiGPUOverhead.toFixed(2)} GB`);
console.log(`Total Used: ${result.usedVRAM.toFixed(2)} GB`);
console.log(`Total VRAM: ${result.totalVRAM.toFixed(2)} GB`);
console.log(`VRAM %: ${result.vramPercentage.toFixed(1)}%`);
console.log();
