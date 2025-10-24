/**
 * Reranking Memory Calculator
 * 
 * Calculates memory requirements for cross-encoder reranking models.
 * These models process query-document pairs by concatenating them and
 * running them through a transformer to produce relevance scores.
 * 
 * Key Points:
 * - Memory scales with BATCH SIZE, not total query-doc pairs
 * - Pairs are processed in batches (e.g., 32 pairs at a time)
 * - Attention memory is for ONE layer at a time (not all layers simultaneously)
 * - Total throughput depends on: total_pairs / (batches_needed * time_per_batch)
 * 
 * Typical Use Cases:
 * - RAG reranking: 1 query × 20-100 docs, batch_size 20-50
 * - Batch processing: 5-10 queries × 50-100 docs, batch_size 32-128
 * - Real-time: 1 query × 10-20 docs, batch_size = num_docs
 */

import {
  RerankingModel,
  GPU,
  InferenceQuantization,
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

export function calculateRerankingMemory(
  model: RerankingModel,
  gpu: GPU,
  quantization: InferenceQuantization,
  numQueries: number,
  docsPerQuery: number,
  maxQueryLength: number,
  maxDocLength: number,
  numGPUs: number = 1,
  batchSize: number = 32
): CalculationResults {
  // Calculate model weights in GB
  const bitsPerParam = QUANTIZATION_BITS[quantization];
  const baseWeights = (model.parameters_millions * 1e6 * bitsPerParam) / 8 / 1e9;

  // Calculate query-document pair processing memory
  // Cross-encoders concatenate query and document, then process together
  const pairLength = Math.min(
    maxQueryLength + maxDocLength,
    Math.min(model.max_query_length, model.max_doc_length)
  );
  
  // Total pairs that will be processed (for throughput calculation)
  const totalPairs = numQueries * docsPerQuery;

  // IMPORTANT: Memory is based on batch size, not total pairs
  // The batch size represents how many pairs are processed simultaneously
  const effectiveBatchSize = Math.min(batchSize, totalPairs);

  // Batch processing memory: each pair in the batch requires hidden_size dimensions
  const pairBatchMemory = (
    effectiveBatchSize * 
    pairLength * 
    model.hidden_size * 
    4 // FP32 intermediate representations
  ) / 1e9;

  // Attention memory for cross-encoder
  // Self-attention over concatenated sequence (per batch)
  // Only one layer's attention is stored at a time during forward pass
  const attentionMemory = (
    effectiveBatchSize * 
    pairLength * pairLength * 
    model.num_heads * 
    4 // FP32 attention scores
  ) / 1e9;

  // Scoring matrix: stores relevance scores for each query-doc pair in batch
  // Typically FP32 scores
  const scoringMemory = (effectiveBatchSize * 4) / 1e9;

  // Intermediate activations (FFN layers, pooling, classification heads)
  // Only for the current batch being processed
  // Typical activations: layer input/output, attention output, FFN intermediate
  const activations = (
    effectiveBatchSize * 
    pairLength * 
    model.hidden_size * 
    4 * // 4x multiplier: attn output, FFN up, FFN down, residual
    4 // FP32 bytes per value
  ) / 1e9;

  // Framework overhead (10% of model weights + activations)
  const frameworkOverhead = (baseWeights + activations) * 0.1;

  // Multi-GPU overhead (2% per additional GPU for communication)
  const multiGPUOverhead = numGPUs > 1
    ? (baseWeights + attentionMemory + activations + frameworkOverhead) * 0.02 * (numGPUs - 1)
    : 0;

  const usedVRAM = baseWeights + pairBatchMemory + attentionMemory + 
                   scoringMemory + activations + frameworkOverhead + multiGPUOverhead;
  const totalVRAM = gpu.vram_gb * numGPUs;
  const vramPercentage = (usedVRAM / totalVRAM) * 100;

  // Calculate throughput estimates
  // Cross-encoders are more computationally intensive than bi-encoders
  const computeIntensity = model.parameters_millions / 50; // Higher intensity factor
  const basePairsPerSecond = (gpu.compute_tflops_fp16 * 1e12) / 
                              (model.hidden_size * model.num_layers * pairLength * computeIntensity);
  
  // Throughput is limited by batch processing
  const batchesNeeded = Math.ceil(totalPairs / effectiveBatchSize);
  const timePerBatch = effectiveBatchSize / basePairsPerSecond; // seconds
  const totalTimeSeconds = batchesNeeded * timePerBatch;
  
  const queryDocPairsPerSecond = totalPairs / totalTimeSeconds;
  const queriesPerSecond = queryDocPairsPerSecond / docsPerQuery;
  const avgLatencyMs = docsPerQuery > 0 ? (1000 / queriesPerSecond) : 0;

  const memoryBreakdown: MemoryBreakdown = {
    baseWeights,
    activations,
    kvCache: 0, // Reranking doesn't use KV cache
    frameworkOverhead,
    multiGPUOverhead,
    pairBatchMemory,
    attentionMemory,
    scoringMemory,
  };

  // Determine status
  let status: 'okay' | 'warning' | 'error' = 'okay';
  let message: string | undefined;

  if (vramPercentage > 100) {
    status = 'error';
    message = `Memory requirements (${usedVRAM.toFixed(2)}GB) exceed GPU capacity (${totalVRAM.toFixed(2)}GB)`;
  } else if (vramPercentage > 90) {
    status = 'warning';
    message = `Memory usage is very high (${vramPercentage.toFixed(1)}%). Consider reducing batch size or docs per query.`;
  } else if (vramPercentage > 80) {
    status = 'warning';
    message = `Memory usage is high (${vramPercentage.toFixed(1)}%). Performance may be suboptimal.`;
  }

  return {
    totalVRAM,
    usedVRAM,
    vramPercentage,
    memoryBreakdown,
    performance: {
      generationSpeed: 0, // Not applicable for reranking
      totalThroughput: queryDocPairsPerSecond,
      perUserSpeed: 0, // Not applicable for reranking
      queryDocPairsPerSecond,
      queriesPerSecond,
      avgLatencyMs,
    },
    status,
    message,
    totalPairs,
    effectiveBatchSize,
  };
}

// Helper function to estimate optimal batch configuration
export function estimateOptimalRerankingBatch(
  model: RerankingModel,
  gpu: GPU,
  quantization: InferenceQuantization,
  docsPerQuery: number,
  maxQueryLength: number,
  maxDocLength: number,
  targetUtilization: number = 0.85
): number {
  let maxBatchSize = 1;

  // Binary search for optimal batch size
  for (let testBatch = 2; testBatch <= 512; testBatch *= 2) {
    const result = calculateRerankingMemory(
      model,
      gpu,
      quantization,
      1, // Single query for testing
      docsPerQuery,
      maxQueryLength,
      maxDocLength,
      1, // Single GPU
      testBatch
    );

    if (result.vramPercentage <= targetUtilization * 100) {
      maxBatchSize = testBatch;
    } else {
      break;
    }
  }

  // Fine-tune
  for (let testBatch = maxBatchSize; testBatch <= maxBatchSize * 2; testBatch += 8) {
    const result = calculateRerankingMemory(
      model,
      gpu,
      quantization,
      1,
      docsPerQuery,
      maxQueryLength,
      maxDocLength,
      1,
      testBatch
    );

    if (result.vramPercentage <= targetUtilization * 100) {
      maxBatchSize = testBatch;
    } else {
      break;
    }
  }

  return Math.max(1, maxBatchSize);
}
