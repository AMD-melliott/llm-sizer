import {
  EmbeddingModel,
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

export function calculateEmbeddingMemory(
  model: EmbeddingModel,
  gpu: GPU,
  quantization: InferenceQuantization,
  batchSize: number,
  documentsPerBatch: number,
  avgDocumentSize: number,
  numGPUs: number = 1
): CalculationResults {
  // Calculate model weights in GB
  const bitsPerParam = QUANTIZATION_BITS[quantization];
  const baseWeights = (model.parameters_millions * 1e6 * bitsPerParam) / 8 / 1e9;

  // Calculate batch input memory
  // Total tokens = batchSize * avgDocumentSize
  // Each token requires hidden_size dimensions stored as fp32 during processing
  const totalTokens = batchSize * avgDocumentSize;
  const batchInputMemory = (totalTokens * model.hidden_size * 4) / 1e9; // 4 bytes for FP32

  // Calculate attention memory for transformer-based embedders
  // Self-attention matrices: (seq_len x seq_len) per head, per layer
  // Using max_tokens as the sequence length limit
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
  // Rough estimate: ~2x hidden size per layer per token
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

  // Calculate throughput estimates
  // Base estimate: tokens processed per second
  // This is a simplified estimation based on GPU compute capability
  const computeIntensity = model.parameters_millions / 100; // Relative compute factor
  const baseTokensPerSecond = (gpu.compute_tflops_fp16 * 1e12) / 
                               (model.hidden_size * model.num_layers * 4 * computeIntensity);
  
  const tokensPerSecond = Math.min(baseTokensPerSecond, totalTokens * 10); // Conservative estimate
  const documentsPerSecond = tokensPerSecond / avgDocumentSize;
  const embeddingsPerSecond = documentsPerSecond * documentsPerBatch;

  const memoryBreakdown: MemoryBreakdown = {
    baseWeights,
    activations,
    kvCache: 0, // Embeddings don't use KV cache
    frameworkOverhead,
    multiGPUOverhead,
  };

  // Determine status
  let status: 'okay' | 'warning' | 'error' = 'okay';
  let message: string | undefined;

  if (vramPercentage > 100) {
    status = 'error';
    message = `Memory requirements (${usedVRAM.toFixed(2)}GB) exceed GPU capacity (${totalVRAM.toFixed(2)}GB)`;
  } else if (vramPercentage > 90) {
    status = 'warning';
    message = `Memory usage is very high (${vramPercentage.toFixed(1)}%). Consider reducing batch size.`;
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
      generationSpeed: 0, // Not applicable for embeddings
      totalThroughput: tokensPerSecond,
      perUserSpeed: 0, // Not applicable for embeddings
      documentsPerSecond,
      tokensPerSecond,
      embeddingsPerSecond,
    },
    status,
    message,
  };
}

// Helper function to estimate optimal batch size
export function estimateOptimalEmbeddingBatchSize(
  model: EmbeddingModel,
  gpu: GPU,
  quantization: InferenceQuantization,
  avgDocumentSize: number,
  targetUtilization: number = 0.85 // Target 85% GPU memory utilization
): number {
  let batchSize = 1;
  let maxBatchSize = 1;

  // Binary search for optimal batch size
  for (let testBatch = 2; testBatch <= 2048; testBatch *= 2) {
    const result = calculateEmbeddingMemory(
      model,
      gpu,
      quantization,
      testBatch,
      1,
      avgDocumentSize
    );

    if (result.vramPercentage <= targetUtilization * 100) {
      maxBatchSize = testBatch;
    } else {
      break;
    }
  }

  // Fine-tune around the maximum
  for (let testBatch = maxBatchSize; testBatch <= maxBatchSize * 2; testBatch += 16) {
    const result = calculateEmbeddingMemory(
      model,
      gpu,
      quantization,
      testBatch,
      1,
      avgDocumentSize
    );

    if (result.vramPercentage <= targetUtilization * 100) {
      batchSize = testBatch;
    } else {
      break;
    }
  }

  return Math.max(1, batchSize);
}
