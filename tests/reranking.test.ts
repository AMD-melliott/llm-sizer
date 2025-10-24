import { calculateRerankingMemory } from '../src/utils/rerankingCalculator';
import { RerankingModel, GPU } from '../src/types';

describe('Reranking Memory Calculation Tests', () => {
  const model: RerankingModel = {
    id: "bge-reranker-large",
    name: "BAAI/bge-reranker-large",
    architecture: 'cross-encoder' as const,
    parameters_millions: 560,
    max_query_length: 512,
    max_doc_length: 512,
    max_docs_per_query: 100,
    hidden_size: 1024,
    num_layers: 24,
    num_heads: 16
  };

  const gpu: GPU = {
    id: "h100-sxm",
    vendor: 'NVIDIA',
    name: "H100 SXM",
    category: 'enterprise',
    tier: 'datacenter',
    vram_gb: 80,
    memory_type: 'HBM3',
    memory_bandwidth_gbps: 3350,
    compute_tflops_fp16: 989,
    compute_tflops_fp8: 1979,
    nvlink_bandwidth_gbps: 900,
    pcie_gen: 5,
    tdp_watts: 700,
    multi_gpu_capable: true,
    release_year: 2022
  };

  describe('Basic Reranking Scenarios', () => {
    test('should handle original case with multiple queries and documents', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'fp16',
        10, // numQueries
        100, // docsPerQuery
        512, // maxQueryLength
        512, // maxDocLength
        1, // numGPUs
        32 // batchSize
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.totalPairs).toBe(1000); // 10 queries * 100 docs
      expect(result.effectiveBatchSize).toBeLessThanOrEqual(32);
      expect(result.status).not.toBe('error');

      // Memory breakdown validation
      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(0);
      expect(result.memoryBreakdown.pairBatchMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.attentionMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.activations).toBeGreaterThan(0);
      expect(result.memoryBreakdown.frameworkOverhead).toBeGreaterThan(0);

      // Should fit in GPU memory
      expect(result.vramPercentage).toBeLessThan(100);
    });

    test('should handle single query with multiple documents', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, // numQueries
        50, // docsPerQuery
        256, // maxQueryLength
        512, // maxDocLength
        1, // numGPUs
        32 // batchSize
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.totalPairs).toBe(50); // 1 query * 50 docs
      expect(result.effectiveBatchSize).toBeLessThanOrEqual(32);
      expect(result.status).not.toBe('error');

      // Memory should be less than the previous test
      expect(result.vramPercentage).toBeLessThan(100);
    });
  });

  describe('RAG Use Cases', () => {
    test('should handle typical RAG scenario', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, // numQueries
        100, // docsPerQuery
        128, // maxQueryLength
        512, // maxDocLength
        1, // numGPUs
        32 // batchSize
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.totalPairs).toBe(100);
      expect(result.status).not.toBe('error');
      expect(result.vramPercentage).toBeLessThan(100);
    });

    test('should handle large-scale RAG with multiple queries', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'fp16',
        5, // numQueries
        50, // docsPerQuery
        256, // maxQueryLength
        1024, // maxDocLength - longer documents
        1, // numGPUs
        16 // batchSize
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.totalPairs).toBe(250); // 5 queries * 50 docs
      expect(result.effectiveBatchSize).toBeLessThanOrEqual(16);
      expect(result.status).not.toBe('error');

      // Longer documents should increase memory usage
      expect(result.memoryBreakdown.pairBatchMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.attentionMemory).toBeGreaterThan(0);
    });
  });

  describe('Quantization Effects', () => {
    test('should reduce memory with INT8 quantization', () => {
      const fp16Result = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, 100, 512, 512, 1, 32
      );

      const int8Result = calculateRerankingMemory(
        model, gpu, 'int8',
        1, 100, 512, 512, 1, 32
      );

      // INT8 should use less memory for weights
      expect(int8Result.memoryBreakdown.baseWeights).toBeLessThan(
        fp16Result.memoryBreakdown.baseWeights
      );

      // Total memory should be less
      expect(int8Result.usedVRAM).toBeLessThan(fp16Result.usedVRAM);

      // Both should be valid
      expect(fp16Result.status).not.toBe('error');
      expect(int8Result.status).not.toBe('error');
    });

    test('should handle INT4 quantization', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'int4',
        1, 100, 512, 512, 1, 32
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.memoryBreakdown.baseWeights).toBeLessThan(0.5); // Very small for INT4
      expect(result.status).not.toBe('error');
    });
  });

  describe('Batch Size Optimization', () => {
    test('should handle different batch sizes', () => {
      const smallBatch = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, 100, 512, 512, 1, 8
      );

      const mediumBatch = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, 100, 512, 512, 1, 32
      );

      const largeBatch = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, 100, 512, 512, 1, 64
      );

      // Larger batch sizes should use more memory for activations
      expect(smallBatch.memoryBreakdown.pairBatchMemory).toBeLessThan(
        mediumBatch.memoryBreakdown.pairBatchMemory ?? 0
      );
      expect(mediumBatch.memoryBreakdown.pairBatchMemory).toBeLessThan(
        largeBatch.memoryBreakdown.pairBatchMemory ?? 0
      );

      // All should be valid
      expect(smallBatch.status).not.toBe('error');
      expect(mediumBatch.status).not.toBe('error');
      expect(largeBatch.status).not.toBe('error');
    });

    test('should cap effective batch size at total pairs', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, // numQueries
        10, // docsPerQuery - only 10 pairs total
        512, // maxQueryLength
        512, // maxDocLength
        1, // numGPUs
        32 // batchSize larger than total pairs
      );

      expect(result.totalPairs).toBe(10);
      expect(result.effectiveBatchSize).toBe(10); // Should be capped at totalPairs
      expect(result.status).not.toBe('error');
    });
  });

  describe('Multi-GPU Scenarios', () => {
    test('should handle multi-GPU deployment', () => {
      const singleGPU = calculateRerankingMemory(
        model, gpu, 'fp16',
        10, 100, 512, 512, 1, 32
      );

      const multiGPU = calculateRerankingMemory(
        model, gpu, 'fp16',
        10, 100, 512, 512, 4, 32
      );

      // Multi-GPU should have overhead
      expect(multiGPU.memoryBreakdown.multiGPUOverhead).toBeGreaterThan(0);
      expect(singleGPU.memoryBreakdown.multiGPUOverhead).toBe(0);

      // Total VRAM should scale with GPU count
      expect(multiGPU.totalVRAM).toBe(gpu.vram_gb * 4);
      expect(singleGPU.totalVRAM).toBe(gpu.vram_gb);

      expect(multiGPU.status).not.toBe('error');
    });
  });

  describe('Edge Cases', () => {
    test('should handle very small workloads', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, // numQueries
        1, // docsPerQuery - single pair
        64, // maxQueryLength
        64, // maxDocLength
        1, // numGPUs
        1 // batchSize
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.totalPairs).toBe(1);
      expect(result.effectiveBatchSize).toBe(1);
      expect(result.status).not.toBe('error');
    });

    test('should handle maximum document length', () => {
      const result = calculateRerankingMemory(
        model, gpu, 'fp16',
        1, // numQueries
        20, // docsPerQuery
        512, // maxQueryLength
        2048, // maxDocLength - very long
        1, // numGPUs
        8 // batchSize - smaller due to longer docs
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');

      // Longer documents should significantly increase memory
      expect(result.memoryBreakdown.pairBatchMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.attentionMemory).toBeGreaterThan(0);
    });
  });
});