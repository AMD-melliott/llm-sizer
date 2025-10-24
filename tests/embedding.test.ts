import { calculateEmbeddingMemory } from '../src/utils/embeddingCalculator';
import { EmbeddingModel, GPU } from '../src/types';

describe('Embedding Memory Calculation Tests', () => {
  const model: EmbeddingModel = {
    id: "bge-large-en-v1.5",
    name: "BAAI/bge-large-en-v1.5",
    architecture: 'transformer' as const,
    parameters_millions: 335,
    dimensions: 1024,
    max_tokens: 512,
    hidden_size: 1024,
    num_layers: 24,
    num_heads: 16
  };

  const gpu: GPU = {
    id: "mi300x",
    vendor: 'AMD',
    name: "AMD Instinct MI300X",
    category: 'enterprise',
    tier: 'datacenter',
    vram_gb: 192,
    memory_type: 'HBM3e',
    memory_bandwidth_gbps: 5300,
    compute_tflops_fp16: 1300,
    compute_tflops_fp8: 2600,
    pcie_gen: 5,
    tdp_watts: 750,
    multi_gpu_capable: true,
    release_year: 2023
  };

  describe('Batch Processing Scenarios', () => {
    test('should handle small batch processing', () => {
      const result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        32, // batchSize
        64, // documentsPerBatch
        512, // avgDocumentSize
        1 // numGPUs
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');
      expect(result.vramPercentage).toBeLessThan(100);

      // Validate memory breakdown
      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(0);
      expect(result.memoryBreakdown.batchInputMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.attentionMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.embeddingStorage).toBeGreaterThan(0);
      expect(result.memoryBreakdown.activations).toBeGreaterThan(0);
      expect(result.memoryBreakdown.frameworkOverhead).toBeGreaterThan(0);
    });

    test('should handle default configuration', () => {
      const result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, // batchSize
        64, // documentsPerBatch
        512, // avgDocumentSize
        1 // numGPUs
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');
      expect(result.vramPercentage).toBeLessThan(100);

      // Larger batch should use more memory
      expect(result.memoryBreakdown.batchInputMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.attentionMemory).toBeGreaterThan(0);
    });

    test('should handle large documents exceeding max tokens', () => {
      const result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        128, // batchSize
        32, // documentsPerBatch
        1024, // avgDocumentSize (exceeds max_tokens)
        1 // numGPUs
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');

      // Documents should be truncated to max_tokens internally
      // The memory should not grow beyond what max_tokens would require
      expect(result.vramPercentage).toBeLessThan(100);
    });
  });

  describe('Quantization Effects', () => {
    test('should reduce memory with INT8 quantization', () => {
      const fp16Result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 64, 512, 1
      );

      const int8Result = calculateEmbeddingMemory(
        model, gpu, 'int8',
        256, 64, 512, 1
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
      const result = calculateEmbeddingMemory(
        model, gpu, 'int4',
        256, 64, 512, 1
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.memoryBreakdown.baseWeights).toBeLessThan(0.2); // Very small for INT4
      expect(result.status).not.toBe('error');
    });
  });

  describe('Batch Size Optimization', () => {
    test('should scale memory with batch size', () => {
      const smallBatch = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        64, 64, 512, 1
      );

      const mediumBatch = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 64, 512, 1
      );

      const largeBatch = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        512, 64, 512, 1
      );

      // Larger batch sizes should use more memory
      expect(smallBatch.memoryBreakdown.batchInputMemory).toBeLessThan(
        mediumBatch.memoryBreakdown.batchInputMemory ?? 0
      );
      expect(mediumBatch.memoryBreakdown.batchInputMemory).toBeLessThan(
        largeBatch.memoryBreakdown.batchInputMemory ?? 0
      );

      // All should be valid
      expect(smallBatch.status).not.toBe('error');
      expect(mediumBatch.status).not.toBe('error');
      expect(largeBatch.status).not.toBe('error');
    });

    test('should handle very large batch sizes', () => {
      const result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        1024, // Very large batch
        128, // Many documents
        512, // avgDocumentSize
        1 // numGPUs
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');

      // Should still fit in the large GPU
      expect(result.vramPercentage).toBeLessThan(100);
    });
  });

  describe('Document Processing', () => {
    test('should handle varying document counts', () => {
      const fewDocs = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 16, 512, 1
      );

      const manyDocs = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 128, 512, 1
      );

      // Embedding storage depends on batch size, not document count
      // Per documentation: batch_size × embedding_dimension × 4 / 10^9
      // Both tests use the same batch size, so embedding storage should be equal
      expect(fewDocs.memoryBreakdown.embeddingStorage).toEqual(
        manyDocs.memoryBreakdown.embeddingStorage ?? 0
      );

      expect(fewDocs.status).not.toBe('error');
      expect(manyDocs.status).not.toBe('error');
    });

    test('should handle different document sizes', () => {
      const shortDocs = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 64, 128, 1 // Short documents
      );

      const longDocs = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 64, 512, 1 // Long documents
      );

      // Longer documents should use more memory (up to max_tokens)
      expect(shortDocs.memoryBreakdown.batchInputMemory).toBeLessThan(
        longDocs.memoryBreakdown.batchInputMemory ?? 0
      );

      expect(shortDocs.status).not.toBe('error');
      expect(longDocs.status).not.toBe('error');
    });
  });

  describe('Multi-GPU Scenarios', () => {
    test('should handle multi-GPU deployment', () => {
      const singleGPU = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 64, 512, 1
      );

      const multiGPU = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 64, 512, 4
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
    test('should handle minimal workload', () => {
      const result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        1, // Minimum batch size
        1, // Single document
        64, // Short document
        1 // Single GPU
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');

      // Even minimal workload needs model weights
      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(0);
    });

    test('should handle maximum supported workload', () => {
      const result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        2048, // Very large batch
        256, // Many documents
        512, // Max token documents
        1 // Single GPU
      );

      expect(result.usedVRAM).toBeGreaterThan(0);

      // This is actually too much for a single GPU even with 192GB
      // The test should recognize realistic limitations
      if (result.vramPercentage > 100) {
        expect(result.status).toBe('error');
      } else {
        expect(result.status).not.toBe('error');
      }
    });

    test('should properly handle documents longer than max tokens', () => {
      const shortDocResult = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        128, 64,
        256, // Short documents
        1
      );

      const longDocResult = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        128, 64,
        2048, // Very long documents (exceeds max_tokens of 512)
        1
      );

      // Memory should be capped due to max_tokens limit
      // The batch input memory should not grow beyond max_tokens
      const maxTokensResult = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        128, 64,
        model.max_tokens, // Exactly max_tokens
        1
      );

      // Long doc result should be similar to max tokens result
      expect(Math.abs(longDocResult.memoryBreakdown.batchInputMemory! - maxTokensResult.memoryBreakdown.batchInputMemory!)).toBeLessThan(0.01);
      expect(longDocResult.status).not.toBe('error');
    });
  });

  describe('Memory Component Validation', () => {
    test('should have all required memory components', () => {
      const result = calculateEmbeddingMemory(
        model, gpu, 'fp16',
        256, 64, 512, 1
      );

      // All components should be present and positive
      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(0);
      expect(result.memoryBreakdown.batchInputMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.attentionMemory).toBeGreaterThan(0);
      expect(result.memoryBreakdown.embeddingStorage).toBeGreaterThan(0);
      expect(result.memoryBreakdown.activations).toBeGreaterThan(0);
      expect(result.memoryBreakdown.frameworkOverhead).toBeGreaterThan(0);
      expect(result.memoryBreakdown.multiGPUOverhead).toBe(0); // Single GPU

      // Total should be sum of components
      const totalCalculated = Object.values(result.memoryBreakdown)
        .reduce((sum, val) => sum + (val ?? 0), 0);
      expect(Math.abs(result.usedVRAM - totalCalculated)).toBeLessThan(0.01);
    });
  });
});