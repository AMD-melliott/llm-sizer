import { describe, test, expect } from '@jest/globals';
import { readFileSync } from 'fs';
import { join } from 'path';
import {
  calculatePartitionMemory,
  analyzePartitionConfiguration,
  getPartitionCapableGPUs,
  getPartitionMode,
  formatMemorySize,
  formatCompute,
  formatBandwidth,
  getModelDisplayName,
} from '../src/utils/partitionCalculator';
import { GPU, PartitionMode, PartitionConfiguration, Model, EmbeddingModel, RerankingModel } from '../src/types';

// Load model data using fs
const modelsData = JSON.parse(readFileSync(join(__dirname, '../src/data/models.json'), 'utf-8'));
const embeddingModelsData = JSON.parse(readFileSync(join(__dirname, '../src/data/embedding-models.json'), 'utf-8'));
const rerankingModelsData = JSON.parse(readFileSync(join(__dirname, '../src/data/reranking-models.json'), 'utf-8'));

describe('Partition Calculator', () => {
  // Mock GPU with partitioning support
  const mockGPU: GPU = {
    id: 'mi300x',
    vendor: 'AMD',
    name: 'AMD Instinct MI300X',
    category: 'enterprise',
    tier: 'datacenter',
    vram_gb: 192,
    memory_type: 'HBM3',
    memory_bandwidth_gbps: 5300,
    compute_tflops_fp16: 1300,
    compute_tflops_fp8: 2600,
    pcie_gen: 5,
    tdp_watts: 750,
    multi_gpu_capable: true,
    release_year: 2023,
    partitioning: {
      supported: true,
      modes: [
        {
          mode: 'SPX',
          name: 'Single Partition eXtended',
          description: 'Full GPU resources',
          partitionCount: 1,
          vramPerPartition: 192,
          bandwidthPerPartition: 5300,
          computeFP16PerPartition: 1300,
          computeFP8PerPartition: 2600,
        },
        {
          mode: 'DPX',
          name: 'Dual Partition eXtended',
          description: '2 isolated partitions',
          partitionCount: 2,
          vramPerPartition: 96,
          bandwidthPerPartition: 2650,
          computeFP16PerPartition: 650,
          computeFP8PerPartition: 1300,
        },
        {
          mode: 'CPX',
          name: 'Compute Partition eXtended',
          description: '8 compute-optimized partitions',
          partitionCount: 8,
          vramPerPartition: 24,
          bandwidthPerPartition: 662.5,
          computeFP16PerPartition: 162.5,
          computeFP8PerPartition: 325,
        },
      ],
    },
  };

  // Mock model for testing
  const mockModel: Model = {
    id: 'llama-3-8b',
    name: 'Llama 3 8B',
    parameters_billions: 8,
    hidden_size: 4096,
    num_layers: 32,
    num_heads: 32,
    default_context_length: 8192,
    architecture: 'transformer',
    modality: 'text',
  };

  const mockEmbeddingModel: EmbeddingModel = {
    id: 'bge-large',
    name: 'BGE Large',
    parameters_millions: 335,
    architecture: 'transformer',
    dimensions: 1024,
    max_tokens: 512,
    hidden_size: 1024,
    num_layers: 24,
    num_heads: 16,
  };

  const mockRerankingModel: RerankingModel = {
    id: 'bge-reranker-large',
    name: 'BGE Reranker Large',
    parameters_millions: 335,
    architecture: 'cross-encoder',
    max_query_length: 512,
    max_doc_length: 512,
    max_docs_per_query: 100,
    hidden_size: 1024,
    num_layers: 24,
    num_heads: 16,
  };

  describe('calculatePartitionMemory', () => {
    test('should calculate memory for text generation model in CPX mode', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const result = calculatePartitionMemory(
        mockModel,
        cpxMode,
        mockGPU,
        'fp8',
        'fp16_bf16',
        1,
        4096,
        1,
        'generation'
      );

      expect(result.memoryAvailable).toBe(24);
      expect(result.memoryUsed).toBeGreaterThan(0);
      expect(result.percentUsed).toBeGreaterThan(0);
      expect(result.status).toBeDefined();
      expect(['fits', 'tight', 'no-fit']).toContain(result.status);
    });

    test('should determine fit status correctly for small model', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const result = calculatePartitionMemory(
        mockModel,
        cpxMode,
        mockGPU,
        'int4', // Smaller quantization
        'int8',
        1,
        2048, // Smaller sequence
        1,
        'generation'
      );

      // With INT4 and small sequence, 8B model should fit in 24GB
      expect(result.fits).toBe(true);
      expect(result.memoryUsed).toBeLessThan(result.memoryAvailable);
    });

    test('should provide a recommended quantization that fits', () => {
      const dpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'DPX')!;
      // Use an intentionally large batch/sequence to force higher memory then see recommendation
      const result = calculatePartitionMemory(
        mockModel,
        dpxMode,
        mockGPU,
        'fp16',
        'fp16_bf16',
        4,
        8192,
        2,
        'generation'
      );
      // Recommended quantization should exist (likely fp16 or lower depending on calc logic)
      expect(result.recommendedQuantization).toBeDefined();
      // If chosen inference quantization doesn't fit, recommended should be lower precision
      if (!result.fits) {
        const order: Record<string, number> = { fp16: 0, fp8: 1, int8: 2, int4: 3 };
        expect(order[result.recommendedQuantization!]).toBeGreaterThan(order['fp16']);
      }
    });

    test('should handle embedding model calculations', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const result = calculatePartitionMemory(
        mockEmbeddingModel,
        cpxMode,
        mockGPU,
        'fp16',
        'fp16_bf16',
        32,
        512,
        1,
        'embedding'
      );

      expect(result.memoryUsed).toBeGreaterThan(0);
      expect(result.memoryUsed).toBeLessThan(result.memoryAvailable);
      expect(result.fits).toBe(true);
    });

    test('should classify tight fit correctly (>80% usage)', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const result = calculatePartitionMemory(
        mockModel,
        cpxMode,
        mockGPU,
        'fp16',
        'fp16_bf16',
        1,
        8192, // Larger sequence
        2, // More users
        'generation'
      );

      // This should be close to or exceed capacity
      if (result.percentUsed > 80 && result.fits) {
        expect(result.status).toBe('tight');
      }
    });
  });

  describe('analyzePartitionConfiguration', () => {
    test('should analyze all models for a configuration', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const configuration: PartitionConfiguration = {
        gpu: mockGPU,
        mode: cpxMode,
        inferenceQuantization: 'fp8',
        kvCacheQuantization: 'fp8_bf16',
        batchSize: 1,
        sequenceLength: 4096,
        concurrentUsers: 1,
      };

      const results = analyzePartitionConfiguration(configuration, 'generation', modelsData.models as Model[]);

      expect(results.compatibleModels).toBeInstanceOf(Array);
      expect(results.compatibleModels.length).toBeGreaterThan(0);
      expect(results.modelType).toBe('generation');
      expect(results.configuration).toEqual(configuration);
    });

    test('should sort models by memory used', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const configuration: PartitionConfiguration = {
        gpu: mockGPU,
        mode: cpxMode,
        inferenceQuantization: 'fp16',
        kvCacheQuantization: 'fp16_bf16',
        batchSize: 1,
        sequenceLength: 2048,
        concurrentUsers: 1,
      };

      const results = analyzePartitionConfiguration(configuration, 'generation', modelsData.models as Model[]);

      // Check that results are sorted by memory used
      for (let i = 1; i < results.compatibleModels.length; i++) {
        expect(results.compatibleModels[i].memoryUsed).toBeGreaterThanOrEqual(
          results.compatibleModels[i - 1].memoryUsed
        );
      }
    });
  });

  describe('getPartitionCapableGPUs', () => {
    test('should filter GPUs with partitioning support', () => {
      const gpus: GPU[] = [
        mockGPU,
        {
          ...mockGPU,
          id: 'rtx4090',
          name: 'RTX 4090',
          vendor: 'NVIDIA',
          partitioning: undefined,
        },
      ];

      const capable = getPartitionCapableGPUs(gpus);

      expect(capable.length).toBe(1);
      expect(capable[0].id).toBe('mi300x');
    });

    test('should return empty array when no GPUs support partitioning', () => {
      const gpus: GPU[] = [
        {
          ...mockGPU,
          partitioning: undefined,
        },
      ];

      const capable = getPartitionCapableGPUs(gpus);

      expect(capable.length).toBe(0);
    });
  });

  describe('getPartitionMode', () => {
    test('should find correct partition mode', () => {
      const cpxMode = getPartitionMode(mockGPU, 'CPX');

      expect(cpxMode).toBeDefined();
      expect(cpxMode?.mode).toBe('CPX');
      expect(cpxMode?.partitionCount).toBe(8);
      expect(cpxMode?.vramPerPartition).toBe(24);
    });

    test('should return undefined for non-existent mode', () => {
      const mode = getPartitionMode(mockGPU, 'INVALID' as any);

      expect(mode).toBeUndefined();
    });
  });

  describe('Utility Functions', () => {
    test('formatMemorySize should format GB correctly', () => {
      expect(formatMemorySize(24)).toBe('24.0 GB');
      expect(formatMemorySize(96)).toBe('96.0 GB');
      expect(formatMemorySize(192)).toBe('192.0 GB');
    });

    test('formatMemorySize should format MB for small values', () => {
      expect(formatMemorySize(0.5)).toBe('512.0 MB');
      expect(formatMemorySize(0.1)).toBe('102.4 MB');
    });

    test('formatCompute should format TFLOPS', () => {
      expect(formatCompute(1300)).toBe('1300.0 TFLOPS');
      expect(formatCompute(162.5)).toBe('162.5 TFLOPS');
    });

    test('getModelDisplayName should format text generation models', () => {
      const name = getModelDisplayName(mockModel);
      expect(name).toContain('Llama 3 8B');
      expect(name).toContain('(8B)');
    });

    test('getModelDisplayName should format embedding models', () => {
      const name = getModelDisplayName(mockEmbeddingModel);
      expect(name).toContain('BGE Large');
      expect(name).toContain('335M');
    });
  });

  describe('Edge Cases', () => {
    test('should handle very large models that do not fit', () => {
      const largeModel: Model = {
        ...mockModel,
        id: 'llama-3-405b',
        name: 'Llama 3 405B',
        parameters_billions: 405,
      };

      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const result = calculatePartitionMemory(
        largeModel,
        cpxMode,
        mockGPU,
        'fp16',
        'fp16_bf16',
        1,
        4096,
        1,
        'generation'
      );

      expect(result.fits).toBe(false);
      expect(result.status).toBe('no-fit');
      expect(result.memoryUsed).toBeGreaterThan(result.memoryAvailable);
    });

    test('should handle SPX mode (full GPU)', () => {
      const spxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'SPX')!;
      
      expect(spxMode.partitionCount).toBe(1);
      expect(spxMode.vramPerPartition).toBe(192);
      
      const result = calculatePartitionMemory(
        mockModel,
        spxMode,
        mockGPU,
        'fp16',
        'fp16_bf16',
        8,
        8192,
        4,
        'generation'
      );

      // Should fit with full GPU resources
      expect(result.memoryAvailable).toBe(192);
    });

    test('should handle DPX mode (dual partition)', () => {
      const dpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'DPX')!;
      
      expect(dpxMode.partitionCount).toBe(2);
      expect(dpxMode.vramPerPartition).toBe(96);
      
      const result = calculatePartitionMemory(
        mockModel,
        dpxMode,
        mockGPU,
        'fp8',
        'fp8_bf16',
        2,
        4096,
        1,
        'generation'
      );

      expect(result.memoryAvailable).toBe(96);
    });
  });

  describe('New Features Tests', () => {
    test('should handle reranking model calculations', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const result = calculatePartitionMemory(
        mockRerankingModel,
        cpxMode,
        mockGPU,
        'fp16',
        'fp16_bf16',
        32,
        512,
        1,
        'reranking'
      );

      expect(result.memoryUsed).toBeGreaterThan(0);
      expect(result.memoryUsed).toBeLessThan(result.memoryAvailable);
      expect(result.memoryBreakdown).toBeDefined();
      expect(result.memoryBreakdown.pairBatchMemory).toBeDefined();
    });

    test('should analyze "all" model types correctly', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const configuration: PartitionConfiguration = {
        gpu: mockGPU,
        mode: cpxMode,
        inferenceQuantization: 'fp8',
        kvCacheQuantization: 'fp8_bf16',
        batchSize: 1,
        sequenceLength: 2048,
        concurrentUsers: 1,
      };

      const allModels = [
        ...modelsData.models.slice(0, 2) as Model[],
        ...embeddingModelsData.models.slice(0, 2) as EmbeddingModel[],
        ...rerankingModelsData.models.slice(0, 2) as RerankingModel[]
      ];

      const results = analyzePartitionConfiguration(configuration, 'all', allModels);

      expect(results.modelType).toBe('all');
      expect(results.compatibleModels.length).toBe(6);
      // Should include models from all types
      const hasGeneration = results.compatibleModels.some(r => 'parameters_billions' in r.model);
      const hasEmbedding = results.compatibleModels.some(r => 'dimensions' in r.model);
      const hasReranking = results.compatibleModels.some(r => 'max_query_length' in r.model);
      expect(hasGeneration).toBe(true);
      expect(hasEmbedding).toBe(true);
      expect(hasReranking).toBe(true);
    });

    test('formatBandwidth should scale to TB/s for large values', () => {
      expect(formatBandwidth(500)).toBe('500.0 GB/s');
      expect(formatBandwidth(1000)).toBe('1.00 TB/s');
      expect(formatBandwidth(5300)).toBe('5.30 TB/s');
      expect(formatBandwidth(2650)).toBe('2.65 TB/s');
    });

    test('should inherit GPU properties correctly in partition', () => {
      const dpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'DPX')!;
      
      const result = calculatePartitionMemory(
        mockModel,
        dpxMode,
        mockGPU,
        'fp8',
        'fp8_bf16',
        1,
        4096,
        1,
        'generation'
      );

      // Verify that memory breakdown exists and is calculated
      expect(result.memoryBreakdown).toBeDefined();
      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(0);
    });

    test('memory breakdown should contain all core components', () => {
      const cpxMode = mockGPU.partitioning!.modes.find(m => m.mode === 'CPX')!;
      
      const result = calculatePartitionMemory(
        mockModel,
        cpxMode,
        mockGPU,
        'fp16',
        'fp16_bf16',
        1,
        4096,
        1,
        'generation'
      );

      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(0);
      expect(result.memoryBreakdown.activations).toBeGreaterThan(0);
      expect(result.memoryBreakdown.kvCache).toBeGreaterThan(0);
      expect(result.memoryBreakdown.frameworkOverhead).toBeGreaterThan(0);
      
      // Sum should approximately match memoryUsed
      const total = result.memoryBreakdown.baseWeights +
                    result.memoryBreakdown.activations +
                    result.memoryBreakdown.kvCache +
                    result.memoryBreakdown.frameworkOverhead +
                    result.memoryBreakdown.multiGPUOverhead;
      
      expect(Math.abs(total - result.memoryUsed)).toBeLessThan(0.1); // Within 100MB
    });

    test('getModelDisplayName should handle reranking models', () => {
      const name = getModelDisplayName(mockRerankingModel);
      expect(name).toContain('BGE Reranker Large');
      expect(name).toContain('335M');
    });
  });
});
