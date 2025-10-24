import { calculateMemoryRequirements } from '../src/utils/memoryCalculator';
import { Model, GPU } from '../src/types';

describe('Multimodal (Vision-Language) Memory Calculation Tests', () => {
  // Test models
  const llava15_7b: Model = {
    id: "llava-1.5-7b",
    name: "LLaVA 1.5 7B",
    parameters_billions: 7.06,
    hidden_size: 4096,
    num_layers: 32,
    num_heads: 32,
    default_context_length: 4096,
    architecture: 'transformer' as const,
    modality: "multimodal",
    vision_config: {
      model_type: "clip_vision_model",
      image_size: 336,
      patch_size: 14,
      num_channels: 3,
      hidden_size: 1024,
      num_layers: 24,
      num_heads: 16,
      intermediate_size: 4096,
      parameters_millions: 303
    },
    multimodal_config: {
      image_token_count: 576,  // (336/14)^2 = 576 tokens per image
      max_images: 1,
      projector_type: "mlp",
      projector_params_millions: 8.4  // 2-layer MLP projector
    }
  };

  const phi3_vision: Model = {
    id: "phi-3-vision-128k",
    name: "Phi-3-vision 128k Instruct",
    parameters_billions: 4.15,
    hidden_size: 3072,
    num_layers: 32,
    num_heads: 32,
    default_context_length: 128000,
    architecture: 'transformer' as const,
    modality: "multimodal",
    vision_config: {
      model_type: "clip_vision_model",
      image_size: 336,
      patch_size: 14,
      num_channels: 3,
      hidden_size: 1024,
      num_layers: 24,
      num_heads: 16,
      intermediate_size: 4096,
      parameters_millions: 303
    },
    multimodal_config: {
      image_token_count: 144,  // Sub-sampled from 576
      max_images: 1,
      projector_type: "mlp",
      projector_params_millions: 6.3
    }
  };

  const florence2_large: Model = {
    id: "florence-2-large",
    name: "Florence-2 Large",
    parameters_billions: 0.771,  // 771M total
    hidden_size: 768,
    num_layers: 12,
    num_heads: 12,
    default_context_length: 1024,
    architecture: 'transformer' as const,
    modality: "multimodal",
    vision_config: {
      model_type: "davit",
      image_size: 768,
      patch_size: [7, 7],  // Hierarchical patch sizes
      num_channels: 3,
      hidden_size: 768,
      num_layers: 12,
      num_heads: 12,
      intermediate_size: 3072,
      parameters_millions: 304
    },
    multimodal_config: {
      image_token_count: 1024,  // DaViT produces more tokens
      max_images: 1,
      projector_type: "linear",
      projector_params_millions: 0.6
    }
  };

  const gpu_h100: GPU = {
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

  const gpu_a100: GPU = {
    id: "a100-80gb",
    vendor: 'NVIDIA',
    name: "A100 80GB",
    category: 'enterprise',
    tier: 'datacenter',
    vram_gb: 80,
    memory_type: 'HBM2e',
    memory_bandwidth_gbps: 2039,
    compute_tflops_fp16: 312,
    compute_tflops_fp8: 624,
    nvlink_bandwidth_gbps: 600,
    pcie_gen: 4,
    tdp_watts: 400,
    multi_gpu_capable: true,
    release_year: 2020
  };

  const gpu_mi300x: GPU = {
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

  describe('Single Image Processing', () => {
    test('should calculate memory for LLaVA 1.5 7B with single image at standard resolution', () => {
      const result = calculateMemoryRequirements(
        llava15_7b, gpu_h100, 'fp16', 'fp16_bf16',
        1, // batchSize
        2048, // sequenceLength (text tokens)
        1, // concurrentUsers
        1, // numGPUs
        1, // numImages
        336 // imageResolution
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.usedVRAM).toBeLessThan(gpu_h100.vram_gb);
      expect(result.vramPercentage).toBeLessThan(100);
      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(10);
      expect(result.memoryBreakdown.visionWeights).toBeGreaterThan(0);
      expect(result.memoryBreakdown.projectorWeights).toBeGreaterThan(0);
      expect(result.memoryBreakdown.imageTokensKV).toBeGreaterThan(0);
      expect(result.memoryBreakdown.visionActivations).toBeGreaterThan(0);
      expect(result.memoryBreakdown.imagePreprocessing).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');
    });

    test('should handle INT8 quantization for batch processing', () => {
      const result = calculateMemoryRequirements(
        llava15_7b, gpu_h100, 'int8', 'int8',
        8, // batchSize - batch processing
        1024, // sequenceLength
        1, // concurrentUsers
        1, // numGPUs
        1, // numImages
        336 // imageResolution
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.memoryBreakdown.baseWeights).toBeLessThan(10); // INT8 should use less memory
      expect(result.memoryBreakdown.kvCache).toBeGreaterThan(0);
      expect(result.memoryBreakdown.imageTokensKV).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');
    });
  });

  describe('Multiple Images Processing', () => {
    test('should handle multiple images at higher resolution', () => {
      const result = calculateMemoryRequirements(
        llava15_7b, gpu_h100, 'fp16', 'fp16_bf16',
        1, // batchSize
        2048, // sequenceLength
        1, // concurrentUsers
        1, // numGPUs
        4, // numImages - multi-image scenario
        448 // imageResolution - higher resolution
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.memoryBreakdown.imageTokensKV).toBeGreaterThan(0);
      expect(result.memoryBreakdown.visionActivations).toBeGreaterThan(0);
      expect(result.memoryBreakdown.imagePreprocessing).toBeGreaterThan(0);
      expect(result.vramPercentage).toBeLessThan(100);
      expect(result.status).not.toBe('error');
    });

    test('should handle stress test with maximum resolution', () => {
      const result = calculateMemoryRequirements(
        llava15_7b, gpu_mi300x, 'int4', 'int8',
        1, // batchSize
        1024, // sequenceLength
        1, // concurrentUsers
        1, // numGPUs
        8, // numImages - stress test
        1024 // imageResolution - very high resolution
      );

      const expectedTokensPerImage = Math.floor((1024/14)**2);
      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.memoryBreakdown.imageTokensKV).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');

      // With INT4 quantization and high GPU memory, should still fit
      expect(result.vramPercentage).toBeLessThan(100);
    });
  });

  describe('Different Model Architectures', () => {
    test('should handle Phi-3-vision with efficient token usage', () => {
      const result = calculateMemoryRequirements(
        phi3_vision, gpu_a100, 'fp16', 'fp16_bf16',
        1, // batchSize
        4096, // sequenceLength - long context
        1, // concurrentUsers
        1, // numGPUs
        1, // numImages
        336 // imageResolution
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.memoryBreakdown.baseWeights).toBeGreaterThan(0);
      expect(result.memoryBreakdown.visionWeights).toBeGreaterThan(0);

      // Phi-3 uses fewer image tokens than LLaVA
      expect(phi3_vision.multimodal_config?.image_token_count).toBeLessThan(
        llava15_7b.multimodal_config?.image_token_count ?? 0
      );

      expect(result.memoryBreakdown.imageTokensKV).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');
    });

    test('should handle Florence-2 with DaViT architecture', () => {
      const result = calculateMemoryRequirements(
        florence2_large, gpu_a100, 'fp16', 'fp16_bf16',
        4, // batchSize
        512, // sequenceLength
        1, // concurrentUsers
        1, // numGPUs
        1, // numImages
        768 // imageResolution - higher for Florence
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.memoryBreakdown.baseWeights).toBeLessThan(2); // Small model
      expect(result.memoryBreakdown.visionWeights).toBeGreaterThan(0);

      // Florence uses more image tokens due to DaViT hierarchical encoding
      expect(florence2_large.multimodal_config?.image_token_count).toBe(1024);

      expect(result.memoryBreakdown.imageTokensKV).toBeGreaterThan(0);
      expect(result.status).not.toBe('error');
    });
  });

  describe('Multi-GPU Deployment', () => {
    test('should handle multi-GPU high throughput scenario', () => {
      const result = calculateMemoryRequirements(
        llava15_7b, gpu_mi300x, 'fp16', 'fp8_bf16',
        16, // batchSize - high throughput
        2048, // sequenceLength
        8, // concurrentUsers
        4, // numGPUs
        2, // numImages per request
        336 // imageResolution
      );

      expect(result.usedVRAM).toBeGreaterThan(0);
      expect(result.totalVRAM).toBe(gpu_mi300x.vram_gb * 4);
      expect(result.memoryBreakdown.multiGPUOverhead).toBeGreaterThan(0);
      
      // With the corrected overhead, this configuration fits comfortably.
      expect(result.vramPercentage).toBeLessThan(100);
      expect(result.status).not.toBe('error');

      const perGPUMemory = result.usedVRAM / 4;
      expect(perGPUMemory).toBeLessThan(gpu_mi300x.vram_gb);
    });
  });

  describe('Text-only vs Multimodal Comparison', () => {
    test('should show multimodal overhead compared to text-only', () => {
      const textOnlyModel: Model = {
        ...llava15_7b,
        modality: "text" as const,
        vision_config: undefined,
        multimodal_config: undefined
      };

      const textResult = calculateMemoryRequirements(
        textOnlyModel, gpu_h100, 'fp16', 'fp16_bf16',
        1, 2048, 1, 1, 0, 0
      );

      const multimodalResult = calculateMemoryRequirements(
        llava15_7b, gpu_h100, 'fp16', 'fp16_bf16',
        1, 2048, 1, 1, 1, 336
      );

      const overhead = multimodalResult.usedVRAM - textResult.usedVRAM;
      const overheadPercent = (overhead / textResult.usedVRAM) * 100;

      expect(overhead).toBeGreaterThan(0);
      // With updated framework overhead (8% vs 5%), the relative overhead is ~5-6%
      // This is realistic for vision encoder + projector weights
      expect(overheadPercent).toBeGreaterThan(5); // At least 5% overhead
      expect(overheadPercent).toBeLessThan(50); // But less than 50%

      // Verify overhead components
      expect(multimodalResult.memoryBreakdown.visionWeights).toBeGreaterThan(0);
      expect(multimodalResult.memoryBreakdown.imageTokensKV).toBeGreaterThan(0);
      expect(multimodalResult.memoryBreakdown.visionActivations).toBeGreaterThan(0);
      expect(multimodalResult.memoryBreakdown.imagePreprocessing).toBeGreaterThan(0);
      expect(multimodalResult.memoryBreakdown.projectorWeights).toBeGreaterThan(0);

      // Text-only model should not have these components
      expect(textResult.memoryBreakdown.visionWeights).toBeUndefined();
      expect(textResult.memoryBreakdown.imageTokensKV).toBeUndefined();
    });
  });
});