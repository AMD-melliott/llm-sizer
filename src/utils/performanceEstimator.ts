import {
  Model,
  GPU,
  InferenceQuantization,
  PerformanceMetrics
} from '../types';

export function estimatePerformance(
  model: Model,
  gpu: GPU,
  inferenceQuantization: InferenceQuantization,
  batchSize: number,
  _sequenceLength: number,
  concurrentUsers: number,
  numGPUs: number,
  vramPercentage: number
): PerformanceMetrics {
  // Get compute capability based on quantization
  let computeTFLOPS: number;
  if (inferenceQuantization === 'fp16') {
    computeTFLOPS = gpu.compute_tflops_fp16;
  } else if (inferenceQuantization === 'fp8' && gpu.compute_tflops_fp8) {
    computeTFLOPS = gpu.compute_tflops_fp8;
  } else if (inferenceQuantization === 'int8' && gpu.compute_tflops_fp8) {
    // Approximate INT8 performance as similar to FP8
    computeTFLOPS = gpu.compute_tflops_fp8 * 1.1;
  } else if (inferenceQuantization === 'int4' && gpu.compute_tflops_fp8) {
    // INT4 can be roughly 2x faster than INT8
    computeTFLOPS = gpu.compute_tflops_fp8 * 2;
  } else {
    // Fallback to FP16 performance with scaling factor
    const scalingFactors: Record<InferenceQuantization, number> = {
      'fp16': 1.0,
      'fp8': 2.0,
      'int8': 2.2,
      'int4': 4.0,
    };
    computeTFLOPS = gpu.compute_tflops_fp16 * scalingFactors[inferenceQuantization];
  }

  // Scale compute for multiple GPUs (not perfectly linear due to communication overhead)
  const multiGPUEfficiency = numGPUs === 1 ? 1.0 : 0.85 + (0.15 / numGPUs);
  const totalCompute = computeTFLOPS * numGPUs * multiGPUEfficiency;

  // Estimate FLOPs per token (simplified)
  // Roughly 2 * model_params FLOPs per token for transformer models
  const flopsPerToken = 2 * model.parameters_billions * 1e9;

  // Calculate theoretical tokens/sec based on compute
  const theoreticalTokensPerSec = (totalCompute * 1e12) / flopsPerToken;

  // Memory bandwidth constraint
  const memoryBandwidth = gpu.memory_bandwidth_gbps * numGPUs * multiGPUEfficiency;

  // Estimate memory-bound tokens/sec
  // Each token needs to load model weights + KV cache
  const bytesPerToken = model.parameters_billions * 2; // Simplified estimate in GB
  const memoryBoundTokensPerSec = memoryBandwidth / bytesPerToken;

  // Take the minimum of compute-bound and memory-bound performance
  let actualTokensPerSec = Math.min(theoreticalTokensPerSec, memoryBoundTokensPerSec);

  // Apply penalties based on VRAM usage
  if (vramPercentage > 95) {
    actualTokensPerSec *= 0.5; // Severe penalty for very high VRAM usage
  } else if (vramPercentage > 90) {
    actualTokensPerSec *= 0.7;
  } else if (vramPercentage > 80) {
    actualTokensPerSec *= 0.85;
  } else if (vramPercentage > 70) {
    actualTokensPerSec *= 0.95;
  }

  // Apply batch size efficiency
  const batchEfficiency = Math.min(1.0, Math.log2(batchSize + 1) / 5);
  actualTokensPerSec *= batchEfficiency;

  // Calculate per-user and total throughput
  const generationSpeed = actualTokensPerSec / concurrentUsers;
  const totalThroughput = actualTokensPerSec;
  const perUserSpeed = generationSpeed;

  return {
    generationSpeed: Math.round(generationSpeed),
    totalThroughput: Math.round(totalThroughput),
    perUserSpeed: Math.round(perUserSpeed),
  };
}

export function getPerformanceRating(tokensPerSec: number): {
  rating: string;
  color: string;
  description: string;
} {
  if (tokensPerSec >= 100) {
    return {
      rating: 'Excellent',
      color: 'text-green-600',
      description: 'Very fast generation, suitable for real-time applications',
    };
  } else if (tokensPerSec >= 50) {
    return {
      rating: 'Good',
      color: 'text-blue-600',
      description: 'Good performance for most use cases',
    };
  } else if (tokensPerSec >= 20) {
    return {
      rating: 'Acceptable',
      color: 'text-yellow-600',
      description: 'Acceptable for batch processing and non-interactive use',
    };
  } else if (tokensPerSec >= 5) {
    return {
      rating: 'Slow',
      color: 'text-orange-600',
      description: 'Slow generation, consider optimization',
    };
  } else {
    return {
      rating: 'Very Slow',
      color: 'text-red-600',
      description: 'Very slow, not suitable for production use',
    };
  }
}

export function estimateLatency(
  tokensPerSec: number,
  outputLength: number = 100
): {
  firstToken: number;
  fullResponse: number;
} {
  // Rough estimate: first token latency is higher due to prompt processing
  const firstToken = 1000 / (tokensPerSec * 0.5); // ms
  const fullResponse = (outputLength * 1000) / tokensPerSec; // ms

  return {
    firstToken: Math.round(firstToken),
    fullResponse: Math.round(fullResponse),
  };
}