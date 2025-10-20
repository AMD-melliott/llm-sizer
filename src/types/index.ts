// GPU Configuration Types
export interface GPU {
  id: string;
  vendor: 'AMD' | 'NVIDIA' | 'Custom';
  name: string;
  category: 'enterprise' | 'consumer';
  vram_gb: number;
  memory_bandwidth_gbps: number;
  compute_tflops_fp16: number;
  compute_tflops_fp8?: number;
}

// Model Configuration Types
export interface Model {
  id: string;
  name: string;
  parameters_billions: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  default_context_length: number;
  architecture: 'transformer' | 'moe' | 'other';
}

// Quantization Types
export type InferenceQuantization = 'fp16' | 'fp8' | 'int8' | 'int4';
export type KVCacheQuantization = 'fp16_bf16' | 'fp8_bf16' | 'int8';

// Memory Breakdown
export interface MemoryBreakdown {
  baseWeights: number;
  activations: number;
  kvCache: number;
  frameworkOverhead: number;
  multiGPUOverhead: number;
}

// Performance Metrics
export interface PerformanceMetrics {
  generationSpeed: number;
  totalThroughput: number;
  perUserSpeed: number;
}

// Calculation Results
export interface CalculationResults {
  totalVRAM: number;
  usedVRAM: number;
  vramPercentage: number;
  memoryBreakdown: MemoryBreakdown;
  performance: PerformanceMetrics;
  status: 'okay' | 'warning' | 'error';
  message?: string;
}

// Application State
export interface AppState {
  // Model Configuration
  selectedModel: string;
  customModelParams?: number;
  customHiddenSize?: number;
  customNumLayers?: number;
  customNumHeads?: number;

  // Quantization
  inferenceQuantization: InferenceQuantization;
  kvCacheQuantization: KVCacheQuantization;

  // Hardware
  selectedGPU: string;
  customVRAM?: number;
  numGPUs: number;

  // Inference Parameters
  batchSize: number;
  sequenceLength: number;
  concurrentUsers: number;
  enableOffloading: boolean;

  // Computed Results
  results: CalculationResults | null;
}

// Store Actions
export interface AppActions {
  setSelectedModel: (modelId: string) => void;
  setCustomModelParams: (params: number | undefined) => void;
  setInferenceQuantization: (quant: InferenceQuantization) => void;
  setKVCacheQuantization: (quant: KVCacheQuantization) => void;
  setSelectedGPU: (gpuId: string) => void;
  setCustomVRAM: (vram: number | undefined) => void;
  setNumGPUs: (num: number) => void;
  setBatchSize: (size: number) => void;
  setSequenceLength: (length: number) => void;
  setConcurrentUsers: (users: number) => void;
  setEnableOffloading: (enable: boolean) => void;
  calculateResults: () => void;
}

export type AppStore = AppState & AppActions;