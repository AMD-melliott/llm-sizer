// GPU Configuration Types
export interface GPU {
  id: string;
  vendor: 'AMD' | 'NVIDIA' | 'Custom';
  name: string;
  category: 'enterprise' | 'consumer'; // Legacy field, kept for backward compatibility
  tier: 'datacenter' | 'professional' | 'consumer' | 'custom';
  vram_gb: number;
  memory_type: 'HBM3e' | 'HBM3' | 'HBM2e' | 'GDDR6X' | 'GDDR6' | 'GDDR5';
  memory_bandwidth_gbps: number;
  compute_tflops_fp16: number;
  compute_tflops_fp8?: number;
  nvlink_bandwidth_gbps?: number;
  pcie_gen: 3 | 4 | 5;
  tdp_watts: number;
  multi_gpu_capable: boolean;
  release_year: number;

  // Partitioning support for AMD Instinct GPUs
  partitioning?: {
    supported: boolean;
    modes: PartitionMode[];
  };
}

// GPU Partitioning Types
export type PartitionModeType = 'SPX' | 'DPX' | 'CPX';

export interface PartitionMode {
  mode: PartitionModeType;
  name: string;
  description: string;
  partitionCount: number;
  vramPerPartition: number;
  bandwidthPerPartition: number;
  computeFP16PerPartition: number;
  computeFP8PerPartition?: number;
}

export interface PartitionConfiguration {
  gpu: GPU;
  mode: PartitionMode;
  inferenceQuantization: InferenceQuantization;
  kvCacheQuantization: KVCacheQuantization;
  batchSize: number;
  sequenceLength: number;
  concurrentUsers: number;
}

export interface PartitionMemoryResult {
  partitionIndex: number;
  model: Model | EmbeddingModel | RerankingModel;
  memoryUsed: number;
  memoryAvailable: number;
  percentUsed: number;
  fits: boolean;
  status: 'fits' | 'tight' | 'no-fit';
  memoryBreakdown: MemoryBreakdown;
  // Recommended quantization that would allow fit (highest quality that fits)
  recommendedQuantization?: InferenceQuantization;
}

export interface PartitionAnalysisResults {
  configuration: PartitionConfiguration;
  compatibleModels: PartitionMemoryResult[];
  modelType: PartitionModelSelection;
}

// Model Type
export type ModelType = 'generation' | 'embedding' | 'reranking';

// Partition Model Selection (includes 'all' option)
export type PartitionModelSelection = ModelType | 'all';

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
  
  // Multimodal support
  modality?: 'text' | 'multimodal';
  vision_config?: {
    model_type: string;
    image_size: number;
    patch_size: number | number[];
    num_channels?: number;
    hidden_size: number;
    num_layers: number;
    num_heads: number;
    intermediate_size?: number;
    parameters_millions: number;
  };
  multimodal_config?: {
    image_token_count?: number;
    max_images?: number;
    projector_type?: string;
    projector_params_millions?: number;
    merge_strategy?: string;
    supports_video?: boolean;
    frames_per_second?: number;
  };
}

// Embedding Model Configuration
export interface EmbeddingModel {
  id: string;
  name: string;
  vendor?: string;
  parameters_millions: number;
  architecture: 'transformer' | 'bert' | 'bi-encoder';
  dimensions: number;
  max_tokens: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  batch_token_limit?: number;
}

// Reranking Model Configuration
export interface RerankingModel {
  id: string;
  name: string;
  vendor?: string;
  parameters_millions: number;
  architecture: 'cross-encoder' | 'late-interaction';
  max_query_length: number;
  max_doc_length: number;
  max_docs_per_query: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
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
  // Optional safety margin added to calculations
  safetyMargin?: number;
  
  // Multimodal memory components
  visionWeights?: number;
  visionActivations?: number;
  projectorWeights?: number;
  imagePreprocessing?: number;
  imageTokensKV?: number;

  // Embedding memory components
  batchInputMemory?: number;
  attentionMemory?: number;
  embeddingStorage?: number;

  // Reranking memory components
  pairBatchMemory?: number;
  scoringMemory?: number;
}

// Performance Metrics
export interface PerformanceMetrics {
  generationSpeed: number;
  totalThroughput: number;
  perUserSpeed: number;
  
  // Embedding-specific metrics
  documentsPerSecond?: number;
  tokensPerSecond?: number;
  embeddingsPerSecond?: number;
  
  // Reranking-specific metrics
  queryDocPairsPerSecond?: number;
  queriesPerSecond?: number;
  avgLatencyMs?: number;
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
  // Reranking specific results
  totalPairs?: number;
  effectiveBatchSize?: number;
}

// Application State
export interface AppState {
  // Model Type Selection
  modelType: ModelType;
  
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

  // Inference Parameters (Generation)
  batchSize: number;
  sequenceLength: number;
  concurrentUsers: number;

  // Multimodal Parameters
  numImages: number;
  imageResolution: number;
  
  // Embedding Parameters
  embeddingBatchSize: number;
  documentsPerBatch: number;
  avgDocumentSize: number;
  chunkSize: number;
  chunkOverlap: number;

  // Reranking Parameters
  rerankingBatchSize: number;
  numQueries: number;
  docsPerQuery: number;
  maxQueryLength: number;
  maxDocLength: number;

  // Partitioning State
  partitioningGPU: string | null;
  partitioningMode: PartitionModeType;
  partitioningModelType: PartitionModelSelection;
  partitioningShowOnlyFits: boolean;

  // Computed Results
  results: CalculationResults | null;
}

// Store Actions
export interface AppActions {
  setModelType: (type: ModelType) => void;
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
  setNumImages: (num: number) => void;
  setImageResolution: (resolution: number) => void;
  setEmbeddingBatchSize: (size: number) => void;
  setDocumentsPerBatch: (count: number) => void;
  setAvgDocumentSize: (size: number) => void;
  setChunkSize: (size: number) => void;
  setChunkOverlap: (overlap: number) => void;
  setRerankingBatchSize: (size: number) => void;
  setNumQueries: (count: number) => void;
  setDocsPerQuery: (count: number) => void;
  setMaxQueryLength: (length: number) => void;
  setMaxDocLength: (length: number) => void;
  calculateResults: () => void;

  // Partitioning Actions
  setPartitioningGPU: (gpuId: string | null) => void;
  setPartitioningMode: (mode: PartitionModeType) => void;
  setPartitioningModelType: (type: PartitionModelSelection) => void;
  setPartitioningShowOnlyFits: (show: boolean) => void;
}

export type AppStore = AppState & AppActions;