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
  hf_model_id?: string; // Full HuggingFace model ID with org prefix (e.g., "meta-llama/Llama-3-70b")
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
  setResults: (results: CalculationResults | null) => void;

  // Partitioning Actions
  setPartitioningGPU: (gpuId: string | null) => void;
  setPartitioningMode: (mode: PartitionModeType) => void;
  setPartitioningModelType: (type: PartitionModelSelection) => void;
  setPartitioningShowOnlyFits: (show: boolean) => void;
}

export type AppStore = AppState & AppActions;

// Container Configuration Types
export interface EngineParameter {
  flag: string;
  type: 'string' | 'number' | 'boolean' | 'select';
  required: boolean;
  description: string;
  default?: string | number | boolean | null;
  options?: EngineParameterOption[];
  source?: 'calculator' | 'manual' | 'computed';
  validation?: {
    min?: number;
    max?: number;
    mustMatchGpuCount?: boolean;
  };
  security_warning?: string;
}

export interface EngineParameterOption {
  value: string;
  label: string;
  description: string;
}

export interface InferenceEngine {
  id: string;
  name: string;
  version: string;
  description: string;
  documentation: string;
  parameters: EngineParameter[];
}

export interface EngineParametersData {
  engines: InferenceEngine[];
}

export interface ContainerImagesData {
  images: ContainerImage[];
}

export interface ContainerImageRequirements {
  minDockerVersion: string;
  requiresContainerToolkit: boolean;
  recommendsContainerToolkit: boolean;
}

export interface ContainerImage {
  engine: string;
  repository: string;
  tag: string;
  fullImage: string;
  entrypoint?: string; // Command to run after the container image (e.g., "python3 -m vllm.entrypoints.openai.api_server")
  stability: 'stable' | 'nightly' | 'experimental';
  rocmVersion: string;
  pythonVersion: string;
  description: string;
  features: string[];
  warnings?: string[];
  requirements: ContainerImageRequirements;
}

export interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readOnly: boolean;
  description?: string;
}

export interface EnvironmentVariable {
  key: string;
  value: string;
  description?: string;
  sensitive?: boolean;
}

export interface PortMapping {
  host: number;
  container: number;
  protocol?: 'tcp' | 'udp';
  description?: string;
}

export interface ResourceLimits {
  shmSize: string;
  memoryLimit?: string;
  cpuLimit?: string;
}

export interface ContainerConfig {
  // Engine and Image
  engine: InferenceEngine;
  image: ContainerImage;
  
  // Container Runtime
  containerName: string;
  useContainerToolkit: boolean;
  autoRemove: boolean;
  
  // GPU Configuration
  gpuIds: string[];
  gpuCount: number;
  
  // Resources
  shmSize: string;
  resourceLimits?: ResourceLimits;
  
  // Volumes
  volumes: VolumeMount[];
  
  // Environment
  environment: EnvironmentVariable[];
  
  // Network
  ports: PortMapping[];
  useHostNetwork: boolean;
  
  // Engine Parameters (mapped from calculator and custom)
  engineParams: Array<{
    flag: string;
    value: string | number | boolean;
  }>;
  
  // Model information from calculator
  model: {
    id: string;
    name: string;
    parameters: number;
  };
  
  // GPU information from calculator
  gpus: Array<{
    id: string;
    name: string;
    vram: number;
  }>;
  
  // Memory calculation from calculator
  memoryUsage: {
    estimated: number;
    available: number;
    percentage: number;
  };
}

export type ValidationLevel = 'error' | 'warning' | 'info' | 'success';

export interface ValidationMessage {
  level: ValidationLevel;
  message: string;
  suggestion?: string;
  field?: string;
}

export interface ConfigValidationResult {
  valid: boolean;
  messages: ValidationMessage[];
  securityIssues: ValidationMessage[];
  recommendations: ValidationMessage[];
}

export type ConfigOutputFormat = 'docker-run' | 'docker-compose';

// Container Store State
export interface ContainerState {
  // Configuration
  selectedEngineId: string;
  selectedImageId: string;
  useContainerToolkit: boolean;
  containerName: string;
  
  // Volumes
  modelPath: string;
  mountModelPath: boolean;
  mountHFCache: boolean;
  customVolumes: VolumeMount[];
  
  // Environment
  customEnvironment: EnvironmentVariable[];
  
  // Ports
  apiPort: number;
  customPorts: PortMapping[];
  
  // Advanced Options
  useHostNetwork: boolean;
  customShmSize?: string;
  trustRemoteCode: boolean;
  enableHealthcheck: boolean;
  autoRemoveContainer: boolean;
  
  // Engine Parameters (custom overrides)
  customEngineParams: Map<string, string | number | boolean>;
  
  // Output
  outputFormat: ConfigOutputFormat;
  
  // Generated config
  generatedConfig: ContainerConfig | null;
  validationResult: ConfigValidationResult | null;
}

// Container Store Actions
export interface ContainerActions {
  // Engine and Image Selection
  setSelectedEngineId: (engineId: string) => void;
  setSelectedImageId: (imageId: string) => void;
  
  // Container Runtime
  setUseContainerToolkit: (use: boolean) => void;
  setContainerName: (name: string) => void;
  
  // Volumes
  setModelPath: (path: string) => void;
  setMountModelPath: (mount: boolean) => void;
  setMountHFCache: (mount: boolean) => void;
  addCustomVolume: (volume: VolumeMount) => void;
  removeCustomVolume: (index: number) => void;
  
  // Environment
  addEnvironmentVariable: (env: EnvironmentVariable) => void;
  removeEnvironmentVariable: (index: number) => void;
  
  // Ports
  setApiPort: (port: number) => void;
  addCustomPort: (port: PortMapping) => void;
  removeCustomPort: (index: number) => void;
  
  // Advanced Options
  setUseHostNetwork: (use: boolean) => void;
  setCustomShmSize: (size: string | undefined) => void;
  setTrustRemoteCode: (trust: boolean) => void;
  setEnableHealthcheck: (enable: boolean) => void;
  setAutoRemoveContainer: (autoRemove: boolean) => void;
  
  // Engine Parameters
  setCustomEngineParam: (flag: string, value: string | number | boolean | null) => void;
  clearCustomEngineParam: (flag: string) => void;
  
  // Output
  setOutputFormat: (format: ConfigOutputFormat) => void;
  
  // Generation
  generateConfig: () => void;
  validateConfig: (config: ContainerConfig) => ConfigValidationResult;
  
  // Reset
  resetToDefaults: () => void;
}

export type ContainerStore = ContainerState & ContainerActions;