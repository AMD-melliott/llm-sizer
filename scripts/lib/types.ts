/**
 * HuggingFace model info from SDK
 */
export interface ModelInfo {
  id: string;
  name: string;
  safetensors?: {
    parameters?: Record<string, number>;
    total?: number;
  };
  tags?: string[];
  [key: string]: unknown;
}

/**
 * Vision encoder configuration
 */
export interface VisionConfig {
  model_type?: string;  // 'clip_vision_tower', 'siglip', 'eva', 'davit', etc.
  image_size?: number;  // Default input resolution
  patch_size?: number | number[];  // Vision transformer patch size (array for hierarchical models)
  num_channels?: number;  // Usually 3 for RGB
  hidden_size?: number;  // Vision encoder hidden dimension
  num_hidden_layers?: number;  // Vision encoder depth
  num_attention_heads?: number;  // Vision encoder attention heads
  intermediate_size?: number;  // Vision FFN size
  num_positions?: number;  // Max position embeddings
  projection_dim?: number;  // Output projection dimension
}

/**
 * HuggingFace model configuration from config.json
 */
export interface HFModelConfig {
  // From config.json
  architectures?: string[];

  // Standard naming (Llama, Mistral, etc.)
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  max_position_embeddings?: number;
  vocab_size?: number;
  model_type?: string;

  // GPT-2 style naming
  n_embd?: number;
  n_layer?: number;
  n_head?: number;
  n_positions?: number;
  n_ctx?: number;

  // MoE specific
  num_local_experts?: number;
  num_experts?: number;
  num_experts_per_tok?: number;

  // Other potential fields
  max_sequence_length?: number;
  sliding_window?: number;

  // Vision-language models configurations
  text_config?: HFModelConfig;
  vision_config?: VisionConfig;
  is_vision_language_model?: boolean;
  image_processor?: string;
  vision_feature_layer?: number;
  vision_feature_select_strategy?: string;
  image_token_index?: number;

  // Multimodal projector config
  mm_hidden_size?: number;
  mm_projector_type?: string;
  mm_vision_tower?: string;
}

/**
 * Combined HuggingFace model data from SDK
 */
export interface HFModelData {
  info: ModelInfo;
  config: HFModelConfig;
  parametersBillions?: number;
}

/**
 * Multimodal configuration for vision-language models
 */
export interface MultimodalConfig {
  image_token_count?: number;  // Tokens per image
  max_images?: number;  // Max images per prompt
  projector_type?: string;  // 'linear', 'mlp', 'resampler'
  projector_params_millions?: number;  // Projector parameters
  merge_strategy?: string;  // 'concatenate', 'cross_attention', 'prefix'
  supports_video?: boolean;  // Video frame support
  frames_per_second?: number;  // For video models
}

/**
 * Model entry in our models.json format
 */
export interface ModelEntry {
  id: string;
  name: string;
  hf_model_id?: string;  // Full HuggingFace model ID with org prefix (e.g., "meta-llama/Llama-3-70b")
  parameters_billions: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  default_context_length: number;
  architecture: 'transformer' | 'moe';

  // Optional fields
  num_kv_heads?: number;
  intermediate_size?: number;
  vocab_size?: number;
  num_experts?: number;
  experts_per_token?: number;

  // Multimodal fields
  modality?: 'text' | 'multimodal';  // Default to 'text' for backward compatibility
  vision_config?: {
    model_type: string;  // Vision encoder type
    image_size: number;  // Default input resolution
    patch_size: number | number[];  // Vision transformer patch size (array for hierarchical models)
    num_channels?: number;  // Usually 3 for RGB
    hidden_size: number;  // Vision encoder hidden dimension
    num_layers: number;  // Vision encoder depth
    num_heads: number;  // Vision encoder attention heads
    intermediate_size?: number;  // Vision FFN size
    parameters_millions: number;  // Vision encoder params in millions
  };
  multimodal_config?: MultimodalConfig;
}

/**
 * CLI options for the import script
 */
export interface ImportOptions {
  url?: string;
  model?: string;
  file?: string;
  dryRun?: boolean;
  params?: number;
  context?: number;
  force?: boolean;
}

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Import result
 */
export interface ImportResult {
  success: boolean;
  model?: ModelEntry;
  errors: string[];
  warnings: string[];
}
