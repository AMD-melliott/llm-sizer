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

  // Vision-language models may have nested text config
  text_config?: HFModelConfig;
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
 * Model entry in our models.json format
 */
export interface ModelEntry {
  id: string;
  name: string;
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
