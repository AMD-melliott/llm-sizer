import { HFModelConfig, ModelEntry } from './types.js';
import chalk from 'chalk';

/**
 * Parse parameter count string to number in billions
 * Handles formats like "7B", "70B", "7000M", "70000000000"
 */
export function parseParameterCount(input: string | number): number {
  if (typeof input === 'number') {
    // If already a number, assume it's in absolute count
    if (input < 1000) return input; // Already in billions
    return input / 1_000_000_000; // Convert to billions
  }

  const str = input.toUpperCase().trim();
  
  if (str.endsWith('B')) {
    return parseFloat(str.replace('B', ''));
  }
  
  if (str.endsWith('M')) {
    return parseFloat(str.replace('M', '')) / 1000;
  }
  
  // Assume raw number
  const num = parseFloat(str);
  if (num < 1000) return num;
  return num / 1_000_000_000;
}

/**
 * Generate a model ID from the model name/repo
 * Example: "meta-llama/Llama-3.3-70B" → "llama-3-70b"
 */
export function generateModelId(modelName: string): string {
  // Remove org prefix if present
  const name = modelName.includes('/') ? modelName.split('/')[1] : modelName;
  
  return name
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

/**
 * Generate a friendly display name
 * Example: "meta-llama/Llama-3.3-70B" → "Llama 3.3 70B"
 */
export function generateModelName(modelName: string): string {
  // Remove org prefix if present
  const name = modelName.includes('/') ? modelName.split('/')[1] : modelName;
  
  // Clean up and format
  return name
    .replace(/-/g, ' ')
    .replace(/\./g, '.')
    .trim();
}

/**
 * Determine architecture type based on config
 */
export function detectArchitecture(config: HFModelConfig): 'transformer' | 'moe' {
  // Handle vision-language models with nested text_config
  // @ts-ignore - text_config may exist on some models
  const textConfig = config.text_config || config;
  
  // Check for MoE indicators
  if (textConfig.num_local_experts || textConfig.num_experts_per_tok || textConfig.num_experts) {
    return 'moe';
  }
  
  if (config.architectures) {
    const arch = config.architectures[0]?.toLowerCase() || '';
    if (arch.includes('mixtral') || arch.includes('moe')) {
      return 'moe';
    }
  }
  
  return 'transformer';
}

/**
 * Transform HuggingFace config to our ModelEntry format
 */
export function transformHFConfig(
  modelId: string,
  modelName: string,
  config: HFModelConfig,
  overrides?: Partial<ModelEntry>
): Partial<ModelEntry> {
  // Handle vision-language models with nested text_config
  // @ts-ignore - text_config may exist on some models
  const textConfig = config.text_config || config;
  
  const architecture = detectArchitecture(config);
  
  // Handle different naming conventions (standard vs GPT-2 style)
  const hidden_size = textConfig.hidden_size || textConfig.n_embd;
  const num_layers = textConfig.num_hidden_layers || textConfig.n_layer;
  const num_heads = textConfig.num_attention_heads || textConfig.n_head;
  const context_length = textConfig.max_position_embeddings || 
                        textConfig.n_positions || 
                        textConfig.n_ctx ||
                        textConfig.max_sequence_length;
  
  const model: Partial<ModelEntry> = {
    id: generateModelId(modelId),
    name: generateModelName(modelName),
    hidden_size,
    num_layers,
    num_heads,
    default_context_length: context_length,
    architecture,
  };

  // Optional fields
  if (textConfig.num_key_value_heads) {
    model.num_kv_heads = textConfig.num_key_value_heads;
  }
  
  if (textConfig.intermediate_size) {
    model.intermediate_size = textConfig.intermediate_size;
  }
  
  if (textConfig.vocab_size) {
    model.vocab_size = textConfig.vocab_size;
  }

  // MoE specific
  if (architecture === 'moe') {
    if (textConfig.num_local_experts || textConfig.num_experts) {
      model.num_experts = textConfig.num_local_experts || textConfig.num_experts;
    }
    if (textConfig.num_experts_per_tok) {
      model.experts_per_token = textConfig.num_experts_per_tok;
    }
  }

  // Apply overrides
  return { ...model, ...overrides };
}

/**
 * Display model information for user confirmation
 */
export function displayModelInfo(model: Partial<ModelEntry>): void {
  console.log(chalk.bold('\nExtracted Model Information:'));
  console.log(chalk.cyan(`  ID:              ${model.id}`));
  console.log(chalk.cyan(`  Name:            ${model.name}`));
  console.log(chalk.cyan(`  Parameters:      ${model.parameters_billions}B`));
  console.log(chalk.cyan(`  Hidden Size:     ${model.hidden_size}`));
  console.log(chalk.cyan(`  Layers:          ${model.num_layers}`));
  console.log(chalk.cyan(`  Attention Heads: ${model.num_heads}`));
  console.log(chalk.cyan(`  Context Length:  ${model.default_context_length}`));
  console.log(chalk.cyan(`  Architecture:    ${model.architecture}`));
  
  if (model.num_kv_heads) {
    console.log(chalk.gray(`  KV Heads:        ${model.num_kv_heads}`));
  }
  if (model.intermediate_size) {
    console.log(chalk.gray(`  Intermediate:    ${model.intermediate_size}`));
  }
  if (model.vocab_size) {
    console.log(chalk.gray(`  Vocab Size:      ${model.vocab_size}`));
  }
  if (model.num_experts) {
    console.log(chalk.gray(`  Num Experts:     ${model.num_experts}`));
  }
  if (model.experts_per_token) {
    console.log(chalk.gray(`  Experts/Token:   ${model.experts_per_token}`));
  }
}
