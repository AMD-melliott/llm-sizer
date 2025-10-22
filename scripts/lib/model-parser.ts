import { HFModelConfig, ModelEntry, VisionConfig, MultimodalConfig } from './types.js';
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
 * Detect if model is multimodal
 */
export function isMultimodal(config: HFModelConfig): boolean {
  return !!(
    config.vision_config ||
    // @ts-ignore - img_processor exists on Phi-3-vision
    config.img_processor ||
    config.is_vision_language_model ||
    config.model_type?.includes('vision') ||
    config.model_type?.includes('image') ||
    config.model_type === 'image-text-to-text' ||
    config.model_type === 'llava' ||
    config.model_type === 'llava_next' ||
    config.model_type === 'florence2' ||
    config.model_type === 'qwen_vl' ||
    config.model_type === 'blip' ||
    config.model_type === 'clip' ||
    config.model_type === 'siglip' ||
    config.model_type === 'phi3_v' ||
    config.model_type === 'paligemma' ||
    config.model_type === 'idefics' ||
    config.model_type === 'idefics2'
  );
}

/**
 * Extract vision encoder configuration
 */
export function extractVisionConfig(config: HFModelConfig): ModelEntry['vision_config'] | undefined {
  // @ts-ignore - img_processor exists on Phi-3-vision and similar models
  const imgProcessor = config.img_processor;
  
  // Handle standard vision_config
  if (config.vision_config) {
    const vc = config.vision_config;
    const visionParams = calculateVisionParameters(vc);

    return {
      model_type: vc.model_type || 'vision_transformer',
      image_size: vc.image_size || 224,
      patch_size: vc.patch_size || 16,
      num_channels: vc.num_channels || 3,
      hidden_size: vc.hidden_size || 768,
      num_layers: vc.num_hidden_layers || 12,
      num_heads: vc.num_attention_heads || 12,
      intermediate_size: vc.intermediate_size,
      parameters_millions: visionParams
    };
  }
  
  // Handle img_processor (Phi-3-vision style)
  if (imgProcessor) {
    // @ts-ignore
    const embedDim = imgProcessor.embd_dim || imgProcessor.embed_dim || 1024;
    // @ts-ignore
    const numHiddenLayers = imgProcessor.num_img_tokens || 144; // Approximate from token count
    // @ts-ignore
    const imageSize = imgProcessor.image_size || 336;
    // @ts-ignore
    const patchSize = imgProcessor.patch_size || 14;

    // Create a pseudo vision config for parameter calculation
    const pseudoVc: VisionConfig = {
      hidden_size: embedDim,
      num_hidden_layers: 24, // CLIP ViT-L/14 standard
      num_attention_heads: embedDim / 64,
      intermediate_size: embedDim * 4,
      image_size: imageSize,
      patch_size: patchSize,
      num_channels: 3
    };

    const visionParams = calculateVisionParameters(pseudoVc);

    return {
      model_type: 'clip_vision_tower',
      image_size: imageSize,
      patch_size: patchSize,
      num_channels: 3,
      hidden_size: embedDim,
      num_layers: 24,
      num_heads: Math.round(embedDim / 64),
      intermediate_size: embedDim * 4,
      parameters_millions: visionParams
    };
  }

  return undefined;
}

/**
 * Calculate vision encoder parameters in millions
 * 
 * NOTE: This function provides an estimation of vision encoder parameters
 * based on common architectural patterns (ViT, CLIP, etc.). Some values
 * use standard conventions:
 * - intermediate_size defaults to hidden_size * 4 (common in transformers)
 * - Position embeddings are approximated from image/patch size
 * - Layer parameters follow standard attention + FFN structure
 * 
 * Actual parameter counts may vary slightly based on implementation details
 * like bias terms, additional projections, or architectural variations.
 */
function calculateVisionParameters(vc: VisionConfig): number {
  const hidden = vc.hidden_size || 768;
  const layers = vc.num_hidden_layers || 12;
  // Intermediate size commonly follows 4x hidden dimension convention
  const intermediate = vc.intermediate_size || hidden * 4;
  const channels = vc.num_channels || 3;

  // Handle standard patch size (single number) or hierarchical patch sizes (array)
  let patchSize: number;
  if (Array.isArray(vc.patch_size)) {
    // For DaViT and similar hierarchical architectures, use first patch size
    // as approximation for parameter calculation
    patchSize = vc.patch_size[0] || 16;
  } else {
    patchSize = vc.patch_size || 16;
  }

  // Patch embedding layer: projects patches to hidden dimension
  const patchEmbed = channels * patchSize * patchSize * hidden;

  // Position embeddings (approximate based on max sequence length)
  const maxPositions = vc.num_positions || ((vc.image_size || 224) / patchSize) ** 2;
  const posEmbed = maxPositions * hidden;

  // Transformer layers: attention (QKV + output) + FFN (2 linear layers)
  const perLayer = (
    4 * hidden * hidden +  // QKV projections + output projection
    2 * hidden * intermediate  // FFN: up projection + down projection
  );
  const transformerParams = layers * perLayer;

  // Layer norms and final normalization (2 per layer + 1 final)
  const otherParams = layers * 2 * hidden + hidden;

  const totalParams = patchEmbed + posEmbed + transformerParams + otherParams;
  return Math.round(totalParams / 1_000_000);
}

/**
 * Extract multimodal configuration from HuggingFace config
 * 
 * This function extracts parameters specific to vision-language model integration:
 * - Image token count: Number of tokens generated per image (varies by resolution/patch size)
 * - Projector parameters: Size of the adapter mapping vision to language space
 * - Max images: Maximum number of images supported per prompt
 * 
 * NOTE: Multimodal architectures vary significantly:
 * - LLaVA: Uses CLIP vision tower + MLP projector
 * - Phi-3-vision: Uses CLIP with img_processor config format
 * - Florence-2: Uses DaViT encoder with hierarchical patches
 * - Qwen-VL: Custom vision encoder with different tokenization
 * 
 * Estimations are based on common patterns but may not capture all architectural details.
 */
export function extractMultimodalConfig(config: HFModelConfig): MultimodalConfig | undefined {
  if (!isMultimodal(config)) {
    return undefined;
  }

  // @ts-ignore - img_processor exists on Phi-3-vision
  const imgProcessor = config.img_processor;

  // Calculate image token count based on vision config
  // Default: 336px with patch_size=14 → (336/14)² = 576 tokens
  let imageTokenCount = 576;
  
  if (config.vision_config) {
    const imageSize = config.vision_config.image_size || 224;
    let patchSize: number;
    
    // Handle array patch sizes (DaViT, etc.)
    if (Array.isArray(config.vision_config.patch_size)) {
      patchSize = config.vision_config.patch_size[0] || 16;
    } else {
      patchSize = config.vision_config.patch_size || 16;
    }
    
    imageTokenCount = Math.floor((imageSize / patchSize) ** 2);
  } else if (imgProcessor) {
    // @ts-ignore
    imageTokenCount = imgProcessor.num_img_tokens || 144;
  }

  // Estimate projector parameters (maps vision features to language model space)
  // NOTE: These are approximations based on common projector architectures
  // Actual implementations may vary in depth, width, and additional components
  let visionHidden = 768;
  if (config.vision_config) {
    visionHidden = config.vision_config.hidden_size || 768;
  } else if (imgProcessor) {
    // @ts-ignore
    visionHidden = imgProcessor.embd_dim || imgProcessor.embed_dim || 1024;
  }

  const textHidden = config.text_config?.hidden_size || config.hidden_size || 4096;
  const projectorType = config.mm_projector_type || 'mlp';

  let projectorParams = 0;
  if (projectorType === 'linear') {
    // Single linear projection layer
    projectorParams = visionHidden * textHidden / 1_000_000;
  } else if (projectorType === 'mlp') {
    // 2-layer MLP projector (common in LLaVA, Phi-3-vision)
    // Assumes: vision_hidden -> text_hidden -> text_hidden
    projectorParams = (visionHidden * textHidden * 2 + textHidden * textHidden) / 1_000_000;
  } else if (projectorType === 'resampler') {
    // Resampler/perceiver projector (more complex, used in some models)
    // Rough estimation: 3x parameter count of linear projection
    projectorParams = (visionHidden * textHidden * 3) / 1_000_000;
  }

  return {
    image_token_count: imageTokenCount,
    max_images: imgProcessor?.max_num_images || 1,  // Default, can be overridden
    projector_type: projectorType,
    projector_params_millions: Math.round(projectorParams * 10) / 10,
    merge_strategy: 'concatenate',
    supports_video: config.model_type?.includes('video') || false
  };
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
  // Detect if this is a multimodal model
  const isMultimodalModel = isMultimodal(config);

  // Handle vision-language models with nested text_config
  // @ts-ignore - text_config may exist on some models
  const textConfig = config.text_config || config;

  const architecture = detectArchitecture(config);

  // Handle different naming conventions (standard vs GPT-2 style vs encoder-decoder)
  // For multimodal models with incomplete text_config, use defaults based on model type
  // @ts-ignore - d_model, decoder_layers, decoder_attention_heads exist on encoder-decoder models
  let hidden_size = textConfig.hidden_size || textConfig.n_embd || textConfig.d_model;
  // @ts-ignore
  let num_layers = textConfig.num_hidden_layers || textConfig.n_layer || textConfig.decoder_layers;
  // @ts-ignore
  let num_heads = textConfig.num_attention_heads || textConfig.n_head || textConfig.decoder_attention_heads;
  let context_length = textConfig.max_position_embeddings ||
                      textConfig.n_positions ||
                      textConfig.n_ctx ||
                      textConfig.max_sequence_length;

  // For LLaVA models with Llama/Vicuna text backbone, apply known defaults
  if (isMultimodalModel && config.model_type === 'llava' && !hidden_size) {
    // LLaVA typically uses Vicuna-7B or Vicuna-13B as text backbone
    // Use 7B defaults if not specified
    hidden_size = 4096;
    num_layers = 32;
    num_heads = 32;
    console.log(chalk.gray('Note: Using Llama/Vicuna-7B defaults for text model configuration'));
  }

  const model: Partial<ModelEntry> = {
    id: generateModelId(modelId),
    name: generateModelName(modelName),
    hidden_size,
    num_layers,
    num_heads,
    default_context_length: context_length,
    architecture,
    modality: isMultimodalModel ? 'multimodal' : 'text',
  };

  // Add multimodal configurations if applicable
  if (isMultimodalModel) {
    const visionConfig = extractVisionConfig(config);
    if (visionConfig) {
      model.vision_config = visionConfig;
    }

    const multimodalConfig = extractMultimodalConfig(config);
    if (multimodalConfig) {
      model.multimodal_config = multimodalConfig;
    }
  }

  // Optional fields
  if (textConfig.num_key_value_heads) {
    model.num_kv_heads = textConfig.num_key_value_heads;
  } else if (isMultimodalModel && config.model_type === 'llava') {
    // LLaVA with Llama/Vicuna backbone uses GQA
    model.num_kv_heads = 32;
  }

  // @ts-ignore - decoder_ffn_dim exists on encoder-decoder models
  if (textConfig.intermediate_size || textConfig.decoder_ffn_dim) {
    // @ts-ignore
    model.intermediate_size = textConfig.intermediate_size || textConfig.decoder_ffn_dim;
  } else if (isMultimodalModel && config.model_type === 'llava') {
    // LLaVA with Llama/Vicuna backbone
    model.intermediate_size = 11008;
  }

  if (textConfig.vocab_size || config.vocab_size) {
    model.vocab_size = textConfig.vocab_size || config.vocab_size;
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
  console.log(chalk.cyan(`  Modality:        ${model.modality || 'text'} ${model.modality === 'multimodal' ? '🖼️' : ''}`));
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

  // Display vision configuration if present
  if (model.vision_config) {
    console.log(chalk.bold('\n  Vision Encoder:'));
    console.log(chalk.yellow(`    Type:          ${model.vision_config.model_type}`));
    console.log(chalk.yellow(`    Image Size:    ${model.vision_config.image_size}px`));
    const patchSizeStr = Array.isArray(model.vision_config.patch_size)
      ? model.vision_config.patch_size.join(',')
      : model.vision_config.patch_size;
    console.log(chalk.yellow(`    Patch Size:    ${patchSizeStr}px`));
    console.log(chalk.yellow(`    Hidden Size:   ${model.vision_config.hidden_size}`));
    console.log(chalk.yellow(`    Layers:        ${model.vision_config.num_layers}`));
    console.log(chalk.yellow(`    Heads:         ${model.vision_config.num_heads}`));
    console.log(chalk.yellow(`    Parameters:    ${model.vision_config.parameters_millions}M`));
  }

  // Display multimodal configuration if present
  if (model.multimodal_config) {
    console.log(chalk.bold('\n  Multimodal Config:'));
    console.log(chalk.magenta(`    Image Tokens:  ${model.multimodal_config.image_token_count}`));
    console.log(chalk.magenta(`    Max Images:    ${model.multimodal_config.max_images}`));
    console.log(chalk.magenta(`    Projector:     ${model.multimodal_config.projector_type}`));
    console.log(chalk.magenta(`    Proj. Params:  ${model.multimodal_config.projector_params_millions}M`));
    if (model.multimodal_config.supports_video) {
      console.log(chalk.magenta(`    Video Support: Yes`));
    }
  }
}
