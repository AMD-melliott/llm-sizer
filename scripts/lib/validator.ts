import Ajv from 'ajv';
import { ModelEntry, ValidationResult } from './types.js';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const ajv = new Ajv();
const schemaPath = join(__dirname, '..', 'schemas', 'model.schema.json');
const schema = JSON.parse(readFileSync(schemaPath, 'utf-8'));
const validate = ajv.compile(schema);

/**
 * Validate a model entry against the schema and business rules
 */
export function validateModel(model: ModelEntry): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // JSON Schema validation
  const valid = validate(model);
  if (!valid && validate.errors) {
    validate.errors.forEach(error => {
      errors.push(`${error.instancePath} ${error.message}`);
    });
  }

  // Business rule validations
  if (model.num_kv_heads && model.num_kv_heads > model.num_heads) {
    errors.push('num_kv_heads cannot be greater than num_heads');
  }

  // Check for MoE specific fields
  if (model.architecture === 'moe') {
    if (!model.num_experts) {
      warnings.push('MoE model missing num_experts field');
    }
    if (!model.experts_per_token) {
      warnings.push('MoE model missing experts_per_token field');
    }
  }

  // Validate hf_model_id format if present
  if (model.hf_model_id) {
    // Format: org/model or simple-name (for legacy models like gpt2)
    if (!model.hf_model_id.match(/^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$|^[a-z0-9-]+$/)) {
      errors.push('hf_model_id contains invalid characters or incorrect format');
    }
    
    // Should contain org prefix for most models
    const legacyModels = ['gpt2', 'distilbert-base-uncased', 'bert-base-uncased'];
    if (!model.hf_model_id.includes('/') && !legacyModels.includes(model.hf_model_id)) {
      warnings.push('hf_model_id should include organization prefix for clarity (e.g., "microsoft/Phi-4")');
    }
  }

  // Warn about missing optional fields
  if (!model.num_kv_heads && model.architecture === 'transformer') {
    warnings.push('Missing num_kv_heads (useful for GQA models)');
  }
  if (!model.intermediate_size) {
    warnings.push('Missing intermediate_size (useful for activation calculations)');
  }
  if (!model.vocab_size) {
    warnings.push('Missing vocab_size (useful for embedding calculations)');
  }

  // Multimodal model validations
  if (model.modality === 'multimodal') {
    if (!model.vision_config) {
      errors.push('Multimodal model must have vision_config');
    } else {
      // Validate vision config fields
      if (!model.vision_config.model_type) {
        errors.push('Vision config missing model_type');
      }
      if (!model.vision_config.image_size) {
        errors.push('Vision config missing image_size');
      }
      if (!model.vision_config.parameters_millions) {
        warnings.push('Vision config missing parameters_millions');
      }
    }

    if (!model.multimodal_config) {
      warnings.push('Multimodal model missing multimodal_config');
    } else {
      // Validate multimodal config fields
      if (!model.multimodal_config.image_token_count) {
        warnings.push('Multimodal config missing image_token_count');
      }
      if (!model.multimodal_config.projector_type) {
        warnings.push('Multimodal config missing projector_type');
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Check if a model ID already exists in the models list
 */
export function isDuplicateModel(modelId: string, existingModels: ModelEntry[]): boolean {
  return existingModels.some(m => m.id === modelId);
}

/**
 * Check if a HuggingFace model ID already exists in the models list
 */
export function isDuplicateHfModel(hfModelId: string, existingModels: ModelEntry[]): boolean {
  return existingModels.some(m => m.hf_model_id === hfModelId);
}
