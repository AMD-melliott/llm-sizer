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
