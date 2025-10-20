import { HFModelConfig } from './types.js';
import chalk from 'chalk';

/**
 * Parse HuggingFace model ID from URL or direct ID
 * Examples:
 *   - "https://huggingface.co/meta-llama/Llama-3.3-70B" → "meta-llama/Llama-3.3-70B"
 *   - "meta-llama/Llama-3.3-70B" → "meta-llama/Llama-3.3-70B"
 */
export function parseModelId(input: string): string {
  const urlPattern = /huggingface\.co\/([^\/]+\/[^\/\?#]+)/;
  const match = input.match(urlPattern);
  
  if (match) {
    return match[1];
  }
  
  // Assume it's already a model ID
  if (input.includes('/')) {
    return input;
  }
  
  throw new Error(`Invalid model URL or ID: ${input}`);
}

/**
 * Fetch model config from HuggingFace
 * This will attempt to fetch config.json from the model repository
 */
export async function fetchModelConfig(modelId: string): Promise<HFModelConfig> {
  const configUrl = `https://huggingface.co/${modelId}/resolve/main/config.json`;
  
  console.log(chalk.gray(`Fetching config from: ${configUrl}`));
  
  try {
    const response = await fetch(configUrl);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const config = await response.json() as HFModelConfig;
    return config;
  } catch (error) {
    throw new Error(`Failed to fetch model config: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Fetch parameter count from model card or API
 * This is a fallback when config.json doesn't have the info
 */
export async function fetchParameterCount(modelId: string): Promise<number | null> {
  try {
    // Try to get from HuggingFace API
    const apiUrl = `https://huggingface.co/api/models/${modelId}`;
    const response = await fetch(apiUrl);
    
    if (!response.ok) {
      return null;
    }
    
    const data = await response.json();
    
    // Look for parameter count in safetensors metadata or model card
    if (data.safetensors?.parameters) {
      const params = data.safetensors.parameters;
      // Handle if it's a number or an object
      if (typeof params === 'number') {
        // Convert from absolute count to billions
        return params / 1_000_000_000;
      } else if (typeof params === 'object' && params.total) {
        return params.total / 1_000_000_000;
      }
    }
    
    // Try to extract from model ID or tags
    const idMatch = modelId.match(/(\d+\.?\d*)([bm])/i);
    if (idMatch) {
      const value = parseFloat(idMatch[1]);
      const unit = idMatch[2].toLowerCase();
      return unit === 'b' ? value : value / 1000;
    }
    
    // Try to extract from tags
    if (data.tags) {
      for (const tag of data.tags) {
        const match = tag.match(/(\d+\.?\d*)([bm])/i);
        if (match) {
          const value = parseFloat(match[1]);
          const unit = match[2].toLowerCase();
          return unit === 'b' ? value : value / 1000;
        }
      }
    }
    
    return null;
  } catch (error) {
    console.warn(chalk.yellow(`Could not fetch parameter count: ${error instanceof Error ? error.message : String(error)}`));
    return null;
  }
}

/**
 * Validate that model exists and is accessible
 */
export async function validateModelExists(modelId: string): Promise<boolean> {
  try {
    const apiUrl = `https://huggingface.co/api/models/${modelId}`;
    const response = await fetch(apiUrl);
    return response.ok;
  } catch (error) {
    return false;
  }
}
