import { listModels, downloadFile } from "@huggingface/hub";
import { HFModelConfig } from './types.js';
import chalk from 'chalk';

// Define a subset of ApiModelInfo that we actually use
// This avoids importing the full type which may not be exported properly
interface ModelInfo {
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
 * Get credentials for HuggingFace API (for gated models)
 */
function getCredentials() {
  return process.env.HF_TOKEN
    ? { accessToken: process.env.HF_TOKEN }
    : undefined;
}

/**
 * Parse HuggingFace model ID from URL or direct ID
 * Examples:
 *   - "https://huggingface.co/meta-llama/Llama-3.3-70B" → "meta-llama/Llama-3.3-70B"
 *   - "meta-llama/Llama-3.3-70B" → "meta-llama/Llama-3.3-70B"
 *   - "gpt2" → "openai-community/gpt2" (simple model names)
 */
export function parseModelId(input: string): string {
  const urlPattern = /huggingface\.co\/([^\/]+\/[^\/\?#]+)/;
  const match = input.match(urlPattern);
  
  if (match) {
    return match[1];
  }
  
  // If it has a slash, assume it's already a model ID
  if (input.includes('/')) {
    return input;
  }
  
  // For simple model names without organization, return as-is
  // The SDK will handle resolving the full path
  return input;
}

/**
 * Fetch model info from HuggingFace using the official SDK
 */
export async function fetchModelInfo(modelId: string): Promise<ModelInfo> {
  const credentials = getCredentials();
  
  console.log(chalk.gray(`Fetching model info for: ${modelId}`));
  
  try {
    // Prepare search parameters
    let owner: string | undefined;
    let query: string;
    
    if (modelId.includes('/')) {
      // Extract owner and model name from ID
      const [ownerPart, ...nameParts] = modelId.split('/');
      owner = ownerPart;
      query = nameParts.join('/');
    } else {
      // Simple model name, search without owner filter
      query = modelId;
    }
    
    // Use listModels with specific search to get the model info
    const models = listModels({
      search: {
        ...(owner && { owner }),
        query
      },
      credentials,
      additionalFields: ["safetensors", "cardData", "config", "transformersInfo", "tags", "library_name"],
      limit: 10 // Get a few to find exact match
    });
    
    // Find exact match
    for await (const model of models) {
      // Match by name (which is the full ID like "openai-community/gpt2")
      // or by the query if it matches the end of the name
      if (model.name === modelId || 
          model.id === modelId || 
          model.name.endsWith(`/${modelId}`) ||
          model.name === query) {
        return model as unknown as ModelInfo;
      }
    }
    
    throw new Error(`Model not found: ${modelId}`);
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('401')) {
        throw new Error(
          'Authentication required. Set HF_TOKEN environment variable.\n' +
          'Get your token at: https://huggingface.co/settings/tokens'
        );
      } else if (error.message.includes('404') || error.message.includes('not found')) {
        throw new Error(`Model not found: ${modelId}`);
      }
    }
    throw new Error(`Failed to fetch model info: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Fetch model config from HuggingFace using the official SDK
 * This will download config.json from the model repository
 */
export async function fetchModelConfig(modelId: string): Promise<HFModelConfig> {
  const credentials = getCredentials();
  
  console.log(chalk.gray(`Fetching config.json for: ${modelId}`));
  
  try {
    const configResponse = await downloadFile({
      repo: modelId,
      path: "config.json",
      credentials
    });
    
    if (!configResponse) {
      throw new Error('Config file not found');
    }
    
    const config = await configResponse.json() as HFModelConfig;
    return config;
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('401')) {
        throw new Error(
          'Authentication required. Set HF_TOKEN environment variable.\n' +
          'Get your token at: https://huggingface.co/settings/tokens'
        );
      } else if (error.message.includes('404')) {
        throw new Error(`Config file not found for model: ${modelId}`);
      }
    }
    throw new Error(`Failed to fetch model config: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Fetch parameter count from model info
 * This uses the safetensors metadata or model ID/tags as fallback
 */
export async function fetchParameterCount(modelId: string): Promise<number | null> {
  try {
    const info = await fetchModelInfo(modelId);
    
    // Try to get from safetensors metadata
    if (info.safetensors?.total) {
      const paramsBillions = info.safetensors.total / 1_000_000_000;
      console.log(chalk.gray(`Found parameter count from safetensors: ${paramsBillions}B`));
      return paramsBillions;
    }
    
    // Try to extract from model ID
    const idMatch = modelId.match(/(\d+\.?\d*)([bm])/i);
    if (idMatch) {
      const value = parseFloat(idMatch[1]);
      const unit = idMatch[2].toLowerCase();
      const paramsBillions = unit === 'b' ? value : value / 1000;
      console.log(chalk.gray(`Extracted parameter count from model ID: ${paramsBillions}B`));
      return paramsBillions;
    }
    
    // Try to extract from tags
    if (info.tags) {
      for (const tag of info.tags) {
        const match = tag.match(/(\d+\.?\d*)([bm])/i);
        if (match) {
          const value = parseFloat(match[1]);
          const unit = match[2].toLowerCase();
          const paramsBillions = unit === 'b' ? value : value / 1000;
          console.log(chalk.gray(`Extracted parameter count from tags: ${paramsBillions}B`));
          return paramsBillions;
        }
      }
    }
    
    console.warn(chalk.yellow('Could not determine parameter count from model info'));
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
    await fetchModelInfo(modelId);
    return true;
  } catch (error) {
    return false;
  }
}
