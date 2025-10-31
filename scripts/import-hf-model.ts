#!/usr/bin/env node

import { Command } from 'commander';
import chalk from 'chalk';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import * as readline from 'readline';
import { parseModelId, fetchModelConfig, fetchParameterCount, validateModelExists } from './lib/hf-client.js';
import { transformHFConfig, parseParameterCount, displayModelInfo, isMultimodal } from './lib/model-parser.js';
import { validateModel, isDuplicateModel, isDuplicateHfModel } from './lib/validator.js';
import { readModelsFile, writeModelsFile, createBackup, addModelToList, showDiff } from './lib/file-handler.js';
import { ImportOptions, ModelEntry } from './lib/types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const program = new Command();

program
  .name('import-hf-model')
  .description('Import HuggingFace models into the LLM Inference Calculator')
  .version('1.0.0')
  .option('-u, --url <url>', 'HuggingFace model URL')
  .option('-m, --model <id>', 'HuggingFace model ID (e.g., meta-llama/Llama-3.3-70B)')
  .option('-f, --file <path>', 'File containing list of model IDs (one per line)')
  .option('--dry-run', 'Preview changes without writing to file')
  .option('--params <number>', 'Override parameter count (in billions)', parseFloat)
  .option('--context <number>', 'Override context length', parseInt)
  .option('--force', 'Skip confirmation prompts')
  .parse(process.argv);

const options = program.opts<ImportOptions>();

/**
 * Main import logic for a single model
 */
async function importModel(modelInput: string, opts: ImportOptions): Promise<boolean> {
  try {
    console.log(chalk.bold(`\n📦 Importing model: ${modelInput}\n`));

    // Parse model ID
    const modelId = parseModelId(modelInput);
    console.log(chalk.gray(`Model ID: ${modelId}\n`));

    // Validate model exists
    console.log(chalk.gray('Validating model exists...'));
    const exists = await validateModelExists(modelId);
    if (!exists) {
      console.error(chalk.red(`❌ Model not found: ${modelId}`));
      return false;
    }
    console.log(chalk.green('✓ Model found\n'));

    // Fetch model config
    console.log(chalk.gray('Fetching model configuration...'));
    const config = await fetchModelConfig(modelId);
    console.log(chalk.green('✓ Config fetched\n'));

    // Fetch parameter count if not in config
    let paramCount = opts.params;
    if (!paramCount) {
      console.log(chalk.gray('Fetching parameter count...'));
      const fetchedParams = await fetchParameterCount(modelId);
      if (fetchedParams) {
        paramCount = fetchedParams;
        console.log(chalk.green(`✓ Found ${paramCount}B parameters\n`));
      } else {
        console.error(chalk.red('❌ Could not determine parameter count'));
        console.log(chalk.yellow('Please provide it with --params <number>'));
        return false;
      }
    }

    // Check if this is a multimodal model and provide appropriate handling
    const isMultimodalModel = isMultimodal(config);

    if (isMultimodalModel) {
      console.log(chalk.bold.yellow('🖼️  Multimodal model detected'));
      console.log(chalk.gray('Processing vision-language model configuration...'));

      // For multimodal models, we need to adjust parameter calculation
      // to account for vision encoder parameters
      if (!paramCount && config.vision_config) {
        console.log(chalk.yellow('Note: Parameter count will include both vision and language components'));
      }
    }

    // Transform config to our format
    const overrides: Partial<ModelEntry> = {
      parameters_billions: paramCount,
    };

    if (opts.context) {
      overrides.default_context_length = opts.context;
    }

    const modelData = transformHFConfig(modelId, modelId, config, overrides);
    
    // Ensure all required fields are present
    const missingFields = [];
    if (!modelData.id) missingFields.push('id');
    if (!modelData.name) missingFields.push('name');
    if (!modelData.parameters_billions) missingFields.push('parameters_billions');
    if (!modelData.hidden_size) missingFields.push('hidden_size');
    if (!modelData.num_layers) missingFields.push('num_layers');
    if (!modelData.num_heads) missingFields.push('num_heads');
    if (!modelData.default_context_length) missingFields.push('default_context_length');
    if (!modelData.architecture) missingFields.push('architecture');

    if (missingFields.length > 0) {
      console.error(chalk.red('❌ Missing required fields:'), missingFields.join(', '));
      console.log(chalk.yellow('\nAvailable data:'));
      console.log(chalk.gray(JSON.stringify(modelData, null, 2)));
      console.log(chalk.yellow('\nYou can provide missing values with:'));
      if (missingFields.includes('parameters_billions')) {
        console.log(chalk.yellow('  --params <number>  : Parameter count in billions'));
      }
      if (missingFields.includes('default_context_length')) {
        console.log(chalk.yellow('  --context <number> : Context length in tokens'));
      }
      return false;
    }

    const model = modelData as ModelEntry;

    // Display extracted info
    displayModelInfo(model);

    // Validate model data
    console.log(chalk.gray('\nValidating model data...'));
    const validation = validateModel(model);
    
    if (!validation.valid) {
      console.error(chalk.red('\n❌ Validation failed:'));
      validation.errors.forEach(err => console.error(chalk.red(`  - ${err}`)));
      return false;
    }
    
    if (validation.warnings.length > 0) {
      console.log(chalk.yellow('\n⚠️  Warnings:'));
      validation.warnings.forEach(warn => console.log(chalk.yellow(`  - ${warn}`)));
    }
    
    console.log(chalk.green('✓ Validation passed\n'));

    // Read existing models
    const modelsPath = join(__dirname, '..', 'src', 'data', 'models.json');
    const modelsData = readModelsFile(modelsPath);

    // Check for duplicates by ID
    if (isDuplicateModel(model.id, modelsData.models)) {
      console.error(chalk.red(`❌ Model with ID "${model.id}" already exists`));
      return false;
    }

    // Check for duplicates by HuggingFace model ID
    if (model.hf_model_id && isDuplicateHfModel(model.hf_model_id, modelsData.models)) {
      console.error(chalk.red(`❌ Model with HuggingFace ID "${model.hf_model_id}" already exists`));
      const existing = modelsData.models.find(m => m.hf_model_id === model.hf_model_id);
      if (existing) {
        console.log(chalk.yellow(`   Existing model ID: "${existing.id}" (${existing.name})`));
      }
      return false;
    }

    // Show diff
    const updatedModels = addModelToList(model, modelsData.models);
    showDiff(modelsData.models, updatedModels);

    // Dry run check
    if (opts.dryRun) {
      console.log(chalk.yellow('\n🔍 Dry run - no changes written'));
      return true;
    }

    // Confirm with user unless --force
    if (!opts.force) {
      const confirmed = await confirm('\nProceed with import?');
      if (!confirmed) {
        console.log(chalk.yellow('Import cancelled'));
        return false;
      }
    }

    // Create backup
    createBackup(modelsPath);

    // Write updated models
    const updatedData = { models: updatedModels };
    writeModelsFile(modelsPath, updatedData);

    console.log(chalk.green('\n✅ Model imported successfully!'));
    console.log(chalk.gray(`Updated: ${modelsPath}\n`));

    return true;
  } catch (error) {
    console.error(chalk.red(`\n❌ Error importing model: ${error instanceof Error ? error.message : String(error)}`));
    if (error instanceof Error && error.stack) {
      console.error(chalk.gray(error.stack));
    }
    return false;
  }
}

/**
 * Prompt user for confirmation
 */
async function confirm(message: string): Promise<boolean> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question(chalk.bold(`${message} (y/N) `), (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
    });
  });
}

/**
 * Main entry point
 */
async function main() {
  if (!options.url && !options.model && !options.file) {
    console.error(chalk.red('Error: Must provide --url, --model, or --file option'));
    program.help();
    process.exit(1);
  }

  let success = true;

  if (options.file) {
    // Batch import from file
    console.log(chalk.yellow('Batch import not yet implemented'));
    process.exit(1);
  } else {
    // Single model import
    const modelInput = options.url || options.model!;
    success = await importModel(modelInput, options);
  }

  process.exit(success ? 0 : 1);
}

main();
