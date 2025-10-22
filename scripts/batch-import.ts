#!/usr/bin/env node

import { Command } from 'commander';
import chalk from 'chalk';
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface ImportResult {
  modelId: string;
  success: boolean;
  error?: string;
  errorType?: 'gated' | 'not_found' | 'multimodal' | 'missing_fields' | 'other';
}

const program = new Command();

program
  .name('batch-import')
  .description('Batch import multiple HuggingFace models')
  .version('1.0.0')
  .option('-f, --file <path>', 'File containing list of model IDs (one per line)', 'models-to-import.txt')
  .option('--dry-run', 'Preview what would be imported without making changes')
  .option('--continue-on-error', 'Continue importing even if some models fail', true)
  .option('--log <path>', 'Path to log file for failed imports', 'import-results.log')
  .parse(process.argv);

const options = program.opts();

/**
 * Parse error type from error message
 */
function parseErrorType(error: string): ImportResult['errorType'] {
  if (error.includes('403') || error.includes('Gated model access')) {
    return 'gated';
  }
  if (error.includes('404') || error.includes('not found')) {
    return 'not_found';
  }
  if (error.includes('Multimodal') || error.includes('vision') || error.includes('image-text')) {
    return 'multimodal';
  }
  if (error.includes('Missing required fields')) {
    return 'missing_fields';
  }
  return 'other';
}

/**
 * Import a single model
 */
async function importModel(modelId: string, dryRun: boolean = false): Promise<ImportResult> {
  try {
    console.log(chalk.bold(`\n📦 Attempting to import: ${modelId}`));

    const args = ['--model', modelId, '--force'];
    if (dryRun) {
      args.push('--dry-run');
    }

    const scriptPath = join(__dirname, 'import-hf-model.ts');
    const command = `npx tsx ${scriptPath} ${args.join(' ')}`;

    execSync(command, {
      stdio: 'inherit',
      cwd: join(__dirname, '..')
    });

    console.log(chalk.green(`✅ Successfully imported: ${modelId}`));
    return { modelId, success: true };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorType = parseErrorType(errorMessage);

    console.error(chalk.red(`❌ Failed to import: ${modelId}`));

    return {
      modelId,
      success: false,
      error: errorMessage,
      errorType
    };
  }
}

/**
 * Read model IDs from file
 */
function readModelIds(filePath: string): string[] {
  try {
    const content = readFileSync(filePath, 'utf-8');
    return content
      .split('\n')
      .map(line => line.trim())
      .filter(line => line && !line.startsWith('#')); // Ignore empty lines and comments
  } catch (error) {
    console.error(chalk.red(`Failed to read file: ${filePath}`));
    console.error(error);
    return [];
  }
}

/**
 * Write import results to log file
 */
function writeResults(results: ImportResult[], logPath: string) {
  const timestamp = new Date().toISOString();
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  // Group failures by type
  const failureGroups = {
    gated: failed.filter(r => r.errorType === 'gated'),
    not_found: failed.filter(r => r.errorType === 'not_found'),
    multimodal: failed.filter(r => r.errorType === 'multimodal'),
    missing_fields: failed.filter(r => r.errorType === 'missing_fields'),
    other: failed.filter(r => r.errorType === 'other')
  };

  let log = `# Batch Import Results\n`;
  log += `Generated: ${timestamp}\n\n`;

  log += `## Summary\n`;
  log += `- Total models: ${results.length}\n`;
  log += `- Successfully imported: ${successful.length}\n`;
  log += `- Failed: ${failed.length}\n`;
  log += `- Success rate: ${((successful.length / results.length) * 100).toFixed(1)}%\n\n`;

  if (successful.length > 0) {
    log += `## Successfully Imported ✅\n\n`;
    successful.forEach(r => {
      log += `- ${r.modelId}\n`;
    });
    log += '\n';
  }

  if (failed.length > 0) {
    log += `## Failed Imports ❌\n\n`;

    if (failureGroups.gated.length > 0) {
      log += `### Gated Access Required (${failureGroups.gated.length})\n`;
      log += `These models require manual access approval on HuggingFace:\n\n`;
      failureGroups.gated.forEach(r => {
        log += `- ${r.modelId}\n`;
        log += `  URL: https://huggingface.co/${r.modelId}\n`;
      });
      log += '\n';
    }

    if (failureGroups.not_found.length > 0) {
      log += `### Model Not Found (${failureGroups.not_found.length})\n`;
      log += `These model IDs could not be found:\n\n`;
      failureGroups.not_found.forEach(r => {
        log += `- ${r.modelId}\n`;
      });
      log += '\n';
    }

    if (failureGroups.multimodal.length > 0) {
      log += `### Multimodal Models (${failureGroups.multimodal.length})\n`;
      log += `These are vision-language models not supported by this tool:\n\n`;
      failureGroups.multimodal.forEach(r => {
        log += `- ${r.modelId}\n`;
      });
      log += '\n';
    }

    if (failureGroups.missing_fields.length > 0) {
      log += `### Missing Configuration Fields (${failureGroups.missing_fields.length})\n`;
      log += `These models have incomplete configurations:\n\n`;
      failureGroups.missing_fields.forEach(r => {
        log += `- ${r.modelId}\n`;
      });
      log += '\n';
    }

    if (failureGroups.other.length > 0) {
      log += `### Other Errors (${failureGroups.other.length})\n\n`;
      failureGroups.other.forEach(r => {
        log += `- ${r.modelId}\n`;
        if (r.error) {
          log += `  Error: ${r.error.substring(0, 200)}\n`;
        }
      });
      log += '\n';
    }
  }

  log += `## Recommendations\n\n`;
  if (failureGroups.gated.length > 0) {
    log += `### For Gated Models:\n`;
    log += `1. Visit each model's HuggingFace page\n`;
    log += `2. Click "Agree and access repository"\n`;
    log += `3. Accept the terms and conditions\n`;
    log += `4. Re-run the import after access is granted\n\n`;
  }

  if (failureGroups.not_found.length > 0) {
    log += `### For Not Found Models:\n`;
    log += `1. Verify the model ID is correct\n`;
    log += `2. Check if the model was renamed or moved\n`;
    log += `3. Search HuggingFace for the correct model ID\n\n`;
  }

  writeFileSync(logPath, log);
  console.log(chalk.gray(`\nResults written to: ${logPath}`));
}

/**
 * Main entry point
 */
async function main() {
  const modelIds = readModelIds(options.file);

  if (modelIds.length === 0) {
    console.error(chalk.red('No model IDs found in file'));
    process.exit(1);
  }

  console.log(chalk.bold(`\n🚀 Starting batch import of ${modelIds.length} models\n`));

  if (options.dryRun) {
    console.log(chalk.yellow('🔍 DRY RUN MODE - No changes will be made\n'));
  }

  const results: ImportResult[] = [];

  for (const modelId of modelIds) {
    const result = await importModel(modelId, options.dryRun);
    results.push(result);

    if (!result.success && !options.continueOnError) {
      console.error(chalk.red('\nStopping due to error (use --continue-on-error to skip failures)'));
      break;
    }
  }

  // Summary
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log(chalk.bold('\n📊 Import Summary:'));
  console.log(chalk.green(`✅ Successful: ${successful.length}`));
  console.log(chalk.red(`❌ Failed: ${failed.length}`));
  console.log(chalk.blue(`📈 Success rate: ${((successful.length / results.length) * 100).toFixed(1)}%`));

  // Write detailed results to log
  writeResults(results, options.log);

  process.exit(failed.length > 0 ? 1 : 0);
}

main();