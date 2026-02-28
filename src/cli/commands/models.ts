import { Command } from 'commander';
import chalk from 'chalk';
import type { Model, EmbeddingModel, RerankingModel, ModelType } from '../../types/index';
import { loadModels, loadEmbeddingModels, loadRerankingModels } from '../utils/data-loader';
import { detectOutputFormat } from '../utils/output';

export interface ListModelsOptions {
  type?: ModelType;
  filter?: string;
  arch?: string;
  modality?: string;
}

/**
 * Returns a filtered array of models based on the provided options.
 */
export function listModels(opts: ListModelsOptions): Array<Model | EmbeddingModel | RerankingModel> {
  const type: ModelType = opts.type ?? 'generation';

  let models: Array<Model | EmbeddingModel | RerankingModel>;

  if (type === 'embedding') {
    models = loadEmbeddingModels();
  } else if (type === 'reranking') {
    models = loadRerankingModels();
  } else {
    models = loadModels();
  }

  // Apply filter (case-insensitive substring match on id and name)
  if (opts.filter) {
    const filterLower = opts.filter.toLowerCase();
    models = models.filter(
      (m) =>
        m.id.toLowerCase().includes(filterLower) ||
        m.name.toLowerCase().includes(filterLower),
    );
  }

  // Apply arch filter (generation models only)
  if (opts.arch && type === 'generation') {
    const archLower = opts.arch.toLowerCase();
    models = models.filter((m) => (m as Model).architecture?.toLowerCase() === archLower);
  }

  // Apply modality filter (generation models only)
  if (opts.modality && type === 'generation') {
    const modalityLower = opts.modality.toLowerCase();
    models = models.filter((m) => {
      const modality = (m as Model).modality;
      return (modality ?? 'text').toLowerCase() === modalityLower;
    });
  }

  return models;
}

/**
 * Finds and returns a single model by ID in the appropriate data source.
 * Returns null if not found.
 */
export function showModel(
  id: string,
  type: ModelType,
): Model | EmbeddingModel | RerankingModel | null {
  let models: Array<Model | EmbeddingModel | RerankingModel>;

  if (type === 'embedding') {
    models = loadEmbeddingModels();
  } else if (type === 'reranking') {
    models = loadRerankingModels();
  } else {
    models = loadModels();
  }

  return models.find((m) => m.id === id) ?? null;
}

function printGenerationTable(models: Model[]): void {
  if (models.length === 0) {
    console.log(chalk.dim('No generation models found.'));
    return;
  }

  const idWidth = Math.max(4, ...models.map((m) => m.id.length));
  const nameWidth = Math.max(4, ...models.map((m) => m.name.length));

  const header =
    chalk.bold('ID'.padEnd(idWidth)) +
    '  ' +
    chalk.bold('Name'.padEnd(nameWidth)) +
    '  ' +
    chalk.bold('Params (B)'.padStart(10)) +
    '  ' +
    chalk.bold('Layers'.padStart(6)) +
    '  ' +
    chalk.bold('Hidden'.padStart(7)) +
    '  ' +
    chalk.bold('Context'.padStart(9)) +
    '  ' +
    chalk.bold('Arch'.padEnd(11)) +
    '  ' +
    chalk.bold('Modality');

  const separator = chalk.dim('-'.repeat(header.replace(/\x1b\[[0-9;]*m/g, '').length));

  console.log('');
  console.log(header);
  console.log(separator);

  for (const m of models) {
    const params =
      m.parameters_billions >= 1
        ? m.parameters_billions.toFixed(1)
        : (m.parameters_billions * 1000).toFixed(0) + 'M (B<1)';

    console.log(
      m.id.padEnd(idWidth) +
        '  ' +
        m.name.padEnd(nameWidth) +
        '  ' +
        params.padStart(10) +
        '  ' +
        String(m.num_layers).padStart(6) +
        '  ' +
        String(m.hidden_size).padStart(7) +
        '  ' +
        String(m.default_context_length).padStart(9) +
        '  ' +
        (m.architecture ?? '').padEnd(11) +
        '  ' +
        (m.modality ?? 'text'),
    );
  }

  console.log('');
  console.log(chalk.dim(`Total: ${models.length} model(s)`));
  console.log('');
}

function printEmbeddingTable(models: EmbeddingModel[]): void {
  if (models.length === 0) {
    console.log(chalk.dim('No embedding models found.'));
    return;
  }

  const idWidth = Math.max(4, ...models.map((m) => m.id.length));
  const nameWidth = Math.max(4, ...models.map((m) => m.name.length));

  const header =
    chalk.bold('ID'.padEnd(idWidth)) +
    '  ' +
    chalk.bold('Name'.padEnd(nameWidth)) +
    '  ' +
    chalk.bold('Params (M)'.padStart(10)) +
    '  ' +
    chalk.bold('Dims'.padStart(6)) +
    '  ' +
    chalk.bold('Max Tokens'.padStart(10)) +
    '  ' +
    chalk.bold('Arch');

  const separator = chalk.dim('-'.repeat(header.replace(/\x1b\[[0-9;]*m/g, '').length));

  console.log('');
  console.log(header);
  console.log(separator);

  for (const m of models) {
    console.log(
      m.id.padEnd(idWidth) +
        '  ' +
        m.name.padEnd(nameWidth) +
        '  ' +
        String(m.parameters_millions).padStart(10) +
        '  ' +
        String(m.dimensions).padStart(6) +
        '  ' +
        String(m.max_tokens).padStart(10) +
        '  ' +
        (m.architecture ?? ''),
    );
  }

  console.log('');
  console.log(chalk.dim(`Total: ${models.length} model(s)`));
  console.log('');
}

function printRerankingTable(models: RerankingModel[]): void {
  if (models.length === 0) {
    console.log(chalk.dim('No reranking models found.'));
    return;
  }

  const idWidth = Math.max(4, ...models.map((m) => m.id.length));
  const nameWidth = Math.max(4, ...models.map((m) => m.name.length));

  const header =
    chalk.bold('ID'.padEnd(idWidth)) +
    '  ' +
    chalk.bold('Name'.padEnd(nameWidth)) +
    '  ' +
    chalk.bold('Params (M)'.padStart(10)) +
    '  ' +
    chalk.bold('Max Query'.padStart(9)) +
    '  ' +
    chalk.bold('Max Doc'.padStart(7)) +
    '  ' +
    chalk.bold('Max Docs/Q'.padStart(10)) +
    '  ' +
    chalk.bold('Arch');

  const separator = chalk.dim('-'.repeat(header.replace(/\x1b\[[0-9;]*m/g, '').length));

  console.log('');
  console.log(header);
  console.log(separator);

  for (const m of models) {
    console.log(
      m.id.padEnd(idWidth) +
        '  ' +
        m.name.padEnd(nameWidth) +
        '  ' +
        String(m.parameters_millions).padStart(10) +
        '  ' +
        String(m.max_query_length).padStart(9) +
        '  ' +
        String(m.max_doc_length).padStart(7) +
        '  ' +
        String(m.max_docs_per_query).padStart(10) +
        '  ' +
        (m.architecture ?? ''),
    );
  }

  console.log('');
  console.log(chalk.dim(`Total: ${models.length} model(s)`));
  console.log('');
}

export function registerModelsCommand(program: Command): void {
  const modelsCmd = program
    .command('models')
    .description('Query the model database');

  // models list subcommand
  modelsCmd
    .command('list')
    .description('List available models')
    .option('--type <type>', 'Model type: generation (default), embedding, or reranking', 'generation')
    .option('--filter <text>', 'Filter by substring match on ID or name')
    .option('--arch <arch>', 'Filter by architecture (generation models only, e.g. moe, transformer)')
    .option('--modality <modality>', 'Filter by modality (generation models only, e.g. text, multimodal)')
    .option('-o, --output <format>', 'Output format: table (default) or json')
    .action((opts) => {
      const type = (opts.type ?? 'generation') as ModelType;
      const format = detectOutputFormat(opts.output);

      const models = listModels({
        type,
        filter: opts.filter,
        arch: opts.arch,
        modality: opts.modality,
      });

      if (format === 'json') {
        console.log(JSON.stringify(models, null, 2));
        return;
      }

      if (type === 'embedding') {
        printEmbeddingTable(models as EmbeddingModel[]);
      } else if (type === 'reranking') {
        printRerankingTable(models as RerankingModel[]);
      } else {
        printGenerationTable(models as Model[]);
      }
    });

  // models show subcommand
  modelsCmd
    .command('show <id>')
    .description('Show details for a specific model')
    .option('--type <type>', 'Model type: generation (default), embedding, or reranking', 'generation')
    .option('-o, --output <format>', 'Output format: table (default) or json')
    .action((id: string, opts) => {
      const type = (opts.type ?? 'generation') as ModelType;
      const format = detectOutputFormat(opts.output);

      const model = showModel(id, type);

      if (!model) {
        process.stderr.write(
          chalk.red('Error: ') + `Model not found: "${id}". Use 'llm-sizer models list' to see available models.\n`,
        );
        process.exit(1);
      }

      if (format === 'json') {
        console.log(JSON.stringify(model, null, 2));
        return;
      }

      // Table format: print key/value pairs
      console.log('');
      console.log(chalk.bold('Model Details'));
      console.log(chalk.dim('-'.repeat(50)));
      for (const [key, value] of Object.entries(model)) {
        if (typeof value === 'object' && value !== null) {
          console.log(chalk.bold(key) + ':');
          for (const [subKey, subValue] of Object.entries(value as Record<string, unknown>)) {
            console.log(`  ${chalk.dim(subKey)}: ${String(subValue)}`);
          }
        } else {
          console.log(chalk.bold(key) + ': ' + String(value));
        }
      }
      console.log('');
    });
}
