import { Command } from 'commander';
import chalk from 'chalk';
import type { InferenceQuantization, KVCacheQuantization, ModelType, CalculationResults, PerformanceMetrics } from '../../types/index';
import { calculateMemoryRequirements } from '../../utils/memoryCalculator';
import { estimatePerformance } from '../../utils/performanceEstimator';
import { resolveModel, resolveGPU } from '../utils/model-resolver';
import { detectOutputFormat } from '../utils/output';

export interface CalculateOptions {
  model: string;
  gpu: string;
  gpus: number;
  quant: InferenceQuantization;
  kvQuant: KVCacheQuantization;
  type: ModelType;
  batchSize: number;
  seqLength: number;
  users: number;
  output?: string;
}

export interface CalculateResult extends CalculationResults {
  inputs: {
    model: string;
    gpu: string;
    gpus: number;
    quant: InferenceQuantization;
    kvQuant: KVCacheQuantization;
    type: ModelType;
    batchSize: number;
    seqLength: number;
    users: number;
  };
}

export function buildCalculateResult(opts: CalculateOptions): CalculateResult {
  const model = resolveModel(opts.model);
  if (!model) {
    throw new Error(`Model not found: "${opts.model}". Use 'llm-sizer models' to list available models.`);
  }

  const gpu = resolveGPU(opts.gpu);
  if (!gpu) {
    throw new Error(`GPU not found: "${opts.gpu}". Use 'llm-sizer gpus' to list available GPUs.`);
  }

  const memResult = calculateMemoryRequirements(
    model,
    gpu,
    opts.quant,
    opts.kvQuant,
    opts.batchSize,
    opts.seqLength,
    opts.users,
    opts.gpus,
  );

  const performance: PerformanceMetrics = estimatePerformance(
    model,
    gpu,
    opts.quant,
    opts.batchSize,
    opts.seqLength,
    opts.users,
    opts.gpus,
    memResult.vramPercentage,
  );

  return {
    ...memResult,
    performance,
    inputs: {
      model: opts.model,
      gpu: opts.gpu,
      gpus: opts.gpus,
      quant: opts.quant,
      kvQuant: opts.kvQuant,
      type: opts.type,
      batchSize: opts.batchSize,
      seqLength: opts.seqLength,
      users: opts.users,
    },
  };
}

function makeBar(pct: number, width = 30): string {
  const filled = Math.min(Math.round((pct / 100) * width), width);
  return '[' + '#'.repeat(filled) + '-'.repeat(width - filled) + ']';
}

function formatGB(gb: number): string {
  return gb.toFixed(2) + ' GB';
}

function printTableOutput(result: CalculateResult): void {
  const { inputs, totalVRAM, usedVRAM, vramPercentage, memoryBreakdown, performance, status, message } = result;

  // Header
  console.log('');
  console.log(chalk.bold('LLM Memory Estimate'));
  console.log(chalk.dim('─'.repeat(60)));
  console.log(
    chalk.bold('Model:'), inputs.model,
    '  ' + chalk.bold('GPU:'), `${inputs.gpus}x ${inputs.gpu}`,
    '  ' + chalk.bold('Quant:'), inputs.quant,
  );
  console.log(
    chalk.bold('Batch:'), inputs.batchSize,
    '  ' + chalk.bold('Seq:'), inputs.seqLength,
    '  ' + chalk.bold('Users:'), inputs.users,
  );
  console.log(chalk.dim('─'.repeat(60)));

  // Status icon and VRAM usage
  let statusIcon: string;
  let usageColor: (s: string) => string;
  if (status === 'error') {
    statusIcon = chalk.red('[OVER]');
    usageColor = chalk.red;
  } else if (status === 'warning') {
    statusIcon = chalk.yellow('[WARN]');
    usageColor = chalk.yellow;
  } else {
    statusIcon = chalk.green('[OK]  ');
    usageColor = chalk.green;
  }

  const bar = makeBar(Math.min(vramPercentage, 100));
  console.log(
    statusIcon,
    usageColor(bar),
    usageColor(`${vramPercentage.toFixed(1)}%`),
    chalk.dim(`(${formatGB(usedVRAM)} / ${formatGB(totalVRAM)})`),
  );

  if (message) {
    console.log(chalk.dim('       ' + message));
  }

  // Memory breakdown
  console.log('');
  console.log(chalk.bold('Memory Breakdown:'));

  const breakdownItems: Array<[string, number | undefined]> = [
    ['  Base Weights', memoryBreakdown.baseWeights],
    ['  KV Cache', memoryBreakdown.kvCache],
    ['  Activations', memoryBreakdown.activations],
    ['  Framework Overhead', memoryBreakdown.frameworkOverhead],
    ['  Multi-GPU Overhead', memoryBreakdown.multiGPUOverhead],
  ];

  if (memoryBreakdown.visionWeights !== undefined) {
    breakdownItems.push(['  Vision Weights', memoryBreakdown.visionWeights]);
  }

  for (const [label, value] of breakdownItems) {
    if (value !== undefined && value > 0) {
      const pct = (value / totalVRAM) * 100;
      const bar = makeBar(pct, 20);
      console.log(
        chalk.dim(label.padEnd(22)),
        chalk.dim(bar),
        formatGB(value).padStart(10),
      );
    }
  }

  // Performance metrics
  console.log('');
  console.log(chalk.bold('Performance Estimate:'));
  console.log(
    '  Total Throughput:',
    chalk.cyan(`${performance.totalThroughput.toLocaleString()} tokens/sec`),
  );
  console.log(
    '  Per-User Speed:  ',
    chalk.cyan(`${performance.perUserSpeed.toLocaleString()} tokens/sec`),
  );
  console.log('');
}

export function registerCalculateCommand(program: Command): void {
  program
    .command('calculate')
    .description('Calculate memory requirements for an LLM inference workload')
    .requiredOption('-m, --model <id>', 'Model ID (e.g. llama-3-70b)')
    .requiredOption('-g, --gpu <id>', 'GPU ID (e.g. mi300x)')
    .option('-n, --gpus <count>', 'Number of GPUs', (v) => parseInt(v, 10), 1)
    .option('--quant <type>', 'Inference quantization (fp16|fp8|int8|int4)', 'fp16')
    .option('--kv-quant <type>', 'KV cache quantization (fp16_bf16|fp8_bf16|int8)', 'fp16_bf16')
    .option('--type <type>', 'Model type (generation|embedding|reranking)', 'generation')
    .option('--batch-size <n>', 'Batch size', (v) => parseInt(v, 10), 1)
    .option('--seq-length <n>', 'Sequence length in tokens', (v) => parseInt(v, 10), 4096)
    .option('--users <n>', 'Concurrent users', (v) => parseInt(v, 10), 1)
    .option('-o, --output <format>', 'Output format: table (default) or json')
    .action((opts) => {
      const format = detectOutputFormat(opts.output);

      const calcOpts: CalculateOptions = {
        model: opts.model,
        gpu: opts.gpu,
        gpus: opts.gpus ?? 1,
        quant: (opts.quant ?? 'fp16') as InferenceQuantization,
        kvQuant: (opts.kvQuant ?? 'fp16_bf16') as KVCacheQuantization,
        type: (opts.type ?? 'generation') as ModelType,
        batchSize: opts.batchSize ?? 1,
        seqLength: opts.seqLength ?? 4096,
        users: opts.users ?? 1,
        output: opts.output,
      };

      let result: CalculateResult;
      try {
        result = buildCalculateResult(calcOpts);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        process.stderr.write(chalk.red('Error: ') + message + '\n');
        process.exit(1);
      }

      if (format === 'json') {
        console.log(JSON.stringify(result, null, 2));
      } else {
        printTableOutput(result);
      }
    });
}
