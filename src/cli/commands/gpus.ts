import { Command } from 'commander';
import chalk from 'chalk';
import type { GPU } from '../../types/index';
import { loadGPUs } from '../utils/data-loader';
import { detectOutputFormat } from '../utils/output';

export interface ListGPUsOptions {
  vendor?: 'AMD' | 'NVIDIA';
  tier?: 'datacenter' | 'professional' | 'consumer';
  minVram?: number;
  partitioning?: boolean;
}

/**
 * Returns a filtered array of GPUs based on the provided options.
 */
export function listGPUs(opts: ListGPUsOptions): GPU[] {
  let gpus = loadGPUs();

  if (opts.vendor) {
    gpus = gpus.filter((g) => g.vendor === opts.vendor);
  }

  if (opts.tier) {
    gpus = gpus.filter((g) => g.tier === opts.tier);
  }

  if (opts.minVram !== undefined) {
    gpus = gpus.filter((g) => g.vram_gb >= opts.minVram!);
  }

  if (opts.partitioning) {
    gpus = gpus.filter((g) => g.partitioning?.supported === true);
  }

  return gpus;
}

/**
 * Finds and returns a single GPU by ID.
 * Returns null if not found.
 */
export function showGPU(id: string): GPU | null {
  const gpus = loadGPUs();
  return gpus.find((g) => g.id === id) ?? null;
}

/**
 * Returns an array of GPUs for the given IDs, skipping any unknown IDs.
 */
export function compareGPUs(ids: string[]): GPU[] {
  return ids.map((id) => showGPU(id)).filter((g): g is GPU => g !== null);
}

function printGPUTable(gpus: GPU[]): void {
  if (gpus.length === 0) {
    console.log(chalk.dim('No GPUs found.'));
    return;
  }

  const idWidth = Math.max(4, ...gpus.map((g) => g.id.length));
  const nameWidth = Math.max(4, ...gpus.map((g) => g.name.length));
  const vendorWidth = Math.max(6, ...gpus.map((g) => g.vendor.length));
  const tierWidth = Math.max(4, ...gpus.map((g) => (g.tier ?? '').length));

  const header =
    chalk.bold('ID'.padEnd(idWidth)) +
    '  ' +
    chalk.bold('Name'.padEnd(nameWidth)) +
    '  ' +
    chalk.bold('Vendor'.padEnd(vendorWidth)) +
    '  ' +
    chalk.bold('VRAM'.padStart(7)) +
    '  ' +
    chalk.bold('Bandwidth'.padStart(11)) +
    '  ' +
    chalk.bold('FP16 TFLOPS'.padStart(11)) +
    '  ' +
    chalk.bold('Tier'.padEnd(tierWidth));

  const separator = chalk.dim('-'.repeat(header.replace(/\x1b\[[0-9;]*m/g, '').length));

  console.log('');
  console.log(header);
  console.log(separator);

  for (const g of gpus) {
    console.log(
      g.id.padEnd(idWidth) +
        '  ' +
        g.name.padEnd(nameWidth) +
        '  ' +
        g.vendor.padEnd(vendorWidth) +
        '  ' +
        `${g.vram_gb} GB`.padStart(7) +
        '  ' +
        `${g.memory_bandwidth_gbps} GB/s`.padStart(11) +
        '  ' +
        String(g.compute_tflops_fp16).padStart(11) +
        '  ' +
        (g.tier ?? '').padEnd(tierWidth),
    );
  }

  console.log('');
  console.log(chalk.dim(`Total: ${gpus.length} GPU(s)`));
  console.log('');
}

function printCompareTable(gpus: GPU[]): void {
  if (gpus.length === 0) {
    console.log(chalk.dim('No GPUs found to compare.'));
    return;
  }

  // Field label width
  const labelWidth = 16;

  // Column width: max of GPU id length and values in that column
  const colWidths = gpus.map((g) => {
    const values = [
      g.id,
      g.name,
      g.vendor,
      `${g.vram_gb} GB`,
      `${g.memory_bandwidth_gbps} GB/s`,
      String(g.compute_tflops_fp16),
      g.compute_tflops_fp8 !== undefined ? String(g.compute_tflops_fp8) : 'N/A',
      g.memory_type,
      `${g.tdp_watts} W`,
      String(g.release_year),
      g.multi_gpu_capable ? 'Yes' : 'No',
      g.partitioning?.supported ? 'Yes' : 'No',
    ];
    return Math.max(g.id.length, ...values.map((v) => v.length)) + 2;
  });

  // Header row: GPU IDs
  const headerRow =
    ' '.repeat(labelWidth) +
    gpus.map((g, i) => chalk.bold(g.id.padEnd(colWidths[i]))).join('');

  // Separator line
  const totalWidth = labelWidth + colWidths.reduce((a, b) => a + b, 0);
  const separator = chalk.dim('\u2500'.repeat(totalWidth));

  console.log('');
  console.log(headerRow);
  console.log(separator);

  const fields: Array<{ label: string; value: (g: GPU) => string }> = [
    { label: 'Name', value: (g) => g.name },
    { label: 'Vendor', value: (g) => g.vendor },
    { label: 'VRAM', value: (g) => `${g.vram_gb} GB` },
    { label: 'Bandwidth', value: (g) => `${g.memory_bandwidth_gbps} GB/s` },
    { label: 'FP16 TFLOPS', value: (g) => String(g.compute_tflops_fp16) },
    {
      label: 'FP8 TFLOPS',
      value: (g) => (g.compute_tflops_fp8 !== undefined ? String(g.compute_tflops_fp8) : 'N/A'),
    },
    { label: 'Memory Type', value: (g) => g.memory_type },
    { label: 'TDP', value: (g) => `${g.tdp_watts} W` },
    { label: 'Release Year', value: (g) => String(g.release_year) },
    { label: 'Multi-GPU', value: (g) => (g.multi_gpu_capable ? 'Yes' : 'No') },
    { label: 'Partitioning', value: (g) => (g.partitioning?.supported ? 'Yes' : 'No') },
  ];

  for (const field of fields) {
    const row =
      chalk.bold(field.label.padEnd(labelWidth)) +
      gpus.map((g, i) => field.value(g).padEnd(colWidths[i])).join('');
    console.log(row);
  }

  console.log('');
}

export function registerGPUsCommand(program: Command): void {
  const gpusCmd = program.command('gpus').description('Query the GPU database');

  // gpus list subcommand
  gpusCmd
    .command('list')
    .description('List available GPUs')
    .option('--vendor <vendor>', 'Filter by vendor: AMD or NVIDIA')
    .option('--tier <tier>', 'Filter by tier: datacenter, professional, or consumer')
    .option('--min-vram <gb>', 'Filter GPUs with at least this much VRAM (GB)', parseFloat)
    .option('--partitioning', 'Only show GPUs with partitioning support')
    .option('-o, --output <format>', 'Output format: table (default) or json')
    .action((opts) => {
      const format = detectOutputFormat(opts.output);

      const gpus = listGPUs({
        vendor: opts.vendor as 'AMD' | 'NVIDIA' | undefined,
        tier: opts.tier as 'datacenter' | 'professional' | 'consumer' | undefined,
        minVram: opts.minVram !== undefined ? Number(opts.minVram) : undefined,
        partitioning: opts.partitioning === true,
      });

      if (format === 'json') {
        console.log(JSON.stringify(gpus, null, 2));
        return;
      }

      printGPUTable(gpus);
    });

  // gpus show subcommand
  gpusCmd
    .command('show <id>')
    .description('Show details for a specific GPU')
    .option('-o, --output <format>', 'Output format: table (default) or json')
    .action((id: string, opts) => {
      const format = detectOutputFormat(opts.output);

      const gpu = showGPU(id);

      if (!gpu) {
        process.stderr.write(
          chalk.red('Error: ') +
            `GPU not found: "${id}". Use 'llm-sizer gpus list' to see available GPUs.\n`,
        );
        process.exit(1);
      }

      if (format === 'json') {
        console.log(JSON.stringify(gpu, null, 2));
        return;
      }

      // Table format: print key/value pairs
      console.log('');
      console.log(chalk.bold('GPU Details'));
      console.log(chalk.dim('-'.repeat(50)));
      for (const [key, value] of Object.entries(gpu)) {
        if (Array.isArray(value)) {
          console.log(chalk.bold(key) + ':');
          value.forEach((item: unknown, idx: number) => {
            console.log(`  [${idx}]:`);
            if (typeof item === 'object' && item !== null) {
              for (const [k, v] of Object.entries(item as Record<string, unknown>)) {
                console.log(`    ${chalk.dim(k)}: ${String(v)}`);
              }
            } else {
              console.log(`    ${String(item)}`);
            }
          });
        } else if (typeof value === 'object' && value !== null) {
          console.log(chalk.bold(key) + ':');
          for (const [subKey, subValue] of Object.entries(value as Record<string, unknown>)) {
            if (Array.isArray(subValue)) {
              console.log(`  ${chalk.dim(subKey)}:`);
              (subValue as unknown[]).forEach((item: unknown, idx: number) => {
                console.log(`    [${idx}]:`);
                if (typeof item === 'object' && item !== null) {
                  for (const [k, v] of Object.entries(item as Record<string, unknown>)) {
                    console.log(`      ${chalk.dim(k)}: ${String(v)}`);
                  }
                } else {
                  console.log(`      ${String(item)}`);
                }
              });
            } else if (typeof subValue === 'object' && subValue !== null) {
              console.log(`  ${chalk.dim(subKey)}:`);
              for (const [k, v] of Object.entries(subValue as Record<string, unknown>)) {
                console.log(`    ${chalk.dim(k)}: ${String(v)}`);
              }
            } else {
              console.log(`  ${chalk.dim(subKey)}: ${String(subValue)}`);
            }
          }
        } else {
          console.log(chalk.bold(key) + ': ' + String(value));
        }
      }
      console.log('');
    });

  // gpus compare subcommand
  gpusCmd
    .command('compare <ids...>')
    .description('Compare multiple GPUs side-by-side')
    .option('-o, --output <format>', 'Output format: table (default) or json')
    .action((ids: string[], opts) => {
      const format = detectOutputFormat(opts.output);

      const gpus = compareGPUs(ids);

      if (gpus.length === 0) {
        process.stderr.write(
          chalk.red('Error: ') +
            `No matching GPUs found for IDs: ${ids.join(', ')}. Use 'llm-sizer gpus list' to see available GPUs.\n`,
        );
        process.exit(1);
      }

      if (format === 'json') {
        console.log(JSON.stringify(gpus, null, 2));
        return;
      }

      printCompareTable(gpus);
    });
}
