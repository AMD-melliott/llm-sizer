#!/usr/bin/env node
/**
 * Batch Calculator Validation Against Profile Results
 * 
 * This script processes multiple profiler outputs, runs them through the calculator,
 * and generates a comprehensive validation report comparing profiled vs calculated results.
 * 
 * Supports both old API-based and new bench-based profile formats.
 * 
 * Usage:
 *   npm run batch-validate -- <profile-dir> [options]
 *   npm run batch-validate -- <profile1.json> <profile2.json> [...] [options]
 *   
 * Options:
 *   --output <file>          Output CSV file (default: validation-results.csv)
 *   --json                   Output as JSON instead of CSV
 *   --detailed               Show detailed breakdown for each profile
 *   --threshold <percent>    Only show profiles with errors above threshold (default: 0)
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { glob } from 'glob';

// Import calculator functions
import { calculateMemoryRequirements } from '../src/utils/memoryCalculator.js';
import type { 
  Model, 
  GPU, 
  InferenceQuantization, 
  KVCacheQuantization,
  CalculationResults 
} from '../src/types/index.js';

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load data files
const modelsData = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../src/data/models.json'), 'utf-8')
);
const gpusData = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../src/data/gpus.json'), 'utf-8')
);

// Profile format types

// v2.0 Enhanced Profile Format (with baseline/post_warmup/peak structure)
interface BenchProfileV2 {
  model_info: {
    name: string;
    path?: string;
    num_parameters: number | null;
    dtype: string;
    quantization?: string;
  };
  benchmark_parameters: {
    batch_size: number;
    input_len: number;
    output_len: number;
    tensor_parallel_size: number;
  };
  gpu_info: {
    gpu_type: string;
    num_gpus: number;
    system_gpus?: number;  // Total GPUs in system (v2.0+)
  };
  memory_measurements: {
    baseline: {
      total_gb: number;
      per_gpu_details: Array<{ device: number; used_gb: number; total_gb: number }>;
    };
    post_warmup: {
      total_gb: number;
      per_gpu_details: Array<{ device: number; used_gb: number; total_gb: number }>;
    };
    peak: {
      total_gb: number;
      per_gpu_details: Array<{ device: number; used_gb: number; total_gb: number }>;
    };
  };
  memory_breakdown: {
    model_weights_gb: number;
    kv_cache_gb: number;
    activations_gb: number;
    framework_overhead_gb: number;
    multi_gpu_overhead_gb?: number;
    total_measured_gb: number;
    baseline_gb?: number;
  };
  latency_stats?: {
    avg_latency: number;
  };
  profiler_version?: string;
  schema_version?: string;
}

// v1.0 Profile Format (legacy, with per_gpu_peak array)
interface BenchProfileV1 {
  model_info: {
    name: string;
    path: string;
    num_parameters: number | null;
    dtype: string;
    quantization?: string;
  };
  benchmark_parameters: {
    batch_size: number;
    input_len: number;
    output_len: number;
    tensor_parallel_size: number;
  };
  gpu_info: {
    gpu_type: string;
    num_gpus: number;
    system_gpus?: number;  // Total GPUs in system (v2.0+)
  };
  memory_measurements: {
    per_gpu_peak: Array<{ device: number; used_gb: number; total_gb: number }>;
  };
  memory_breakdown: {
    model_weights_gb: number;
    kv_cache_gb: number;
    activations_gb: number;
    framework_overhead_gb: number;
    multi_gpu_overhead_gb?: number;
    total_measured_gb: number;
  };
  latency_stats?: {
    avg_latency: number;
  };
}

type BenchProfile = BenchProfileV1 | BenchProfileV2;

interface APIProfile {
  model_info: {
    model_name: string;
    model_id: string;
    num_parameters: number | null;
    batch_size: number;
    total_sequence_length: number;
  };
  gpu_info: {
    gpu_type: string;
    num_gpus: number;
    system_gpus?: number;  // Total GPUs in system (v2.0+)
    gpu_memory_baseline: Array<{ device: number; used_gb: number; total_gb: number }>;
  };
  memory_breakdown: {
    model_weights_gb: number;
    kv_cache_gb: number;
    activations_gb: number;
    framework_overhead_gb: number;
    multi_gpu_overhead_gb?: number;
    total_gb: number;
  };
}

type ProfileData = BenchProfile | APIProfile;

interface NormalizedProfile {
  model_name: string;
  model_path: string;
  num_parameters: number | null;
  dtype: string;
  batch_size: number;
  input_len: number;
  output_len: number;
  sequence_length: number;
  tensor_parallel_size: number;
  gpu_type: string;
  num_gpus: number;
  vram_per_gpu: number;
  profiled: {
    weights_gb: number;
    kv_cache_gb: number;
    activations_gb: number;
    overhead_gb: number;
    multi_gpu_overhead_gb: number;
    total_gb: number;
  };
  latency_ms?: number;
}

interface ValidationResult {
  profile_file: string;
  model_name: string;
  params_b: number;
  gpus: number;
  gpu_type: string;
  batch: number;
  input_len: number;
  output_len: number;
  seq_len: number;
  dtype: string;
  profiled_total_gb: number;
  profiled_weights_gb: number;
  profiled_kv_gb: number;
  profiled_activations_gb: number;
  profiled_overhead_gb: number;
  calculator_total_gb: number;
  calculator_weights_gb: number;
  calculator_kv_gb: number;
  calculator_activations_gb: number;
  calculator_overhead_gb: number;
  diff_total_gb: number;
  diff_total_pct: number;
  diff_weights_pct: number;
  diff_kv_pct: number;
  diff_activations_pct: number;
  diff_overhead_pct: number;
  match_quality: string;
  matched_model: string | null;
  matched_gpu: string | null;
  inference_quant: string;
  kv_quant: string;
  latency_ms?: number;
}

/**
 * Detect profile format and normalize to common structure
 */
function normalizeProfile(profile: ProfileData): NormalizedProfile {
  // Check if it's bench format (has benchmark_parameters)
  if ('benchmark_parameters' in profile) {
    const bench = profile as BenchProfile;
    
    // Detect v2.0 vs v1.0 schema
    const isV2 = 'profiler_version' in bench || 'schema_version' in bench ||
                 ('memory_measurements' in bench && 'baseline' in bench.memory_measurements);
    
    // Estimate parameters from weights if not provided
    let numParams = bench.model_info.num_parameters;
    if (!numParams && bench.memory_breakdown.model_weights_gb) {
      const dtype = bench.model_info.dtype.toLowerCase();
      const bytesPerParam = dtype.includes('float16') || dtype.includes('fp16') ? 2 : 
                           dtype.includes('float32') || dtype.includes('fp32') ? 4 : 1;
      numParams = (bench.memory_breakdown.model_weights_gb * 1e9) / bytesPerParam;
    }
    
    // Get VRAM per GPU based on schema version
    let vramPerGpu = 0;
    if (isV2) {
      const v2 = bench as BenchProfileV2;
      // v2.0: Use peak.per_gpu_details
      vramPerGpu = v2.memory_measurements.peak?.per_gpu_details?.[0]?.total_gb || 
                   v2.memory_measurements.post_warmup?.per_gpu_details?.[0]?.total_gb || 
                   v2.memory_measurements.baseline?.per_gpu_details?.[0]?.total_gb || 0;
    } else {
      const v1 = bench as BenchProfileV1;
      // v1.0: Use per_gpu_peak array
      vramPerGpu = v1.memory_measurements.per_gpu_peak?.[0]?.total_gb || 0;
    }
    
    return {
      model_name: bench.model_info.name,
      model_path: (bench.model_info as any).path || bench.model_info.name,
      num_parameters: numParams,
      dtype: bench.model_info.dtype,
      batch_size: bench.benchmark_parameters.batch_size,
      input_len: bench.benchmark_parameters.input_len,
      output_len: bench.benchmark_parameters.output_len,
      sequence_length: bench.benchmark_parameters.input_len + bench.benchmark_parameters.output_len,
      tensor_parallel_size: bench.benchmark_parameters.tensor_parallel_size,
      gpu_type: bench.gpu_info.gpu_type,
      num_gpus: bench.gpu_info.num_gpus,
      vram_per_gpu: vramPerGpu,
      profiled: {
        weights_gb: bench.memory_breakdown.model_weights_gb,
        kv_cache_gb: bench.memory_breakdown.kv_cache_gb,
        activations_gb: bench.memory_breakdown.activations_gb,
        overhead_gb: bench.memory_breakdown.framework_overhead_gb,
        multi_gpu_overhead_gb: bench.memory_breakdown.multi_gpu_overhead_gb || 0,
        total_gb: bench.memory_breakdown.total_measured_gb,
      },
      latency_ms: bench.latency_stats?.avg_latency,
    };
  } else {
    // Old API format
    const api = profile as APIProfile;
    const totalSeq = api.model_info.total_sequence_length;
    
    // Estimate parameters from weights if not provided
    let numParams = api.model_info.num_parameters;
    if (!numParams && api.memory_breakdown.model_weights_gb) {
      const bytesPerParam = 2; // Assume FP16
      numParams = (api.memory_breakdown.model_weights_gb * 1e9) / bytesPerParam;
    }
    
    return {
      model_name: api.model_info.model_name,
      model_path: api.model_info.model_id,
      num_parameters: numParams,
      dtype: 'float16', // Assume from old profiles
      batch_size: api.model_info.batch_size,
      input_len: Math.floor(totalSeq / 2), // Estimate
      output_len: Math.ceil(totalSeq / 2),
      sequence_length: totalSeq,
      tensor_parallel_size: api.gpu_info.num_gpus, // Assume TP = num_gpus
      gpu_type: api.gpu_info.gpu_type,
      num_gpus: api.gpu_info.num_gpus,
      vram_per_gpu: api.gpu_info.gpu_memory_baseline[0]?.total_gb || 0,
      profiled: {
        weights_gb: api.memory_breakdown.model_weights_gb,
        kv_cache_gb: api.memory_breakdown.kv_cache_gb,
        activations_gb: api.memory_breakdown.activations_gb,
        overhead_gb: api.memory_breakdown.framework_overhead_gb,
        multi_gpu_overhead_gb: api.memory_breakdown.multi_gpu_overhead_gb || 0,
        total_gb: api.memory_breakdown.total_gb,
      },
    };
  }
}

/**
 * Find matching model in calculator data
 */
function findMatchingModel(
  modelName: string,
  modelPath: string,
  numParams: number | null
): Model | null {
  const models = modelsData.models as Model[];

  // Try exact path match
  let match = models.find(m => m.id && modelPath && (m.id === modelPath || modelPath.includes(m.id)));
  if (match) return match;

  // Try name match
  const normalizedName = modelName.toLowerCase().replace(/[^a-z0-9]/g, '');
  match = models.find(m => {
    if (!m.name) return false;
    const modelNormalized = m.name.toLowerCase().replace(/[^a-z0-9]/g, '');
    return modelNormalized.includes(normalizedName) || normalizedName.includes(modelNormalized);
  });
  if (match) return match;

  // Try parameter-based match (within 15% tolerance for estimates)
  if (numParams) {
    const paramsBillions = numParams / 1e9;
    match = models.find(m => {
      const diff = Math.abs(m.parameters_billions - paramsBillions);
      const tolerance = m.parameters_billions * 0.15;
      return diff <= tolerance;
    });
    if (match) return match;
  }

  // Create custom model using estimated/provided parameters
  // This avoids issues with MoE models where total params != active params
  if (numParams) {
    return {
      id: 'custom',
      name: modelName,
      parameters_billions: numParams / 1e9,
      context_window: 4096, // Default
      supported_modalities: ['text'],
      quantization_support: ['fp16', 'fp8', 'int8', 'int4'],
    } as Model;
  }

  return null;
}

/**
 * Find matching GPU in calculator data
 */
function findMatchingGPU(gpuType: string, vramPerGpu: number): GPU | null {
  const gpus = gpusData.gpus as GPU[];
  const vramGB = Math.round(vramPerGpu);

  // AMD MI300X
  if (gpuType === 'rocm' && vramGB >= 180 && vramGB <= 210) {
    return gpus.find(g => g.id === 'mi300x') || null;
  }

  // AMD MI325X
  if (gpuType === 'rocm' && vramGB >= 240 && vramGB <= 260) {
    return gpus.find(g => g.id === 'mi325x') || null;
  }

  // NVIDIA H100
  if (gpuType === 'cuda' && vramGB >= 75 && vramGB <= 85) {
    return gpus.find(g => g.id === 'h100-sxm') || null;
  }

  // NVIDIA H200
  if (gpuType === 'cuda' && vramGB >= 135 && vramGB <= 145) {
    return gpus.find(g => g.id === 'h200-sxm') || null;
  }

  // Fuzzy match by VRAM
  const match = gpus.find(g => {
    const diff = Math.abs(g.vram_gb - vramGB);
    return diff <= g.vram_gb * 0.1;
  });

  return match || null;
}

/**
 * Infer quantization from dtype and weights
 */
function inferQuantization(
  dtype: string,
  numParams: number | null,
  weightsGB: number
): { inference: InferenceQuantization; kvCache: KVCacheQuantization } {
  // Check dtype first
  const dtypeLower = dtype.toLowerCase();
  if (dtypeLower.includes('int4') || dtypeLower.includes('4bit')) {
    return { inference: 'int4', kvCache: 'fp16_bf16' };
  }
  if (dtypeLower.includes('int8') || dtypeLower.includes('8bit')) {
    return { inference: 'int8', kvCache: 'int8' };
  }
  if (dtypeLower.includes('fp8') || dtypeLower.includes('float8')) {
    return { inference: 'fp8', kvCache: 'fp8_bf16' };
  }

  // Calculate from weights if possible
  if (numParams) {
    const bytesPerParam = (weightsGB * 1e9) / numParams;
    if (bytesPerParam <= 0.6) return { inference: 'int4', kvCache: 'fp16_bf16' };
    if (bytesPerParam <= 1.2) return { inference: 'int8', kvCache: 'int8' };
    if (bytesPerParam <= 1.5) return { inference: 'fp8', kvCache: 'fp8_bf16' };
  }

  return { inference: 'fp16', kvCache: 'fp16_bf16' };
}

/**
 * Validate a single profile
 */
function validateProfile(
  profilePath: string,
  profileData: ProfileData
): ValidationResult | null {
  try {
    const normalized = normalizeProfile(profileData);

    // Find matching model and GPU
    const matchedModel = findMatchingModel(
      normalized.model_name,
      normalized.model_path,
      normalized.num_parameters
    );
    const matchedGPU = findMatchingGPU(normalized.gpu_type, normalized.vram_per_gpu);

    if (!matchedModel || !matchedGPU) {
      console.error(
        `⚠️  Skipping ${path.basename(profilePath)}: ` +
        `${!matchedModel ? 'model not found' : 'GPU not found'}`
      );
      return null;
    }

    // Infer quantization
    const quant = inferQuantization(
      normalized.dtype,
      normalized.num_parameters,
      normalized.profiled.weights_gb
    );

    // Run calculator
    const calculated = calculateMemoryRequirements(
      matchedModel,
      matchedGPU,
      quant.inference,
      quant.kvCache,
      normalized.batch_size,
      normalized.sequence_length,
      1, // concurrent users
      normalized.num_gpus
    );

    // Calculate differences
    const diffTotal = normalized.profiled.total_gb - calculated.usedVRAM;
    const diffTotalPct = (diffTotal / calculated.usedVRAM) * 100;

    const diffWeightsPct =
      ((normalized.profiled.weights_gb - calculated.memoryBreakdown.baseWeights) /
        calculated.memoryBreakdown.baseWeights) *
      100;

    const diffKvPct =
      ((normalized.profiled.kv_cache_gb - calculated.memoryBreakdown.kvCache) /
        calculated.memoryBreakdown.kvCache) *
      100;

    const diffActivationsPct =
      ((normalized.profiled.activations_gb - calculated.memoryBreakdown.activations) /
        calculated.memoryBreakdown.activations) *
      100;

    const diffOverheadPct =
      ((normalized.profiled.overhead_gb - calculated.memoryBreakdown.frameworkOverhead) /
        calculated.memoryBreakdown.frameworkOverhead) *
      100;

    // Match quality
    const absDiffPct = Math.abs(diffTotalPct);
    let matchQuality: string;
    if (absDiffPct < 5) matchQuality = 'Excellent ✓✓';
    else if (absDiffPct < 10) matchQuality = 'Good ✓';
    else if (absDiffPct < 20) matchQuality = 'Fair ⚠';
    else matchQuality = 'Poor ✗';

    return {
      profile_file: path.basename(profilePath),
      model_name: normalized.model_name,
      params_b: normalized.num_parameters ? normalized.num_parameters / 1e9 : 0,
      gpus: normalized.num_gpus,
      gpu_type: matchedGPU.name,
      batch: normalized.batch_size,
      input_len: normalized.input_len,
      output_len: normalized.output_len,
      seq_len: normalized.sequence_length,
      dtype: normalized.dtype,
      profiled_total_gb: normalized.profiled.total_gb,
      profiled_weights_gb: normalized.profiled.weights_gb,
      profiled_kv_gb: normalized.profiled.kv_cache_gb,
      profiled_activations_gb: normalized.profiled.activations_gb,
      profiled_overhead_gb: normalized.profiled.overhead_gb,
      calculator_total_gb: calculated.usedVRAM,
      calculator_weights_gb: calculated.memoryBreakdown.baseWeights,
      calculator_kv_gb: calculated.memoryBreakdown.kvCache,
      calculator_activations_gb: calculated.memoryBreakdown.activations,
      calculator_overhead_gb: calculated.memoryBreakdown.frameworkOverhead,
      diff_total_gb: diffTotal,
      diff_total_pct: diffTotalPct,
      diff_weights_pct: diffWeightsPct,
      diff_kv_pct: diffKvPct,
      diff_activations_pct: diffActivationsPct,
      diff_overhead_pct: diffOverheadPct,
      match_quality: matchQuality,
      matched_model: matchedModel.name,
      matched_gpu: matchedGPU.name,
      inference_quant: quant.inference,
      kv_quant: quant.kvCache,
      latency_ms: normalized.latency_ms,
    };
  } catch (error) {
    console.error(`Error validating ${profilePath}:`, error);
    return null;
  }
}

/**
 * Format results as CSV
 */
function formatCSV(results: ValidationResult[]): string {
  const headers = [
    'profile_file',
    'model_name',
    'params_b',
    'gpus',
    'gpu_type',
    'batch',
    'input_len',
    'output_len',
    'seq_len',
    'dtype',
    'profiled_total_gb',
    'profiled_weights_gb',
    'profiled_kv_gb',
    'profiled_activations_gb',
    'profiled_overhead_gb',
    'calculator_total_gb',
    'calculator_weights_gb',
    'calculator_kv_gb',
    'calculator_activations_gb',
    'calculator_overhead_gb',
    'diff_total_gb',
    'diff_total_pct',
    'diff_weights_pct',
    'diff_kv_pct',
    'diff_activations_pct',
    'diff_overhead_pct',
    'match_quality',
    'matched_model',
    'matched_gpu',
    'inference_quant',
    'kv_quant',
    'latency_ms',
  ];

  const rows = results.map(r =>
    [
      r.profile_file,
      r.model_name,
      r.params_b.toFixed(1),
      r.gpus,
      r.gpu_type,
      r.batch,
      r.input_len,
      r.output_len,
      r.seq_len,
      r.dtype,
      r.profiled_total_gb.toFixed(2),
      r.profiled_weights_gb.toFixed(2),
      r.profiled_kv_gb.toFixed(2),
      r.profiled_activations_gb.toFixed(2),
      r.profiled_overhead_gb.toFixed(2),
      r.calculator_total_gb.toFixed(2),
      r.calculator_weights_gb.toFixed(2),
      r.calculator_kv_gb.toFixed(2),
      r.calculator_activations_gb.toFixed(2),
      r.calculator_overhead_gb.toFixed(2),
      r.diff_total_gb.toFixed(2),
      r.diff_total_pct.toFixed(1),
      r.diff_weights_pct.toFixed(1),
      r.diff_kv_pct.toFixed(1),
      r.diff_activations_pct.toFixed(1),
      r.diff_overhead_pct.toFixed(1),
      r.match_quality,
      r.matched_model || '',
      r.matched_gpu || '',
      r.inference_quant,
      r.kv_quant,
      r.latency_ms?.toFixed(2) || '',
    ].join(',')
  );

  return [headers.join(','), ...rows].join('\n');
}

/**
 * Format results as summary table
 */
function formatSummary(results: ValidationResult[], detailed: boolean = false): string {
  const lines: string[] = [];

  lines.push('═'.repeat(100));
  lines.push('BATCH VALIDATION SUMMARY');
  lines.push('═'.repeat(100));
  lines.push('');

  // Overall stats
  const excellent = results.filter(r => r.match_quality.includes('✓✓')).length;
  const good = results.filter(r => r.match_quality.includes('✓') && !r.match_quality.includes('✓✓')).length;
  const fair = results.filter(r => r.match_quality.includes('⚠')).length;
  const poor = results.filter(r => r.match_quality.includes('✗')).length;

  const avgError =
    results.reduce((sum, r) => sum + Math.abs(r.diff_total_pct), 0) / results.length;

  lines.push(`Total Profiles:  ${results.length}`);
  lines.push(`  Excellent ✓✓:  ${excellent} (${((excellent / results.length) * 100).toFixed(0)}%)`);
  lines.push(`  Good ✓:        ${good} (${((good / results.length) * 100).toFixed(0)}%)`);
  lines.push(`  Fair ⚠:        ${fair} (${((fair / results.length) * 100).toFixed(0)}%)`);
  lines.push(`  Poor ✗:        ${poor} (${((poor / results.length) * 100).toFixed(0)}%)`);
  lines.push(`  Avg Error:     ${avgError.toFixed(1)}%`);
  lines.push('');

  // Per-profile results
  lines.push('INDIVIDUAL RESULTS');
  lines.push('─'.repeat(100));
  lines.push(
    `${'Profile'.padEnd(35)} ` +
      `${'Model'.padEnd(20)} ` +
      `${'GPUs'.padStart(4)} ` +
      `${'Profiled'.padStart(10)} ` +
      `${'Calc'.padStart(10)} ` +
      `${'Diff %'.padStart(8)} ` +
      `${'Quality'.padStart(12)}`
  );
  lines.push('─'.repeat(100));

  for (const result of results) {
    const profileShort =
      result.profile_file.length > 35
        ? result.profile_file.substring(0, 32) + '...'
        : result.profile_file;
    const modelShort =
      result.model_name.length > 20
        ? result.model_name.substring(0, 17) + '...'
        : result.model_name;

    lines.push(
      `${profileShort.padEnd(35)} ` +
        `${modelShort.padEnd(20)} ` +
        `${result.gpus.toString().padStart(4)} ` +
        `${result.profiled_total_gb.toFixed(2).padStart(8)} GB ` +
        `${result.calculator_total_gb.toFixed(2).padStart(8)} GB ` +
        `${(result.diff_total_pct >= 0 ? '+' : '') + result.diff_total_pct.toFixed(1).padStart(6)}% ` +
        `${result.match_quality.padStart(12)}`
    );

    if (detailed) {
      lines.push(`  Weights:     ${result.diff_weights_pct.toFixed(1)}%`);
      lines.push(`  KV Cache:    ${result.diff_kv_pct.toFixed(1)}%`);
      lines.push(`  Activations: ${result.diff_activations_pct.toFixed(1)}%`);
      lines.push(`  Overhead:    ${result.diff_overhead_pct.toFixed(1)}%`);
      lines.push('');
    }
  }

  lines.push('═'.repeat(100));

  return lines.join('\n');
}

/**
 * Main function
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
Batch Calculator Validation

Usage:
  npm run batch-validate -- <profile-dir> [options]
  npm run batch-validate -- <profile1.json> <profile2.json> [...] [options]

Options:
  --output <file>       Output CSV file (default: validation-results.csv)
  --json                Output as JSON instead of CSV
  --detailed            Show detailed breakdown
  --threshold <pct>     Only show profiles with errors above threshold (default: 0)

Examples:
  # Validate all profiles in a directory
  npm run batch-validate -- results/memory-profiles/
  
  # Validate specific profiles
  npm run batch-validate -- profile1.json profile2.json
  
  # Output to custom CSV
  npm run batch-validate -- results/ --output my-validation.csv
  
  # Show only problematic profiles (>10% error)
  npm run batch-validate -- results/ --threshold 10 --detailed
`);
    process.exit(0);
  }

  // Parse options
  let outputFile = 'validation-results.csv';
  let outputJson = false;
  let detailed = false;
  let threshold = 0;
  const profilePaths: string[] = [];

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--output' && i + 1 < args.length) {
      outputFile = args[++i];
    } else if (args[i] === '--json') {
      outputJson = true;
    } else if (args[i] === '--detailed') {
      detailed = true;
    } else if (args[i] === '--threshold' && i + 1 < args.length) {
      threshold = parseFloat(args[++i]);
    } else if (!args[i].startsWith('--')) {
      profilePaths.push(args[i]);
    }
  }

  // Gather profile files
  let files: string[] = [];
  for (const p of profilePaths) {
    if (fs.statSync(p).isDirectory()) {
      // Get all JSON files in directory
      const jsonFiles = await glob(path.join(p, '**/*.json'));
      files.push(...jsonFiles);
    } else {
      files.push(p);
    }
  }

  if (files.length === 0) {
    console.error('Error: No profile files found');
    process.exit(1);
  }

  console.log(`\nValidating ${files.length} profile(s)...\n`);

  // Validate each profile
  const results: ValidationResult[] = [];
  for (const file of files) {
    try {
      const data = JSON.parse(fs.readFileSync(file, 'utf-8'));
      const result = validateProfile(file, data);
      if (result) {
        results.push(result);
        console.log(
          `✓ ${path.basename(file)}: ${result.match_quality} (${result.diff_total_pct >= 0 ? '+' : ''}${result.diff_total_pct.toFixed(1)}%)`
        );
      }
    } catch (error) {
      console.error(`✗ Error processing ${file}:`, error);
    }
  }

  if (results.length === 0) {
    console.error('\nNo valid results generated');
    process.exit(1);
  }

  // Filter by threshold
  const filtered = results.filter(r => Math.abs(r.diff_total_pct) >= threshold);

  console.log('');

  // Output results
  if (outputJson) {
    console.log(JSON.stringify(filtered, null, 2));
  } else {
    // Print summary to console
    console.log(formatSummary(filtered, detailed));

    // Write CSV
    fs.writeFileSync(outputFile, formatCSV(filtered));
    console.log(`\n✓ Results saved to: ${outputFile}`);
    console.log(`\nView with: column -t -s, ${outputFile} | less -S\n`);
  }
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
