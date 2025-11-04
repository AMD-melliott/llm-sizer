#!/usr/bin/env node
/**
 * Validate Calculator Against Real-World Measurements
 * 
 * This script takes memory profiler output JSON, runs it through the LLM Sizer
 * calculator, and compares the results to validate accuracy.
 * 
 * Usage:
 *   npm run validate-calculator -- <profile.json> [options]
 *   
 * Options:
 *   --model-id <id>          Force specific model ID from calculator
 *   --gpu-id <id>            Force specific GPU ID from calculator  
 *   --inference-quant <type> Override inference quantization (fp16, fp8, int8, int4)
 *   --kv-quant <type>        Override KV cache quantization (fp16_bf16, fp8_bf16, int8)
 *   --json                   Output as JSON
 *   --detailed               Show detailed breakdown
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Import calculator functions
// Note: We'll need to compile TypeScript or use ts-node
import { calculateMemoryRequirements } from '../src/utils/memoryCalculator.js';
import type { 
  Model, 
  GPU, 
  InferenceQuantization, 
  KVCacheQuantization,
  CalculationResults,
  MemoryBreakdown 
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

interface ProfilerOutput {
  memory_breakdown: {
    model_weights_gb: number;
    kv_cache_gb: number;
    activations_gb: number;
    framework_overhead_gb: number;
    multi_gpu_overhead_gb: number;
    total_gb: number;
    estimation_method: string;
  };
  model_info: {
    model_name: string;
    model_id: string;
    num_parameters: number | null;
    prompt: string;
    max_tokens: number;
    batch_size: number;
    prompt_tokens: number;
    completion_tokens: number;
    total_sequence_length: number;
  };
  gpu_info: {
    gpu_type: string;
    num_gpus: number;
    gpu_memory_baseline: Array<{
      device: number;
      used_gb: number;
      total_gb: number;
      utilization_pct: number;
    }>;
    total_memory_gb: number;
  };
  inference_stats?: {
    inference_time_sec: number;
    tokens_per_second: number;
  };
}

interface ComparisonResult {
  component: string;
  actual_gb: number;
  calculated_gb: number;
  difference_gb: number;
  percent_diff: number;
  match: 'exact' | 'good' | 'fair' | 'poor';
}

interface ValidationReport {
  profile_file: string;
  model_info: {
    profiled: string;
    matched: string | null;
    parameters_profiled: number | null;
    parameters_matched: number | null;
  };
  gpu_info: {
    profiled: string;
    matched: string | null;
    num_gpus: number;
    vram_per_gpu: number;
  };
  configuration: {
    batch_size: number;
    sequence_length: number;
    inference_quantization: InferenceQuantization;
    kv_cache_quantization: KVCacheQuantization;
  };
  comparison: ComparisonResult[];
  summary: {
    total_actual_gb: number;
    total_calculated_gb: number;
    total_difference_gb: number;
    total_percent_diff: number;
    overall_match: 'exact' | 'good' | 'fair' | 'poor';
  };
  recommendations: string[];
}

/**
 * Match profiler model name to calculator model
 */
function findMatchingModel(
  modelName: string,
  modelId: string,
  numParams: number | null,
  forceModelId?: string
): Model | null {
  const models = modelsData.models as Model[];

  // If forced, use that
  if (forceModelId) {
    const forced = models.find(m => m.id === forceModelId);
    if (forced) return forced;
  }

  // Try exact model_id match first
  let match = models.find(m => m.id === modelId);
  if (match) return match;

  // Try fuzzy name match
  const normalizedName = modelName.toLowerCase().replace(/[^a-z0-9]/g, '');
  match = models.find(m => {
    const modelNormalized = m.name.toLowerCase().replace(/[^a-z0-9]/g, '');
    return modelNormalized.includes(normalizedName) || normalizedName.includes(modelNormalized);
  });
  if (match) return match;

  // Try parameter-based match (within 10% tolerance)
  if (numParams) {
    const paramsBillions = numParams / 1e9;
    match = models.find(m => {
      const diff = Math.abs(m.parameters_billions - paramsBillions);
      const tolerance = m.parameters_billions * 0.1;
      return diff <= tolerance;
    });
    if (match) return match;
  }

  return null;
}

/**
 * Match profiler GPU to calculator GPU
 */
function findMatchingGPU(
  gpuType: string,
  vramPerGpu: number,
  forceGpuId?: string
): GPU | null {
  const gpus = gpusData.gpus as GPU[];

  // If forced, use that
  if (forceGpuId) {
    const forced = gpus.find(g => g.id === forceGpuId);
    if (forced) return forced;
  }

  // Try to detect from type and VRAM
  const vramGB = Math.round(vramPerGpu);

  // AMD MI300X detection
  if (gpuType === 'rocm' && vramGB >= 180 && vramGB <= 210) {
    return gpus.find(g => g.id === 'mi300x') || null;
  }

  // AMD MI325X detection  
  if (gpuType === 'rocm' && vramGB >= 240 && vramGB <= 260) {
    return gpus.find(g => g.id === 'mi325x') || null;
  }

  // NVIDIA H100 detection
  if (gpuType === 'cuda' && vramGB >= 75 && vramGB <= 85) {
    return gpus.find(g => g.id === 'h100-sxm') || null;
  }

  // NVIDIA H200 detection
  if (gpuType === 'cuda' && vramGB >= 135 && vramGB <= 145) {
    return gpus.find(g => g.id === 'h200-sxm') || null;
  }

  // Try fuzzy match by VRAM (within 10%)
  const match = gpus.find(g => {
    const diff = Math.abs(g.vram_gb - vramGB);
    const tolerance = g.vram_gb * 0.1;
    return diff <= tolerance;
  });

  return match || null;
}

/**
 * Infer quantization from memory profile
 */
function inferQuantization(
  modelParams: number | null,
  modelWeightsGB: number
): { inference: InferenceQuantization; kvCache: KVCacheQuantization } {
  if (!modelParams) {
    return { inference: 'fp16', kvCache: 'fp16_bf16' };
  }

  // Calculate bytes per parameter
  const bytesPerParam = (modelWeightsGB * 1e9) / modelParams;

  // Determine quantization
  let inference: InferenceQuantization;
  if (bytesPerParam <= 0.6) {
    inference = 'int4';
  } else if (bytesPerParam <= 1.2) {
    inference = 'int8';
  } else if (bytesPerParam <= 1.5) {
    inference = 'fp8';
  } else {
    inference = 'fp16';
  }

  // For now, assume KV cache matches inference quantization
  const kvCache: KVCacheQuantization = 
    inference === 'fp8' ? 'fp8_bf16' : 
    inference === 'int8' ? 'int8' : 
    'fp16_bf16';

  return { inference, kvCache };
}

/**
 * Compare two values and determine match quality
 */
function compareValues(
  actual: number,
  calculated: number,
  component: string
): ComparisonResult {
  const diff = actual - calculated;
  const percentDiff = calculated > 0 ? (diff / calculated) * 100 : 
    (actual > 0 ? Infinity : 0);

  let match: 'exact' | 'good' | 'fair' | 'poor';
  const absDiff = Math.abs(percentDiff);
  
  if (absDiff < 5) match = 'exact';
  else if (absDiff < 15) match = 'good';
  else if (absDiff < 30) match = 'fair';
  else match = 'poor';

  return {
    component,
    actual_gb: actual,
    calculated_gb: calculated,
    difference_gb: diff,
    percent_diff: percentDiff,
    match,
  };
}

/**
 * Generate recommendations based on comparison
 */
function generateRecommendations(
  comparison: ComparisonResult[],
  profile: ProfilerOutput
): string[] {
  const recommendations: string[] = [];

  for (const comp of comparison) {
    const absDiff = Math.abs(comp.percent_diff);
    if (absDiff < 15) continue; // Only recommend for significant differences

    const component = comp.component;
    const isOver = comp.percent_diff > 0;

    if (component === 'Model Weights') {
      if (isOver) {
        recommendations.push(
          `Model Weights are ${absDiff.toFixed(1)}% higher than calculated. ` +
          `Verify quantization settings (currently assuming ${profile.memory_breakdown.estimation_method}).`
        );
      } else {
        recommendations.push(
          `Model Weights are ${absDiff.toFixed(1)}% lower than calculated. ` +
          `Model may use weight sharing or more aggressive compression.`
        );
      }
    } else if (component === 'KV Cache') {
      if (isOver) {
        recommendations.push(
          `KV Cache is ${absDiff.toFixed(1)}% higher than calculated. ` +
          `Check if model uses different attention mechanism (GQA, MQA) or verify sequence length.`
        );
      } else {
        recommendations.push(
          `KV Cache is ${absDiff.toFixed(1)}% lower than calculated. ` +
          `Framework may use PagedAttention or KV cache compression.`
        );
      }
    } else if (component === 'Activations') {
      if (isOver) {
        recommendations.push(
          `Activations are ${absDiff.toFixed(1)}% higher than calculated. ` +
          `Verify batch size and hidden dimensions match the profiled model.`
        );
      } else {
        recommendations.push(
          `Activations are ${absDiff.toFixed(1)}% lower than calculated. ` +
          `Framework may use activation checkpointing or recomputation.`
        );
      }
    } else if (component === 'Framework Overhead') {
      if (isOver) {
        recommendations.push(
          `Framework Overhead is ${absDiff.toFixed(1)}% higher than calculated. ` +
          `Consider increasing overhead percentage in calculator (currently 8%).`
        );
      }
    } else if (component === 'Multi-GPU Overhead') {
      if (isOver && profile.gpu_info.num_gpus > 1) {
        recommendations.push(
          `Multi-GPU Overhead is ${absDiff.toFixed(1)}% higher than calculated. ` +
          `Tensor parallelism may require more communication buffer space.`
        );
      }
    }
  }

  // Overall recommendation
  const totalComp = comparison.find(c => c.component === 'Total');
  if (totalComp && Math.abs(totalComp.percent_diff) > 20) {
    recommendations.push(
      `\n⚠️  OVERALL: Calculator is ${totalComp.percent_diff > 0 ? 'under' : 'over'}estimating ` +
      `by ${Math.abs(totalComp.percent_diff).toFixed(1)}%. Review individual component recommendations above.`
    );
  } else if (totalComp && Math.abs(totalComp.percent_diff) < 10) {
    recommendations.push(
      `\n✅ OVERALL: Calculator accuracy is excellent (${totalComp.percent_diff > 0 ? '+' : ''}${totalComp.percent_diff.toFixed(1)}% difference).`
    );
  }

  return recommendations;
}

/**
 * Main validation function
 */
async function validateCalculator(
  profilePath: string,
  options: {
    modelId?: string;
    gpuId?: string;
    inferenceQuant?: InferenceQuantization;
    kvQuant?: KVCacheQuantization;
    detailed?: boolean;
    json?: boolean;
  } = {}
): Promise<ValidationReport> {
  // Load profile
  const profileData = fs.readFileSync(profilePath, 'utf-8');
  const profile: ProfilerOutput = JSON.parse(profileData);

  // Match model
  const matchedModel = findMatchingModel(
    profile.model_info.model_name,
    profile.model_info.model_id,
    profile.model_info.num_parameters,
    options.modelId
  );

  if (!matchedModel) {
    throw new Error(
      `Could not find matching model for: ${profile.model_info.model_name} (${profile.model_info.model_id}). ` +
      `Try specifying --model-id manually.`
    );
  }

  // Match GPU
  const vramPerGpu = profile.gpu_info.gpu_memory_baseline[0]?.total_gb || 0;
  const matchedGPU = findMatchingGPU(
    profile.gpu_info.gpu_type,
    vramPerGpu,
    options.gpuId
  );

  if (!matchedGPU) {
    throw new Error(
      `Could not find matching GPU for: ${profile.gpu_info.gpu_type} with ${vramPerGpu.toFixed(0)}GB VRAM. ` +
      `Try specifying --gpu-id manually.`
    );
  }

  // Infer quantization if not specified
  const quantization = options.inferenceQuant && options.kvQuant ? 
    { inference: options.inferenceQuant, kvCache: options.kvQuant } :
    inferQuantization(
      profile.model_info.num_parameters,
      profile.memory_breakdown.model_weights_gb
    );

  // Extract configuration
  const batchSize = profile.model_info.batch_size;
  const sequenceLength = profile.model_info.total_sequence_length;
  const numGPUs = profile.gpu_info.num_gpus;
  const concurrentUsers = 1; // Assume 1 for profiling

  // Run calculator
  const calculatedResults: CalculationResults = calculateMemoryRequirements(
    matchedModel,
    matchedGPU,
    quantization.inference,
    quantization.kvCache,
    batchSize,
    sequenceLength,
    concurrentUsers,
    numGPUs
  );

  // Compare results
  const comparison: ComparisonResult[] = [
    compareValues(
      profile.memory_breakdown.model_weights_gb,
      calculatedResults.memoryBreakdown.baseWeights,
      'Model Weights'
    ),
    compareValues(
      profile.memory_breakdown.kv_cache_gb,
      calculatedResults.memoryBreakdown.kvCache,
      'KV Cache'
    ),
    compareValues(
      profile.memory_breakdown.activations_gb,
      calculatedResults.memoryBreakdown.activations,
      'Activations'
    ),
    compareValues(
      profile.memory_breakdown.framework_overhead_gb,
      calculatedResults.memoryBreakdown.frameworkOverhead,
      'Framework Overhead'
    ),
    compareValues(
      profile.memory_breakdown.multi_gpu_overhead_gb || 0,
      calculatedResults.memoryBreakdown.multiGPUOverhead,
      'Multi-GPU Overhead'
    ),
  ];

  // Total comparison
  const totalActual = profile.memory_breakdown.total_gb;
  const totalCalculated = calculatedResults.usedVRAM;
  const totalComparison = compareValues(totalActual, totalCalculated, 'Total');
  comparison.push(totalComparison);

  // Generate recommendations
  const recommendations = generateRecommendations(comparison, profile);

  // Build report
  const report: ValidationReport = {
    profile_file: path.basename(profilePath),
    model_info: {
      profiled: profile.model_info.model_name,
      matched: matchedModel.name,
      parameters_profiled: profile.model_info.num_parameters,
      parameters_matched: matchedModel.parameters_billions * 1e9,
    },
    gpu_info: {
      profiled: profile.gpu_info.gpu_type,
      matched: matchedGPU.name,
      num_gpus: numGPUs,
      vram_per_gpu: vramPerGpu,
    },
    configuration: {
      batch_size: batchSize,
      sequence_length: sequenceLength,
      inference_quantization: quantization.inference,
      kv_cache_quantization: quantization.kvCache,
    },
    comparison,
    summary: {
      total_actual_gb: totalActual,
      total_calculated_gb: totalCalculated,
      total_difference_gb: totalComparison.difference_gb,
      total_percent_diff: totalComparison.percent_diff,
      overall_match: totalComparison.match,
    },
    recommendations,
  };

  return report;
}

/**
 * Format report as text
 */
function formatReportText(report: ValidationReport, detailed: boolean = false): string {
  const lines: string[] = [];
  
  lines.push('═'.repeat(80));
  lines.push('LLM SIZER CALCULATOR VALIDATION REPORT');
  lines.push('═'.repeat(80));
  lines.push('');
  
  lines.push(`Profile: ${report.profile_file}`);
  lines.push('');
  
  lines.push('MODEL INFORMATION');
  lines.push('─'.repeat(80));
  lines.push(`  Profiled:  ${report.model_info.profiled}`);
  lines.push(`  Matched:   ${report.model_info.matched}`);
  if (report.model_info.parameters_profiled) {
    lines.push(`  Parameters: ${(report.model_info.parameters_profiled / 1e9).toFixed(2)}B (profiled) ` +
      `vs ${(report.model_info.parameters_matched! / 1e9).toFixed(2)}B (calculator)`);
  }
  lines.push('');
  
  lines.push('GPU INFORMATION');
  lines.push('─'.repeat(80));
  lines.push(`  Type:      ${report.gpu_info.profiled} → ${report.gpu_info.matched}`);
  lines.push(`  Count:     ${report.gpu_info.num_gpus}x GPUs`);
  lines.push(`  VRAM:      ${report.gpu_info.vram_per_gpu.toFixed(0)}GB per GPU`);
  lines.push('');
  
  lines.push('CONFIGURATION');
  lines.push('─'.repeat(80));
  lines.push(`  Batch Size:         ${report.configuration.batch_size}`);
  lines.push(`  Sequence Length:    ${report.configuration.sequence_length}`);
  lines.push(`  Inference Quant:    ${report.configuration.inference_quantization}`);
  lines.push(`  KV Cache Quant:     ${report.configuration.kv_cache_quantization}`);
  lines.push('');
  
  lines.push('MEMORY COMPARISON');
  lines.push('═'.repeat(80));
  lines.push(
    `${'Component'.padEnd(20)} ` +
    `${'Actual'.padStart(12)} ` +
    `${'Calculated'.padStart(12)} ` +
    `${'Diff'.padStart(12)} ` +
    `${'% Diff'.padStart(10)} ` +
    `${'Match'.padStart(8)}`
  );
  lines.push('─'.repeat(80));
  
  const matchIcons = {
    exact: '✓✓',
    good: '✓',
    fair: '⚠',
    poor: '✗',
  };
  
  for (const comp of report.comparison) {
    const icon = matchIcons[comp.match];
    lines.push(
      `${comp.component.padEnd(20)} ` +
      `${comp.actual_gb.toFixed(2).padStart(10)} GB ` +
      `${comp.calculated_gb.toFixed(2).padStart(10)} GB ` +
      `${(comp.difference_gb >= 0 ? '+' : '') + comp.difference_gb.toFixed(2).padStart(10)} GB ` +
      `${(comp.percent_diff >= 0 ? '+' : '') + comp.percent_diff.toFixed(1).padStart(8)}% ` +
      `${icon.padStart(8)}`
    );
  }
  
  lines.push('═'.repeat(80));
  lines.push('');
  
  lines.push('SUMMARY');
  lines.push('─'.repeat(80));
  lines.push(`  Overall Match:  ${report.summary.overall_match.toUpperCase()} (${matchIcons[report.summary.overall_match]})`);
  lines.push(`  Total Error:    ${(report.summary.total_percent_diff >= 0 ? '+' : '')}${report.summary.total_percent_diff.toFixed(1)}%`);
  lines.push(`  Absolute Error: ${Math.abs(report.summary.total_difference_gb).toFixed(2)} GB`);
  lines.push('');
  
  if (report.recommendations.length > 0) {
    lines.push('RECOMMENDATIONS');
    lines.push('─'.repeat(80));
    for (const rec of report.recommendations) {
      // Wrap long lines
      const words = rec.split(' ');
      let currentLine = '  ';
      for (const word of words) {
        if (currentLine.length + word.length + 1 > 78) {
          lines.push(currentLine);
          currentLine = '  ' + word;
        } else {
          currentLine += (currentLine.length > 2 ? ' ' : '') + word;
        }
      }
      if (currentLine.length > 2) lines.push(currentLine);
      lines.push('');
    }
  }
  
  lines.push('═'.repeat(80));
  
  return lines.join('\n');
}

/**
 * CLI entry point
 */
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
LLM Sizer Calculator Validator

Usage:
  npm run validate-calculator -- <profile.json> [options]
  
Options:
  --model-id <id>          Force specific model ID from calculator
  --gpu-id <id>            Force specific GPU ID from calculator
  --inference-quant <type> Override inference quantization (fp16, fp8, int8, int4)
  --kv-quant <type>        Override KV cache quantization (fp16_bf16, fp8_bf16, int8)
  --json                   Output as JSON
  --detailed               Show detailed breakdown

Examples:
  # Basic validation
  npm run validate-calculator -- results/memory-profiles/vllm-inference_20251031_162615.json
  
  # Force specific model match
  npm run validate-calculator -- profile.json --model-id llama-3-70b
  
  # Output as JSON for programmatic use
  npm run validate-calculator -- profile.json --json > validation-report.json
`);
    process.exit(0);
  }
  
  const profilePath = args[0];
  
  if (!fs.existsSync(profilePath)) {
    console.error(`Error: Profile file not found: ${profilePath}`);
    process.exit(1);
  }
  
  // Parse options
  const options: any = {
    detailed: args.includes('--detailed'),
    json: args.includes('--json'),
  };
  
  const modelIdIdx = args.indexOf('--model-id');
  if (modelIdIdx >= 0 && args[modelIdIdx + 1]) {
    options.modelId = args[modelIdIdx + 1];
  }
  
  const gpuIdIdx = args.indexOf('--gpu-id');
  if (gpuIdIdx >= 0 && args[gpuIdIdx + 1]) {
    options.gpuId = args[gpuIdIdx + 1];
  }
  
  const inferenceQuantIdx = args.indexOf('--inference-quant');
  if (inferenceQuantIdx >= 0 && args[inferenceQuantIdx + 1]) {
    options.inferenceQuant = args[inferenceQuantIdx + 1] as InferenceQuantization;
  }
  
  const kvQuantIdx = args.indexOf('--kv-quant');
  if (kvQuantIdx >= 0 && args[kvQuantIdx + 1]) {
    options.kvQuant = args[kvQuantIdx + 1] as KVCacheQuantization;
  }
  
  try {
    const report = await validateCalculator(profilePath, options);
    
    if (options.json) {
      console.log(JSON.stringify(report, null, 2));
    } else {
      console.log(formatReportText(report, options.detailed));
    }
    
    // Exit code based on match quality
    const exitCode = report.summary.overall_match === 'poor' ? 1 : 0;
    process.exit(exitCode);
    
  } catch (error) {
    console.error(`Error: ${(error as Error).message}`);
    process.exit(1);
  }
}

// Run if called directly (ES module check)
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { validateCalculator, ValidationReport };
