import type {
  ContainerConfig,
  ConfigValidationResult,
  ValidationMessage,
} from '../types';

/**
 * Comprehensive configuration validator for Docker container configs
 * Handles memory, security, compatibility, and parameter validation
 */

// ============================================================================
// Memory Validation
// ============================================================================

export function validateMemory(config: ContainerConfig): ValidationMessage[] {
  const messages: ValidationMessage[] = [];
  const memoryPercent = config.memoryUsage.percentage;

  if (memoryPercent > 95) {
    messages.push({
      level: 'error',
      message: `Memory usage (${memoryPercent.toFixed(1)}%) exceeds safe limits`,
      suggestion: 'Consider using a lower quantization or smaller model',
      field: 'memory',
    });
  } else if (memoryPercent > 85) {
    messages.push({
      level: 'warning',
      message: `Memory usage (${memoryPercent.toFixed(1)}%) is high`,
      suggestion: 'Monitor memory usage closely during inference, or consider quantization',
      field: 'memory',
    });
  } else if (memoryPercent > 75) {
    messages.push({
      level: 'info',
      message: `Memory usage is ${memoryPercent.toFixed(1)}%`,
      suggestion: 'Usage is healthy, but watch for potential OOM with larger batches',
      field: 'memory',
    });
  } else if (memoryPercent < 50) {
    messages.push({
      level: 'info',
      message: `Memory usage is only ${memoryPercent.toFixed(1)}%`,
      suggestion: 'You could potentially use a larger model, higher batch size, or longer context',
      field: 'memory',
    });
  } else {
    // Success case
    messages.push({
      level: 'success',
      message: `Memory usage (${memoryPercent.toFixed(1)}%) is optimal`,
      field: 'memory',
    });
  }

  return messages;
}

// ============================================================================
// GPU & Hardware Validation
// ============================================================================

export function validateGPUConfiguration(config: ContainerConfig): ValidationMessage[] {
  const messages: ValidationMessage[] = [];

  // GPU count mismatch
  if (config.gpuCount !== config.gpuIds.length) {
    messages.push({
      level: 'error',
      message: 'GPU count mismatch',
      suggestion: 'Ensure GPU IDs array length matches GPU count',
      field: 'gpu',
    });
  }

  // Tensor parallel size validation
  const tensorParallelParam = config.engineParams.find(
    (p) => p.flag === '--tensor-parallel-size'
  );
  if (tensorParallelParam && typeof tensorParallelParam.value === 'number') {
    if (tensorParallelParam.value !== config.gpuCount) {
      messages.push({
        level: 'error',
        message: `Tensor parallel size (${tensorParallelParam.value}) must match GPU count (${config.gpuCount})`,
        suggestion: `Set --tensor-parallel-size to ${config.gpuCount}`,
        field: 'gpu',
      });
    }
  }

  // Container Toolkit recommendation
  if (!config.useContainerToolkit) {
    messages.push({
      level: 'info',
      message: 'AMD Container Toolkit not enabled',
      suggestion: 'Consider using Container Toolkit for easier GPU management and better compatibility',
      field: 'runtime',
    });
  }

  return messages;
}

// ============================================================================
// Shared Memory Validation
// ============================================================================

export interface SharedMemoryCalculation {
  gpuBased: number;
  modelBased: number;
  recommended: number;
  reasoning: {
    gpuReasoning: string;
    modelReasoning: string;
    finalReasoning: string;
  };
}

export function calculateSharedMemory(
  gpuCount: number,
  modelParams: number
): SharedMemoryCalculation {
  // GPU-based calculation: 8GB per GPU minimum
  const gpuBased = gpuCount * 8;

  // Model-based calculation
  let modelBased = 16;
  let modelReasoning = '';

  if (modelParams < 13) {
    modelBased = 16;
    modelReasoning = 'Small model (<13B parameters) - 16GB baseline';
  } else if (modelParams <= 70) {
    modelBased = 32;
    modelReasoning = 'Medium model (13B-70B parameters) - 32GB for adequate workspace';
  } else if (modelParams <= 200) {
    modelBased = 64;
    modelReasoning = 'Large model (70B-200B parameters) - 64GB for IPC and tensor ops';
  } else {
    modelBased = 128;
    modelReasoning = 'Very large model (>200B parameters) - 128GB for distributed ops';
  }

  const recommended = Math.max(gpuBased, modelBased);
  const gpuReasoning = `${gpuCount} GPU${gpuCount > 1 ? 's' : ''} × 8GB = ${gpuBased}GB (inter-process communication)`;

  let finalReasoning = '';
  if (recommended === gpuBased && recommended === modelBased) {
    finalReasoning = 'Both GPU and model calculations align';
  } else if (recommended === gpuBased) {
    finalReasoning = `Using GPU-based calculation (higher than model-based ${modelBased}GB)`;
  } else {
    finalReasoning = `Using model-based calculation (higher than GPU-based ${gpuBased}GB)`;
  }

  return {
    gpuBased,
    modelBased,
    recommended,
    reasoning: {
      gpuReasoning,
      modelReasoning,
      finalReasoning,
    },
  };
}

export function validateSharedMemory(config: ContainerConfig): ValidationMessage[] {
  const messages: ValidationMessage[] = [];
  const shmGB = parseInt(config.shmSize);

  const calculation = calculateSharedMemory(config.gpuCount, config.model.parameters);

  // Check critical threshold first (80% of recommended)
  if (shmGB < calculation.recommended * 0.8) {
    messages.push({
      level: 'error',
      message: `Shared memory (${config.shmSize}) is critically low`,
      suggestion: `Minimum ${calculation.recommended}g required. Current setting will likely cause failures.`,
      field: 'resources',
    });
  } else if (shmGB < calculation.recommended) {
    messages.push({
      level: 'warning',
      message: `Shared memory (${config.shmSize}) may be insufficient`,
      suggestion: `Recommend at least ${calculation.recommended}g based on ${config.gpuCount} GPU(s) and ${config.model.parameters}B parameter model. May cause IPC issues or OOM during multi-GPU operations.`,
      field: 'resources',
    });
  } else {
    messages.push({
      level: 'success',
      message: `Shared memory (${config.shmSize}) is adequate`,
      field: 'resources',
    });
  }

  return messages;
}

// ============================================================================
// Volume & Path Validation
// ============================================================================

export interface PathValidation {
  isAbsolute: boolean;
  isSensitive: boolean;
  isSystemPath: boolean;
  warnings: string[];
  suggestions: string[];
}

export function validateHostPath(path: string): PathValidation {
  const warnings: string[] = [];
  const suggestions: string[] = [];
  
  // Check if absolute path
  const isAbsolute = path.startsWith('/') || !!path.match(/^[A-Za-z]:\\/);
  
  // Detect sensitive system paths
  const sensitivePatterns = [
    { pattern: /^\/etc/, desc: 'System configuration directory' },
    { pattern: /^\/sys/, desc: 'System information directory' },
    { pattern: /^\/proc/, desc: 'Process information directory' },
    { pattern: /^\/dev/, desc: 'Device directory' },
    { pattern: /^\/boot/, desc: 'Boot directory' },
    { pattern: /^\/root/, desc: 'Root user home' },
    { pattern: /^\/usr\/bin/, desc: 'System binaries' },
    { pattern: /^\/usr\/sbin/, desc: 'System admin binaries' },
    { pattern: /^\/bin/, desc: 'Essential binaries' },
    { pattern: /^\/sbin/, desc: 'System binaries' },
  ];

  let isSensitive = false;
  let isSystemPath = false;

  for (const { pattern, desc } of sensitivePatterns) {
    if (pattern.test(path)) {
      isSensitive = true;
      isSystemPath = true;
      warnings.push(`Mounting sensitive path: ${desc}`);
      suggestions.push('Consider mounting as read-only unless write access is required');
      break;
    }
  }

  // Check for $HOME or ~ expansion
  if (path.includes('$HOME') || path.startsWith('~')) {
    warnings.push('Path uses home directory expansion');
    suggestions.push('Ensure the path expands correctly in your environment');
  }

  // Recommend relative paths for portability
  if (isAbsolute && !isSystemPath && !path.includes('$HOME') && !path.startsWith('~')) {
    suggestions.push('Consider using relative paths (e.g., ./models) for better portability');
  }

  return {
    isAbsolute,
    isSensitive,
    isSystemPath,
    warnings,
    suggestions,
  };
}

export function validateVolumes(config: ContainerConfig): ValidationMessage[] {
  const messages: ValidationMessage[] = [];

  for (const volume of config.volumes) {
    const validation = validateHostPath(volume.hostPath);

    if (validation.isSensitive) {
      if (!volume.readOnly) {
        messages.push({
          level: 'warning',
          message: `Volume "${volume.hostPath}" is writable and may pose security risks`,
          suggestion: 'Mount sensitive system paths as read-only',
          field: 'volumes',
        });
      }

      messages.push({
        level: 'warning',
        message: `Mounting sensitive path: ${volume.hostPath}`,
        suggestion: validation.suggestions[0] || 'Review access requirements',
        field: 'volumes',
      });
    }

    // Check for potential conflicts
    if (volume.hostPath === volume.containerPath) {
      messages.push({
        level: 'info',
        message: `Volume uses same path on host and container: ${volume.hostPath}`,
        suggestion: 'Ensure this is intentional',
        field: 'volumes',
      });
    }
  }

  return messages;
}

// ============================================================================
// Network & Port Validation
// ============================================================================

export function validateNetwork(config: ContainerConfig): ValidationMessage[] {
  const messages: ValidationMessage[] = [];

  // Host network warning
  if (config.useHostNetwork) {
    messages.push({
      level: 'warning',
      message: 'Using host network mode reduces container isolation',
      suggestion: 'Consider using explicit port mappings instead for better security',
      field: 'network',
    });
  }

  // Privileged port validation
  const privilegedPorts = config.ports.filter((p) => p.host < 1024);
  if (privilegedPorts.length > 0) {
    messages.push({
      level: 'warning',
      message: `Using privileged ports (< 1024): ${privilegedPorts.map(p => p.host).join(', ')}`,
      suggestion: 'May require root privileges or CAP_NET_BIND_SERVICE capability',
      field: 'ports',
    });
  }

  // Port conflict detection (basic)
  const hostPorts = config.ports.map((p) => p.host);
  const duplicates = hostPorts.filter((port, index) => hostPorts.indexOf(port) !== index);
  if (duplicates.length > 0) {
    messages.push({
      level: 'error',
      message: `Duplicate host ports detected: ${[...new Set(duplicates)].join(', ')}`,
      suggestion: 'Each host port must be unique',
      field: 'ports',
    });
  }

  return messages;
}

// ============================================================================
// Security Validation
// ============================================================================

export function validateSecurity(config: ContainerConfig): ValidationMessage[] {
  const messages: ValidationMessage[] = [];

  // Image stability warnings
  if (config.image.stability === 'nightly') {
    messages.push({
      level: 'warning',
      message: 'Using nightly/development container image',
      suggestion: 'Nightly builds may be unstable. Use stable tags for production deployments.',
      field: 'image',
    });
  } else if (config.image.stability === 'experimental') {
    messages.push({
      level: 'warning',
      message: 'Using experimental container image',
      suggestion: 'Experimental images are not recommended for production use',
      field: 'image',
    });
  }

  // Trust remote code warning
  const trustRemoteCode = config.engineParams.find((p) => p.flag === '--trust-remote-code');
  if (trustRemoteCode && trustRemoteCode.value === true) {
    messages.push({
      level: 'warning',
      message: 'Trust remote code is enabled',
      suggestion: 'Only enable for models from trusted sources. This allows execution of arbitrary code from HuggingFace model repos.',
      field: 'security',
    });
  }

  // Environment variable security
  const sensitiveVars = config.environment.filter((env) => env.sensitive);
  const hardcodedSecrets = sensitiveVars.filter((env) => {
    // Check if value is a placeholder syntax
    if (env.value.startsWith('${') && env.value.endsWith('}')) {
      return false;
    }

    // Check if value looks like a placeholder/example (common patterns)
    const placeholderPatterns = [
      /^your[-_]?/i,                    // "your-token-here", "your_api_key"
      /^(replace|change|enter|insert)[-_]?/i,  // "replace-with-token"
      /^<.*>$/,                          // "<your-token>"
      /^\[.*\]$/,                        // "[your-token]"
      /^example/i,                       // "example-token"
      /^test/i,                          // "test-token"
      /^placeholder/i,                   // "placeholder"
      /^(xxx|yyy|zzz)+$/i,              // "xxxxxxxxxxxx"
      /^[\*]+$/,                         // "************"
      /^[•]+$/,                          // "••••••••••••"
    ];

    const isPlaceholder = placeholderPatterns.some(pattern => pattern.test(env.value));

    // If it's not a placeholder and has substantial length, it's likely hardcoded
    // Real tokens are typically > 20 chars, so check for that
    return !isPlaceholder && env.value.length > 20;
  });

  if (hardcodedSecrets.length > 0) {
    messages.push({
      level: 'error',
      message: 'Sensitive environment variables appear to contain hardcoded values',
      suggestion: 'Use environment variable substitution (e.g., ${HF_TOKEN}) or .env file instead of hardcoded secrets',
      field: 'security',
    });
  }

  return messages;
}

// ============================================================================
// Parameter Compatibility Validation
// ============================================================================

export function validateParameterCompatibility(config: ContainerConfig): ValidationMessage[] {
  const messages: ValidationMessage[] = [];

  // Check quantization parameter
  const quantizationParam = config.engineParams.find((p) => p.flag === '--quantization');
  const dtypeParam = config.engineParams.find((p) => p.flag === '--dtype');
  const kvCacheParam = config.engineParams.find((p) => p.flag === '--kv-cache-dtype');

  // Quantization compatibility
  if (quantizationParam) {
    if (dtypeParam && dtypeParam.value !== 'float16' && dtypeParam.value !== 'auto') {
      messages.push({
        level: 'warning',
        message: `Quantization with --dtype ${dtypeParam.value} may have limited support`,
        suggestion: 'Consider using float16 or auto with quantization',
        field: 'parameters',
      });
    }
  }

  // KV cache FP8 memory implication
  if (kvCacheParam && kvCacheParam.value === 'fp8') {
    messages.push({
      level: 'info',
      message: 'Using FP8 KV cache reduces memory usage by ~50%',
      suggestion: 'Monitor inference quality; some models may have accuracy degradation',
      field: 'parameters',
    });
  }

  // Max model length validation
  const maxModelLen = config.engineParams.find((p) => p.flag === '--max-model-len');
  if (maxModelLen && typeof maxModelLen.value === 'number') {
    if (maxModelLen.value > 32768 && config.gpuCount < 2) {
      messages.push({
        level: 'warning',
        message: `Large context length (${maxModelLen.value}) on single GPU`,
        suggestion: 'Consider using multiple GPUs or reducing context length',
        field: 'parameters',
      });
    }
  }

  // GPU memory utilization parameter
  const gpuMemUtil = config.engineParams.find((p) => p.flag === '--gpu-memory-utilization');
  if (gpuMemUtil && typeof gpuMemUtil.value === 'number') {
    if (gpuMemUtil.value > 0.95) {
      messages.push({
        level: 'warning',
        message: `High GPU memory utilization (${gpuMemUtil.value})`,
        suggestion: 'May cause OOM errors. Consider 0.9 or lower for stability',
        field: 'parameters',
      });
    }
  }

  return messages;
}

// ============================================================================
// Main Validation Function
// ============================================================================

export function validateContainerConfig(config: ContainerConfig): ConfigValidationResult {
  const messages: ValidationMessage[] = [];
  const securityIssues: ValidationMessage[] = [];
  const recommendations: ValidationMessage[] = [];

  // Run all validation checks
  const memoryMessages = validateMemory(config);
  const gpuMessages = validateGPUConfiguration(config);
  const shmMessages = validateSharedMemory(config);
  const volumeMessages = validateVolumes(config);
  const networkMessages = validateNetwork(config);
  const securityMessages = validateSecurity(config);
  const paramMessages = validateParameterCompatibility(config);

  // Categorize messages
  const allMessages = [
    ...memoryMessages,
    ...gpuMessages,
    ...shmMessages,
    ...volumeMessages,
    ...networkMessages,
    ...securityMessages,
    ...paramMessages,
  ];

  for (const msg of allMessages) {
    if (msg.field === 'security' || msg.field === 'image') {
      securityIssues.push(msg);
    } else if (msg.level === 'info' || msg.level === 'success') {
      recommendations.push(msg);
    } else {
      messages.push(msg);
    }
  }

  // Determine overall validity
  const hasErrors =
    messages.filter((m) => m.level === 'error').length > 0 ||
    securityIssues.filter((m) => m.level === 'error').length > 0;

  const valid = !hasErrors;

  return {
    valid,
    messages,
    securityIssues,
    recommendations,
  };
}
