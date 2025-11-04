"""
Calculator Expected Memory Formulas

This module implements the CURRENT calculator's formulas so we can compare
profiled actual values against expected calculator estimates.

Version: 1.0 - Implements current (potentially incorrect) formulas
"""

from typing import Dict, Any, Optional


def calculate_expected_memory(
    model_params: float,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    input_len: int,
    output_len: int,
    batch_size: int,
    weight_dtype: str = 'float16',
    kv_cache_dtype: Optional[str] = None,
    num_gpus: int = 1,
    framework: str = 'vllm'
) -> Dict[str, Any]:
    """
    Calculate expected memory using CURRENT calculator formulas
    
    This implements the existing (potentially incorrect) calculator logic
    so we can validate against actual profiled values.
    
    Returns dict with:
        - weights_gb
        - kv_cache_gb
        - activations_gb
        - overhead_gb
        - total_gb
        - formula_version
    """
    
    # Bytes per parameter by dtype
    BYTES_PER_PARAM = {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2,
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'fp8': 1,
        'int4': 0.5,
    }
    
    # KV cache dtype multipliers (relative to fp32 baseline)
    KV_MULTIPLIERS = {
        'float32': 1.0, 'fp32': 1.0,
        'float16': 0.5, 'fp16': 0.5,
        'bfloat16': 0.5, 'bf16': 0.5,
        'fp8': 0.25,  # FP8 is half of FP16
    }
    
    weight_bytes = BYTES_PER_PARAM.get(weight_dtype.lower(), 2)
    kv_dtype = kv_cache_dtype or weight_dtype
    kv_multiplier = KV_MULTIPLIERS.get(kv_dtype.lower(), 0.5)
    
    seq_len = input_len + output_len
    
    # ========================================
    # CURRENT CALCULATOR FORMULAS (v1.0)
    # These may be incorrect! That's the point of validation.
    # ========================================
    
    # 1. Model Weights (sharded across GPUs if multi-GPU)
    weights_gb = (model_params * weight_bytes) / (num_gpus * 1e9)
    
    # 2. KV Cache
    # Formula: 2 (key+value) × layers × hidden × seq_len × batch × 4 bytes (fp32 baseline)
    # Then apply dtype multiplier
    kv_cache_base = 2 * num_layers * hidden_size * seq_len * batch_size * 4 / 1e9
    kv_cache_gb = kv_cache_base * kv_multiplier / num_gpus
    
    # 3. Activations (very rough estimate)
    # Current formula: batch × seq × hidden × 8 (activation factor) × 2 bytes (fp16)
    activations_gb = (batch_size * seq_len * hidden_size * 8 * 2) / 1e9
    
    # 4. Framework Overhead (CURRENT FORMULA - likely wrong!)
    # Current: Fixed 8% of model memory
    model_memory = weights_gb + kv_cache_gb + activations_gb
    overhead_gb = model_memory * 0.08  # 8% fixed percentage
    
    # Total memory
    total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb
    
    return {
        'weights_gb': round(weights_gb, 2),
        'kv_cache_gb': round(kv_cache_gb, 2),
        'activations_gb': round(activations_gb, 2),
        'overhead_gb': round(overhead_gb, 2),
        'total_gb': round(total_gb, 2),
        'formula_version': 'calculator_v1.0_current',
        'notes': [
            'Uses CURRENT calculator formulas (may be incorrect)',
            'Overhead is fixed 8% of model memory',
            'Does not account for vLLM baseline overhead',
            'Batch size does not affect overhead'
        ]
    }


def calculate_proposed_memory(
    model_params: float,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    input_len: int,
    output_len: int,
    batch_size: int,
    weight_dtype: str = 'float16',
    kv_cache_dtype: Optional[str] = None,
    num_gpus: int = 1,
    framework: str = 'vllm'
) -> Dict[str, Any]:
    """
    Calculate expected memory using PROPOSED improved formulas
    
    This implements the new formulas based on our validation findings.
    """
    
    BYTES_PER_PARAM = {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2,
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'fp8': 1,
        'int4': 0.5,
    }
    
    KV_MULTIPLIERS = {
        'float32': 1.0, 'fp32': 1.0,
        'float16': 0.5, 'fp16': 0.5,
        'bfloat16': 0.5, 'bf16': 0.5,
        'fp8': 0.25,
    }
    
    weight_bytes = BYTES_PER_PARAM.get(weight_dtype.lower(), 2)
    kv_dtype = kv_cache_dtype or weight_dtype
    kv_multiplier = KV_MULTIPLIERS.get(kv_dtype.lower(), 0.5)
    
    seq_len = input_len + output_len
    
    # ========================================
    # PROPOSED NEW FORMULAS (v2.0)
    # Based on validation findings
    # ========================================
    
    # 1. Model Weights (same as before)
    weights_gb = (model_params * weight_bytes) / (num_gpus * 1e9)
    
    # 2. KV Cache (same formula, but with FP8 support)
    kv_cache_base = 2 * num_layers * hidden_size * seq_len * batch_size * 4 / 1e9
    kv_cache_gb = kv_cache_base * kv_multiplier / num_gpus
    
    # 3. Activations (same estimate)
    activations_gb = (batch_size * seq_len * hidden_size * 8 * 2) / 1e9
    
    # 4. Framework Overhead (NEW FORMULA!)
    import math
    
    model_memory = weights_gb + kv_cache_gb + activations_gb
    
    # Baseline overhead (vLLM engine, CUDA context, memory pools)
    if framework == 'vllm':
        baseline_overhead = 14.0  # GB
        proportional_factor = 0.05  # 5%
        batch_scaling = 0.002  # 0.2% per log2(batch)
    else:
        # Default conservative
        baseline_overhead = 10.0
        proportional_factor = 0.08
        batch_scaling = 0.001
    
    overhead_gb = baseline_overhead
    overhead_gb += model_memory * proportional_factor
    
    # Batch size scaling (logarithmic)
    if batch_size > 1:
        overhead_gb += model_memory * batch_scaling * math.log2(batch_size)
    
    # Multi-GPU overhead
    if num_gpus > 1:
        multi_gpu_overhead = (weights_gb / num_gpus) * 0.15 * (num_gpus - 1)
        overhead_gb += multi_gpu_overhead
    
    total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb
    
    return {
        'weights_gb': round(weights_gb, 2),
        'kv_cache_gb': round(kv_cache_gb, 2),
        'activations_gb': round(activations_gb, 2),
        'overhead_gb': round(overhead_gb, 2),
        'total_gb': round(total_gb, 2),
        'formula_version': 'calculator_v2.0_proposed',
        'notes': [
            'Uses PROPOSED new formulas based on validation',
            f'Baseline overhead: {baseline_overhead} GB',
            'Proportional overhead: 5% of model memory',
            'Batch scaling: logarithmic',
            'Multi-GPU overhead: 15% per additional GPU'
        ]
    }
