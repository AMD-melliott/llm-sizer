#!/usr/bin/env python3
"""
Comprehensive Profile Analysis Tool
Analyzes memory profiles and provides calculator validation insights
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import csv

def load_profile(profile_path: str) -> Dict[str, Any]:
    """Load a profile JSON file"""
    with open(profile_path, 'r') as f:
        return json.load(f)

def analyze_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a profile"""
    model_info = profile.get('model_info', {})
    bench_params = profile.get('benchmark_parameters', {})
    memory = profile.get('memory_breakdown', {})
    dtype_meta = profile.get('dtype_metadata', {})
    
    return {
        'model_name': model_info.get('name', 'Unknown'),
        'model_params_b': model_info.get('num_parameters', 0) / 1e9,
        'num_layers': model_info.get('num_layers'),
        'hidden_size': model_info.get('hidden_size'),
        'num_heads': model_info.get('num_heads'),
        'head_dim': model_info.get('head_dim'),
        
        # Benchmark configuration
        'input_len': bench_params.get('input_len'),
        'output_len': bench_params.get('output_len'),
        'batch_size': bench_params.get('batch_size'),
        'tensor_parallel': bench_params.get('tensor_parallel_size', 1),
        
        # Memory breakdown
        'total_memory_gb': memory.get('total_measured_gb', 0),
        'weights_gb': memory.get('model_weights_gb', 0),
        'kv_cache_gb': memory.get('kv_cache_gb', 0),
        'activations_gb': memory.get('activations_gb', 0),
        'overhead_gb': memory.get('framework_overhead_gb', 0),
        'overhead_pct': memory.get('overhead_percentage', 0),
        
        # Data types
        'weight_dtype': dtype_meta.get('weight_dtype', 'unknown'),
        'activation_dtype': dtype_meta.get('activation_dtype', 'unknown'),
        'kv_cache_dtype': dtype_meta.get('kv_cache_dtype', 'unknown'),
        
        # Confidence
        'confidence': memory.get('confidence_levels', {}).get('overall', 'unknown'),
        
        # Latency
        'latency_ms': profile.get('latency_stats', {}).get('avg_latency', 0) * 1000
    }

def generate_calculator_inputs(analysis: Dict[str, Any]) -> str:
    """Generate formatted calculator input instructions"""
    seq_length = (analysis['input_len'] or 0) + (analysis['output_len'] or 0)
    
    lines = [
        "=" * 80,
        f"CALCULATOR INPUTS FOR: {analysis['model_name']}",
        "=" * 80,
        "",
        "Model Configuration:",
        f"  • Model: {analysis['model_name']}",
        f"  • Parameters: {analysis['model_params_b']:.2f}B",
        f"  • Layers: {analysis['num_layers']}",
        f"  • Hidden Size: {analysis['hidden_size']}",
        f"  • Attention Heads: {analysis['num_heads']}",
        "",
        "Inference Configuration:",
        f"  • Batch Size: {analysis['batch_size']}",
        f"  • Input Length: {analysis['input_len']} tokens",
        f"  • Output Length: {analysis['output_len']} tokens", 
        f"  • Total Sequence: {seq_length} tokens",
        f"  • Data Type: {analysis['weight_dtype']}",
        f"  • KV Cache Type: {analysis['kv_cache_dtype']}",
        "",
        "GPU Configuration:",
        f"  • GPUs: {analysis['tensor_parallel']}",
        f"  • Tensor Parallel: {'Yes' if analysis['tensor_parallel'] > 1 else 'No'}",
        "",
        "=" * 80,
        "MEASURED MEMORY (Ground Truth)",
        "=" * 80,
        f"  Total Memory:      {analysis['total_memory_gb']:>8.2f} GB",
        f"  Model Weights:     {analysis['weights_gb']:>8.2f} GB",
        f"  KV Cache:          {analysis['kv_cache_gb']:>8.2f} GB",
        f"  Activations:       {analysis['activations_gb']:>8.2f} GB",
        f"  Framework Overhead:{analysis['overhead_gb']:>8.2f} GB ({analysis['overhead_pct']:.1f}%)",
        "",
        "=" * 80,
        "EXPECTED CALCULATOR BREAKDOWN",
        "=" * 80,
    ]
    
    # Calculate expected values
    expected_weights = analysis['model_params_b'] * 2  # FP16 = 2 bytes/param
    if analysis['weight_dtype'] == 'bfloat16':
        expected_weights = analysis['model_params_b'] * 2
    elif analysis['kv_cache_dtype'] == 'fp8':
        expected_kv_multiplier = 0.5
    else:
        expected_kv_multiplier = 1.0
    
    # KV cache calculation
    bytes_per_element = 2 if analysis['kv_cache_dtype'] in ['float16', 'bfloat16'] else 1
    seq_len = (analysis['input_len'] or 0) + (analysis['output_len'] or 0)
    expected_kv = (2 * analysis['num_layers'] * analysis['hidden_size'] * 
                   seq_len * analysis['batch_size'] * bytes_per_element) / 1e9
    
    lines.extend([
        f"  Expected Weights:  {expected_weights:>8.2f} GB (params × 2 bytes)",
        f"  Expected KV Cache: {expected_kv:>8.2f} GB (formula-based)",
        f"  Expected Total:    ~{expected_weights + expected_kv + 1:>7.2f} GB (rough estimate)",
        "",
        "🎯 Enter these values into the calculator and compare results!",
        "=" * 80,
        ""
    ])
    
    return "\n".join(lines)

def compare_with_calculator(analysis: Dict[str, Any], calc_total: float) -> str:
    """Generate comparison between measured and calculator"""
    measured = analysis['total_memory_gb']
    diff = calc_total - measured
    diff_pct = (diff / measured) * 100 if measured > 0 else 0
    
    # Determine match quality
    abs_pct = abs(diff_pct)
    if abs_pct <= 2:
        quality = "✓✓ EXCELLENT"
        symbol = "✓✓"
    elif abs_pct <= 5:
        quality = "✓ GOOD"
        symbol = "✓"
    elif abs_pct <= 10:
        quality = "✓ ACCEPTABLE"
        symbol = "✓"
    elif abs_pct <= 20:
        quality = "⚠ FAIR"
        symbol = "⚠"
    else:
        quality = "✗ POOR"
        symbol = "✗"
    
    lines = [
        "=" * 80,
        "VALIDATION RESULT",
        "=" * 80,
        f"  Measured (Profile): {measured:>8.2f} GB",
        f"  Calculator:         {calc_total:>8.2f} GB",
        f"  Difference:         {diff:>8.2f} GB ({diff_pct:+.1f}%)",
        f"  Match Quality:      {quality} {symbol}",
        "=" * 80,
        ""
    ]
    
    if abs_pct > 10:
        lines.append("\n⚠️  RECOMMENDATIONS:")
        if diff > 0:
            lines.append("  • Calculator is OVER-estimating memory")
            lines.append("  • Check framework overhead multiplier")
            lines.append("  • Verify KV cache calculations")
        else:
            lines.append("  • Calculator is UNDER-estimating memory")
            lines.append("  • May need to increase overhead factor")
            lines.append("  • Check for missing memory components")
    
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze-profiles.py <profile.json> [--calc-total <GB>]")
        sys.exit(1)
    
    profile_path = sys.argv[1]
    calc_total = None
    
    # Parse optional calculator total
    if '--calc-total' in sys.argv:
        idx = sys.argv.index('--calc-total')
        if idx + 1 < len(sys.argv):
            calc_total = float(sys.argv[idx + 1])
    
    # Load and analyze profile
    profile = load_profile(profile_path)
    analysis = analyze_profile(profile)
    
    # Generate output
    print(generate_calculator_inputs(analysis))
    
    if calc_total:
        print(compare_with_calculator(analysis, calc_total))
    else:
        print("\n💡 Run again with --calc-total <GB> to compare with calculator results")
        print(f"   Example: python {sys.argv[0]} {profile_path} --calc-total 20.5")

if __name__ == '__main__':
    main()
