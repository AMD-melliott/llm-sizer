#!/usr/bin/env python3
"""
Profile vs Calculator Validation Tool

This tool analyzes a single profile and provides clear inputs for calculator validation.
It's designed for systematically validating the calculator against real-world measurements.

Usage:
    python scripts/validate-profile.py <profile.json>
    python scripts/validate-profile.py <profile.json> --calculator-total 145.2
    python scripts/validate-profile.py <profile.json> --record results.csv
"""

import json
import sys
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def load_profile(file_path: str) -> Dict[str, Any]:
    """Load profile JSON"""
    with open(file_path) as f:
        return json.load(f)


def get_calculator_inputs(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Extract calculator input parameters from profile"""
    
    model_info = profile.get('model_info', {})
    bench_params = profile.get('benchmark_parameters', {})
    gpu_info = profile.get('gpu_info', {})
    mem_measurements = profile.get('memory_measurements', {})
    mem_breakdown = profile.get('memory_breakdown', {})
    
    # Estimate model size from weights
    weights_gb = mem_breakdown.get('model_weights_gb', 0)
    dtype = model_info.get('dtype', 'float16')
    bytes_per_param = 2 if dtype == 'float16' else 4 if dtype == 'float32' else 1
    estimated_params_b = (weights_gb * 1e9) / bytes_per_param / 1e9
    
    # Get GPU info
    per_gpu_peak = mem_measurements.get('per_gpu_peak', [])
    vram_per_gpu = per_gpu_peak[0].get('total_gb', 0) if per_gpu_peak else 0
    
    # Determine GPU type
    gpu_type_name = "AMD Instinct MI300X" if gpu_info.get('gpu_type') == 'rocm' else "NVIDIA H100"
    if vram_per_gpu >= 240:
        gpu_type_name = "AMD Instinct MI325X"
    
    return {
        'model_name': model_info.get('name', 'Unknown'),
        'estimated_params_b': round(estimated_params_b, 1),
        'gpu_type': gpu_type_name,
        'num_gpus': gpu_info.get('num_gpus', 1),
        'vram_per_gpu_gb': vram_per_gpu,
        'batch_size': bench_params.get('batch_size', 1),
        'input_len': bench_params.get('input_len', 256),
        'output_len': bench_params.get('output_len', 256),
        'total_seq_len': bench_params.get('input_len', 0) + bench_params.get('output_len', 0),
        'dtype': dtype.upper(),
        'quantization': model_info.get('quantization') or 'None',
        'tensor_parallel': bench_params.get('tensor_parallel_size', 1),
    }


def get_profiled_results(profile: Dict[str, Any]) -> Dict[str, float]:
    """Extract profiled memory results"""
    
    mem_breakdown = profile.get('memory_breakdown', {})
    latency_stats = profile.get('latency_stats', {})
    
    return {
        'total_gb': mem_breakdown.get('total_measured_gb', 0),
        'weights_gb': mem_breakdown.get('model_weights_gb', 0),
        'kv_cache_gb': mem_breakdown.get('kv_cache_gb', 0),
        'activations_gb': mem_breakdown.get('activations_gb', 0),
        'overhead_gb': mem_breakdown.get('framework_overhead_gb', 0),
        'latency_ms': latency_stats.get('avg_latency', 0),
        'latency_p50_ms': latency_stats.get('percentiles', {}).get('50', 0),
        'latency_p99_ms': latency_stats.get('percentiles', {}).get('99', 0),
    }


def validate_comparison(
    profiled: Dict[str, float],
    calculator_total: Optional[float] = None,
    calculator_breakdown: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Compare profiled vs calculator results"""
    
    if not calculator_total:
        return None
    
    diff_gb = calculator_total - profiled['total_gb']
    pct_diff = (diff_gb / profiled['total_gb'] * 100) if profiled['total_gb'] > 0 else 0
    
    # Determine match quality
    abs_pct = abs(pct_diff)
    if abs_pct <= 5:
        match = 'Excellent ✓✓'
    elif abs_pct <= 10:
        match = 'Good ✓'
    elif abs_pct <= 20:
        match = 'Fair ⚠'
    else:
        match = 'Poor ✗'
    
    result = {
        'calculator_total_gb': calculator_total,
        'profiled_total_gb': profiled['total_gb'],
        'difference_gb': diff_gb,
        'percent_diff': pct_diff,
        'match_quality': match,
    }
    
    # Add component comparisons if provided
    if calculator_breakdown:
        result['components'] = {}
        for key in ['weights_gb', 'kv_cache_gb', 'activations_gb', 'overhead_gb']:
            calc_key = key.replace('_gb', '')  # Remove _gb for lookup
            if calc_key in calculator_breakdown:
                calc_val = calculator_breakdown[calc_key]
                prof_val = profiled[key]
                comp_diff = calc_val - prof_val
                comp_pct = (comp_diff / prof_val * 100) if prof_val > 0 else 0
                result['components'][key] = {
                    'calculator': calc_val,
                    'profiled': prof_val,
                    'diff_gb': comp_diff,
                    'pct_diff': comp_pct,
                }
    
    return result


def print_validation_report(
    file_name: str,
    calc_inputs: Dict[str, Any],
    profiled: Dict[str, float],
    validation: Optional[Dict[str, Any]] = None
):
    """Print formatted validation report"""
    
    print()
    print("=" * 80)
    print("CALCULATOR VALIDATION REPORT")
    print("=" * 80)
    print()
    print(f"Profile: {file_name}")
    print()
    
    # Model Information
    print("MODEL INFORMATION")
    print("-" * 80)
    print(f"  Model:              {calc_inputs['model_name']}")
    print(f"  Est. Parameters:    {calc_inputs['estimated_params_b']:.1f}B")
    print(f"  Data Type:          {calc_inputs['dtype']}")
    print(f"  Quantization:       {calc_inputs['quantization']}")
    print()
    
    # Configuration
    print("TEST CONFIGURATION")
    print("-" * 80)
    print(f"  GPU Type:           {calc_inputs['gpu_type']}")
    print(f"  Number of GPUs:     {calc_inputs['num_gpus']}")
    print(f"  VRAM per GPU:       {calc_inputs['vram_per_gpu_gb']:.0f} GB")
    print(f"  Tensor Parallel:    {calc_inputs['tensor_parallel']}")
    print(f"  Batch Size:         {calc_inputs['batch_size']}")
    print(f"  Input Tokens:       {calc_inputs['input_len']}")
    print(f"  Output Tokens:      {calc_inputs['output_len']}")
    print(f"  Total Sequence:     {calc_inputs['total_seq_len']}")
    print()
    
    # Profiled Results
    print("PROFILED RESULTS (Ground Truth)")
    print("-" * 80)
    print(f"  Total Memory:       {profiled['total_gb']:>8.2f} GB  ← Target for calculator")
    print(f"  Model Weights:      {profiled['weights_gb']:>8.2f} GB")
    print(f"  KV Cache:           {profiled['kv_cache_gb']:>8.2f} GB")
    print(f"  Activations:        {profiled['activations_gb']:>8.2f} GB")
    print(f"  Framework Overhead: {profiled['overhead_gb']:>8.2f} GB")
    if profiled['latency_ms'] > 0:
        print(f"  Latency (avg):      {profiled['latency_ms']:>8.2f} ms")
        print(f"  Latency (p50):      {profiled['latency_p50_ms']:>8.2f} ms")
        print(f"  Latency (p99):      {profiled['latency_p99_ms']:>8.2f} ms")
    print()
    
    # Calculator Inputs Section
    print("═" * 80)
    print("CALCULATOR INPUTS")
    print("═" * 80)
    print()
    print("Copy these values into the LLM Sizer calculator:")
    print()
    print("  1. Model Selection:")
    print(f"     - Search for: {calc_inputs['model_name']}")
    print(f"     - Or use Custom with {calc_inputs['estimated_params_b']:.1f}B parameters")
    print()
    print("  2. GPU Configuration:")
    print(f"     - GPU Type: {calc_inputs['gpu_type']}")
    print(f"     - Number of GPUs: {calc_inputs['num_gpus']}")
    print(f"     - Partition Mode: Tensor Parallel")
    print()
    print("  3. Inference Settings:")
    print(f"     - Batch Size: {calc_inputs['batch_size']}")
    print(f"     - Input Sequence: {calc_inputs['input_len']} tokens")
    print(f"     - Output Sequence: {calc_inputs['output_len']} tokens")
    print(f"     - Data Type: {calc_inputs['dtype']}")
    if calc_inputs['quantization'] != 'None':
        print(f"     - Quantization: {calc_inputs['quantization']}")
    print()
    print("  4. After calculator computes:")
    print(f"     - Record the Total Memory estimate")
    print(f"     - Target value: {profiled['total_gb']:.2f} GB")
    print()
    
    # Validation Results (if provided)
    if validation:
        print("═" * 80)
        print("VALIDATION RESULTS")
        print("═" * 80)
        print()
        
        calc_total = validation['calculator_total_gb']
        prof_total = validation['profiled_total_gb']
        diff = validation['difference_gb']
        pct = validation['percent_diff']
        match = validation['match_quality']
        
        print(f"  Calculator Total:   {calc_total:>8.2f} GB")
        print(f"  Profiled Total:     {prof_total:>8.2f} GB")
        print(f"  Difference:         {diff:>+8.2f} GB  ({pct:>+6.1f}%)")
        print(f"  Match Quality:      {match}")
        print()
        
        # Component breakdown if available
        if 'components' in validation:
            print("  Component Breakdown:")
            print()
            print(f"  {'Component':<18} {'Calculator':>10} {'Profiled':>10} {'Diff':>10} {'% Diff':>8}")
            print("  " + "-" * 68)
            
            for comp_name, comp_data in validation['components'].items():
                name = comp_name.replace('_gb', '').replace('_', ' ').title()
                print(f"  {name:<18} {comp_data['calculator']:>8.2f} GB {comp_data['profiled']:>8.2f} GB "
                      f"{comp_data['diff_gb']:>+8.2f} GB {comp_data['pct_diff']:>+6.1f}%")
            print()
        
        # Recommendations
        print("  Recommendations:")
        print()
        
        abs_pct = abs(pct)
        if abs_pct <= 5:
            print("  ✓ Calculator is highly accurate for this configuration!")
            print("  ✓ No adjustments needed.")
        elif abs_pct <= 10:
            print("  ✓ Calculator accuracy is good.")
            print("  → Consider minor refinements for this model size/GPU combination.")
        elif abs_pct <= 20:
            if pct > 0:
                print("  ⚠ Calculator is underestimating memory.")
            else:
                print("  ⚠ Calculator is overestimating memory.")
            print(f"  → Review component breakdown to identify adjustment areas.")
            print(f"  → Test additional configurations (different batch/sequence lengths).")
        else:
            if pct > 0:
                print("  ✗ Calculator significantly underestimates memory.")
            else:
                print("  ✗ Calculator significantly overestimates memory.")
            print(f"  → Critical: Review calculation formulas.")
            print(f"  → Check if model uses non-standard architecture.")
            print(f"  → Verify profiling captured correct configuration.")
        print()
    
    print("=" * 80)
    print()


def append_to_csv(
    csv_path: str,
    file_name: str,
    calc_inputs: Dict[str, Any],
    profiled: Dict[str, float],
    validation: Optional[Dict[str, Any]] = None
):
    """Append validation results to CSV for tracking"""
    
    csv_file = Path(csv_path)
    file_exists = csv_file.exists()
    
    row = {
        'timestamp': datetime.now().isoformat(),
        'profile': file_name,
        'model': calc_inputs['model_name'],
        'params_b': calc_inputs['estimated_params_b'],
        'gpus': calc_inputs['num_gpus'],
        'batch': calc_inputs['batch_size'],
        'input_len': calc_inputs['input_len'],
        'output_len': calc_inputs['output_len'],
        'profiled_total_gb': profiled['total_gb'],
        'profiled_weights_gb': profiled['weights_gb'],
        'profiled_kv_gb': profiled['kv_cache_gb'],
        'profiled_activations_gb': profiled['activations_gb'],
        'profiled_overhead_gb': profiled['overhead_gb'],
        'latency_ms': profiled['latency_ms'],
    }
    
    if validation:
        row.update({
            'calculator_total_gb': validation['calculator_total_gb'],
            'diff_gb': validation['difference_gb'],
            'pct_diff': validation['percent_diff'],
            'match_quality': validation['match_quality'],
        })
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"✓ Results appended to: {csv_path}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate-profile.py <profile.json> [options]")
        print()
        print("Options:")
        print("  --calculator-total <GB>     Calculator's total memory estimate")
        print("  --record <csv-file>         Append results to CSV file")
        print()
        print("Examples:")
        print("  # View profile and get calculator inputs")
        print("  python scripts/validate-profile.py results/glm-profile.json")
        print()
        print("  # Validate with calculator result")
        print("  python scripts/validate-profile.py results/glm-profile.json --calculator-total 95.3")
        print()
        print("  # Track results over time")
        print("  python scripts/validate-profile.py results/glm-profile.json \\")
        print("      --calculator-total 95.3 --record validation-results.csv")
        sys.exit(1)
    
    profile_path = sys.argv[1]
    
    # Parse options
    calculator_total = None
    record_csv = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--calculator-total' and i + 1 < len(sys.argv):
            calculator_total = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--record' and i + 1 < len(sys.argv):
            record_csv = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    # Load profile
    try:
        profile = load_profile(profile_path)
    except FileNotFoundError:
        print(f"Error: File not found: {profile_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {profile_path}: {e}")
        sys.exit(1)
    
    # Extract data
    calc_inputs = get_calculator_inputs(profile)
    profiled = get_profiled_results(profile)
    
    # Validate if calculator total provided
    validation = None
    if calculator_total:
        validation = validate_comparison(profiled, calculator_total)
    
    # Print report
    file_name = Path(profile_path).name
    print_validation_report(file_name, calc_inputs, profiled, validation)
    
    # Record to CSV if requested
    if record_csv:
        append_to_csv(record_csv, file_name, calc_inputs, profiled, validation)


if __name__ == '__main__':
    main()
