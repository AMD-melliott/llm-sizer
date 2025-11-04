#!/usr/bin/env python3
"""
Quick comparison tool for bench-based profile outputs

This compares the profiled metrics against expected calculator values.
Since the TypeScript validator needs updating for the new format,
this provides immediate comparison capability.

Usage:
    python scripts/compare-bench-profiles.py scripts/results/profile.json
    python scripts/compare-bench-profiles.py scripts/results/*.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_profile(file_path: str) -> Dict[str, Any]:
    """Load profile JSON"""
    with open(file_path) as f:
        return json.load(f)


def analyze_profile(profile: Dict[str, Any], file_name: str):
    """Analyze and display profile metrics"""
    
    print("=" * 80)
    print(f"Profile Analysis: {file_name}")
    print("=" * 80)
    print()
    
    # Model Info
    model_info = profile.get('model_info', {})
    print("MODEL INFORMATION")
    print("-" * 80)
    print(f"  Model:           {model_info.get('name', 'Unknown')}")
    print(f"  Parameters:      {model_info.get('num_parameters', 'Not specified')}")
    print(f"  Data Type:       {model_info.get('dtype', 'Unknown')}")
    print(f"  Quantization:    {model_info.get('quantization') or 'None'}")
    print()
    
    # Benchmark Parameters
    bench_params = profile.get('benchmark_parameters', {})
    print("BENCHMARK CONFIGURATION")
    print("-" * 80)
    print(f"  Input Length:    {bench_params.get('input_len', 'Unknown')} tokens")
    print(f"  Output Length:   {bench_params.get('output_len', 'Unknown')} tokens")
    print(f"  Batch Size:      {bench_params.get('batch_size', 'Unknown')}")
    print(f"  Tensor Parallel: {bench_params.get('tensor_parallel_size', 'Unknown')} GPUs")
    print(f"  Total Seq Len:   {bench_params.get('input_len', 0) + bench_params.get('output_len', 0)} tokens")
    print()
    
    # GPU Info
    gpu_info = profile.get('gpu_info', {})
    mem_measurements = profile.get('memory_measurements', {})
    per_gpu_peak = mem_measurements.get('per_gpu_peak', [])
    
    print("GPU INFORMATION")
    print("-" * 80)
    print(f"  GPU Type:        {gpu_info.get('gpu_type', 'Unknown').upper()}")
    print(f"  Number of GPUs:  {gpu_info.get('num_gpus', 'Unknown')}")
    if per_gpu_peak:
        total_vram = per_gpu_peak[0].get('total_gb', 0)
        print(f"  VRAM per GPU:    {total_vram} GB")
        print(f"  Total VRAM:      {total_vram * len(per_gpu_peak)} GB")
    print()
    
    # Memory Breakdown
    mem_breakdown = profile.get('memory_breakdown', {})
    total_mem = mem_breakdown.get('total_measured_gb', 0)
    
    print("MEMORY BREAKDOWN (Profiled)")
    print("-" * 80)
    print(f"  Total Memory:       {total_mem:>8.2f} GB  (100%)")
    
    components = [
        ('Model Weights', 'model_weights_gb'),
        ('KV Cache', 'kv_cache_gb'),
        ('Activations', 'activations_gb'),
        ('Framework Overhead', 'framework_overhead_gb'),
    ]
    
    for name, key in components:
        value = mem_breakdown.get(key, 0)
        percentage = (value / total_mem * 100) if total_mem > 0 else 0
        print(f"  {name:<20} {value:>8.2f} GB  ({percentage:>5.1f}%)")
    
    print()
    print(f"  Estimation Method:  {mem_breakdown.get('estimation_method', 'Unknown')}")
    print()
    
    # Per-GPU Distribution
    if per_gpu_peak:
        print("PER-GPU MEMORY DISTRIBUTION")
        print("-" * 80)
        
        gpu_memories = [gpu.get('used_gb', 0) for gpu in per_gpu_peak]
        avg_memory = sum(gpu_memories) / len(gpu_memories)
        min_memory = min(gpu_memories)
        max_memory = max(gpu_memories)
        
        for gpu in per_gpu_peak:
            device = gpu.get('device', '?')
            used = gpu.get('used_gb', 0)
            bar_length = int((used / max_memory) * 40) if max_memory > 0 else 0
            bar = '█' * bar_length
            print(f"  GPU {device}:  {used:>6.2f} GB  {bar}")
        
        print()
        print(f"  Average:    {avg_memory:>6.2f} GB")
        print(f"  Min:        {min_memory:>6.2f} GB")
        print(f"  Max:        {max_memory:>6.2f} GB")
        print(f"  Imbalance:  {((max_memory - min_memory) / avg_memory * 100):>5.1f}% variation")
        print()
    
    # Latency Stats
    latency_stats = profile.get('latency_stats', {})
    if latency_stats:
        print("LATENCY STATISTICS")
        print("-" * 80)
        print(f"  Mean Latency:    {latency_stats.get('avg_latency', 0):>8.2f} ms")
        
        percentiles = latency_stats.get('percentiles', {})
        if percentiles:
            print(f"  p50 (Median):    {percentiles.get('50', 0):>8.2f} ms")
            print(f"  p90:             {percentiles.get('90', 0):>8.2f} ms")
            print(f"  p99:             {percentiles.get('99', 0):>8.2f} ms")
        
        num_iters = bench_params.get('num_iters', 0)
        if num_iters:
            print(f"  Iterations:      {num_iters}")
        print()
    
    # Calculator Comparison Inputs
    print("CALCULATOR COMPARISON INPUTS")
    print("=" * 80)
    print()
    print("To validate against the LLM Sizer calculator, input these values:")
    print()
    
    # Estimate model size if not provided
    weights_gb = mem_breakdown.get('model_weights_gb', 0)
    dtype = model_info.get('dtype', 'float16')
    bytes_per_param = 2 if dtype == 'float16' else 4
    estimated_params = (weights_gb * 1e9) / bytes_per_param / 1e9
    
    print("Model Selection:")
    if model_info.get('num_parameters'):
        param_billions = model_info['num_parameters'] / 1e9
        print(f"  - Search for model with {param_billions:.1f}B parameters")
    else:
        print(f"  - Estimated ~{estimated_params:.1f}B parameters (from {weights_gb:.1f} GB weights)")
    print(f"  - Or use Custom model input")
    print()
    
    print("GPU Configuration:")
    gpu_type_name = "AMD Instinct MI300X" if gpu_info.get('gpu_type') == 'rocm' else "NVIDIA H100"
    print(f"  - GPU Type: {gpu_type_name}")
    print(f"  - Number of GPUs: {gpu_info.get('num_gpus', 1)}")
    print(f"  - Partition Mode: Tensor Parallel")
    print()
    
    print("Inference Configuration:")
    print(f"  - Batch Size: {bench_params.get('batch_size', 1)}")
    print(f"  - Input Sequence: {bench_params.get('input_len', 256)} tokens")
    print(f"  - Output Sequence: {bench_params.get('output_len', 256)} tokens")
    print(f"  - Data Type: {dtype.upper()}")
    print(f"  - KV Cache Quantization: None (FP16)")
    if model_info.get('quantization'):
        print(f"  - Model Quantization: {model_info['quantization'].upper()}")
    print()
    
    print("Expected Calculator Outputs (to compare):")
    print(f"  - Total Memory:       ~{total_mem:.2f} GB")
    print(f"  - Model Weights:      ~{mem_breakdown.get('model_weights_gb', 0):.2f} GB")
    print(f"  - KV Cache:           ~{mem_breakdown.get('kv_cache_gb', 0):.2f} GB")
    print(f"  - Activations:        ~{mem_breakdown.get('activations_gb', 0):.2f} GB")
    print(f"  - Framework Overhead: ~{mem_breakdown.get('framework_overhead_gb', 0):.2f} GB")
    print()
    
    # Notes
    notes = mem_breakdown.get('notes', [])
    if notes:
        print("NOTES")
        print("-" * 80)
        for note in notes:
            print(f"  • {note}")
        print()
    
    print("=" * 80)
    print()


def compare_two_profiles(profile1: Dict[str, Any], profile2: Dict[str, Any], 
                        name1: str, name2: str):
    """Compare two profiles side by side"""
    
    print("=" * 80)
    print(f"PROFILE COMPARISON: {name1} vs {name2}")
    print("=" * 80)
    print()
    
    mem1 = profile1.get('memory_breakdown', {})
    mem2 = profile2.get('memory_breakdown', {})
    
    bench1 = profile1.get('benchmark_parameters', {})
    bench2 = profile2.get('benchmark_parameters', {})
    
    gpu1 = profile1.get('gpu_info', {})
    gpu2 = profile2.get('gpu_info', {})
    
    # Configuration comparison
    print("CONFIGURATION")
    print("-" * 80)
    print(f"{'Parameter':<25} {name1:<20} {name2:<20} {'Difference':<15}")
    print("-" * 80)
    
    config_items = [
        ('Model', profile1.get('model_info', {}).get('name', 'Unknown'),
                 profile2.get('model_info', {}).get('name', 'Unknown')),
        ('Batch Size', bench1.get('batch_size', 0),
                      bench2.get('batch_size', 0)),
        ('Input Length', bench1.get('input_len', 0),
                        bench2.get('input_len', 0)),
        ('Output Length', bench1.get('output_len', 0),
                         bench2.get('output_len', 0)),
        ('Tensor Parallel', bench1.get('tensor_parallel_size', 0),
                           bench2.get('tensor_parallel_size', 0)),
        ('GPUs', gpu1.get('num_gpus', 0), gpu2.get('num_gpus', 0)),
    ]
    
    for name, val1, val2 in config_items:
        diff = ""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 != val2:
                diff = f"{val2 - val1:+.0f}"
        print(f"{name:<25} {str(val1):<20} {str(val2):<20} {diff:<15}")
    
    print()
    
    # Memory comparison
    print("MEMORY BREAKDOWN")
    print("-" * 80)
    print(f"{'Component':<25} {name1:<15} {name2:<15} {'Diff (GB)':<12} {'% Change':<10}")
    print("-" * 80)
    
    components = [
        ('Total Memory', 'total_measured_gb'),
        ('Model Weights', 'model_weights_gb'),
        ('KV Cache', 'kv_cache_gb'),
        ('Activations', 'activations_gb'),
        ('Framework Overhead', 'framework_overhead_gb'),
    ]
    
    for name, key in components:
        val1 = mem1.get(key, 0)
        val2 = mem2.get(key, 0)
        diff = val2 - val1
        pct_change = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
        
        print(f"{name:<25} {val1:>8.2f} GB    {val2:>8.2f} GB    {diff:>+7.2f} GB   {pct_change:>+6.1f}%")
    
    print()
    
    # Latency comparison
    latency1 = profile1.get('latency_stats', {}).get('avg_latency', 0)
    latency2 = profile2.get('latency_stats', {}).get('avg_latency', 0)
    
    if latency1 and latency2:
        print("LATENCY")
        print("-" * 80)
        print(f"Profile 1: {latency1:>8.2f} ms")
        print(f"Profile 2: {latency2:>8.2f} ms")
        print(f"Difference: {latency2 - latency1:>+7.2f} ms ({(latency2/latency1 - 1)*100:>+6.1f}%)")
        print()
    
    # Per-GPU memory comparison
    mem_meas1 = profile1.get('memory_measurements', {})
    mem_meas2 = profile2.get('memory_measurements', {})
    
    peak1 = mem_meas1.get('per_gpu_peak', [])
    peak2 = mem_meas2.get('per_gpu_peak', [])
    
    if peak1 and peak2:
        print("PER-GPU MEMORY")
        print("-" * 80)
        
        avg1 = sum(g.get('used_gb', 0) for g in peak1) / len(peak1) if peak1 else 0
        avg2 = sum(g.get('used_gb', 0) for g in peak2) / len(peak2) if peak2 else 0
        
        print(f"Average per GPU:")
        print(f"  Profile 1: {avg1:>6.2f} GB ({len(peak1)} GPUs)")
        print(f"  Profile 2: {avg2:>6.2f} GB ({len(peak2)} GPUs)")
        print()
    
    print("=" * 80)
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare-bench-profiles.py <profile1.json> [profile2.json]")
        print()
        print("Examples:")
        print("  python scripts/compare-bench-profiles.py results/glm-profile.json")
        print("  python scripts/compare-bench-profiles.py results/glm-4gpu.json results/glm-8gpu.json")
        sys.exit(1)
    
    profile_files = sys.argv[1:]
    
    # Load profiles
    profiles = []
    for file_path in profile_files:
        try:
            profile = load_profile(file_path)
            profiles.append((profile, Path(file_path).name))
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}")
            sys.exit(1)
    
    # Analyze each profile
    for profile, name in profiles:
        analyze_profile(profile, name)
    
    # If two profiles, do comparison
    if len(profiles) == 2:
        compare_two_profiles(
            profiles[0][0], profiles[1][0],
            profiles[0][1], profiles[1][1]
        )


if __name__ == '__main__':
    main()
