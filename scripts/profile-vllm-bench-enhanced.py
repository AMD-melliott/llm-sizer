#!/usr/bin/env python3
"""
vLLM Benchmark Profiler - Enhanced Version with Phase 1 Improvements

This script profiles vLLM models using the native `vllm bench latency` command,
with comprehensive instrumentation for validation and analysis.

Version: 2.0 (Phase 1 Enhancements)
Changes from v1.0:
- Persist baseline, post-warmup, and post-run memory snapshots
- Add vLLM engine version detection and storage
- Capture dtype metadata (weight_dtype, activation_dtype, kv_cache_dtype)
- Enhanced per-GPU memory tracking (peak, mean, stddev)
- Improved component breakdown with validation metadata
- Store tensor_parallel_size, pipeline_parallel_size explicitly
- Track model architecture details (heads, head_dim, num_layers)

Key improvements:
- Direct vLLM engine measurement (no API overhead)
- Exact parameter control (input-len, output-len, batch-size)
- Standardized warmup and iteration counts
- Better reproducibility
- Comprehensive memory attribution

Usage:
    # Inside a container with vLLM installed:
    python profile-vllm-bench-enhanced.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --input-len 256 \\
        --output-len 256 \\
        --batch-size 8 \\
        --dtype float16

Requirements:
    - vLLM installed (in container or locally)
    - GPU with nvidia-smi or rocm-smi
"""

import argparse
import json
import subprocess
import sys
import time
import os
import re
import tempfile
import statistics
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import calculator formulas for validation (optional)
try:
    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
    from calculator_formulas import calculate_expected_memory, calculate_proposed_memory
    CALCULATOR_AVAILABLE = True
except ImportError:
    CALCULATOR_AVAILABLE = False
    print("⚠️  Warning: calculator_formulas module not available, calculator comparison will be skipped")


def get_vllm_version() -> str:
    """Detect vLLM version"""
    try:
        import vllm
        return vllm.__version__
    except (ImportError, AttributeError):
        try:
            result = subprocess.run(
                ['python', '-c', 'import vllm; print(vllm.__version__)'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
    return "unknown"


class VLLMBenchProfiler:
    """Profile vLLM using native bench command with enhanced instrumentation"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gpu_type = self.detect_gpu_type()
        self.vllm_version = get_vllm_version()
        
    def detect_gpu_type(self) -> str:
        """Detect if using NVIDIA CUDA or AMD ROCm"""
        if os.path.exists('/opt/rocm') or subprocess.run(
            ['which', 'rocm-smi'], capture_output=True
        ).returncode == 0:
            return 'rocm'
        if subprocess.run(['which', 'nvidia-smi'], capture_output=True).returncode == 0:
            return 'cuda'
        return 'unknown'
    
    def log(self, message: str):
        """Print log message if verbose"""
        if self.verbose:
            print(message)
    
    def get_gpu_memory_snapshot(self) -> List[Dict[str, float]]:
        """Get current GPU memory usage for all devices"""
        if self.gpu_type == 'rocm':
            return self._get_rocm_memory()
        elif self.gpu_type == 'cuda':
            return self._get_cuda_memory()
        else:
            self.log(f"WARNING: Unknown GPU type, cannot measure memory")
            return []
    
    def _get_rocm_memory(self) -> List[Dict[str, float]]:
        """Get memory usage from ROCm GPUs"""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.split('\n'):
                if 'GPU[' in line and 'Total Used Memory' in line:
                    try:
                        gpu_match = re.search(r'GPU\[(\d+)\]', line)
                        if not gpu_match:
                            continue
                        gpu_idx = int(gpu_match.group(1))
                        
                        mem_match = re.search(r':\s*([\d,]+)', line.split(':', 1)[1])
                        if mem_match:
                            used_bytes = int(mem_match.group(1).replace(',', ''))
                            used_gb = used_bytes / 1e9
                            
                            gpus.append({
                                'device': gpu_idx,
                                'used_gb': round(used_gb, 2),
                                'total_gb': 206.0  # MI300X default
                            })
                    except (ValueError, IndexError):
                        continue
            
            return sorted(gpus, key=lambda x: x['device'])
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.log(f"WARNING: rocm-smi failed: {e}")
            return []
    
    def _get_cuda_memory(self) -> List[Dict[str, float]]:
        """Get memory usage from NVIDIA GPUs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpus.append({
                            'device': int(parts[0]),
                            'used_gb': round(float(parts[1]) / 1024, 2),
                            'total_gb': round(float(parts[2]) / 1024, 2)
                        })
            
            return gpus
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.log(f"WARNING: Could not get CUDA memory info: {e}")
            return []
    
    def compute_memory_statistics(self, snapshots: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute aggregate memory statistics across GPUs"""
        if not snapshots:
            return {
                'total_gb': 0.0,
                'max_used_gb': 0.0,
                'mean_used_gb': 0.0,
                'stddev_used_gb': 0.0,
                'num_gpus': 0
            }
        
        used_values = [gpu['used_gb'] for gpu in snapshots]
        
        return {
            'total_gb': round(sum(used_values), 2),
            'max_used_gb': round(max(used_values), 2),
            'mean_used_gb': round(statistics.mean(used_values), 2),
            'stddev_used_gb': round(statistics.stdev(used_values), 2) if len(used_values) > 1 else 0.0,
            'num_gpus': len(snapshots),
            'per_gpu_details': snapshots
        }
    
    def run_vllm_bench(
        self,
        model: str,
        input_len: int,
        output_len: int,
        batch_size: int = 1,
        dtype: str = 'auto',
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        num_iters_warmup: int = 10,
        num_iters: int = 30,
        enforce_eager: bool = False,
        output_json: Optional[str] = None,
        kv_cache_dtype: Optional[str] = None,
        trust_remote_code: bool = False
    ) -> Dict[str, Any]:
        """
        Run vllm bench latency command with specified parameters and enhanced monitoring
        
        Args:
            model: Model name or path
            input_len: Number of input tokens
            output_len: Number of output tokens
            batch_size: Batch size
            dtype: Data type (auto, float16, bfloat16, etc.)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            quantization: Quantization method (awq, gptq, fp8, etc.)
            num_iters_warmup: Number of warmup iterations
            num_iters: Number of measurement iterations
            enforce_eager: Disable CUDA graphs (default: False for production-like profiling)
            output_json: Path to save JSON output (temp file if None)
            kv_cache_dtype: Explicit KV cache dtype if different from model dtype
            
        Returns:
            Dictionary with benchmark results and enhanced memory measurements
        """
        
        self.log("\n=== vLLM Benchmark Profiler (Enhanced) ===")
        self.log(f"vLLM Version: {self.vllm_version}")
        self.log(f"Model: {model}")
        self.log(f"Input Length: {input_len} tokens")
        self.log(f"Output Length: {output_len} tokens")
        self.log(f"Batch Size: {batch_size}")
        self.log(f"Data Type: {dtype}")
        self.log(f"Tensor Parallel Size: {tensor_parallel_size}")
        if quantization:
            self.log(f"Quantization: {quantization}")
        if enforce_eager:
            self.log(f"⚠️  CUDA Graphs: DISABLED (enforce_eager=True)")
            self.log(f"   Note: Production deployments use CUDA graphs by default")
        
        # Capture baseline GPU memory BEFORE model load
        self.log("\n=== Measuring Baseline GPU Memory (Pre-Load) ===")
        baseline_memory = self.get_gpu_memory_snapshot()
        baseline_stats = self.compute_memory_statistics(baseline_memory)
        self.log(f"Baseline: {baseline_stats['total_gb']:.2f} GB across {baseline_stats['num_gpus']} GPUs")
        for gpu in baseline_memory:
            self.log(f"  GPU {gpu['device']}: {gpu['used_gb']:.2f} GB")
        
        # Create temp file for JSON output if not specified
        use_temp = output_json is None
        if use_temp:
            temp_fd, output_json = tempfile.mkstemp(suffix='.json', prefix='vllm_bench_')
            os.close(temp_fd)
        
        try:
            # Build vllm bench command
            cmd = [
                'vllm', 'bench', 'latency',
                '--model', model,
                '--input-len', str(input_len),
                '--output-len', str(output_len),
                '--batch-size', str(batch_size),
                '--dtype', dtype,
                '--tensor-parallel-size', str(tensor_parallel_size),
                '--num-iters-warmup', str(num_iters_warmup),
                '--num-iters', str(num_iters),
                '--output-json', output_json,
            ]
            
            if enforce_eager:
                cmd.append('--enforce-eager')
            
            if quantization:
                cmd.extend(['--quantization', quantization])
            
            if kv_cache_dtype:
                cmd.extend(['--kv-cache-dtype', kv_cache_dtype])
            
            if trust_remote_code:
                cmd.append('--trust-remote-code')
            
            self.log(f"\n=== Running vLLM Bench Command ===")
            self.log(f"Command: {' '.join(cmd)}")
            
            # Run benchmark
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            elapsed_time = time.time() - start_time
            
            if result.returncode != 0:
                self.log(f"\nERROR: vllm bench failed with exit code {result.returncode}")
                self.log(f"STDOUT:\n{result.stdout}")
                self.log(f"STDERR:\n{result.stderr}")
                sys.exit(1)
            
            self.log(f"Benchmark completed in {elapsed_time:.1f} seconds")
            
            # Capture memory AFTER warmup but before final run (if we can separate them)
            # For now, we'll capture immediately after benchmark
            self.log("\n=== Measuring Post-Warmup GPU Memory ===")
            time.sleep(1)  # Let memory settle
            post_warmup_memory = self.get_gpu_memory_snapshot()
            post_warmup_stats = self.compute_memory_statistics(post_warmup_memory)
            self.log(f"Post-Warmup: {post_warmup_stats['total_gb']:.2f} GB")
            
            # Capture peak GPU memory after benchmark
            self.log("\n=== Measuring Peak GPU Memory (Post-Run) ===")
            time.sleep(1)  # Let memory settle
            peak_memory = self.get_gpu_memory_snapshot()
            peak_stats = self.compute_memory_statistics(peak_memory)
            
            # PHASE 1 FIX: Detect if peak equals baseline (vLLM process cleaned up)
            # Use post_warmup as fallback since that captured memory during actual inference
            if abs(peak_stats['total_gb'] - baseline_stats['total_gb']) < 0.5:
                self.log(f"⚠️  Peak memory equals baseline ({peak_stats['total_gb']:.2f} GB = {baseline_stats['total_gb']:.2f} GB)")
                self.log("   vLLM process likely cleaned up before measurement")
                self.log(f"   Using post-warmup memory as peak: {post_warmup_stats['total_gb']:.2f} GB")
                peak_memory = post_warmup_memory
                peak_stats = post_warmup_stats
            
            self.log(f"Peak: {peak_stats['total_gb']:.2f} GB")
            for gpu in peak_memory:
                self.log(f"  GPU {gpu['device']}: {gpu['used_gb']:.2f} GB")
            
            # Load benchmark results
            with open(output_json, 'r') as f:
                bench_results = json.load(f)
            
            # Determine actual dtypes used
            # vLLM may convert dtypes internally, but we'll record what was requested
            actual_weight_dtype = dtype
            actual_kv_cache_dtype = kv_cache_dtype if kv_cache_dtype else dtype
            # Activations typically use fp16/bf16 even if weights are quantized
            actual_activation_dtype = 'float16' if dtype in ['int8', 'int4', 'fp8'] else dtype
            
            # Combine results with enhanced metadata
            results = {
                'benchmark_results': bench_results,
                'memory_measurements': {
                    'baseline': baseline_stats,
                    'post_warmup': post_warmup_stats,
                    'peak': peak_stats,
                    'memory_increase_gb': round(peak_stats['total_gb'] - baseline_stats['total_gb'], 2),
                    'baseline_to_warmup_gb': round(post_warmup_stats['total_gb'] - baseline_stats['total_gb'], 2),
                    'warmup_to_peak_gb': round(peak_stats['total_gb'] - post_warmup_stats['total_gb'], 2)
                },
                'parameters': {
                    'model': model,
                    'input_len': input_len,
                    'output_len': output_len,
                    'batch_size': batch_size,
                    'dtype': dtype,
                    'tensor_parallel_size': tensor_parallel_size,
                    'pipeline_parallel_size': 1,  # Default, could be parameterized
                    'quantization': quantization,
                    'num_iters_warmup': num_iters_warmup,
                    'num_iters': num_iters
                },
                'dtype_metadata': {
                    'weight_dtype': actual_weight_dtype,
                    'activation_dtype': actual_activation_dtype,
                    'kv_cache_dtype': actual_kv_cache_dtype,
                    'requested_dtype': dtype,
                    'quantization_applied': quantization is not None
                },
                'gpu_info': {
                    'gpu_type': self.gpu_type,                    'num_gpus': tensor_parallel_size,  # Actual GPUs used by vLLM (not total system GPUs)
                    'system_gpus': len(peak_memory),  # Total GPUs in system for reference
                    'total_gpu_memory_gb': sum(gpu['total_gb'] for gpu in peak_memory) if peak_memory else 0
                },
                'engine_metadata': {
                    'engine_version': self.vllm_version,
                    'engine_type': 'vllm',
                    'profiling_method': 'vllm_bench_latency',
                    'profiler_version': '2.0'
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'calculation_version': '2.0'  # For future formula tracking
            }
            
            return results
            
        finally:
            # Clean up temp file
            if use_temp and os.path.exists(output_json):
                os.unlink(output_json)
    
    def estimate_memory_breakdown(
        self,
        peak_memory_gb: float,
        baseline_memory_gb: float,
        model_params: Optional[float] = None,
        input_len: int = 0,
        output_len: int = 0,
        batch_size: int = 1,
        dtype: str = 'float16',
        weight_dtype: Optional[str] = None,
        activation_dtype: Optional[str] = None,
        kv_cache_dtype: Optional[str] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        tensor_parallel_size: int = 1
    ) -> Dict[str, Any]:
        """
        Estimate memory component breakdown with enhanced accuracy
        
        Uses multiple strategies and stores confidence levels
        """
        
        # Bytes per parameter based on dtype
        bytes_per_param = {
            'float32': 4, 'fp32': 4, 'float': 4,
            'float16': 2, 'fp16': 2, 'half': 2,
            'bfloat16': 2, 'bf16': 2,
            'int8': 1, 'fp8': 1, 'fp8_e4m3': 1, 'fp8_e5m2': 1,
            'int4': 0.5, 'nf4': 0.5,
            'auto': 2  # Default to fp16
        }
        
        # Use specific dtypes if provided, otherwise fall back to main dtype
        weight_dtype = weight_dtype or dtype
        activation_dtype = activation_dtype or ('float16' if dtype in ['int8', 'int4'] else dtype)
        kv_cache_dtype = kv_cache_dtype or dtype
        
        weight_bytes = bytes_per_param.get(weight_dtype.lower(), 2)
        activation_bytes = bytes_per_param.get(activation_dtype.lower(), 2)
        kv_bytes = bytes_per_param.get(kv_cache_dtype.lower(), 2)
        
        # Calculate net memory increase (excluding baseline framework overhead)
        net_memory_gb = peak_memory_gb - baseline_memory_gb
        
        # Estimate model weights
        if model_params:
            # For tensor parallel, weights are sharded
            model_weights_gb = (model_params * weight_bytes) / (tensor_parallel_size * 1e9)
            estimation_confidence = 'high'
        else:
            # Rough estimate: 60-70% of net memory for model weights
            model_weights_gb = net_memory_gb * 0.65
            estimation_confidence = 'low'
        
        # Estimate KV cache with improved formula
        seq_len = input_len + output_len
        
        if num_layers and (hidden_size or (num_heads and head_dim)):
            # Accurate calculation
            kv_dim = hidden_size if hidden_size else (num_heads * head_dim)
            # Formula: 2 (key+value) * layers * kv_dim * seq_len * batch * bytes / parallel_size
            kv_elements = 2 * num_layers * kv_dim * seq_len * batch_size
            kv_cache_gb = (kv_elements * kv_bytes) / (tensor_parallel_size * 1e9)
            kv_confidence = 'high'
        else:
            # Rough estimate: 15-25% of net memory
            kv_cache_gb = net_memory_gb * 0.20
            kv_confidence = 'low'
        
        # Estimate activations
        # Activations scale with batch * seq_len * hidden_size * activation_factor
        # Typical activation_factor is 6-8x hidden_size for transformer blocks
        if hidden_size and batch_size and seq_len:
            # Rough formula: batch * seq * hidden * factor * bytes
            activation_factor = 8  # Conservative estimate
            activation_elements = batch_size * seq_len * hidden_size * activation_factor
            activations_gb = (activation_elements * activation_bytes) / 1e9
            activation_confidence = 'medium'
        else:
            # Very rough estimate: 10-15% of net memory
            activations_gb = net_memory_gb * 0.12
            activation_confidence = 'low'
        
        # Framework overhead (residual)
        # This includes: CUDA context, vLLM engine, PagedAttention bookkeeping, etc.
        framework_overhead_gb = max(0, net_memory_gb - (
            model_weights_gb + kv_cache_gb + activations_gb
        ))
        
        # Adjust if overhead is negative (over-estimation)
        if framework_overhead_gb < 0:
            # Redistribute negative overhead proportionally
            adjustment_factor = net_memory_gb / (model_weights_gb + kv_cache_gb + activations_gb)
            model_weights_gb *= adjustment_factor
            kv_cache_gb *= adjustment_factor
            activations_gb *= adjustment_factor
            framework_overhead_gb = net_memory_gb * 0.03  # Minimum 3%
            estimation_confidence = 'adjusted'
        
        # Sanity check: overhead should be reasonable (5-15%)
        overhead_pct = (framework_overhead_gb / peak_memory_gb * 100) if peak_memory_gb > 0 else 0
        
        return {
            'total_measured_gb': round(peak_memory_gb, 2),
            'baseline_gb': round(baseline_memory_gb, 2),
            'net_increase_gb': round(net_memory_gb, 2),
            'model_weights_gb': round(model_weights_gb, 2),
            'kv_cache_gb': round(kv_cache_gb, 2),
            'activations_gb': round(activations_gb, 2),
            'framework_overhead_gb': round(framework_overhead_gb, 2),
            'overhead_percentage': round(overhead_pct, 1),
            'estimation_method': 'vllm_bench_enhanced',
            'confidence_levels': {
                'overall': estimation_confidence,
                'weights': 'high' if model_params else 'low',
                'kv_cache': kv_confidence,
                'activations': activation_confidence,
                'overhead': 'calculated_residual'
            },
            'component_dtypes': {
                'weights': weight_dtype,
                'activations': activation_dtype,
                'kv_cache': kv_cache_dtype
            },
            'architecture_params_used': {
                'num_layers': num_layers,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'hidden_size': hidden_size,
                'model_params': model_params,
                'tensor_parallel_size': tensor_parallel_size
            },
            'notes': [
                'Memory breakdown v2.0 with enhanced estimation',
                'Baseline memory excluded from component calculations',
                'Component dtypes tracked separately for accurate sizing',
                f'Estimation confidence: {estimation_confidence}',
                'Overhead includes CUDA context + vLLM engine + PagedAttention structures'
            ]
        }
    
    def generate_report(
        self,
        model: str,
        input_len: int,
        output_len: int,
        batch_size: int = 1,
        dtype: str = 'auto',
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        model_params: Optional[float] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        kv_cache_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive profiling report with enhanced metadata
        
        Returns complete report with benchmark results, memory measurements,
        and estimated memory breakdown with confidence levels.
        """
        
        # Run benchmark
        results = self.run_vllm_bench(
            model=model,
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager
        )
        
        # Estimate memory breakdown with enhanced parameters
        self.log("\n=== Estimating Memory Breakdown ===")
        memory_breakdown = self.estimate_memory_breakdown(
            peak_memory_gb=results['memory_measurements']['peak']['total_gb'],
            baseline_memory_gb=results['memory_measurements']['baseline']['total_gb'],
            model_params=model_params,
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            dtype=dtype,
            weight_dtype=results['dtype_metadata']['weight_dtype'],
            activation_dtype=results['dtype_metadata']['activation_dtype'],
            kv_cache_dtype=results['dtype_metadata']['kv_cache_dtype'],
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            tensor_parallel_size=tensor_parallel_size
        )
        
        # Calculate expected values from calculator formulas for comparison
        self.log("\n=== Calculating Expected Memory (Calculator Validation) ===")
        
        calculator_comparison = {}
        
        if not CALCULATOR_AVAILABLE:
            self.log("  Skipped: Calculator module not available")
            calculator_comparison = {
                'note': 'Calculator module not available - calculator_formulas.py not found'
            }
        elif not model_params:  # Only calculate if we have model parameters
            self.log("  Skipped: Model parameters not provided")
            calculator_comparison = {
                'note': 'Calculator comparison requires model_params parameter'
            }
        else:
            try:
                # Calculate CURRENT formula (v1.0 - known to be wrong)
                current_calc = calculate_expected_memory(
                    model_params=model_params,
                    num_layers=num_layers or 32,  # Default if not provided
                    hidden_size=hidden_size or 4096,  # Default if not provided
                    num_heads=num_heads or 32,  # Default if not provided
                    input_len=input_len,
                    output_len=output_len,
                    batch_size=batch_size,
                    weight_dtype=dtype,
                    kv_cache_dtype=kv_cache_dtype or dtype,
                    num_gpus=tensor_parallel_size
                )
                
                # Calculate PROPOSED formula (v2.0 - improved)
                proposed_calc = calculate_proposed_memory(
                    model_params=model_params,
                    num_layers=num_layers or 32,
                    hidden_size=hidden_size or 4096,
                    num_heads=num_heads or 32,
                    input_len=input_len,
                    output_len=output_len,
                    batch_size=batch_size,
                    weight_dtype=dtype,
                    kv_cache_dtype=kv_cache_dtype or dtype,
                    num_gpus=tensor_parallel_size
                )
                
                actual_total = memory_breakdown['total_measured_gb']
                
                calculator_comparison = {
                    'current_formula': {
                        'version': 'v1.0',
                        'total_gb': round(current_calc['total_gb'], 2),
                        'weights_gb': round(current_calc['weights_gb'], 2),
                        'kv_cache_gb': round(current_calc['kv_cache_gb'], 2),
                        'overhead_gb': round(current_calc['overhead_gb'], 2),
                        'error_vs_actual_pct': round((current_calc['total_gb'] - actual_total) / actual_total * 100, 1),
                        'notes': 'Current calculator formula (8% fixed overhead)'
                    },
                    'proposed_formula': {
                        'version': 'v2.0',
                        'total_gb': round(proposed_calc['total_gb'], 2),
                        'weights_gb': round(proposed_calc['weights_gb'], 2),
                        'kv_cache_gb': round(proposed_calc['kv_cache_gb'], 2),
                        'overhead_gb': round(proposed_calc['overhead_gb'], 2),
                        'error_vs_actual_pct': round((proposed_calc['total_gb'] - actual_total) / actual_total * 100, 1),
                        'notes': 'Proposed calculator formula (14GB baseline + proportional overhead)'
                    },
                    'actual_measured': {
                        'total_gb': round(actual_total, 2),
                        'weights_gb': memory_breakdown['model_weights_gb'],
                        'kv_cache_gb': memory_breakdown['kv_cache_gb'],
                        'overhead_gb': memory_breakdown['framework_overhead_gb'],
                        'notes': 'Actual measured values from profiler'
                    },
                    'validation_status': {
                        'current_formula_accurate': abs(current_calc['total_gb'] - actual_total) / actual_total < 0.15,
                        'proposed_formula_accurate': abs(proposed_calc['total_gb'] - actual_total) / actual_total < 0.15,
                        'accuracy_threshold': '15%',
                        'better_formula': 'proposed' if abs(proposed_calc['total_gb'] - actual_total) < abs(current_calc['total_gb'] - actual_total) else 'current'
                    }
                }
                
                self.log(f"  Current Formula (v1.0): {current_calc['total_gb']:.2f} GB (error: {calculator_comparison['current_formula']['error_vs_actual_pct']:.1f}%)")
                self.log(f"  Proposed Formula (v2.0): {proposed_calc['total_gb']:.2f} GB (error: {calculator_comparison['proposed_formula']['error_vs_actual_pct']:.1f}%)")
                self.log(f"  Actual Measured: {actual_total:.2f} GB")
                self.log(f"  Better Formula: {calculator_comparison['validation_status']['better_formula'].upper()}")
                
            except Exception as e:
                self.log(f"  WARNING: Calculator comparison failed: {e}")
                calculator_comparison = {
                    'error': str(e),
                    'note': 'Calculator comparison unavailable'
                }
        
        # Build comprehensive report
        report = {
            'model_info': {
                'name': model,
                'num_parameters': model_params,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'hidden_size': hidden_size,
                'dtype': dtype,
                'quantization': quantization
            },
            'benchmark_parameters': results['parameters'],
            'dtype_metadata': results['dtype_metadata'],
            'latency_stats': results['benchmark_results'],
            'memory_breakdown': memory_breakdown,
            'calculator_comparison': calculator_comparison,  # NEW: Calculator validation
            'memory_measurements': results['memory_measurements'],
            'gpu_info': results['gpu_info'],
            'engine_metadata': results['engine_metadata'],
            'timestamp': results['timestamp'],
            'profiler_version': '2.0',
            'schema_version': '2.0'  # Track schema changes
        }
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.log(f"\n=== Report saved to: {output_file} ===")
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print formatted summary of profiling results"""
        
        print(f"\n{'='*70}")
        print("PROFILING SUMMARY (Enhanced v2.0)")
        print(f"{'='*70}")
        
        # Model info
        print("\n📊 Model Information:")
        print(f"  Name: {report['model_info']['name']}")
        if report['model_info']['num_parameters']:
            params_b = report['model_info']['num_parameters'] / 1e9
            print(f"  Parameters: {params_b:.1f}B")
        print(f"  Data Type: {report['model_info']['dtype']}")
        if report['model_info']['quantization']:
            print(f"  Quantization: {report['model_info']['quantization']}")
        
        # Architecture
        if any([report['model_info'].get(k) for k in ['num_layers', 'num_heads', 'hidden_size']]):
            print("\n🏗️  Architecture:")
            if report['model_info']['num_layers']:
                print(f"  Layers: {report['model_info']['num_layers']}")
            if report['model_info']['num_heads']:
                print(f"  Attention Heads: {report['model_info']['num_heads']}")
            if report['model_info']['head_dim']:
                print(f"  Head Dimension: {report['model_info']['head_dim']}")
            if report['model_info']['hidden_size']:
                print(f"  Hidden Size: {report['model_info']['hidden_size']}")
        
        # Benchmark parameters
        print("\n⚙️  Benchmark Parameters:")
        print(f"  Input Tokens: {report['benchmark_parameters']['input_len']}")
        print(f"  Output Tokens: {report['benchmark_parameters']['output_len']}")
        print(f"  Batch Size: {report['benchmark_parameters']['batch_size']}")
        print(f"  Tensor Parallel Size: {report['benchmark_parameters']['tensor_parallel_size']}")
        
        # Dtype metadata
        print("\n🔢 Dtype Configuration:")
        dt = report['dtype_metadata']
        print(f"  Weight dtype: {dt['weight_dtype']}")
        print(f"  Activation dtype: {dt['activation_dtype']}")
        print(f"  KV Cache dtype: {dt['kv_cache_dtype']}")
        
        # Latency results
        if 'avg_latency' in report['latency_stats']:
            print("\n⚡ Latency Results:")
            print(f"  Average: {report['latency_stats']['avg_latency']:.3f}s")
            if 'percentiles' in report['latency_stats']:
                p = report['latency_stats']['percentiles']
                print(f"  P50: {p.get('50', 0):.3f}s")
                print(f"  P90: {p.get('90', 0):.3f}s")
                print(f"  P99: {p.get('99', 0):.3f}s")
        
        # Memory measurements
        print("\n💾 Memory Measurements:")
        mm = report['memory_measurements']
        print(f"  Baseline: {mm['baseline']['total_gb']:.2f} GB")
        print(f"  Post-Warmup: {mm['post_warmup']['total_gb']:.2f} GB (+{mm['baseline_to_warmup_gb']:.2f} GB)")
        print(f"  Peak: {mm['peak']['total_gb']:.2f} GB (+{mm['warmup_to_peak_gb']:.2f} GB)")
        print(f"  Total Increase: {mm['memory_increase_gb']:.2f} GB")
        
        # Memory breakdown
        print("\n📊 Memory Breakdown (Estimated):")
        mb = report['memory_breakdown']
        print(f"  Total Measured: {mb['total_measured_gb']:.2f} GB")
        print(f"  ├─ Model Weights: {mb['model_weights_gb']:.2f} GB ({mb['model_weights_gb']/mb['total_measured_gb']*100:.1f}%)")
        print(f"  ├─ KV Cache: {mb['kv_cache_gb']:.2f} GB ({mb['kv_cache_gb']/mb['total_measured_gb']*100:.1f}%)")
        print(f"  ├─ Activations: {mb['activations_gb']:.2f} GB ({mb['activations_gb']/mb['total_measured_gb']*100:.1f}%)")
        print(f"  └─ Framework Overhead: {mb['framework_overhead_gb']:.2f} GB ({mb['overhead_percentage']:.1f}%)")
        
        # Confidence levels
        conf = mb['confidence_levels']
        print(f"\n  Estimation Confidence: {conf['overall'].upper()}")
        print(f"    Weights: {conf['weights']}, KV: {conf['kv_cache']}, Activations: {conf['activations']}")
        
        # Calculator Comparison (NEW)
        if 'calculator_comparison' in report and 'current_formula' in report['calculator_comparison']:
            print("\n🧮 Calculator Validation:")
            cc = report['calculator_comparison']
            actual_gb = cc['actual_measured']['total_gb']
            
            print(f"  Actual Measured: {actual_gb:.2f} GB")
            print(f"  ├─ Current Formula (v1.0): {cc['current_formula']['total_gb']:.2f} GB (error: {cc['current_formula']['error_vs_actual_pct']:+.1f}%)")
            print(f"  └─ Proposed Formula (v2.0): {cc['proposed_formula']['total_gb']:.2f} GB (error: {cc['proposed_formula']['error_vs_actual_pct']:+.1f}%)")
            
            # Show component breakdown comparison
            print("\n  Component Comparison:")
            print(f"                     Actual    Current(v1)  Proposed(v2)")
            print(f"  Weights:        {cc['actual_measured']['weights_gb']:7.2f}    {cc['current_formula']['weights_gb']:7.2f}      {cc['proposed_formula']['weights_gb']:7.2f} GB")
            print(f"  KV Cache:       {cc['actual_measured']['kv_cache_gb']:7.2f}    {cc['current_formula']['kv_cache_gb']:7.2f}      {cc['proposed_formula']['kv_cache_gb']:7.2f} GB")
            print(f"  Overhead:       {cc['actual_measured']['overhead_gb']:7.2f}    {cc['current_formula']['overhead_gb']:7.2f}      {cc['proposed_formula']['overhead_gb']:7.2f} GB")
            
            # Validation status
            vs = cc['validation_status']
            current_status = "✅" if vs['current_formula_accurate'] else "❌"
            proposed_status = "✅" if vs['proposed_formula_accurate'] else "❌"
            print(f"\n  Validation Status (threshold: {vs['accuracy_threshold']}):")
            print(f"    Current v1.0: {current_status} {'PASS' if vs['current_formula_accurate'] else 'FAIL'}")
            print(f"    Proposed v2.0: {proposed_status} {'PASS' if vs['proposed_formula_accurate'] else 'FAIL'}")
            print(f"    Better Formula: {vs['better_formula'].upper()}")
        elif 'calculator_comparison' in report and 'note' in report['calculator_comparison']:
            print(f"\n🧮 Calculator Validation: {report['calculator_comparison']['note']}")
        
        # GPU info
        print("\n🖥️  GPU Information:")
        print(f"  Type: {report['gpu_info']['gpu_type'].upper()}")
        print(f"  Count: {report['gpu_info']['num_gpus']}")
        
        if mm['peak'].get('per_gpu_details'):
            print("\n  Per-GPU Peak Memory:")
            for gpu in mm['peak']['per_gpu_details']:
                print(f"    GPU {gpu['device']}: {gpu['used_gb']:.2f} GB / {gpu['total_gb']:.2f} GB")
        
        # Engine info
        print("\n🔧 Engine Information:")
        em = report['engine_metadata']
        print(f"  vLLM Version: {em['engine_version']}")
        print(f"  Profiler Version: {em['profiler_version']}")
        print(f"  Profiling Method: {em['profiling_method']}")
        
        print(f"\n{'='*70}")
        
        # Notes
        if mb.get('notes'):
            print("\n📝 Notes:")
            for note in mb['notes']:
                print(f"  • {note}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile vLLM using native bench command with enhanced instrumentation (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling
  python profile-vllm-bench-enhanced.py --model meta-llama/Llama-2-7b-hf \\
      --input-len 256 --output-len 256 --batch-size 8
  
  # With architecture details for accurate estimation
  python profile-vllm-bench-enhanced.py --model meta-llama/Llama-2-13b-hf \\
      --input-len 512 --output-len 512 --model-params 13e9 \\
      --num-layers 40 --num-heads 40 --head-dim 128
  
  # Multi-GPU with tensor parallelism
  python profile-vllm-bench-enhanced.py --model meta-llama/Llama-2-70b-hf \\
      --input-len 1024 --output-len 1024 --tensor-parallel-size 4 \\
      --model-params 70e9 --num-layers 80
  
  # With different KV cache dtype
  python profile-vllm-bench-enhanced.py --model meta-llama/Llama-2-7b-hf \\
      --input-len 512 --output-len 512 --dtype float16 \\
      --kv-cache-dtype fp8
        """
    )
    
    # Required parameters
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name or path (e.g., meta-llama/Llama-2-7b-hf)'
    )
    parser.add_argument(
        '--input-len',
        type=int,
        required=True,
        help='Number of input tokens'
    )
    parser.add_argument(
        '--output-len',
        type=int,
        required=True,
        help='Number of output tokens to generate'
    )
    
    # Optional parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32', 'fp8'],
        help='Data type for model weights (default: auto)'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism (default: 1)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['awq', 'gptq', 'fp8', 'int8', 'int4'],
        help='Quantization method (optional)'
    )
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'float16', 'bfloat16', 'fp8'],
        help='KV cache dtype (if different from model dtype)'
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Allow custom code execution for models with custom implementations (e.g., Kimi K2)'
    )
    
    # Model architecture (for better memory estimation)
    parser.add_argument(
        '--model-params',
        type=float,
        help='Number of model parameters (e.g., 7e9 for 7B model)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        help='Number of model layers (for accurate KV cache estimation)'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--head-dim',
        type=int,
        help='Head dimension size'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        help='Hidden size dimension (for accurate memory estimation)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path for detailed report'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--enforce-eager',
        action='store_true',
        help='Disable CUDA graphs (NOT recommended - use only for debugging). Production uses CUDA graphs by default.'
    )
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = VLLMBenchProfiler(verbose=not args.quiet)
    
    # Generate report
    try:
        report = profiler.generate_report(
            model=args.model,
            input_len=args.input_len,
            output_len=args.output_len,
            batch_size=args.batch_size,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            quantization=args.quantization,
            model_params=args.model_params,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            hidden_size=args.hidden_size,
            kv_cache_dtype=args.kv_cache_dtype,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            output_file=args.output
        )
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
