#!/usr/bin/env python3
"""
vLLM Benchmark Profiler - Version 3.0 with Enhanced vLLM Metrics Capture

This script extends v2.1 with critical enhancements to capture vLLM's internal
memory allocation decisions and comprehensive GPU metrics.

Version: 3.0 (vLLM Internal Metrics + AMD GPU Enhancements)

NEW in v3.0:
- Real-time memory monitoring during benchmark execution
- vLLM log parsing for internal metrics (KV cache blocks, attention backend, etc.)
- Extended AMD GPU metrics using modern amd-smi monitor
- Multi-phase memory capture timeline
- Comprehensive power, temperature, and utilization tracking

Key Improvements Over v2.1:
✅ Captures vLLM's internal memory allocation decisions
✅ Real-time memory timeline (not just 3 snapshots)
✅ AMD GPU extended metrics (power, temperature, utilization, clocks)
✅ vLLM engine logs with KV cache block statistics
✅ Attention backend detection
✅ Better timing - memory captured during execution, not after cleanup

Addresses Root Cause of Calculator Validation Failures:
- Previously: Only captured external GPU memory snapshots
- Now: Captures vLLM's internal view of memory allocation
- Result: Can validate KV cache calculations against vLLM's actual block allocation

Usage:
    # Auto-load model with enhanced profiling
    python profile-vllm-bench-v3.py \\
        --hf-model-id meta-llama/Llama-3.2-1B-Instruct \\
        --input-len 256 \\
        --output-len 256 \\
        --batch-size 8 \\
        --output results.json

Requirements:
    - vLLM installed (in container or locally)
    - GPU with nvidia-smi or amd-smi
    - Access to src/data/models.json (for --hf-model-id)
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
import threading
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

# Import model loader for automatic parameter loading (optional)
try:
    from model_loader import get_profiling_params, validate_model_in_db
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False


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


def parse_vllm_initialization_logs(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Extract vLLM's internal memory allocation from logs

    This is the SOURCE OF TRUTH for how vLLM is allocating memory.
    Parses vLLM engine initialization logs to extract:
    - Number of GPU/CPU blocks allocated
    - Block size (critical for KV cache calculations)
    - Total KV cache size (from vLLM's calculation)
    - Attention backend being used
    - Memory breakdown (weights, KV cache, activations)
    - Engine configuration

    Example vLLM log output:
        INFO 11-04 10:23:22 model_runner.py:1049]
          # GPU blocks: 7326, # CPU blocks: 2048
          Block size: 16 tokens
          Total GPU KV cache: 117216 tokens (7.28 GB)
    """
    combined = stdout + "\n" + stderr

    info = {
        'num_gpu_blocks': None,
        'num_cpu_blocks': None,
        'block_size': None,
        'total_gpu_kv_cache_tokens': None,
        'total_gpu_kv_cache_gb': None,
        'total_cpu_kv_cache_tokens': None,
        'total_cpu_kv_cache_gb': None,
        'attention_backend': None,
        'model_weights_gb': None,
        'activations_gb': None,
        'total_allocated_gb': None,
        'gpu_memory_utilization': None,
        'max_model_len': None,
        'cuda_graphs_enabled': None,
        'vllm_version': None
    }

    # Pattern: # GPU blocks: 7326, # CPU blocks: 2048
    blocks_match = re.search(
        r'#\s*GPU blocks:\s*(\d+),\s*#\s*CPU blocks:\s*(\d+)',
        combined,
        re.IGNORECASE
    )
    if blocks_match:
        info['num_gpu_blocks'] = int(blocks_match.group(1))
        info['num_cpu_blocks'] = int(blocks_match.group(2))

    # Pattern: Block size: 16 tokens
    block_size_match = re.search(r'Block size:\s*(\d+)\s*tokens?', combined, re.IGNORECASE)
    if block_size_match:
        info['block_size'] = int(block_size_match.group(1))

    # Pattern: Total GPU KV cache: 117216 tokens (7.28 GB)
    kv_cache_match = re.search(
        r'Total GPU KV cache:\s*([\d,]+)\s*tokens?\s*\(([0-9.]+)\s*GB\)',
        combined,
        re.IGNORECASE
    )
    if kv_cache_match:
        info['total_gpu_kv_cache_tokens'] = int(kv_cache_match.group(1).replace(',', ''))
        info['total_gpu_kv_cache_gb'] = float(kv_cache_match.group(2))

    # Pattern: Total CPU KV cache: 32768 tokens (2.03 GB)
    cpu_kv_match = re.search(
        r'Total CPU KV cache:\s*([\d,]+)\s*tokens?\s*\(([0-9.]+)\s*GB\)',
        combined,
        re.IGNORECASE
    )
    if cpu_kv_match:
        info['total_cpu_kv_cache_tokens'] = int(cpu_kv_match.group(1).replace(',', ''))
        info['total_cpu_kv_cache_gb'] = float(cpu_kv_match.group(2))

    # Pattern: Using FlashAttention-2 backend
    attn_match = re.search(r'Using\s+(\S+(?:\s+\S+)?)\s+backend', combined, re.IGNORECASE)
    if attn_match:
        info['attention_backend'] = attn_match.group(1)

    # Pattern: Model weights: 2.47 GB
    weights_match = re.search(r'Model weights:\s*([0-9.]+)\s*GB', combined, re.IGNORECASE)
    if weights_match:
        info['model_weights_gb'] = float(weights_match.group(1))

    # Pattern: Activations: 0.82 GB
    activations_match = re.search(r'Activations:\s*([0-9.]+)\s*GB', combined, re.IGNORECASE)
    if activations_match:
        info['activations_gb'] = float(activations_match.group(1))

    # Pattern: Total: 10.57 GB (total allocated)
    total_match = re.search(r'Total:\s*([0-9.]+)\s*GB', combined, re.IGNORECASE)
    if total_match:
        info['total_allocated_gb'] = float(total_match.group(1))

    # Pattern: gpu_memory_utilization=0.9
    gpu_mem_util_match = re.search(r'gpu_memory_utilization[=:\s]+([0-9.]+)', combined, re.IGNORECASE)
    if gpu_mem_util_match:
        info['gpu_memory_utilization'] = float(gpu_mem_util_match.group(1))

    # Pattern: max_model_len=8192
    max_len_match = re.search(r'max_model_len[=:\s]+(\d+)', combined, re.IGNORECASE)
    if max_len_match:
        info['max_model_len'] = int(max_len_match.group(1))

    # Pattern: CUDA graphs detection (either enabled or disabled messages)
    if re.search(r'CUDA graphs? (enabled|capturing)', combined, re.IGNORECASE):
        info['cuda_graphs_enabled'] = True
    elif re.search(r'CUDA graphs? disabled|enforce.?eager', combined, re.IGNORECASE):
        info['cuda_graphs_enabled'] = False

    # Pattern: vLLM version in logs
    version_match = re.search(r'vLLM\s+(?:version\s+)?v?([0-9.]+(?:\.post\d+)?)', combined, re.IGNORECASE)
    if version_match:
        info['vllm_version'] = version_match.group(1)

    return info


class BackgroundMemoryMonitor:
    """
    Monitors GPU memory in background thread during vLLM execution

    For AMD GPUs: Uses amd-smi to capture comprehensive metrics
    For NVIDIA GPUs: Uses nvidia-smi for memory only

    This solves the critical timing issue where vLLM process cleanup
    happens before we can measure peak memory.
    """

    def __init__(
        self,
        interval_ms: int = 100,
        gpu_type: str = 'rocm',
        capture_extended: bool = True,
        gpu_memory_fn = None
    ):
        self.interval = interval_ms / 1000.0
        self.gpu_type = gpu_type
        self.capture_extended = capture_extended  # Power, temp, utilization
        self.samples = []
        self.running = False
        self.thread = None
        self.gpu_memory_fn = gpu_memory_fn  # Function to get GPU memory

    def start(self):
        """Start background monitoring thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop background monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background thread monitoring loop"""
        while self.running:
            timestamp = time.time()

            if self.capture_extended and self.gpu_type == 'rocm':
                # Capture full amd-smi metrics
                sample = self._get_amd_extended_sample()
            else:
                # Basic memory only (fallback)
                if self.gpu_memory_fn:
                    memory = self.gpu_memory_fn()
                    sample = {'memory_gb': sum(gpu.get('used_gb', 0) for gpu in memory)}
                else:
                    sample = {'memory_gb': 0.0}

            sample['timestamp'] = timestamp
            self.samples.append(sample)
            time.sleep(self.interval)

    def _get_amd_extended_sample(self) -> Dict[str, Any]:
        """Get comprehensive AMD GPU metrics for this sample"""
        try:
            result = subprocess.run(
                ['amd-smi', 'monitor', '-putmqv', '--json'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Aggregate across all GPUs
                if not data:
                    return {'memory_gb': 0.0}

                total_mem = sum(
                    gpu.get('vram_used', {}).get('value', 0) for gpu in data
                ) / 1024  # MB to GB

                avg_power = statistics.mean([
                    gpu.get('power_usage', {}).get('value', 0) for gpu in data
                ]) if data else 0

                avg_temp = statistics.mean([
                    gpu.get('hotspot_temperature', {}).get('value', 0) for gpu in data
                ]) if data else 0

                return {
                    'memory_gb': total_mem,
                    'power_w': avg_power,
                    'temp_c': avg_temp,
                    'gpus': data  # Full per-GPU breakdown
                }
        except Exception:
            pass

        # Fallback to basic memory if amd-smi fails
        if self.gpu_memory_fn:
            memory = self.gpu_memory_fn()
            return {'memory_gb': sum(gpu.get('used_gb', 0) for gpu in memory)}
        return {'memory_gb': 0.0}

    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """Get statistics from all collected samples"""
        if not self.samples:
            return None

        memories = [s.get('memory_gb', 0) for s in self.samples]
        stats = {
            'min_gb': min(memories),
            'max_gb': max(memories),
            'mean_gb': statistics.mean(memories),
            'median_gb': statistics.median(memories),
            'p95_gb': statistics.quantiles(memories, n=20)[18] if len(memories) >= 20 else max(memories),
            'num_samples': len(self.samples),
            'timeline': self.samples  # Full timeline for detailed analysis
        }

        # Add extended metrics if available
        if self.capture_extended and self.samples[0].get('power_w') is not None:
            powers = [s.get('power_w', 0) for s in self.samples if s.get('power_w')]
            temps = [s.get('temp_c', 0) for s in self.samples if s.get('temp_c')]

            if powers and temps:
                stats['extended'] = {
                    'power': {
                        'min_w': min(powers),
                        'max_w': max(powers),
                        'mean_w': statistics.mean(powers)
                    },
                    'temperature': {
                        'min_c': min(temps),
                        'max_c': max(temps),
                        'mean_c': statistics.mean(temps)
                    }
                }

        return stats


class VLLMBenchProfilerV3:
    """
    Enhanced vLLM Profiler v3.0 with comprehensive metrics capture

    Key improvements:
    - Real-time memory monitoring during benchmark
    - vLLM log parsing for internal metrics
    - Extended AMD ROCm metrics
    - Multi-phase memory capture
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gpu_type = self.detect_gpu_type()
        self.vllm_version = get_vllm_version()

    def detect_gpu_type(self) -> str:
        """Detect if using NVIDIA CUDA or AMD ROCm"""
        # Check for amd-smi first (modern AMD tool)
        if subprocess.run(['which', 'amd-smi'], capture_output=True).returncode == 0:
            return 'rocm'
        # Fallback to rocm-smi (legacy)
        if os.path.exists('/opt/rocm') or subprocess.run(
            ['which', 'rocm-smi'], capture_output=True
        ).returncode == 0:
            return 'rocm'
        # Check for NVIDIA
        if subprocess.run(['which', 'nvidia-smi'], capture_output=True).returncode == 0:
            return 'cuda'
        return 'unknown'

    def log(self, message: str):
        """Print log message if verbose"""
        if self.verbose:
            print(message)

    def get_amd_smi_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive AMD GPU metrics using modern amd-smi tool

        Captures: memory, power, temperature, utilization, clocks, processes

        This is significantly better than legacy rocm-smi:
        - Single call gets all metrics
        - Process tracking - see which processes use GPU memory
        - Official tool - rocm-smi is deprecated
        - Better JSON structure with units included
        """
        try:
            # Get comprehensive monitoring data (single call)
            result = subprocess.run(
                ['amd-smi', 'monitor', '-putmqv', '--json'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )

            amd_data = json.loads(result.stdout)

            # Parse per-GPU metrics
            gpus = []
            for gpu_data in amd_data:
                gpu_info = {
                    'gpu_id': gpu_data.get('gpu', 0),

                    # Memory metrics
                    'vram_used_mb': gpu_data.get('vram_used', {}).get('value', 0),
                    'vram_free_mb': gpu_data.get('vram_free', {}).get('value', 0),
                    'vram_total_mb': gpu_data.get('vram_total', {}).get('value', 0),
                    'vram_percent': gpu_data.get('vram_percent', {}).get('value', 0),

                    # Power and thermal
                    'power_watts': gpu_data.get('power_usage', {}).get('value', 0),
                    'hotspot_temp_c': gpu_data.get('hotspot_temperature', {}).get('value', 0),
                    'memory_temp_c': gpu_data.get('memory_temperature', {}).get('value', 0),

                    # Utilization
                    'gfx_utilization_pct': gpu_data.get('gfx', {}).get('value', 0),
                    'mem_utilization_pct': gpu_data.get('mem', {}).get('value', 0),

                    # Clock speeds
                    'gfx_clock_mhz': gpu_data.get('gfx_clk', {}).get('value', 0),
                    'mem_clock_mhz': gpu_data.get('mem_clock', {}).get('value', 0),

                    # Process info
                    'processes': gpu_data.get('process_list', [])
                }
                gpus.append(gpu_info)

            return {
                'tool': 'amd-smi',
                'gpus': gpus,
                'raw_json': amd_data,
                'timestamp': time.time()
            }

        except subprocess.CalledProcessError as e:
            self.log(f"WARNING: amd-smi failed with code {e.returncode}: {e.stderr}")
            return {'error': 'amd-smi_failed', 'details': str(e)}
        except json.JSONDecodeError as e:
            self.log(f"WARNING: Failed to parse amd-smi JSON: {e}")
            return {'error': 'json_parse_failed', 'details': str(e)}
        except Exception as e:
            self.log(f"WARNING: Unexpected error getting AMD GPU metrics: {e}")
            return {'error': 'unexpected_error', 'details': str(e)}

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
        """Get memory usage from ROCm GPUs (tries amd-smi first, falls back to rocm-smi)"""
        # Try modern amd-smi first
        try:
            result = subprocess.run(
                ['amd-smi', 'monitor', '-m', '--json'],
                capture_output=True,
                text=True,
                check=True,
                timeout=3
            )

            data = json.loads(result.stdout)
            gpus = []
            for gpu_data in data:
                gpus.append({
                    'device': gpu_data.get('gpu', 0),
                    'used_gb': round(gpu_data.get('vram_used', {}).get('value', 0) / 1024, 2),
                    'total_gb': round(gpu_data.get('vram_total', {}).get('value', 0) / 1024, 2)
                })
            return sorted(gpus, key=lambda x: x['device'])

        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            # Fall back to legacy rocm-smi
            pass

        # Legacy rocm-smi fallback
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
        Run vllm bench latency command with ENHANCED monitoring (v3.0)

        NEW in v3.0:
        - Real-time memory monitoring during execution
        - vLLM log parsing for internal metrics
        - Extended AMD GPU metrics capture
        - Memory timeline statistics

        Returns:
            Dictionary with benchmark results and comprehensive memory measurements
        """

        self.log("\n=== vLLM Benchmark Profiler v3.0 (Enhanced) ===")
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

        # Phase 0: System baseline
        self.log("\n=== Phase 0: Measuring System Baseline (Pre-vLLM) ===")
        baseline_memory = self.get_gpu_memory_snapshot()
        baseline_stats = self.compute_memory_statistics(baseline_memory)
        self.log(f"Baseline: {baseline_stats['total_gb']:.2f} GB across {baseline_stats['num_gpus']} GPUs")

        # Phase 1: Start background memory monitoring
        self.log("\n=== Phase 1: Starting Real-Time Memory Monitor ===")
        monitor = BackgroundMemoryMonitor(
            interval_ms=100,
            gpu_type=self.gpu_type,
            capture_extended=True,
            gpu_memory_fn=self.get_gpu_memory_snapshot
        )
        monitor.start()
        self.log("Background monitor started (sampling every 100ms)")

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

            self.log(f"\n=== Phase 2: Running vLLM Bench Command ===")
            self.log(f"Command: {' '.join(cmd)}")

            # Run benchmark with log capture
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,  # CRITICAL: Capture logs!
                text=True  # CRITICAL: Get strings not bytes
            )
            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                self.log(f"\nERROR: vllm bench failed with exit code {result.returncode}")
                self.log(f"STDOUT:\n{result.stdout}")
                self.log(f"STDERR:\n{result.stderr}")
                sys.exit(1)

            self.log(f"Benchmark completed in {elapsed_time:.1f} seconds")

        finally:
            # Phase 3: Stop background monitoring
            self.log("\n=== Phase 3: Stopping Memory Monitor ===")
            monitor.stop()
            memory_timeline = monitor.get_statistics()

            if memory_timeline:
                self.log(f"Captured {memory_timeline['num_samples']} memory samples")
                self.log(f"Memory range: {memory_timeline['min_gb']:.2f} - {memory_timeline['max_gb']:.2f} GB")
                self.log(f"Peak memory: {memory_timeline['max_gb']:.2f} GB")

        # Phase 4: Parse vLLM logs for internal metrics
        self.log("\n=== Phase 4: Parsing vLLM Internal Metrics ===")
        vllm_metrics = parse_vllm_initialization_logs(
            result.stdout,
            result.stderr
        )

        # Log what we found
        if vllm_metrics.get('num_gpu_blocks'):
            self.log(f"✓ GPU Blocks: {vllm_metrics['num_gpu_blocks']}")
        if vllm_metrics.get('block_size'):
            self.log(f"✓ Block Size: {vllm_metrics['block_size']} tokens")
        if vllm_metrics.get('total_gpu_kv_cache_gb'):
            self.log(f"✓ KV Cache (vLLM reported): {vllm_metrics['total_gpu_kv_cache_gb']:.2f} GB")
        if vllm_metrics.get('attention_backend'):
            self.log(f"✓ Attention Backend: {vllm_metrics['attention_backend']}")
        if vllm_metrics.get('model_weights_gb'):
            self.log(f"✓ Model Weights (vLLM reported): {vllm_metrics['model_weights_gb']:.2f} GB")

        # Phase 5: Get detailed GPU metrics (if AMD)
        detailed_gpu = None
        if self.gpu_type == 'rocm':
            self.log("\n=== Phase 5: Capturing Extended AMD GPU Metrics ===")
            detailed_gpu = self.get_amd_smi_detailed_metrics()
            if 'error' not in detailed_gpu:
                self.log(f"✓ Captured detailed metrics for {len(detailed_gpu.get('gpus', []))} GPUs")
                if detailed_gpu.get('gpus'):
                    for gpu in detailed_gpu['gpus']:
                        self.log(f"  GPU {gpu['gpu_id']}: {gpu['vram_used_mb']/1024:.2f}GB, "
                                f"{gpu['power_watts']}W, {gpu['hotspot_temp_c']}°C")

        # Capture final snapshot
        self.log("\n=== Phase 6: Final Memory Snapshot ===")
        time.sleep(1)
        final_memory = self.get_gpu_memory_snapshot()
        final_stats = self.compute_memory_statistics(final_memory)
        self.log(f"Final: {final_stats['total_gb']:.2f} GB")

        # Load benchmark results
        with open(output_json, 'r') as f:
            bench_results = json.load(f)

        # Clean up temp file
        if use_temp and os.path.exists(output_json):
            os.unlink(output_json)

        # Determine actual dtypes used
        actual_weight_dtype = dtype
        actual_kv_cache_dtype = kv_cache_dtype if kv_cache_dtype else dtype
        actual_activation_dtype = 'float16' if dtype in ['int8', 'int4', 'fp8'] else dtype

        # Build comprehensive results with v3.0 enhancements
        results = {
            'benchmark_results': bench_results,

            # v3.0 NEW: vLLM internal metrics (source of truth!)
            'vllm_internal_metrics': vllm_metrics,

            # v3.0 NEW: Memory timeline from background monitoring
            'memory_timeline': memory_timeline,

            # v3.0 NEW: Extended AMD GPU metrics
            'detailed_gpu_metrics': detailed_gpu if detailed_gpu else None,

            # Enhanced memory measurements
            'memory_measurements': {
                'baseline': baseline_stats,
                'peak_from_timeline': {
                    'total_gb': memory_timeline['max_gb'] if memory_timeline else final_stats['total_gb'],
                    'mean_gb': memory_timeline['mean_gb'] if memory_timeline else final_stats['total_gb'],
                    'source': 'real_time_monitoring' if memory_timeline else 'final_snapshot'
                },
                'final_snapshot': final_stats,
                'memory_increase_gb': round(
                    (memory_timeline['max_gb'] if memory_timeline else final_stats['total_gb']) - baseline_stats['total_gb'],
                    2
                )
            },

            'parameters': {
                'model': model,
                'input_len': input_len,
                'output_len': output_len,
                'batch_size': batch_size,
                'dtype': dtype,
                'tensor_parallel_size': tensor_parallel_size,
                'pipeline_parallel_size': 1,
                'quantization': quantization,
                'num_iters_warmup': num_iters_warmup,
                'num_iters': num_iters,
                'enforce_eager': enforce_eager
            },

            'dtype_metadata': {
                'weight_dtype': actual_weight_dtype,
                'activation_dtype': actual_activation_dtype,
                'kv_cache_dtype': actual_kv_cache_dtype,
                'requested_dtype': dtype,
                'quantization_applied': quantization is not None
            },

            'gpu_info': {
                'gpu_type': self.gpu_type,
                'num_gpus': tensor_parallel_size,
                'system_gpus': len(final_memory),
                'total_gpu_memory_gb': sum(gpu['total_gb'] for gpu in final_memory) if final_memory else 0
            },

            'engine_metadata': {
                'engine_version': self.vllm_version,
                'engine_type': 'vllm',
                'profiling_method': 'vllm_bench_latency_v3',
                'profiler_version': '3.0',
                'enhancements': [
                    'real_time_memory_monitoring',
                    'vllm_log_parsing',
                    'extended_amd_metrics',
                    'memory_timeline_capture'
                ]
            },

            # v3.0 NEW: Raw logs for future analysis
            'raw_logs': {
                'vllm_stdout': result.stdout,
                'vllm_stderr': result.stderr
            },

            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'profiler_version': '3.0',
            'schema_version': '3.0'
        }

        return results

    def estimate_memory_breakdown_v3(
        self,
        peak_memory_gb: float,
        baseline_memory_gb: float,
        vllm_metrics: Dict[str, Any],
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
        Enhanced memory breakdown using vLLM's internal metrics (v3.0)

        v3.0: Prioritizes vLLM's reported values as source of truth
        """

        # Bytes per parameter based on dtype
        bytes_per_param = {
            'float32': 4, 'fp32': 4, 'float': 4,
            'float16': 2, 'fp16': 2, 'half': 2,
            'bfloat16': 2, 'bf16': 2,
            'int8': 1, 'fp8': 1, 'fp8_e4m3': 1, 'fp8_e5m2': 1,
            'int4': 0.5, 'nf4': 0.5,
            'auto': 2
        }

        weight_dtype = weight_dtype or dtype
        activation_dtype = activation_dtype or ('float16' if dtype in ['int8', 'int4'] else dtype)
        kv_cache_dtype = kv_cache_dtype or dtype

        weight_bytes = bytes_per_param.get(weight_dtype.lower(), 2)

        # Calculate net memory increase
        net_memory_gb = peak_memory_gb - baseline_memory_gb

        # v3.0: Use vLLM's reported values when available (HIGH CONFIDENCE)
        if vllm_metrics.get('model_weights_gb'):
            model_weights_gb = vllm_metrics['model_weights_gb']
            weights_confidence = 'high_from_vllm_logs'
        elif model_params:
            model_weights_gb = (model_params * weight_bytes) / (tensor_parallel_size * 1e9)
            weights_confidence = 'high_from_params'
        else:
            model_weights_gb = net_memory_gb * 0.65
            weights_confidence = 'low_estimated'

        # v3.0: Use vLLM's KV cache when available (SOURCE OF TRUTH!)
        if vllm_metrics.get('total_gpu_kv_cache_gb'):
            kv_cache_gb = vllm_metrics['total_gpu_kv_cache_gb']
            kv_confidence = 'high_from_vllm_logs'
        elif num_layers and (hidden_size or (num_heads and head_dim)):
            seq_len = input_len + output_len
            kv_dim = hidden_size if hidden_size else (num_heads * head_dim)
            kv_bytes = bytes_per_param.get(kv_cache_dtype.lower(), 2)
            kv_elements = 2 * num_layers * kv_dim * seq_len * batch_size
            kv_cache_gb = (kv_elements * kv_bytes) / (tensor_parallel_size * 1e9)
            kv_confidence = 'high_calculated'
        else:
            kv_cache_gb = net_memory_gb * 0.20
            kv_confidence = 'low_estimated'

        # v3.0: Use vLLM's activations when available
        if vllm_metrics.get('activations_gb'):
            activations_gb = vllm_metrics['activations_gb']
            activation_confidence = 'high_from_vllm_logs'
        else:
            activations_gb = net_memory_gb * 0.12
            activation_confidence = 'low_estimated'

        # Framework overhead (residual)
        framework_overhead_gb = max(0, net_memory_gb - (
            model_weights_gb + kv_cache_gb + activations_gb
        ))

        # Adjust if negative
        if framework_overhead_gb < 0:
            adjustment_factor = net_memory_gb / (model_weights_gb + kv_cache_gb + activations_gb)
            model_weights_gb *= adjustment_factor
            kv_cache_gb *= adjustment_factor
            activations_gb *= adjustment_factor
            framework_overhead_gb = net_memory_gb * 0.03

        overhead_pct = (framework_overhead_gb / peak_memory_gb * 100) if peak_memory_gb > 0 else 0

        # Determine overall confidence
        if vllm_metrics.get('total_gpu_kv_cache_gb') and vllm_metrics.get('model_weights_gb'):
            overall_confidence = 'high_from_vllm_logs'
        elif model_params and num_layers and hidden_size:
            overall_confidence = 'high_calculated'
        else:
            overall_confidence = 'low_estimated'

        return {
            'total_measured_gb': round(peak_memory_gb, 2),
            'baseline_gb': round(baseline_memory_gb, 2),
            'net_increase_gb': round(net_memory_gb, 2),
            'model_weights_gb': round(model_weights_gb, 2),
            'kv_cache_gb': round(kv_cache_gb, 2),
            'activations_gb': round(activations_gb, 2),
            'framework_overhead_gb': round(framework_overhead_gb, 2),
            'overhead_percentage': round(overhead_pct, 1),
            'estimation_method': 'vllm_bench_enhanced_v3',
            'confidence_levels': {
                'overall': overall_confidence,
                'weights': weights_confidence,
                'kv_cache': kv_confidence,
                'activations': activation_confidence,
                'overhead': 'calculated_residual'
            },
            'component_dtypes': {
                'weights': weight_dtype,
                'activations': activation_dtype,
                'kv_cache': kv_cache_dtype
            },
            'vllm_reported_values_used': {
                'weights': vllm_metrics.get('model_weights_gb') is not None,
                'kv_cache': vllm_metrics.get('total_gpu_kv_cache_gb') is not None,
                'activations': vllm_metrics.get('activations_gb') is not None
            },
            'notes': [
                'Memory breakdown v3.0 with vLLM internal metrics',
                'vLLM-reported values used when available (highest confidence)',
                'Baseline memory excluded from component calculations',
                f'Overall estimation confidence: {overall_confidence}',
                'Overhead includes CUDA context + vLLM engine + PagedAttention + CUDA graphs'
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
        Generate comprehensive profiling report with v3.0 enhancements
        """

        # Run enhanced benchmark
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

        # Get peak memory (prefer timeline if available)
        peak_gb = results['memory_measurements']['peak_from_timeline']['total_gb']
        baseline_gb = results['memory_measurements']['baseline']['total_gb']

        # Estimate memory breakdown with v3.0 enhancements
        self.log("\n=== Estimating Memory Breakdown (v3.0) ===")
        memory_breakdown = self.estimate_memory_breakdown_v3(
            peak_memory_gb=peak_gb,
            baseline_memory_gb=baseline_gb,
            vllm_metrics=results['vllm_internal_metrics'],
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
            'vllm_internal_metrics': results['vllm_internal_metrics'],
            'memory_timeline': results['memory_timeline'],
            'detailed_gpu_metrics': results['detailed_gpu_metrics'],
            'memory_measurements': results['memory_measurements'],
            'gpu_info': results['gpu_info'],
            'engine_metadata': results['engine_metadata'],
            'raw_logs': results['raw_logs'],
            'timestamp': results['timestamp'],
            'profiler_version': '3.0',
            'schema_version': '3.0'
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
        """Print formatted summary of profiling results with v3.0 enhancements"""

        print(f"\n{'='*70}")
        print("PROFILING SUMMARY v3.0 (Enhanced with vLLM Internal Metrics)")
        print(f"{'='*70}")

        # Model info
        print("\n📊 Model Information:")
        print(f"  Name: {report['model_info']['name']}")
        if report['model_info']['num_parameters']:
            params_b = report['model_info']['num_parameters'] / 1e9
            print(f"  Parameters: {params_b:.1f}B")
        print(f"  Data Type: {report['model_info']['dtype']}")

        # vLLM Internal Metrics (NEW in v3.0)
        vllm = report.get('vllm_internal_metrics', {})
        if vllm.get('num_gpu_blocks') or vllm.get('total_gpu_kv_cache_gb'):
            print("\n🔍 vLLM Internal Metrics (Source of Truth):")
            if vllm.get('num_gpu_blocks'):
                print(f"  GPU Blocks: {vllm['num_gpu_blocks']}")
            if vllm.get('block_size'):
                print(f"  Block Size: {vllm['block_size']} tokens")
            if vllm.get('total_gpu_kv_cache_gb'):
                print(f"  KV Cache (vLLM reported): {vllm['total_gpu_kv_cache_gb']:.2f} GB")
            if vllm.get('model_weights_gb'):
                print(f"  Model Weights (vLLM reported): {vllm['model_weights_gb']:.2f} GB")
            if vllm.get('activations_gb'):
                print(f"  Activations (vLLM reported): {vllm['activations_gb']:.2f} GB")
            if vllm.get('attention_backend'):
                print(f"  Attention Backend: {vllm['attention_backend']}")

        # Memory Timeline (NEW in v3.0)
        timeline = report.get('memory_timeline')
        if timeline:
            print("\n📈 Memory Timeline (Real-Time Monitoring):")
            print(f"  Samples Captured: {timeline['num_samples']}")
            print(f"  Min: {timeline['min_gb']:.2f} GB")
            print(f"  Mean: {timeline['mean_gb']:.2f} GB")
            print(f"  Peak: {timeline['max_gb']:.2f} GB")
            print(f"  P95: {timeline['p95_gb']:.2f} GB")

            if timeline.get('extended'):
                ext = timeline['extended']
                print(f"\n  Extended Metrics:")
                if ext.get('power'):
                    print(f"    Power: {ext['power']['mean_w']:.0f}W (range: {ext['power']['min_w']:.0f}-{ext['power']['max_w']:.0f}W)")
                if ext.get('temperature'):
                    print(f"    Temperature: {ext['temperature']['mean_c']:.0f}°C (range: {ext['temperature']['min_c']:.0f}-{ext['temperature']['max_c']:.0f}°C)")

        # Memory breakdown
        print("\n💾 Memory Breakdown:")
        mb = report['memory_breakdown']
        print(f"  Total Measured: {mb['total_measured_gb']:.2f} GB")
        print(f"  ├─ Model Weights: {mb['model_weights_gb']:.2f} GB ({mb['model_weights_gb']/mb['total_measured_gb']*100:.1f}%)")
        print(f"  ├─ KV Cache: {mb['kv_cache_gb']:.2f} GB ({mb['kv_cache_gb']/mb['total_measured_gb']*100:.1f}%)")
        print(f"  ├─ Activations: {mb['activations_gb']:.2f} GB ({mb['activations_gb']/mb['total_measured_gb']*100:.1f}%)")
        print(f"  └─ Framework Overhead: {mb['framework_overhead_gb']:.2f} GB ({mb['overhead_percentage']:.1f}%)")

        # Confidence levels with v3.0 source indicators
        conf = mb['confidence_levels']
        vllm_used = mb.get('vllm_reported_values_used', {})
        print(f"\n  Estimation Confidence: {conf['overall'].upper()}")
        weights_source = " (vLLM reported)" if vllm_used.get('weights') else ""
        kv_source = " (vLLM reported)" if vllm_used.get('kv_cache') else ""
        act_source = " (vLLM reported)" if vllm_used.get('activations') else ""
        print(f"    Weights: {conf['weights']}{weights_source}")
        print(f"    KV Cache: {conf['kv_cache']}{kv_source}")
        print(f"    Activations: {conf['activations']}{act_source}")

        # GPU info
        print("\n🖥️  GPU Information:")
        print(f"  Type: {report['gpu_info']['gpu_type'].upper()}")
        print(f"  Count: {report['gpu_info']['num_gpus']}")

        # Engine info with v3.0 enhancements
        print("\n🔧 Engine Information:")
        em = report['engine_metadata']
        print(f"  vLLM Version: {em['engine_version']}")
        print(f"  Profiler Version: {em['profiler_version']}")
        print(f"  Enhancements: {', '.join(em.get('enhancements', []))}")

        print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile vLLM with enhanced internal metrics capture (v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-load model with enhanced profiling
  python profile-vllm-bench-v3.py --hf-model-id meta-llama/Llama-3.2-1B-Instruct \\
      --input-len 256 --output-len 256 --batch-size 8 --output results.json

  # Manual specification with all parameters
  python profile-vllm-bench-v3.py --model meta-llama/Llama-2-7b-hf \\
      --input-len 512 --output-len 512 --model-params 7e9 \\
      --num-layers 32 --num-heads 32 --hidden-size 4096
        """
    )

    # Model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str, help='Model name or path')
    model_group.add_argument('--hf-model-id', type=str, help='HuggingFace model ID (auto-loads from models.json)')

    parser.add_argument('--input-len', type=int, required=True, help='Number of input tokens')
    parser.add_argument('--output-len', type=int, required=True, help='Number of output tokens')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16', 'float32', 'fp8'])
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--quantization', type=str, choices=['awq', 'gptq', 'fp8', 'int8', 'int4'])
    parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'float16', 'bfloat16', 'fp8'])
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--enforce-eager', action='store_true')

    # Model architecture
    parser.add_argument('--model-params', type=float, help='Number of model parameters (e.g., 7e9)')
    parser.add_argument('--num-layers', type=int, help='Number of model layers')
    parser.add_argument('--num-heads', type=int, help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, help='Head dimension size')
    parser.add_argument('--hidden-size', type=int, help='Hidden size dimension')

    # Output
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Handle model specification (same as v2.1)
    model_name = None
    model_params = args.model_params
    num_layers = args.num_layers
    num_heads = args.num_heads
    head_dim = args.head_dim
    hidden_size = args.hidden_size

    if args.hf_model_id:
        if not MODEL_LOADER_AVAILABLE:
            print("ERROR: --hf-model-id requires model_loader module")
            sys.exit(1)

        print(f"Loading model architecture from models.json: {args.hf_model_id}")
        try:
            arch_params = get_profiling_params(args.hf_model_id)
            model_name = args.hf_model_id
            model_params = model_params or arch_params['model_params']
            num_layers = num_layers or arch_params['num_layers']
            num_heads = num_heads or arch_params['num_heads']
            head_dim = head_dim or arch_params['head_dim']
            hidden_size = hidden_size or arch_params['hidden_size']
            print(f"  ✓ Loaded: {model_params/1e9:.1f}B params, {num_layers} layers, {hidden_size} hidden_size")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        model_name = args.model

    # Create v3.0 profiler
    profiler = VLLMBenchProfilerV3(verbose=not args.quiet)

    # Generate report with v3.0 enhancements
    try:
        report = profiler.generate_report(
            model=model_name,
            input_len=args.input_len,
            output_len=args.output_len,
            batch_size=args.batch_size,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            quantization=args.quantization,
            model_params=model_params,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
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
