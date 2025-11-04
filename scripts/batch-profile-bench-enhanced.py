#!/usr/bin/env python3
"""
Batch Profiling Script using Enhanced vLLM Bench Approach (v2.0)

This script automates profiling multiple models using the enhanced vLLM bench profiler
which captures comprehensive metadata for improved validation and analysis.

Key enhancements in v2.0:
- Uses profile-vllm-bench-enhanced.py with full metadata capture
- Extracts and stores engine version, dtype metadata
- Reports per-GPU memory statistics
- Includes confidence levels for component estimates
- Enhanced CSV output with new fields for analysis

Usage:
    # Profile with config file
    python scripts/batch-profile-bench-enhanced.py --config models.yaml
    
    # Profile specific container with architecture details
    python scripts/batch-profile-bench-enhanced.py \\
        --container vllm-inference \\
        --model meta-llama/Llama-2-7b-hf \\
        --model-params 6.7e9 \\
        --num-layers 32 \\
        --num-heads 32 \\
        --hidden-size 4096
    
    # Dry run to see what would be done
    python scripts/batch-profile-bench-enhanced.py --config models.yaml --dry-run

Config file format (YAML):
    models:
      - model_id: "meta-llama/Llama-2-7b-hf"
        container_name: "vllm-inference"
        input_lengths: [256, 512, 1024]
        output_lengths: [256, 512]
        batch_sizes: [1, 8, 16]
        dtype: "float16"
        tensor_parallel_size: 1
        quantization: null
        model_params: 6.7e9
        num_layers: 32
        num_heads: 32
        hidden_size: 4096
        head_dim: 128
"""

import argparse
import json
import subprocess
import sys
import time
import yaml
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for a single model profiling run"""
    model_id: str
    container_name: str
    input_lengths: List[int]
    output_lengths: List[int]
    batch_sizes: List[int]
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None
    model_params: Optional[float] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_size: Optional[int] = None
    kv_cache_dtype: Optional[str] = None
    trust_remote_code: bool = False
    gpu_ids: Optional[str] = None


@dataclass
class ProfileResult:
    """Result from a single profiling run with enhanced metadata"""
    model_id: str
    container_name: str
    input_len: int
    output_len: int
    batch_size: int
    profile_file: str
    success: bool
    error_message: Optional[str] = None
    
    # Memory metrics (enhanced)
    total_memory_gb: Optional[float] = None
    baseline_memory_gb: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    memory_increase_gb: Optional[float] = None
    
    # Component breakdown
    weights_gb: Optional[float] = None
    kv_cache_gb: Optional[float] = None
    activations_gb: Optional[float] = None
    overhead_gb: Optional[float] = None
    
    # Metadata
    engine_version: Optional[str] = None
    weight_dtype: Optional[str] = None
    activation_dtype: Optional[str] = None
    kv_cache_dtype_actual: Optional[str] = None
    
    # Performance
    latency_ms: Optional[float] = None
    duration_seconds: Optional[float] = None
    
    # Confidence
    estimation_confidence: Optional[str] = None


class ProgressTicker:
    """Display elapsed time progress for long-running operations"""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.start_time = None
        self.stop_flag = threading.Event()
        self.thread = None
    
    def start(self):
        """Start the progress ticker"""
        self.start_time = time.time()
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._ticker, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the progress ticker and return elapsed time"""
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1)
        return time.time() - self.start_time if self.start_time else 0
    
    def _ticker(self):
        """Background thread that updates progress"""
        while not self.stop_flag.is_set():
            elapsed = time.time() - self.start_time
            # Print on same line, carriage return to overwrite
            print(f"\r{self.prefix} ⏳ {elapsed:.0f}s...", end='', flush=True)
            time.sleep(2)  # Update every 2 seconds


class ContainerManager:
    """Manage Docker container operations"""
    
    @staticmethod
    def is_running(container_name: str) -> bool:
        """Check if container is running"""
        result = subprocess.run(
            ['docker', 'ps', '-q', '-f', f'name=^{container_name}$'],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    
    @staticmethod
    def exists(container_name: str) -> bool:
        """Check if container exists"""
        result = subprocess.run(
            ['docker', 'ps', '-aq', '-f', f'name=^{container_name}$'],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    
    @staticmethod
    def exec_command(container_name: str, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Execute command inside container"""
        cmd = ['docker', 'exec', container_name] + command
        return subprocess.run(cmd, capture_output=True, text=True, check=check)
    
    @staticmethod
    def copy_to_container(container_name: str, src: Path, dest: str) -> bool:
        """Copy file to container"""
        result = subprocess.run(
            ['docker', 'cp', str(src), f'{container_name}:{dest}'],
            capture_output=True
        )
        return result.returncode == 0
    
    @staticmethod
    def copy_from_container(container_name: str, src: str, dest: Path) -> bool:
        """Copy file from container"""
        result = subprocess.run(
            ['docker', 'cp', f'{container_name}:{src}', str(dest)],
            capture_output=True
        )
        return result.returncode == 0


class EnhancedBenchProfiler:
    """Run enhanced vLLM bench profiling in containers"""
    
    def __init__(self, results_dir: Path, script_dir: Path):
        self.results_dir = results_dir
        self.script_dir = script_dir
        self.results: List[ProfileResult] = []
        self.profiler_script = script_dir / 'profile-vllm-bench-enhanced.py'
        self.calculator_module = script_dir / 'lib' / 'calculator_formulas.py'
        
        if not self.profiler_script.exists():
            raise FileNotFoundError(f"Enhanced profiler script not found: {self.profiler_script}")
        
        if not self.calculator_module.exists():
            print(f"⚠️  Warning: Calculator module not found: {self.calculator_module}")
            print("    Calculator comparison will be unavailable in profiles")
            self.calculator_module = None
    
    def profile_model(
        self,
        config: ModelConfig,
        input_len: int,
        output_len: int,
        batch_size: int,
        config_num: int = 1,
        total_configs: int = 1
    ) -> ProfileResult:
        """Profile a single model configuration using enhanced vllm bench profiler"""
        
        # Extract model name (last part of path)
        model_name = config.model_id.split('/')[-1]
        
        # Get model size in billions (handle string or float from YAML)
        size_str = "?"
        if config.model_params:
            try:
                # Convert to float if it's a string (from YAML scientific notation)
                params = float(config.model_params) if isinstance(config.model_params, str) else config.model_params
                size_b = params / 1e9
                if size_b >= 1:
                    size_str = f"{size_b:.0f}B"
                else:
                    size_str = f"{size_b*1000:.0f}M"
            except (ValueError, TypeError):
                size_str = "?"
        
        # Format KV cache dtype (default to same as weight dtype if not specified)
        kv_dtype = config.kv_cache_dtype or config.dtype
        kv_str = f"KV:{kv_dtype}" if kv_dtype != config.dtype else ""
        
        # Concise single-line header
        header = f"[{config_num}/{total_configs}] {model_name} ({size_str}) | TP:{config.tensor_parallel_size} | {config.dtype}"
        if kv_str:
            header += f" | {kv_str}"
        
        print(f"\n{header}")
        print(f"    {input_len}→{output_len} tokens, BS={batch_size} | ", end='', flush=True)
        
        start_time = time.time()
        
        # Copy enhanced profiler script to container (silent)
        if not ContainerManager.copy_to_container(
            config.container_name,
            self.profiler_script,
            '/tmp/profile-vllm-bench-enhanced.py'
        ):
            error_msg = "Failed to copy profiler script to container"
            print(f"❌ {error_msg}")
            return self._create_error_result(
                config, input_len, output_len, batch_size,
                error_msg, time.time() - start_time
            )
        
        # Copy calculator module to container if available
        if self.calculator_module:
            # Create lib directory in container
            ContainerManager.exec_command(
                config.container_name,
                ['mkdir', '-p', '/tmp/lib'],
                check=False
            )
            
            # Copy calculator_formulas.py
            if not ContainerManager.copy_to_container(
                config.container_name,
                self.calculator_module,
                '/tmp/lib/calculator_formulas.py'
            ):
                print(f"  ⚠️  Warning: Failed to copy calculator module (calculator comparison will be unavailable)")
        
        # Build profiler command to run inside container
        output_filename = f'{config.container_name}_{input_len}in_{output_len}out_bs{batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        container_output = f'/tmp/{output_filename}'
        
        cmd = [
            'python3', '/tmp/profile-vllm-bench-enhanced.py',
            '--model', config.model_id,
            '--input-len', str(input_len),
            '--output-len', str(output_len),
            '--batch-size', str(batch_size),
            '--dtype', config.dtype,
            '--tensor-parallel-size', str(config.tensor_parallel_size),
            '--output', container_output
        ]
        
        if config.quantization:
            cmd.extend(['--quantization', config.quantization])
        
        if config.model_params:
            cmd.extend(['--model-params', str(config.model_params)])
        
        if config.num_layers:
            cmd.extend(['--num-layers', str(config.num_layers)])
        
        if config.num_heads:
            cmd.extend(['--num-heads', str(config.num_heads)])
        
        if config.head_dim:
            cmd.extend(['--head-dim', str(config.head_dim)])
        
        if config.hidden_size:
            cmd.extend(['--hidden-size', str(config.hidden_size)])
        
        if config.kv_cache_dtype:
            cmd.extend(['--kv-cache-dtype', config.kv_cache_dtype])
        
        if config.trust_remote_code:
            cmd.append('--trust-remote-code')
        
        # Start progress ticker
        ticker = ProgressTicker(prefix="    ")
        ticker.start()
        
        # Execute profiler inside container
        result = ContainerManager.exec_command(
            config.container_name,
            cmd,
            check=False
        )
        
        # Stop ticker and get elapsed time
        elapsed = ticker.stop()
        print()  # New line after ticker
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            print(f"    ❌ Profiling failed: {error_msg[:100]}")
            return self._create_error_result(
                config, input_len, output_len, batch_size,
                error_msg, elapsed
            )
        
        # Copy results from container
        local_output = self.results_dir / output_filename
        
        if not ContainerManager.copy_from_container(
            config.container_name,
            container_output,
            local_output
        ):
            error_msg = "Failed to copy results from container"
            print(f"    ❌ {error_msg}")
            return self._create_error_result(
                config, input_len, output_len, batch_size,
                error_msg, elapsed
            )
        
        # Parse enhanced results
        try:
            with open(local_output) as f:
                profile_data = json.load(f)
            
            # Extract memory measurements
            mem_meas = profile_data.get('memory_measurements', {})
            baseline_stats = mem_meas.get('baseline', {})
            peak_stats = mem_meas.get('peak', {})
            
            # Extract memory breakdown
            mem_breakdown = profile_data.get('memory_breakdown', {})
            
            # Extract dtype metadata
            dtype_meta = profile_data.get('dtype_metadata', {})
            
            # Extract engine metadata
            engine_meta = profile_data.get('engine_metadata', {})
            
            # Extract latency
            latency_stats = profile_data.get('latency_stats', {})
            mean_latency_ms = latency_stats.get('mean_latency_ms') or latency_stats.get('avg_latency', 0) * 1000
            
            # Concise single-line result
            engine_version = engine_meta.get('engine_version', 'unknown')
            print(f"    ✓ {peak_stats.get('total_gb', 0):.1f} GB peak (+{mem_meas.get('memory_increase_gb', 0):.1f} GB) | {elapsed:.1f}s | vLLM {engine_version}")
            
            # Display calculator comparison if available
            calc_comp = profile_data.get('calculator_comparison', {})
            if 'current_formula' in calc_comp:
                actual_gb = calc_comp['actual_measured']['total_gb']
                current_gb = calc_comp['current_formula']['total_gb']
                current_err = calc_comp['current_formula']['error_vs_actual_pct']
                
                # Determine status symbol (threshold: ±15%)
                threshold = 15.0
                current_pass = abs(current_err) < threshold
                status_symbol = "✅" if current_pass else "❌"
                
                print(f"    🧮 Calculator: Actual={actual_gb:.1f}GB | Estimated={current_gb:.1f}GB ({current_err:+.1f}%) {status_symbol}")
            elif 'note' in calc_comp:
                print(f"    🧮 Calculator: {calc_comp['note']}")
            
            return ProfileResult(
                model_id=config.model_id,
                container_name=config.container_name,
                input_len=input_len,
                output_len=output_len,
                batch_size=batch_size,
                profile_file=str(local_output),
                success=True,
                total_memory_gb=peak_stats.get('total_gb'),
                baseline_memory_gb=baseline_stats.get('total_gb'),
                peak_memory_gb=peak_stats.get('total_gb'),
                memory_increase_gb=mem_meas.get('memory_increase_gb'),
                weights_gb=mem_breakdown.get('model_weights_gb'),
                kv_cache_gb=mem_breakdown.get('kv_cache_gb'),
                activations_gb=mem_breakdown.get('activations_gb'),
                overhead_gb=mem_breakdown.get('framework_overhead_gb'),
                engine_version=engine_meta.get('engine_version'),
                weight_dtype=dtype_meta.get('weight_dtype'),
                activation_dtype=dtype_meta.get('activation_dtype'),
                kv_cache_dtype_actual=dtype_meta.get('kv_cache_dtype'),
                latency_ms=mean_latency_ms,
                duration_seconds=elapsed,
                estimation_confidence=mem_breakdown.get('confidence_levels', {}).get('overall')
            )
            
        except Exception as e:
            error_msg = f"Failed to parse profile: {e}"
            print(f"    ⚠️  {error_msg}")
            return ProfileResult(
                model_id=config.model_id,
                container_name=config.container_name,
                input_len=input_len,
                output_len=output_len,
                batch_size=batch_size,
                profile_file=str(local_output) if local_output.exists() else "",
                success=True,  # File exists but parsing failed
                duration_seconds=elapsed,
                error_message=error_msg
            )
    
    def _create_error_result(
        self,
        config: ModelConfig,
        input_len: int,
        output_len: int,
        batch_size: int,
        error_msg: str,
        duration: float
    ) -> ProfileResult:
        """Helper to create error result"""
        return ProfileResult(
            model_id=config.model_id,
            container_name=config.container_name,
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            profile_file="",
            success=False,
            error_message=error_msg,
            duration_seconds=duration
        )
    
    def _cleanup_gpu_memory(self, container_name: str) -> bool:
        """
        Force cleanup of GPU memory by terminating vLLM processes
        
        This ensures a clean baseline for the next profiling run by:
        1. Killing any lingering vLLM bench processes
        2. Killing any vLLM worker processes
        3. Clearing PyTorch CUDA cache
        4. Waiting for memory to settle
        
        Returns:
            bool: True if cleanup succeeded, False otherwise
        """
        print("  🧹 Cleaning up GPU memory...")
        
        try:
            # Kill vLLM bench processes
            print("    Terminating vLLM processes...")
            ContainerManager.exec_command(
                container_name,
                ['pkill', '-9', '-f', 'vllm'],
                check=False
            )
            
            # Give processes time to terminate
            time.sleep(2)
            
            # Force PyTorch CUDA cache cleanup
            print("    Clearing CUDA cache...")
            ContainerManager.exec_command(
                container_name,
                ['python3', '-c', 
                 'import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None'],
                check=False
            )
            
            # Wait for memory to stabilize
            print("    Waiting for memory to stabilize (5s)...")
            time.sleep(5)
            
            # Verify cleanup by checking GPU memory
            result = ContainerManager.exec_command(
                container_name,
                ['rocm-smi', '--showmeminfo', 'vram'],
                check=False
            )
            
            if result.returncode == 0:
                # Parse memory usage to verify it's low
                import re
                total_used_gb = 0
                for line in result.stdout.split('\n'):
                    if 'Total Used Memory' in line and 'GPU[' in line:
                        # Extract memory value
                        mem_match = re.search(r':\s*([\d,]+)', line.split(':', 1)[1])
                        if mem_match:
                            used_bytes = int(mem_match.group(1).replace(',', ''))
                            total_used_gb += used_bytes / 1e9
                
                print(f"    ✓ GPU memory after cleanup: {total_used_gb:.2f} GB")
                
                # Warn if memory is still high (> 5 GB suggests model still loaded)
                if total_used_gb > 5.0:
                    print(f"    ⚠️  WARNING: Memory usage still high ({total_used_gb:.2f} GB)")
                    print("       Model may not have fully unloaded")
                    return False
            
            print("    ✓ GPU memory cleanup complete")
            return True
            
        except Exception as e:
            print(f"    ⚠️  Cleanup failed: {e}")
            return False
    
    def profile_all_configs(self, config: ModelConfig) -> List[ProfileResult]:
        """Profile all parameter combinations for a model"""
        
        results = []
        total_configs = len(config.input_lengths) * len(config.output_lengths) * len(config.batch_sizes)
        current = 0
        
        for input_len in config.input_lengths:
            for output_len in config.output_lengths:
                for batch_size in config.batch_sizes:
                    current += 1
                    
                    result = self.profile_model(
                        config, input_len, output_len, batch_size,
                        config_num=current,
                        total_configs=total_configs
                    )
                    results.append(result)
                    self.results.append(result)
                    
                    # Cleanup GPU memory between configs (except after last)
                    if current < total_configs:
                        cleanup_success = self._cleanup_gpu_memory(config.container_name)
                        if not cleanup_success:
                            print("    ⚠️  WARNING: GPU cleanup may have failed")
                            print("       Next baseline measurement may be contaminated")
        
        return results
    
    def generate_report(self, output_file: Path):
        """Generate comprehensive text report"""
        
        report = []
        report.append("=" * 80)
        report.append("ENHANCED BATCH PROFILING REPORT (v2.0)")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Profiles: {len(self.results)}")
        report.append(f"Successful: {sum(1 for r in self.results if r.success)}")
        report.append(f"Failed: {sum(1 for r in self.results if not r.success)}")
        report.append("")
        
        # Group by model
        by_model: Dict[str, List[ProfileResult]] = {}
        for result in self.results:
            if result.model_id not in by_model:
                by_model[result.model_id] = []
            by_model[result.model_id].append(result)
        
        # Report per model
        for model_id, model_results in by_model.items():
            report.append("=" * 80)
            report.append(f"MODEL: {model_id}")
            report.append("=" * 80)
            
            # Get engine version from first successful result
            engine_version = next(
                (r.engine_version for r in model_results if r.engine_version),
                "unknown"
            )
            report.append(f"Engine Version: vLLM {engine_version}")
            report.append(f"Configurations: {len(model_results)}")
            report.append(f"Successful: {sum(1 for r in model_results if r.success)}")
            report.append("")
            
            for result in model_results:
                report.append(f"Configuration: input={result.input_len}, output={result.output_len}, batch={result.batch_size}")
                
                if result.success:
                    if result.total_memory_gb:
                        report.append(f"  Total Memory: {result.total_memory_gb:.2f} GB")
                    if result.baseline_memory_gb:
                        report.append(f"  Baseline: {result.baseline_memory_gb:.2f} GB")
                    if result.memory_increase_gb:
                        report.append(f"  Increase: {result.memory_increase_gb:.2f} GB")
                    
                    # Component breakdown
                    if result.weights_gb:
                        report.append(f"  Weights: {result.weights_gb:.2f} GB")
                    if result.kv_cache_gb:
                        report.append(f"  KV Cache: {result.kv_cache_gb:.2f} GB")
                    if result.activations_gb:
                        report.append(f"  Activations: {result.activations_gb:.2f} GB")
                    if result.overhead_gb:
                        report.append(f"  Overhead: {result.overhead_gb:.2f} GB")
                    
                    # Dtype info
                    if result.weight_dtype:
                        report.append(f"  Weight dtype: {result.weight_dtype}")
                    if result.estimation_confidence:
                        report.append(f"  Confidence: {result.estimation_confidence}")
                    
                    if result.latency_ms:
                        report.append(f"  Mean Latency: {result.latency_ms:.2f} ms")
                    if result.profile_file:
                        report.append(f"  Profile: {Path(result.profile_file).name}")
                else:
                    report.append(f"  Error: {result.error_message}")
                
                if result.duration_seconds:
                    report.append(f"  Duration: {result.duration_seconds:.1f}s")
                
                report.append("")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n📝 Report saved to: {output_file}")
    
    def generate_csv(self, output_file: Path):
        """Generate CSV for analysis with enhanced fields"""
        
        import csv
        
        fieldnames = [
            'model_id', 'container_name', 'input_len', 'output_len', 'batch_size',
            'success', 'profile_file',
            'total_memory_gb', 'baseline_memory_gb', 'memory_increase_gb',
            'weights_gb', 'kv_cache_gb', 'activations_gb', 'overhead_gb',
            'engine_version', 'weight_dtype', 'activation_dtype', 'kv_cache_dtype',
            'estimation_confidence', 'latency_ms', 'duration_seconds',
            'error_message'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow({
                    'model_id': result.model_id,
                    'container_name': result.container_name,
                    'input_len': result.input_len,
                    'output_len': result.output_len,
                    'batch_size': result.batch_size,
                    'success': result.success,
                    'profile_file': result.profile_file,
                    'total_memory_gb': result.total_memory_gb or '',
                    'baseline_memory_gb': result.baseline_memory_gb or '',
                    'memory_increase_gb': result.memory_increase_gb or '',
                    'weights_gb': result.weights_gb or '',
                    'kv_cache_gb': result.kv_cache_gb or '',
                    'activations_gb': result.activations_gb or '',
                    'overhead_gb': result.overhead_gb or '',
                    'engine_version': result.engine_version or '',
                    'weight_dtype': result.weight_dtype or '',
                    'activation_dtype': result.activation_dtype or '',
                    'kv_cache_dtype': result.kv_cache_dtype_actual or '',
                    'estimation_confidence': result.estimation_confidence or '',
                    'latency_ms': result.latency_ms or '',
                    'duration_seconds': result.duration_seconds or '',
                    'error_message': result.error_message or ''
                })
        
        print(f"📊 CSV saved to: {output_file}")


def load_config_file(config_path: Path) -> List[ModelConfig]:
    """Load model configurations from YAML file"""
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    configs = []
    for model_data in data.get('models', []):
        config = ModelConfig(**model_data)
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Batch profile models using enhanced vLLM bench approach (v2.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='YAML configuration file with model specifications'
    )
    parser.add_argument(
        '--container',
        type=str,
        help='Container name (for single model profiling)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model ID (for single model profiling)'
    )
    parser.add_argument(
        '--input-len',
        type=int,
        nargs='+',
        default=[256],
        help='Input sequence lengths to test (default: [256])'
    )
    parser.add_argument(
        '--output-len',
        type=int,
        nargs='+',
        default=[256],
        help='Output sequence lengths to test (default: [256])'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        nargs='+',
        default=[1],
        help='Batch sizes to test (default: [1])'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        help='Data type (default: float16)'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Tensor parallel size (default: 1)'
    )
    parser.add_argument(
        '--model-params',
        type=float,
        help='Number of model parameters (e.g., 7e9 for 7B)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        help='Number of model layers'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        help='Hidden size dimension'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results/memory-profiles'),
        help='Directory to save profile results (default: results/memory-profiles)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create configuration
    configs = []
    
    if args.config:
        print("📋 Loading configuration from file...")
        configs = load_config_file(args.config)
        print(f"   Found {len(configs)} model(s) to profile\n")
    elif args.container and args.model:
        print("📋 Creating single model configuration...")
        config = ModelConfig(
            model_id=args.model,
            container_name=args.container,
            input_lengths=args.input_len,
            output_lengths=args.output_len,
            batch_sizes=args.batch_size,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            model_params=args.model_params,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_size=args.hidden_size
        )
        configs = [config]
        print(f"   Model: {args.model}")
        print(f"   Container: {args.container}\n")
    else:
        print("Error: Either --config or (--container and --model) must be provided")
        parser.print_help()
        sys.exit(1)
    
    if args.dry_run:
        print("🔍 DRY RUN - No actual profiling will be performed\n")
        for config in configs:
            print(f"Would profile: {config.model_id}")
            print(f"  Container: {config.container_name}")
            print(f"  Input lengths: {config.input_lengths}")
            print(f"  Output lengths: {config.output_lengths}")
            print(f"  Batch sizes: {config.batch_sizes}")
            print(f"  Total configs: {len(config.input_lengths) * len(config.output_lengths) * len(config.batch_sizes)}")
            print()
        return
    
    # Verify containers exist
    print("🔍 Verifying containers...")
    for config in configs:
        if not ContainerManager.is_running(config.container_name):
            print(f"❌ Container not running: {config.container_name}")
            print(f"   Start it with: docker start {config.container_name}")
            sys.exit(1)
        print(f"   ✓ {config.container_name} is running")
    print()
    
    # Initialize enhanced profiler
    profiler = EnhancedBenchProfiler(results_dir, script_dir)
    
    # Process each model
    for i, config in enumerate(configs, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(configs)}] Processing: {config.model_id}")
        print("=" * 80)
        print(f"Container: {config.container_name}")
        print(f"Configurations to test: {len(config.input_lengths) * len(config.output_lengths) * len(config.batch_sizes)}")
        
        try:
            profiler.profile_all_configs(config)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        
        except Exception as e:
            print(f"❌ Error processing {config.model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f'batch-bench-report-enhanced-{timestamp}.txt'
    csv_file = results_dir / f'batch-bench-results-{timestamp}.csv'
    
    profiler.generate_report(report_file)
    profiler.generate_csv(csv_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("ENHANCED BATCH PROFILING COMPLETE (v2.0)")
    print("=" * 80)
    print(f"Total Profiles: {len(profiler.results)}")
    print(f"Successful: {sum(1 for r in profiler.results if r.success)}")
    print(f"Failed: {sum(1 for r in profiler.results if not r.success)}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Report: {report_file}")
    print(f"CSV: {csv_file}")
    print("\nNext steps:")
    print("  1. Review profile results with enhanced metadata")
    print("  2. Run validation: npm run batch-validate")
    print("  3. Analyze with: python scripts/analyze-validation.py")
    print("  4. Compare component breakdown accuracy")
    print("=" * 80)


if __name__ == '__main__':
    main()
