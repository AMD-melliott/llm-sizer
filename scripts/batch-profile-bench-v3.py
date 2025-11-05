#!/usr/bin/env python3
"""
Batch Profiling Script using vLLM Bench v3.0 Profiler

This script automates profiling multiple models using the enhanced v3.0 vLLM bench profiler
which captures comprehensive vLLM internal metrics, real-time memory monitoring, and
extended AMD GPU metrics.

Key enhancements in v3.0:
- Real-time memory monitoring during benchmark execution
- vLLM log parsing for internal metrics (KV cache blocks, attention backend)
- Extended AMD GPU metrics using modern amd-smi monitor
- Multi-phase memory capture timeline
- Comprehensive power, temperature, and utilization tracking
- Distinction between KV cache allocated vs used memory

Usage:
    # Profile with config file
    python scripts/batch-profile-bench-v3.py --config scripts/configs/quick.yaml

    # Profile specific container with architecture details
    python scripts/batch-profile-bench-v3.py \
        --container vllm-inference \
        --model meta-llama/Llama-2-7b-hf \
        --input-len 256 \
        --output-len 256

    # Dry run to see what would be done
    python scripts/batch-profile-bench-v3.py --config scripts/configs/quick.yaml --dry-run

Config file format (YAML):
    models:
      - hf_model_id: "meta-llama/Llama-3.2-1B-Instruct"
        container_name: "vllm-inference"
        input_lengths: [256, 512, 1024]
        output_lengths: [256, 512]
        batch_sizes: [1, 8, 16]
        dtype: "float16"
        tensor_parallel_size: 1
        quantization: null
        kv_cache_dtype: "fp8"
        trust_remote_code: false
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
    """
    Configuration for a single model profiling run

    Model architecture parameters are auto-loaded from src/data/models.json by hf_model_id.
    Only test configuration parameters should be specified here.
    """
    hf_model_id: str  # HuggingFace model ID (references models.json)
    container_name: str
    input_lengths: List[int]
    output_lengths: List[int]
    batch_sizes: List[int]
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None
    kv_cache_dtype: Optional[str] = None
    trust_remote_code: bool = False
    enforce_eager: bool = False
    gpu_ids: Optional[str] = None


@dataclass
class ProfileResult:
    """Result from a single profiling run with v3.0 enhanced metadata"""
    model_id: str
    container_name: str
    input_len: int
    output_len: int
    batch_size: int
    profile_file: str
    success: bool
    error_message: Optional[str] = None

    # Memory metrics (enhanced v3.0)
    total_memory_gb: Optional[float] = None
    baseline_memory_gb: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    memory_increase_gb: Optional[float] = None

    # v3.0 NEW: Memory timeline statistics
    memory_timeline_min_gb: Optional[float] = None
    memory_timeline_max_gb: Optional[float] = None
    memory_timeline_mean_gb: Optional[float] = None
    memory_timeline_p95_gb: Optional[float] = None
    memory_timeline_samples: Optional[int] = None

    # Component breakdown (v3.0 enhanced)
    weights_gb: Optional[float] = None
    kv_cache_allocated_gb: Optional[float] = None  # v3.0 NEW: Pre-allocated pool
    kv_cache_used_gb: Optional[float] = None       # v3.0 NEW: Actual usage
    graph_capture_gb: Optional[float] = None        # v3.0 NEW: CUDA graphs
    activations_gb: Optional[float] = None
    overhead_gb: Optional[float] = None

    # v3.0 NEW: vLLM internal metrics
    vllm_num_gpu_blocks: Optional[int] = None
    vllm_block_size: Optional[int] = None
    vllm_kv_cache_tokens: Optional[int] = None
    vllm_kv_cache_gb: Optional[float] = None
    vllm_attention_backend: Optional[str] = None
    vllm_log_format_version: Optional[str] = None

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

    # v3.0 NEW: Extended metrics (AMD GPUs)
    power_mean_w: Optional[float] = None
    power_max_w: Optional[float] = None
    temp_mean_c: Optional[float] = None
    temp_max_c: Optional[float] = None


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


def ensure_latest_scripts(
    container_name: str,
    repo_url: str = "https://github.com/AMD-melliott/llm-sizer.git",
    target_path: str = "/app/llm-sizer",
    branch: str = "calculator-adjustments"
) -> str:
    """
    Clone or pull latest llm-sizer code into container

    Args:
        container_name: Name of Docker container
        repo_url: Git repository URL
        target_path: Path in container to clone/pull to
        branch: Git branch to checkout (default: calculator-adjustments)

    Returns:
        Current git commit hash for tracking

    Raises:
        subprocess.CalledProcessError: If git operations fail
    """
    # Check if repo exists in container
    result = ContainerManager.exec_command(
        container_name,
        ['test', '-d', f'{target_path}/.git'],
        check=False
    )

    if result.returncode != 0:
        # Clone repo
        print(f"  📥 Cloning llm-sizer repository to {target_path}...")
        ContainerManager.exec_command(
            container_name,
            ['git', 'clone', '-b', branch, repo_url, target_path],
            check=True
        )
        print(f"  ✓ Repository cloned (branch: {branch})")
    else:
        # Fetch and checkout branch
        print(f"  🔄 Fetching latest changes...")
        ContainerManager.exec_command(
            container_name,
            ['git', '-C', target_path, 'fetch', 'origin'],
            check=True
        )

        # Checkout the specified branch
        ContainerManager.exec_command(
            container_name,
            ['git', '-C', target_path, 'checkout', branch],
            check=True
        )

        # Pull latest changes
        ContainerManager.exec_command(
            container_name,
            ['git', '-C', target_path, 'pull', 'origin', branch],
            check=True
        )
        print(f"  ✓ Repository updated (branch: {branch})")

    # Get current commit hash for tracking
    result = ContainerManager.exec_command(
        container_name,
        ['git', '-C', target_path, 'rev-parse', 'HEAD'],
        check=True
    )
    commit_hash = result.stdout.strip()[:8]  # Short hash

    # Verify we're on the correct branch
    result = ContainerManager.exec_command(
        container_name,
        ['git', '-C', target_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
        check=True
    )
    current_branch = result.stdout.strip()

    print(f"  ✓ Using commit {commit_hash} on branch {current_branch}")

    return commit_hash


class V3BenchProfiler:
    """Run v3.0 vLLM bench profiling in containers"""

    def __init__(self, results_dir: Path, script_dir: Path, use_git_pull: bool = True):
        self.results_dir = results_dir
        self.script_dir = script_dir
        self.results: List[ProfileResult] = []
        self.use_git_pull = use_git_pull

    def profile_model(
        self,
        config: ModelConfig,
        input_len: int,
        output_len: int,
        batch_size: int,
        config_num: int = 1,
        total_configs: int = 1
    ) -> ProfileResult:
        """Profile a single model configuration using v3.0 vllm bench profiler"""

        # Extract model name (last part of path)
        model_name = config.hf_model_id.split('/')[-1]

        # Try to get model size from models.json
        size_str = "?"
        try:
            # Import model_loader to get model info
            sys.path.insert(0, str(self.script_dir / 'lib'))
            from model_loader import get_model_info

            model_info = get_model_info(config.hf_model_id)
            if model_info and 'parameters_billions' in model_info:
                size_b = model_info['parameters_billions']
                if size_b >= 1:
                    size_str = f"{size_b:.0f}B"
                else:
                    size_str = f"{size_b*1000:.0f}M"
        except (ImportError, ValueError, TypeError, FileNotFoundError):
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

        # Ensure scripts are available in container
        if self.use_git_pull:
            # Use git clone/pull for latest code
            try:
                commit_hash = ensure_latest_scripts(config.container_name)
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to sync repository: {e}"
                print(f"    ❌ {error_msg}")
                return self._create_error_result(
                    config, input_len, output_len, batch_size,
                    error_msg, time.time() - start_time
                )

            # Use v3.0 profiler from cloned repo
            profiler_path = '/app/llm-sizer/scripts/profile-vllm-bench-v3.py'
        else:
            error_msg = "Legacy docker cp mode not supported for v3.0 profiler (use --no-git-pull=false)"
            print(f"    ❌ {error_msg}")
            return self._create_error_result(
                config, input_len, output_len, batch_size,
                error_msg, time.time() - start_time
            )

        # Build profiler command to run inside container
        output_filename = f'{config.container_name}_{input_len}in_{output_len}out_bs{batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        container_output = f'/tmp/{output_filename}'

        # Use --hf-model-id to auto-load architecture from models.json
        cmd = [
            'python3', profiler_path,
            '--hf-model-id', config.hf_model_id,
            '--input-len', str(input_len),
            '--output-len', str(output_len),
            '--batch-size', str(batch_size),
            '--dtype', config.dtype,
            '--tensor-parallel-size', str(config.tensor_parallel_size),
            '--output', container_output
        ]

        if config.quantization:
            cmd.extend(['--quantization', config.quantization])

        if config.kv_cache_dtype:
            cmd.extend(['--kv-cache-dtype', config.kv_cache_dtype])

        if config.trust_remote_code:
            cmd.append('--trust-remote-code')

        if config.enforce_eager:
            cmd.append('--enforce-eager')

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

        # Parse v3.0 enhanced results
        try:
            with open(local_output) as f:
                profile_data = json.load(f)

            # Extract memory measurements
            mem_meas = profile_data.get('memory_measurements', {})
            baseline_stats = mem_meas.get('baseline', {})
            peak_stats = mem_meas.get('peak_from_timeline', {})

            # Extract v3.0 memory timeline
            mem_timeline = profile_data.get('memory_timeline', {})

            # Extract memory breakdown (v3.0 enhanced)
            mem_breakdown = profile_data.get('memory_breakdown', {})

            # Extract v3.0 vLLM internal metrics
            vllm_metrics = profile_data.get('vllm_internal_metrics', {})

            # Extract dtype metadata
            dtype_meta = profile_data.get('dtype_metadata', {})

            # Extract engine metadata
            engine_meta = profile_data.get('engine_metadata', {})

            # Extract latency
            latency_stats = profile_data.get('latency_stats', {})
            mean_latency_ms = latency_stats.get('mean_latency_ms') or latency_stats.get('avg_latency', 0) * 1000

            # Extract v3.0 extended metrics
            extended_metrics = mem_timeline.get('extended', {}) if mem_timeline else {}
            power_metrics = extended_metrics.get('power', {})
            temp_metrics = extended_metrics.get('temperature', {})

            # Concise single-line result with v3.0 enhancements
            engine_version = engine_meta.get('engine_version', 'unknown')
            peak_gb = peak_stats.get('total_gb', 0)
            increase_gb = mem_meas.get('memory_increase_gb', 0)

            result_str = f"    ✓ {peak_gb:.1f} GB peak (+{increase_gb:.1f} GB) | {elapsed:.1f}s | vLLM {engine_version}"

            # Add v3.0 log format indicator
            log_version = vllm_metrics.get('log_format_version', 'unknown')
            if log_version != 'unknown':
                result_str += f" | {log_version}"

            print(result_str)

            # Display v3.0 vLLM internal metrics if available
            if vllm_metrics.get('num_gpu_blocks') or vllm_metrics.get('total_gpu_kv_cache_tokens'):
                vllm_info = []
                if vllm_metrics.get('num_gpu_blocks'):
                    vllm_info.append(f"Blocks:{vllm_metrics['num_gpu_blocks']}")
                if vllm_metrics.get('total_gpu_kv_cache_tokens'):
                    vllm_info.append(f"KV:{vllm_metrics['total_gpu_kv_cache_tokens']:,}tok")
                if vllm_metrics.get('attention_backend'):
                    vllm_info.append(f"{vllm_metrics['attention_backend']}")
                print(f"    🔍 vLLM: {' | '.join(vllm_info)}")

            # Display memory timeline summary
            if mem_timeline and mem_timeline.get('num_samples'):
                timeline_info = f"Timeline: {mem_timeline['num_samples']} samples | "
                timeline_info += f"{mem_timeline.get('min_gb', 0):.1f}-{mem_timeline.get('max_gb', 0):.1f} GB"
                if power_metrics:
                    timeline_info += f" | {power_metrics.get('mean_w', 0):.0f}W"
                if temp_metrics:
                    timeline_info += f" | {temp_metrics.get('mean_c', 0):.0f}°C"
                print(f"    📈 {timeline_info}")

            return ProfileResult(
                model_id=config.hf_model_id,
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

                # v3.0 Memory timeline
                memory_timeline_min_gb=mem_timeline.get('min_gb'),
                memory_timeline_max_gb=mem_timeline.get('max_gb'),
                memory_timeline_mean_gb=mem_timeline.get('mean_gb'),
                memory_timeline_p95_gb=mem_timeline.get('p95_gb'),
                memory_timeline_samples=mem_timeline.get('num_samples'),

                # v3.0 Enhanced component breakdown
                weights_gb=mem_breakdown.get('model_weights_gb'),
                kv_cache_allocated_gb=mem_breakdown.get('kv_cache_allocated_gb'),
                kv_cache_used_gb=mem_breakdown.get('kv_cache_used_gb'),
                graph_capture_gb=mem_breakdown.get('graph_capture_gb'),
                activations_gb=mem_breakdown.get('activations_gb'),
                overhead_gb=mem_breakdown.get('framework_overhead_gb'),

                # v3.0 vLLM internal metrics
                vllm_num_gpu_blocks=vllm_metrics.get('num_gpu_blocks'),
                vllm_block_size=vllm_metrics.get('block_size'),
                vllm_kv_cache_tokens=vllm_metrics.get('total_gpu_kv_cache_tokens'),
                vllm_kv_cache_gb=vllm_metrics.get('total_gpu_kv_cache_gb'),
                vllm_attention_backend=vllm_metrics.get('attention_backend'),
                vllm_log_format_version=vllm_metrics.get('log_format_version'),

                # Standard metadata
                engine_version=engine_meta.get('engine_version'),
                weight_dtype=dtype_meta.get('weight_dtype'),
                activation_dtype=dtype_meta.get('activation_dtype'),
                kv_cache_dtype_actual=dtype_meta.get('kv_cache_dtype'),
                latency_ms=mean_latency_ms,
                duration_seconds=elapsed,
                estimation_confidence=mem_breakdown.get('confidence_levels', {}).get('overall'),

                # v3.0 Extended metrics
                power_mean_w=power_metrics.get('mean_w'),
                power_max_w=power_metrics.get('max_w'),
                temp_mean_c=temp_metrics.get('mean_c'),
                temp_max_c=temp_metrics.get('max_c')
            )

        except Exception as e:
            error_msg = f"Failed to parse profile: {e}"
            print(f"    ⚠️  {error_msg}")
            return ProfileResult(
                model_id=config.hf_model_id,
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
            model_id=config.hf_model_id,
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

        Returns:
            bool: True if cleanup succeeded, False otherwise
        """
        print("  🧹 Cleaning up GPU memory...")

        try:
            # Kill vLLM processes
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

        return results

    def generate_report(self, output_file: Path):
        """Generate comprehensive text report with v3.0 enhancements"""

        report = []
        report.append("=" * 80)
        report.append("BATCH PROFILING REPORT v3.0 (Enhanced with vLLM Internal Metrics)")
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
                    # Memory measurements
                    if result.total_memory_gb:
                        report.append(f"  Total Memory: {result.total_memory_gb:.2f} GB")
                    if result.baseline_memory_gb:
                        report.append(f"  Baseline: {result.baseline_memory_gb:.2f} GB")
                    if result.memory_increase_gb:
                        report.append(f"  Increase: {result.memory_increase_gb:.2f} GB")

                    # v3.0 Memory timeline
                    if result.memory_timeline_samples:
                        report.append(f"  Memory Timeline: {result.memory_timeline_samples} samples")
                        report.append(f"    Min: {result.memory_timeline_min_gb:.2f} GB")
                        report.append(f"    Mean: {result.memory_timeline_mean_gb:.2f} GB")
                        report.append(f"    Max: {result.memory_timeline_max_gb:.2f} GB")
                        report.append(f"    P95: {result.memory_timeline_p95_gb:.2f} GB")

                    # v3.0 vLLM internal metrics
                    if result.vllm_num_gpu_blocks or result.vllm_kv_cache_tokens:
                        report.append(f"  vLLM Internal Metrics ({result.vllm_log_format_version or 'unknown'}):")
                        if result.vllm_num_gpu_blocks:
                            report.append(f"    GPU Blocks: {result.vllm_num_gpu_blocks}")
                        if result.vllm_block_size:
                            report.append(f"    Block Size: {result.vllm_block_size} tokens")
                        if result.vllm_kv_cache_tokens:
                            report.append(f"    KV Cache Tokens: {result.vllm_kv_cache_tokens:,}")
                        if result.vllm_kv_cache_gb:
                            report.append(f"    KV Cache GB: {result.vllm_kv_cache_gb:.2f}")
                        if result.vllm_attention_backend:
                            report.append(f"    Attention Backend: {result.vllm_attention_backend}")

                    # Component breakdown (v3.0 enhanced)
                    if result.weights_gb:
                        report.append(f"  Weights: {result.weights_gb:.2f} GB")
                    if result.kv_cache_allocated_gb:
                        report.append(f"  KV Cache (allocated): {result.kv_cache_allocated_gb:.2f} GB")
                    if result.kv_cache_used_gb:
                        report.append(f"  KV Cache (used): {result.kv_cache_used_gb:.2f} GB")
                    if result.graph_capture_gb:
                        report.append(f"  CUDA Graphs: {result.graph_capture_gb:.2f} GB")
                    if result.activations_gb:
                        report.append(f"  Activations: {result.activations_gb:.2f} GB")
                    if result.overhead_gb:
                        report.append(f"  Overhead: {result.overhead_gb:.2f} GB")

                    # v3.0 Extended metrics
                    if result.power_mean_w or result.temp_mean_c:
                        report.append(f"  Extended Metrics:")
                        if result.power_mean_w:
                            report.append(f"    Power: {result.power_mean_w:.0f}W (max: {result.power_max_w:.0f}W)")
                        if result.temp_mean_c:
                            report.append(f"    Temperature: {result.temp_mean_c:.0f}°C (max: {result.temp_max_c:.0f}°C)")

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
        """Generate CSV for analysis with v3.0 enhanced fields"""

        import csv

        fieldnames = [
            'model_id', 'container_name', 'input_len', 'output_len', 'batch_size',
            'success', 'profile_file',
            'total_memory_gb', 'baseline_memory_gb', 'memory_increase_gb',

            # v3.0 Memory timeline
            'memory_timeline_min_gb', 'memory_timeline_max_gb', 'memory_timeline_mean_gb',
            'memory_timeline_p95_gb', 'memory_timeline_samples',

            # v3.0 Enhanced breakdown
            'weights_gb', 'kv_cache_allocated_gb', 'kv_cache_used_gb',
            'graph_capture_gb', 'activations_gb', 'overhead_gb',

            # v3.0 vLLM internal metrics
            'vllm_num_gpu_blocks', 'vllm_block_size', 'vllm_kv_cache_tokens',
            'vllm_kv_cache_gb', 'vllm_attention_backend', 'vllm_log_format_version',

            # Standard metadata
            'engine_version', 'weight_dtype', 'activation_dtype', 'kv_cache_dtype',
            'estimation_confidence', 'latency_ms', 'duration_seconds',

            # v3.0 Extended metrics
            'power_mean_w', 'power_max_w', 'temp_mean_c', 'temp_max_c',

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

                    # v3.0 Memory timeline
                    'memory_timeline_min_gb': result.memory_timeline_min_gb or '',
                    'memory_timeline_max_gb': result.memory_timeline_max_gb or '',
                    'memory_timeline_mean_gb': result.memory_timeline_mean_gb or '',
                    'memory_timeline_p95_gb': result.memory_timeline_p95_gb or '',
                    'memory_timeline_samples': result.memory_timeline_samples or '',

                    # v3.0 Enhanced breakdown
                    'weights_gb': result.weights_gb or '',
                    'kv_cache_allocated_gb': result.kv_cache_allocated_gb or '',
                    'kv_cache_used_gb': result.kv_cache_used_gb or '',
                    'graph_capture_gb': result.graph_capture_gb or '',
                    'activations_gb': result.activations_gb or '',
                    'overhead_gb': result.overhead_gb or '',

                    # v3.0 vLLM internal metrics
                    'vllm_num_gpu_blocks': result.vllm_num_gpu_blocks or '',
                    'vllm_block_size': result.vllm_block_size or '',
                    'vllm_kv_cache_tokens': result.vllm_kv_cache_tokens or '',
                    'vllm_kv_cache_gb': result.vllm_kv_cache_gb or '',
                    'vllm_attention_backend': result.vllm_attention_backend or '',
                    'vllm_log_format_version': result.vllm_log_format_version or '',

                    # Standard metadata
                    'engine_version': result.engine_version or '',
                    'weight_dtype': result.weight_dtype or '',
                    'activation_dtype': result.activation_dtype or '',
                    'kv_cache_dtype': result.kv_cache_dtype_actual or '',
                    'estimation_confidence': result.estimation_confidence or '',
                    'latency_ms': result.latency_ms or '',
                    'duration_seconds': result.duration_seconds or '',

                    # v3.0 Extended metrics
                    'power_mean_w': result.power_mean_w or '',
                    'power_max_w': result.power_max_w or '',
                    'temp_mean_c': result.temp_mean_c or '',
                    'temp_max_c': result.temp_max_c or '',

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
        description='Batch profile models using v3.0 vLLM bench profiler',
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
            hf_model_id=args.model,
            container_name=args.container,
            input_lengths=args.input_len,
            output_lengths=args.output_len,
            batch_sizes=args.batch_size,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size
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
            print(f"Would profile: {config.hf_model_id}")
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

    # Initialize v3.0 profiler (use git pull for latest code)
    profiler = V3BenchProfiler(results_dir, script_dir, use_git_pull=True)

    # Process each model
    for i, config in enumerate(configs, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(configs)}] Processing: {config.hf_model_id}")
        print("=" * 80)
        print(f"Container: {config.container_name}")
        print(f"Configurations to test: {len(config.input_lengths) * len(config.output_lengths) * len(config.batch_sizes)}")

        try:
            profiler.profile_all_configs(config)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break

        except Exception as e:
            print(f"❌ Error processing {config.hf_model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f'batch-bench-report-v3-{timestamp}.txt'
    csv_file = results_dir / f'batch-bench-results-v3-{timestamp}.csv'

    profiler.generate_report(report_file)
    profiler.generate_csv(csv_file)

    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROFILING COMPLETE v3.0")
    print("=" * 80)
    print(f"Total Profiles: {len(profiler.results)}")
    print(f"Successful: {sum(1 for r in profiler.results if r.success)}")
    print(f"Failed: {sum(1 for r in profiler.results if not r.success)}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Report: {report_file}")
    print(f"CSV: {csv_file}")
    print("\nv3.0 Enhancements:")
    print("  ✓ Real-time memory monitoring during execution")
    print("  ✓ vLLM internal metrics (KV cache blocks, attention backend)")
    print("  ✓ Extended AMD GPU metrics (power, temperature)")
    print("  ✓ Memory timeline statistics")
    print("  ✓ Distinction between KV cache allocated vs used")
    print("=" * 80)


if __name__ == '__main__':
    main()
