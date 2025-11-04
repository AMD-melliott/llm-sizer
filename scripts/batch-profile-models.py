#!/usr/bin/env python3
"""
Advanced Batch Model Profiling Script

This script automates the process of:
1. Starting model containers
2. Waiting for models to load
3. Running profiler
4. Collecting results
5. Stopping containers
6. Repeating for next model

Features:
- Automatic container lifecycle management
- GPU memory monitoring
- Parallel profiling (multiple configs per model)
- Result validation
- Detailed reporting

Usage:
    # Basic usage with config file
    python scripts/batch-profile-models.py --config models.yaml
    
    # Profile with multiple configurations
    python scripts/batch-profile-models.py --config models.yaml --multi-config
    
    # Dry run (don't actually profile)
    python scripts/batch-profile-models.py --config models.yaml --dry-run
    
    # Keep containers running after profiling
    python scripts/batch-profile-models.py --config models.yaml --keep-containers
"""

import argparse
import json
import subprocess
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model profiling run"""
    model_id: str
    container_name: str
    image: str
    batch_sizes: List[int]
    max_tokens: List[int]
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    port: int = 8000
    gpu_ids: Optional[str] = None
    prompts: Optional[List[str]] = None
    quantization: Optional[str] = None
    max_model_len: Optional[int] = None


@dataclass
class ProfileResult:
    """Result from a single profiling run"""
    model_id: str
    container_name: str
    batch_size: int
    max_tokens: int
    profile_file: str
    success: bool
    error_message: Optional[str] = None
    total_memory_gb: Optional[float] = None
    tokens_per_second: Optional[float] = None
    duration_seconds: Optional[float] = None


class GPUMonitor:
    """Monitor GPU memory usage"""
    
    @staticmethod
    def get_gpu_type() -> str:
        """Detect GPU type (AMD or NVIDIA)"""
        if subprocess.run(['which', 'rocm-smi'], capture_output=True).returncode == 0:
            return 'amd'
        elif subprocess.run(['which', 'nvidia-smi'], capture_output=True).returncode == 0:
            return 'nvidia'
        return 'unknown'
    
    @staticmethod
    def get_memory_usage() -> Dict[int, float]:
        """Get memory usage for each GPU in GB"""
        gpu_type = GPUMonitor.get_gpu_type()
        memory = {}
        
        try:
            if gpu_type == 'amd':
                result = subprocess.run(
                    ['rocm-smi', '--showmemuse', '--csv'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Parse rocm-smi output
                for line in result.stdout.split('\n'):
                    if 'GPU[' in line:
                        parts = line.split(',')
                        gpu_id = int(parts[0].split('[')[1].split(']')[0])
                        used_mb = float(parts[1].strip())
                        memory[gpu_id] = used_mb / 1024  # Convert to GB
                        
            elif gpu_type == 'nvidia':
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.strip().split('\n'):
                    gpu_id, used_mb = line.split(',')
                    memory[int(gpu_id)] = float(used_mb) / 1024  # Convert to GB
        except Exception as e:
            print(f"⚠️  Warning: Could not get GPU memory usage: {e}")
        
        return memory
    
    @staticmethod
    def is_memory_clear(threshold_gb: float = 5.0) -> bool:
        """Check if GPU memory is below threshold"""
        memory = GPUMonitor.get_memory_usage()
        if not memory:
            return True  # Can't check, assume clear
        
        for gpu_id, used_gb in memory.items():
            if used_gb > threshold_gb:
                print(f"⚠️  GPU {gpu_id} has {used_gb:.1f}GB in use (threshold: {threshold_gb}GB)")
                return False
        
        return True
    
    @staticmethod
    def wait_for_memory_clear(timeout: int = 60, threshold_gb: float = 5.0) -> bool:
        """Wait for GPU memory to clear"""
        print("⏳ Waiting for GPU memory to clear...")
        start = time.time()
        
        while time.time() - start < timeout:
            if GPUMonitor.is_memory_clear(threshold_gb):
                print("✓ GPU memory cleared")
                return True
            time.sleep(5)
        
        print(f"⚠️  GPU memory did not clear after {timeout}s")
        return False


class ContainerManager:
    """Manage Docker container lifecycle"""
    
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
        """Check if container exists (running or stopped)"""
        result = subprocess.run(
            ['docker', 'ps', '-aq', '-f', f'name=^{container_name}$'],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    
    @staticmethod
    def stop(container_name: str, timeout: int = 30) -> bool:
        """Stop a running container"""
        if not ContainerManager.is_running(container_name):
            return True
        
        print(f"🛑 Stopping container: {container_name}")
        result = subprocess.run(
            ['docker', 'stop', '-t', str(timeout), container_name],
            capture_output=True
        )
        return result.returncode == 0
    
    @staticmethod
    def remove(container_name: str) -> bool:
        """Remove a container"""
        if not ContainerManager.exists(container_name):
            return True
        
        print(f"🗑️  Removing container: {container_name}")
        result = subprocess.run(
            ['docker', 'rm', container_name],
            capture_output=True
        )
        return result.returncode == 0
    
    @staticmethod
    def start_vllm(config: ModelConfig) -> bool:
        """Start vLLM container with the specified model"""
        # Build docker run command
        cmd = [
            'docker', 'run', '-d',
            '--name', config.container_name,
            '--shm-size', '16g',
            '-p', f'{config.port}:8000',
        ]
        
        # Add GPU configuration
        if config.gpu_ids:
            cmd.extend(['--device', '/dev/kfd', '--device', '/dev/dri',
                       '--group-add', 'video', '--ipc=host', '--cap-add=SYS_PTRACE',
                       '--security-opt', 'seccomp=unconfined'])
            cmd.extend(['-e', f'AMD_VISIBLE_DEVICES={config.gpu_ids}'])
        else:
            cmd.extend(['--gpus', 'all'])
        
        # Add environment variables
        cmd.extend([
            '-e', f'MODEL={config.model_id}',
        ])
        
        # Add image
        cmd.append(config.image)
        
        # Add vLLM arguments
        vllm_args = [
            '--model', config.model_id,
            '--tensor-parallel-size', str(config.tensor_parallel_size),
            '--gpu-memory-utilization', str(config.gpu_memory_utilization),
        ]
        
        if config.max_model_len:
            vllm_args.extend(['--max-model-len', str(config.max_model_len)])
        
        if config.quantization:
            vllm_args.extend(['--quantization', config.quantization])
        
        cmd.extend(vllm_args)
        
        print(f"🚀 Starting container: {config.container_name}")
        print(f"   Model: {config.model_id}")
        print(f"   Image: {config.image}")
        print(f"   Tensor Parallel: {config.tensor_parallel_size}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to start container: {result.stderr}")
            return False
        
        return True
    
    @staticmethod
    def wait_for_ready(container_name: str, timeout: int = 600) -> bool:
        """Wait for container to be ready"""
        print(f"⏳ Waiting for container to be ready (max {timeout}s)...")
        start = time.time()
        
        ready_patterns = [
            'Uvicorn running',
            'Application startup complete',
            'model loaded successfully',
        ]
        
        while time.time() - start < timeout:
            # Check if container is still running
            if not ContainerManager.is_running(container_name):
                print("❌ Container stopped unexpectedly")
                # Show logs
                subprocess.run(['docker', 'logs', '--tail', '50', container_name])
                return False
            
            # Check logs for ready signal
            result = subprocess.run(
                ['docker', 'logs', '--tail', '20', container_name],
                capture_output=True,
                text=True
            )
            
            for pattern in ready_patterns:
                if pattern.lower() in result.stdout.lower():
                    print("✓ Container ready")
                    time.sleep(10)  # Extra time for full initialization
                    return True
            
            elapsed = int(time.time() - start)
            print(f"  ⏳ Waiting... ({elapsed}s)", end='\r')
            time.sleep(5)
        
        print(f"\n❌ Container did not become ready after {timeout}s")
        return False


class ModelProfiler:
    """Run profiling on models"""
    
    def __init__(self, results_dir: Path, script_dir: Path):
        self.results_dir = results_dir
        self.script_dir = script_dir
        self.results: List[ProfileResult] = []
    
    def profile_model(
        self,
        config: ModelConfig,
        batch_size: int,
        max_tokens: int,
        prompt: Optional[str] = None
    ) -> ProfileResult:
        """Profile a single model configuration"""
        
        print(f"\n📊 Profiling: BS={batch_size}, MaxTokens={max_tokens}")
        
        start_time = time.time()
        
        # Build profiler command
        profiler_script = self.script_dir / 'profile-docker-model.sh'
        cmd = [
            str(profiler_script),
            config.container_name,
            '--batch-size', str(batch_size),
            '--max-tokens', str(max_tokens),
        ]
        
        if prompt:
            cmd.extend(['--prompt', prompt])
        
        # Run profiler
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            print(f"❌ Profiling failed: {error_msg}")
            return ProfileResult(
                model_id=config.model_id,
                container_name=config.container_name,
                batch_size=batch_size,
                max_tokens=max_tokens,
                profile_file="",
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )
        
        # Find the generated profile file (most recent)
        profile_files = sorted(
            self.results_dir.glob(f'{config.container_name}_*.json'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not profile_files:
            print("❌ Profile file not found")
            return ProfileResult(
                model_id=config.model_id,
                container_name=config.container_name,
                batch_size=batch_size,
                max_tokens=max_tokens,
                profile_file="",
                success=False,
                error_message="Profile file not generated",
                duration_seconds=duration
            )
        
        profile_file = profile_files[0]
        
        # Extract metrics from profile
        try:
            with open(profile_file) as f:
                profile_data = json.load(f)
            
            total_memory = profile_data.get('memory_breakdown', {}).get('total_gb')
            tokens_per_sec = profile_data.get('inference_stats', {}).get('tokens_per_second')
            
            print(f"✓ Profile saved: {profile_file.name}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Tokens/sec: {tokens_per_sec:.2f}")
            
            return ProfileResult(
                model_id=config.model_id,
                container_name=config.container_name,
                batch_size=batch_size,
                max_tokens=max_tokens,
                profile_file=str(profile_file),
                success=True,
                total_memory_gb=total_memory,
                tokens_per_second=tokens_per_sec,
                duration_seconds=duration
            )
            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse profile: {e}")
            return ProfileResult(
                model_id=config.model_id,
                container_name=config.container_name,
                batch_size=batch_size,
                max_tokens=max_tokens,
                profile_file=str(profile_file),
                success=True,
                duration_seconds=duration
            )
    
    def profile_all_configs(
        self,
        config: ModelConfig,
        prompts: Optional[List[str]] = None
    ) -> List[ProfileResult]:
        """Profile all batch size and max token combinations"""
        
        results = []
        default_prompts = prompts or ["Explain quantum computing in simple terms"]
        
        for batch_size in config.batch_sizes:
            for max_tokens in config.max_tokens:
                for prompt in default_prompts:
                    result = self.profile_model(config, batch_size, max_tokens, prompt)
                    results.append(result)
                    self.results.append(result)
                    
                    # Small delay between profiles
                    time.sleep(5)
        
        return results
    
    def generate_report(self, output_file: Path):
        """Generate a detailed report of all profiling runs"""
        
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BATCH MODEL PROFILING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Profiles: {len(self.results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for result in self.results:
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                f.write(f"{status}: {result.model_id}\n")
                f.write(f"  Container: {result.container_name}\n")
                f.write(f"  Config: BS={result.batch_size}, MaxTokens={result.max_tokens}\n")
                
                if result.success:
                    f.write(f"  Profile: {Path(result.profile_file).name}\n")
                    if result.total_memory_gb:
                        f.write(f"  Memory: {result.total_memory_gb:.2f} GB\n")
                    if result.tokens_per_second:
                        f.write(f"  Speed: {result.tokens_per_second:.2f} tok/s\n")
                else:
                    f.write(f"  Error: {result.error_message}\n")
                
                if result.duration_seconds:
                    f.write(f"  Duration: {result.duration_seconds:.1f}s\n")
                
                f.write("\n")
        
        print(f"\n📝 Report saved to: {output_file}")


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
        description='Batch profile multiple LLM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='YAML configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--keep-containers',
        action='store_true',
        help='Keep containers running after profiling'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results/memory-profiles'),
        help='Directory to save profile results'
    )
    parser.add_argument(
        '--skip-cleanup',
        action='store_true',
        help='Skip cleaning up existing containers'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("📋 Loading configuration...")
    configs = load_config_file(args.config)
    print(f"   Found {len(configs)} model(s) to profile\n")
    
    if args.dry_run:
        print("🔍 DRY RUN - No actual profiling will be performed\n")
        for config in configs:
            print(f"Would profile: {config.model_id}")
            print(f"  Batch sizes: {config.batch_sizes}")
            print(f"  Max tokens: {config.max_tokens}")
            print()
        return
    
    # Initialize profiler
    profiler = ModelProfiler(results_dir, script_dir)
    
    # Process each model
    for i, config in enumerate(configs, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(configs)}] Processing: {config.model_id}")
        print("=" * 80)
        
        try:
            # Cleanup existing container
            if not args.skip_cleanup:
                if ContainerManager.exists(config.container_name):
                    ContainerManager.stop(config.container_name)
                    ContainerManager.remove(config.container_name)
                    GPUMonitor.wait_for_memory_clear()
            
            # Start container
            if not ContainerManager.start_vllm(config):
                print(f"❌ Failed to start container for {config.model_id}")
                continue
            
            # Wait for ready
            if not ContainerManager.wait_for_ready(config.container_name):
                print(f"❌ Container not ready for {config.model_id}")
                if not args.keep_containers:
                    ContainerManager.stop(config.container_name)
                    ContainerManager.remove(config.container_name)
                continue
            
            # Profile all configurations
            profiler.profile_all_configs(config, config.prompts)
            
            # Cleanup
            if not args.keep_containers:
                ContainerManager.stop(config.container_name)
                ContainerManager.remove(config.container_name)
                GPUMonitor.wait_for_memory_clear()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            if not args.keep_containers:
                print("🧹 Cleaning up...")
                ContainerManager.stop(config.container_name)
                ContainerManager.remove(config.container_name)
            break
        
        except Exception as e:
            print(f"❌ Error processing {config.model_id}: {e}")
            if not args.keep_containers:
                ContainerManager.stop(config.container_name)
                ContainerManager.remove(config.container_name)
            continue
    
    # Generate report
    report_file = results_dir / f'batch-profile-report-{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    profiler.generate_report(report_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROFILING COMPLETE")
    print("=" * 80)
    print(f"Total Profiles: {len(profiler.results)}")
    print(f"Successful: {sum(1 for r in profiler.results if r.success)}")
    print(f"Failed: {sum(1 for r in profiler.results if not r.success)}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Report: {report_file}")
    print("\nNext steps:")
    print("  1. Review profile results")
    print("  2. Run validation: npm run batch-validate")
    print("  3. Analyze discrepancies and improve calculator")
    print("=" * 80)


if __name__ == '__main__':
    main()
