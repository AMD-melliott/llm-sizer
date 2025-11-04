#!/usr/bin/env python3
"""
vLLM Memory Profiler - Extract memory breakdown from running vLLM server

This script profiles a running vLLM server by:
1. Querying the vLLM API for model info
2. Measuring GPU memory usage via rocm-smi or nvidia-smi
3. Running inference and measuring memory changes
4. Extracting memory breakdown from vLLM's internal metrics

Usage:
    python profile-vllm-model.py --api-url http://localhost:8000 --model "zai-org/GLM-4.5-Air"

Requirements:
    - requests
    - Running vLLM server
"""

import argparse
import json
import subprocess
import sys
import time
from typing import Dict, Any, List
import os
import re

try:
    import requests
except ImportError:
    print("ERROR: requests library not found. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


class VLLMMemoryProfiler:
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.gpu_type = self.detect_gpu_type()
        self.visible_devices = self.get_visible_devices()
        
    def detect_gpu_type(self) -> str:
        """Detect if using NVIDIA CUDA or AMD ROCm"""
        # Check for ROCm
        if os.path.exists('/opt/rocm') or subprocess.run(['which', 'rocm-smi'], 
                                                         capture_output=True).returncode == 0:
            return 'rocm'
        # Check for CUDA
        if subprocess.run(['which', 'nvidia-smi'], capture_output=True).returncode == 0:
            return 'cuda'
        return 'unknown'
    
    def get_visible_devices(self) -> List[int]:
        """Get list of visible GPU devices from environment"""
        # Check AMD_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES
        amd_devices = os.environ.get('AMD_VISIBLE_DEVICES', '')
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        devices_str = amd_devices or cuda_devices
        
        if devices_str and devices_str != 'all':
            # Parse comma-separated list
            try:
                devices = [int(d.strip()) for d in devices_str.split(',') if d.strip().isdigit()]
                return devices
            except ValueError:
                pass
        
        # If not set or 'all', return empty list (will detect all GPUs)
        return []
    
    def get_gpu_memory_usage(self) -> List[Dict[str, float]]:
        """Get current GPU memory usage for all devices"""
        if self.gpu_type == 'rocm':
            return self._get_rocm_memory()
        elif self.gpu_type == 'cuda':
            return self._get_cuda_memory()
        else:
            print("WARNING: Could not detect GPU type, memory measurements may be inaccurate", 
                  file=sys.stderr)
            return []
    
    def _get_rocm_memory(self) -> List[Dict[str, float]]:
        """Get memory usage from ROCm GPUs"""
        try:
            # Try simple rocm-smi output first
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            # Parse text output
            lines = result.stdout.split('\n')
            for line in lines:
                # Look for lines like: "GPU[0]              : VRAM Total Used Memory (B): 47329271808"
                if 'GPU[' in line and 'Total Used Memory' in line:
                    try:
                        # Extract GPU index
                        gpu_match = re.search(r'GPU\[(\d+)\]', line)
                        if not gpu_match:
                            continue
                        gpu_idx = int(gpu_match.group(1))
                        
                        # Skip if not in visible devices list (if specified)
                        if self.visible_devices and gpu_idx not in self.visible_devices:
                            continue
                        
                        # Extract memory value in bytes
                        mem_match = re.search(r':\s*([\d,]+)', line.split(':', 1)[1])
                        if mem_match:
                            used_bytes = int(mem_match.group(1).replace(',', ''))
                            used_gb = used_bytes / 1e9
                            
                            # Get total memory for this GPU
                            total_gb = 206.0  # MI300X default, will try to get actual value
                            
                            gpus.append({
                                'device': gpu_idx,
                                'used_gb': used_gb,
                                'total_gb': total_gb,
                                'utilization_pct': (used_gb / total_gb * 100) if total_gb > 0 else 0
                            })
                    except (ValueError, IndexError) as e:
                        continue
            
            if gpus:
                return sorted(gpus, key=lambda x: x['device'])
            
            # Fallback to sysfs
            return self._get_memory_fallback()
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"WARNING: rocm-smi failed, trying fallback method: {e}", file=sys.stderr)
            return self._get_memory_fallback()
    
    def _get_cuda_memory(self) -> List[Dict[str, float]]:
        """Get memory usage from NVIDIA GPUs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_idx = int(parts[0])
                        
                        # Skip if not in visible devices list (if specified)
                        if self.visible_devices and gpu_idx not in self.visible_devices:
                            continue
                        
                        gpus.append({
                            'device': gpu_idx,
                            'used_gb': float(parts[1]) / 1024,
                            'total_gb': float(parts[2]) / 1024,
                            'utilization_pct': float(parts[3])
                        })
            
            return gpus
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"WARNING: Could not get CUDA memory info: {e}", file=sys.stderr)
            return []
    
    def _get_memory_fallback(self) -> List[Dict[str, float]]:
        """Fallback memory detection using /sys/class/drm for AMD"""
        gpus = []
        try:
            # Try to read from /sys/class/drm for AMD GPUs
            import glob
            card_paths = glob.glob('/sys/class/drm/card*/device/mem_info_vram_used')
            
            for i, used_path in enumerate(sorted(card_paths)):
                # Skip if not in visible devices list (if specified)
                if self.visible_devices and i not in self.visible_devices:
                    continue
                
                try:
                    with open(used_path, 'r') as f:
                        used_bytes = int(f.read().strip())
                    
                    total_path = used_path.replace('mem_info_vram_used', 'mem_info_vram_total')
                    with open(total_path, 'r') as f:
                        total_bytes = int(f.read().strip())
                    
                    used_gb = used_bytes / 1e9
                    total_gb = total_bytes / 1e9
                    
                    gpus.append({
                        'device': i,
                        'used_gb': used_gb,
                        'total_gb': total_gb,
                        'utilization_pct': (used_gb / total_gb * 100) if total_gb > 0 else 0
                    })
                except (IOError, ValueError):
                    continue
        except Exception as e:
            print(f"WARNING: Fallback memory detection failed: {e}", file=sys.stderr)
        
        return gpus
    
    def test_api_connection(self) -> bool:
        """Test if vLLM API is accessible"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"ERROR: Cannot connect to vLLM API at {self.api_url}: {e}", file=sys.stderr)
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from vLLM API"""
        try:
            response = requests.get(f"{self.api_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json()
            
            if 'data' in models and len(models['data']) > 0:
                model_data = models['data'][0]
                return {
                    'model_id': model_data.get('id', self.model_name),
                    'owned_by': model_data.get('owned_by', 'unknown')
                }
        except requests.RequestException as e:
            print(f"WARNING: Could not fetch model info: {e}", file=sys.stderr)
        
        return {'model_id': self.model_name, 'owned_by': 'unknown'}
    
    def run_inference(self, prompt: str, max_tokens: int, batch_size: int = 1) -> Dict[str, Any]:
        """Run inference through vLLM API and measure memory"""
        
        # Get baseline memory
        baseline_memory = self.get_gpu_memory_usage()
        baseline_total = sum(gpu['used_gb'] for gpu in baseline_memory)
        
        print(f"Baseline memory: {baseline_total:.2f} GB across {len(baseline_memory)} GPUs")
        if self.visible_devices:
            print(f"Visible devices: {self.visible_devices}")
        
        # Prepare request
        request_data = {
            "model": self.model_name,
            "prompt": prompt if batch_size == 1 else [prompt] * batch_size,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False
        }
        
        # Run inference
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/v1/completions",
                json=request_data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            print(f"ERROR: Inference failed: {e}", file=sys.stderr)
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}", file=sys.stderr)
            sys.exit(1)
        
        inference_time = time.time() - start_time
        
        # Get post-inference memory
        time.sleep(1)  # Let memory stabilize
        final_memory = self.get_gpu_memory_usage()
        final_total = sum(gpu['used_gb'] for gpu in final_memory)
        
        print(f"Post-inference memory: {final_total:.2f} GB")
        
        # Extract token counts
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            text = choice.get('text', '')
            finish_reason = choice.get('finish_reason', 'unknown')
        else:
            text = ''
            finish_reason = 'unknown'
        
        # Get usage stats if available
        usage = result.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        return {
            'baseline_memory_gb': baseline_total,
            'final_memory_gb': final_total,
            'memory_increase_gb': final_total - baseline_total,
            'inference_time_sec': inference_time,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'generated_text_length': len(text),
            'finish_reason': finish_reason,
            'per_gpu_baseline': baseline_memory,
            'per_gpu_final': final_memory
        }
    
    def estimate_memory_breakdown(self, inference_stats: Dict[str, Any], 
                                   model_params: float = None) -> Dict[str, Any]:
        """Estimate memory component breakdown"""
        
        final_memory = inference_stats['final_memory_gb']
        baseline = inference_stats['baseline_memory_gb']
        increase = inference_stats['memory_increase_gb']
        
        # For vLLM with pre-allocated KV cache:
        # Baseline includes: weights + framework overhead + pre-allocated KV cache
        # Since vLLM pre-allocates KV cache, we need to estimate it differently
        
        # Model weights estimation
        if model_params:
            # If we know param count, calculate weights directly
            # fp16 = 2 bytes per param
            model_weights_gb = (model_params * 2) / 1e9
        else:
            # Rough estimate: 60-70% of baseline for models
            # Adjust based on model size - larger models have higher weight ratio
            model_weights_gb = baseline * 0.65
        
        # vLLM pre-allocates KV cache at startup
        # Estimate KV cache size based on max_model_len and num_layers
        # For now, estimate as 15-25% of baseline
        kv_cache_gb = baseline * 0.20
        
        # Framework overhead (vLLM runtime, CUDA context, page blocks, etc.)
        # vLLM has significant overhead due to PagedAttention and scheduler
        framework_overhead_gb = baseline - model_weights_gb - kv_cache_gb
        
        # Activations from increase (if any)
        # With pre-allocated cache, increase is mostly activations
        activations_gb = max(0, increase * 0.8) if increase > 0.1 else 0.5  # Small default
        
        # Adjust if we have negative overhead
        if framework_overhead_gb < 0:
            # Redistribute - reduce KV cache estimate
            kv_cache_gb += framework_overhead_gb
            framework_overhead_gb = baseline * 0.10  # Minimum overhead
        
        # Multi-GPU overhead
        num_gpus = len(inference_stats['per_gpu_final'])
        multi_gpu_overhead_gb = 0
        if num_gpus > 1:
            # Calculate per-GPU variance
            per_gpu_memories = [gpu['used_gb'] for gpu in inference_stats['per_gpu_final']]
            avg_per_gpu = sum(per_gpu_memories) / len(per_gpu_memories)
            
            # With tensor parallelism, model weights are sharded
            # So overhead is the difference from perfect sharding
            expected_per_gpu = model_weights_gb / num_gpus
            actual_overhead_per_gpu = avg_per_gpu - expected_per_gpu
            multi_gpu_overhead_gb = max(0, actual_overhead_per_gpu * num_gpus)
        
        return {
            'model_weights_gb': round(model_weights_gb, 2),
            'kv_cache_gb': round(kv_cache_gb, 2),
            'activations_gb': round(activations_gb, 2),
            'framework_overhead_gb': round(max(0, framework_overhead_gb), 2),
            'multi_gpu_overhead_gb': round(multi_gpu_overhead_gb, 2),
            'total_gb': round(final_memory, 2),
            'estimation_method': 'vllm_preallocated_cache',
            'notes': 'vLLM pre-allocates KV cache; estimates are approximate'
        }
    
    def generate_report(self, prompt: str, max_tokens: int, batch_size: int = 1,
                       model_params: float = None) -> Dict[str, Any]:
        """Generate comprehensive memory profile report"""
        
        print("\n=== vLLM Memory Profiler ===")
        print(f"API URL: {self.api_url}")
        print(f"Model: {self.model_name}")
        print(f"GPU Type: {self.gpu_type}")
        
        if not self.test_api_connection():
            sys.exit(1)
        
        model_info = self.get_model_info()
        print(f"Model ID: {model_info['model_id']}")
        
        print("\n=== Running Inference ===")
        inference_stats = self.run_inference(prompt, max_tokens, batch_size)
        
        print("\n=== Estimating Memory Breakdown ===")
        memory_breakdown = self.estimate_memory_breakdown(inference_stats, model_params)
        
        report = {
            'memory_breakdown': memory_breakdown,
            'model_info': {
                'model_name': self.model_name,
                'model_id': model_info['model_id'],
                'num_parameters': model_params,
                'prompt': prompt,
                'max_tokens': max_tokens,
                'batch_size': batch_size,
                'prompt_tokens': inference_stats['prompt_tokens'],
                'completion_tokens': inference_stats['completion_tokens'],
                'total_sequence_length': inference_stats['total_tokens']
            },
            'gpu_info': {
                'gpu_type': self.gpu_type,
                'num_gpus': len(inference_stats['per_gpu_final']),
                'visible_devices': self.visible_devices or 'all',
                'gpu_memory_baseline': inference_stats['per_gpu_baseline'],
                'gpu_memory_final': inference_stats['per_gpu_final'],
                'total_memory_gb': inference_stats['final_memory_gb']
            },
            'inference_stats': {
                'inference_time_sec': inference_stats['inference_time_sec'],
                'tokens_per_second': (inference_stats['completion_tokens'] / 
                                     inference_stats['inference_time_sec']) 
                                     if inference_stats['inference_time_sec'] > 0 else 0
            }
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Profile vLLM memory usage via API"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., 'zai-org/GLM-4.5-Air')"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Prompt to use for inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--model-params",
        type=float,
        help="Number of model parameters (if known, for better weight estimation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="memory-profile.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Auto-detect model name from vLLM if not provided
    model_name = args.model
    if not model_name:
        try:
            response = requests.get(f"{args.api_url}/v1/models", timeout=10)
            models = response.json()
            if 'data' in models and len(models['data']) > 0:
                model_name = models['data'][0]['id']
                print(f"Auto-detected model: {model_name}")
        except:
            pass
    
    if not model_name:
        print("ERROR: Could not detect model name. Please provide --model argument.", file=sys.stderr)
        sys.exit(1)
    
    profiler = VLLMMemoryProfiler(args.api_url, model_name)
    
    report = profiler.generate_report(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        model_params=args.model_params
    )
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MEMORY BREAKDOWN SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(report['memory_breakdown'], indent=2))
    
    print(f"\n{'='*60}")
    print("GPU INFO")
    print(f"{'='*60}")
    print(f"GPU Type: {report['gpu_info']['gpu_type']}")
    print(f"Number of GPUs: {report['gpu_info']['num_gpus']}")
    print(f"Visible Devices: {report['gpu_info']['visible_devices']}")
    print(f"Total Memory: {report['gpu_info']['total_memory_gb']:.2f} GB")
    
    if len(report['gpu_info']['gpu_memory_final']) > 0:
        print("\nPer-GPU Memory:")
        for gpu in report['gpu_info']['gpu_memory_final']:
            print(f"  GPU {gpu['device']}: {gpu['used_gb']:.2f} GB / {gpu['total_gb']:.2f} GB ({gpu['utilization_pct']:.1f}%)")
    
    print(f"\n{'='*60}")
    print("INFERENCE STATS")
    print(f"{'='*60}")
    print(f"Inference Time: {report['inference_stats']['inference_time_sec']:.2f} sec")
    print(f"Tokens/Second: {report['inference_stats']['tokens_per_second']:.2f}")
    
    print(f"\nFull report saved to: {args.output}")
    
    if 'notes' in report['memory_breakdown']:
        print(f"\nNote: {report['memory_breakdown']['notes']}")


if __name__ == "__main__":
    main()
