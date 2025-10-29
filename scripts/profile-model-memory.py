#!/usr/bin/env python3
"""
LLM Memory Profiler - Extract detailed memory breakdown for comparison with calculator

This script runs inside a container with a loaded LLM model and extracts:
- Base Model Weights
- KV Cache
- Activations
- Framework Overhead
- Multi-GPU Overhead (if applicable)

Usage:
    python profile-model-memory.py --model <model_path> --prompt "Hello" --max-tokens 100

Requirements:
    - torch
    - transformers (for HuggingFace models)
    - Optional: vllm, sglang (detected automatically)
"""

import argparse
import json
import sys
from typing import Dict, Any
import gc

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not found. This script requires torch.", file=sys.stderr)
    sys.exit(1)


class MemoryProfiler:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.measurements: Dict[str, Any] = {}

    def reset_memory_stats(self):
        """Reset GPU memory statistics"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_memory_allocated_gb(self) -> float:
        """Get current allocated memory in GB"""
        return torch.cuda.memory_allocated() / 1e9

    def get_peak_memory_gb(self) -> float:
        """Get peak allocated memory in GB"""
        return torch.cuda.max_memory_allocated() / 1e9

    def measure_model_weights(self, model) -> float:
        """Measure actual model weights in GPU memory"""
        self.reset_memory_stats()

        # Model should already be loaded, measure its footprint
        model_memory = 0
        for param in model.parameters():
            model_memory += param.nelement() * param.element_size()

        model_memory_gb = model_memory / 1e9
        self.measurements['model_weights_gb'] = model_memory_gb

        # Cross-check with actual GPU allocation
        actual_allocated = self.get_memory_allocated_gb()
        self.measurements['model_weights_actual_gb'] = actual_allocated

        return model_memory_gb

    def measure_kv_cache_and_inference(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """Measure KV cache and activation memory during inference"""

        # Prepare inputs
        inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Measure baseline (just model + inputs)
        self.reset_memory_stats()
        baseline_memory = self.get_memory_allocated_gb()

        print(f"Baseline memory (model + inputs): {baseline_memory:.2f} GB")

        # Run inference with cache
        with torch.no_grad():
            # First forward pass - measures activation memory
            self.reset_memory_stats()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=tokenizer.eos_token_id
            )

            peak_with_cache = self.get_peak_memory_gb()
            final_memory = self.get_memory_allocated_gb()

        print(f"Peak memory during generation: {peak_with_cache:.2f} GB")
        print(f"Final memory (with KV cache): {final_memory:.2f} GB")

        # Estimate KV cache size
        # The KV cache persists after generation
        kv_cache_estimate = final_memory - baseline_memory

        # Activation memory is the peak spike during computation
        activation_estimate = peak_with_cache - final_memory

        self.measurements['kv_cache_gb'] = max(0, kv_cache_estimate)
        self.measurements['activations_gb'] = max(0, activation_estimate)
        self.measurements['peak_memory_gb'] = peak_with_cache

        # Get sequence info for validation
        generated_length = outputs.sequences.shape[1]
        prompt_length = inputs['input_ids'].shape[1]

        return {
            'kv_cache_gb': self.measurements['kv_cache_gb'],
            'activations_gb': self.measurements['activations_gb'],
            'prompt_length': prompt_length,
            'generated_length': generated_length,
            'total_sequence_length': generated_length,
            'batch_size': batch_size
        }

    def measure_framework_overhead(self) -> float:
        """Calculate framework overhead as total - (weights + kv + activations)"""
        total_allocated = self.get_memory_allocated_gb()

        accounted_memory = (
            self.measurements.get('model_weights_actual_gb', 0) +
            self.measurements.get('kv_cache_gb', 0) +
            self.measurements.get('activations_gb', 0)
        )

        overhead = total_allocated - accounted_memory
        self.measurements['framework_overhead_gb'] = max(0, overhead)

        return self.measurements['framework_overhead_gb']

    def get_multi_gpu_info(self) -> Dict[str, Any]:
        """Get multi-GPU information if available"""
        if not torch.cuda.is_available():
            return {}

        num_gpus = torch.cuda.device_count()
        multi_gpu_info = {
            'num_gpus': num_gpus,
            'gpu_memory_per_device': []
        }

        if num_gpus > 1:
            for i in range(num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                multi_gpu_info['gpu_memory_per_device'].append({
                    'device': i,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved
                })

            # Calculate multi-GPU overhead (difference in total vs single GPU equivalent)
            total_allocated = sum(d['allocated_gb'] for d in multi_gpu_info['gpu_memory_per_device'])
            # Overhead is total memory across GPUs minus what would be needed on single GPU
            single_gpu_equivalent = self.measurements.get('peak_memory_gb', 0)
            multi_gpu_overhead = total_allocated - single_gpu_equivalent

            multi_gpu_info['multi_gpu_overhead_gb'] = max(0, multi_gpu_overhead)

        return multi_gpu_info

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report"""

        # Calculate total
        total_memory = (
            self.measurements.get('model_weights_actual_gb', 0) +
            self.measurements.get('kv_cache_gb', 0) +
            self.measurements.get('activations_gb', 0) +
            self.measurements.get('framework_overhead_gb', 0)
        )

        report = {
            'memory_breakdown': {
                'model_weights_gb': round(self.measurements.get('model_weights_actual_gb', 0), 2),
                'kv_cache_gb': round(self.measurements.get('kv_cache_gb', 0), 2),
                'activations_gb': round(self.measurements.get('activations_gb', 0), 2),
                'framework_overhead_gb': round(self.measurements.get('framework_overhead_gb', 0), 2),
                'total_gb': round(total_memory, 2)
            },
            'gpu_info': self.get_multi_gpu_info(),
            'all_measurements': self.measurements
        }

        return report


def profile_huggingface_model(
    model_name_or_path: str,
    prompt: str,
    max_tokens: int,
    batch_size: int = 1,
    dtype: str = "auto"
) -> Dict[str, Any]:
    """Profile a HuggingFace Transformers model"""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers library not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {model_name_or_path}")
    print(f"Using dtype: {dtype}")

    profiler = MemoryProfiler()
    profiler.reset_memory_stats()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype if dtype != "auto" else "auto",
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Step 1: Measuring Model Weights ===")
    model_weights = profiler.measure_model_weights(model)
    print(f"Model weights: {model_weights:.2f} GB")

    print("\n=== Step 2: Measuring KV Cache and Activations ===")
    inference_stats = profiler.measure_kv_cache_and_inference(
        model, tokenizer, prompt, max_tokens, batch_size
    )
    print(f"KV cache: {inference_stats['kv_cache_gb']:.2f} GB")
    print(f"Activations: {inference_stats['activations_gb']:.2f} GB")
    print(f"Sequence length: {inference_stats['total_sequence_length']} tokens")

    print("\n=== Step 3: Measuring Framework Overhead ===")
    overhead = profiler.measure_framework_overhead()
    print(f"Framework overhead: {overhead:.2f} GB")

    print("\n=== Generating Report ===")
    report = profiler.generate_report()

    # Add model info
    report['model_info'] = {
        'model_name': model_name_or_path,
        'dtype': str(model.dtype),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'prompt': prompt,
        'max_tokens': max_tokens,
        'batch_size': batch_size,
        **inference_stats
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Profile LLM memory usage with detailed breakdown"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (HuggingFace format)"
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
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="memory-profile.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Run profiling
    report = profile_huggingface_model(
        model_name_or_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        dtype=args.dtype
    )

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, indent=2, fp=f)

    print(f"\n{'='*60}")
    print("MEMORY BREAKDOWN SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(report['memory_breakdown'], indent=2))

    if report['gpu_info'].get('num_gpus', 1) > 1:
        print(f"\n{'='*60}")
        print("MULTI-GPU INFO")
        print(f"{'='*60}")
        print(json.dumps(report['gpu_info'], indent=2))

    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
