# LLM Memory Profiling Scripts

This directory contains scripts to measure actual memory usage of running LLM models for validation against the calculator.

## 🚀 Quick Start (Recommended)

### NEW: Native vLLM Benchmark Profiling

For the most accurate profiling, use the new `profile-vllm-bench.py` script that leverages vLLM's native `bench latency` command:

```bash
# Basic profiling with exact parameters matching calculator
python scripts/profile-vllm-bench.py \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 256 \
    --output-len 256 \
    --batch-size 8 \
    --dtype float16 \
    --output profile_results.json

# With quantization
python scripts/profile-vllm-bench.py \
    --model meta-llama/Llama-2-13b-hf \
    --input-len 512 \
    --output-len 512 \
    --quantization awq

# Multi-GPU with tensor parallelism
python scripts/profile-vllm-bench.py \
    --model meta-llama/Llama-2-70b-hf \
    --input-len 1024 \
    --output-len 1024 \
    --tensor-parallel-size 4
```

**Why this is better:**
- ✅ Direct vLLM engine measurement (no API overhead)
- ✅ Exact parameter control matching calculator inputs
- ✅ Standardized warmup and iteration counts
- ✅ More reproducible results
- ✅ Better alignment with vLLM best practices

See `VLLM-PROFILING-RECOMMENDATIONS.md` for detailed analysis and comparison.

## Alternative Methods

### Option 1: Easy Mode - Using the Wrapper Script (API-based)

For quick profiling of already-running vLLM containers:

```bash
# Profile a running container (auto-detects model path)
./scripts/profile-docker-model.sh my-llama-container

# With custom parameters
./scripts/profile-docker-model.sh my-llama-container \
    --max-tokens 200 \
    --batch-size 2 \
    --prompt "Explain quantum computing"
```

### Option 2: Manual Mode - Inside a Model Container

```bash
# Copy script into container
docker cp scripts/profile-model-memory.py <container-name>:/tmp/

# Run profiling
docker exec -it <container-name> python /tmp/profile-model-memory.py \
    --model meta-llama/Llama-2-7b-hf \
    --prompt "Hello, how are you today?" \
    --max-tokens 100 \
    --output /tmp/memory-profile.json

# Copy results out
docker cp <container-name>:/tmp/memory-profile.json ./results/
```

### Complete Workflow: Measure → Calculate → Compare

```bash
# 1. Profile actual memory usage (NEW METHOD)
python scripts/profile-vllm-bench.py \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 256 \
    --output-len 256 \
    --batch-size 8 \
    --dtype float16 \
    --output results/llama2-7b-profile.json

# 2. Open calculator, input same parameters:
#    - Model: Llama 2 7B
#    - Batch: 8
#    - Input Sequence: 256
#    - Output Sequence: 256
#    - Data Type: FP16
#    - Record calculator estimates

# 3. Compare actual vs calculated
python scripts/compare-estimates.py \
    results/llama2-7b-profile.json \
    --calc-total 15.2 \
    --show-breakdown
```

## Output Format

The profiling scripts generate a JSON report with this structure:

```json
{
  "model_info": {
    "name": "meta-llama/Llama-2-7b-hf",
    "num_parameters": 6738415616,
    "dtype": "float16"
  },
  "benchmark_parameters": {
    "input_len": 256,
    "output_len": 256,
    "batch_size": 8,
    "tensor_parallel_size": 1
  },
  "latency_stats": {
    "avg_latency": 0.45,
    "p50_latency": 0.44,
    "p90_latency": 0.48,
    "p99_latency": 0.52
  },
  "memory_breakdown": {
    "total_measured_gb": 14.2,
    "model_weights_gb": 13.5,
    "kv_cache_gb": 0.4,
    "activations_gb": 0.2,
    "framework_overhead_gb": 0.1
  },
  "gpu_info": {
    "gpu_type": "cuda",
    "num_gpus": 1
  }
}
```

## Memory Component Breakdown

### 1. **Model Weights** ✅ Most Accurate
- **Method**: Direct calculation from model parameters
- **Accuracy**: ~99% accurate
- Measures actual GPU memory occupied by model parameters
- Formula: `num_parameters × bytes_per_param`

### 2. **KV Cache** ⚠️ Good Estimate
- **Method**: Calculated based on sequence length and batch size
- **Accuracy**: ~85-95% accurate
- vLLM pre-allocates KV cache based on `max_model_len`
- Formula: `2 × num_layers × hidden_size × seq_len × batch_size × bytes_per_element`

### 3. **Activations** ⚠️ Approximate
- **Method**: Peak memory during forward pass
- **Accuracy**: ~70-85% accurate
- Includes temporary tensors, gradients, intermediate computations
- Typically 10-15% of total memory

### 4. **Framework Overhead** ⚠️ Residual
- **Method**: Total allocated - (weights + KV + activations)
- **Accuracy**: 60-80% accurate
- Includes: CUDA kernels, memory pools, framework buffers
- PagedAttention structures in vLLM

## Measurement Approach

### vLLM Bench Method (Recommended)

Uses vLLM's native `bench latency` command:

```bash
vllm bench latency \
    --model <model_path> \
    --input-len <N> \
    --output-len <N> \
    --batch-size <N> \
    --dtype <type> \
    --num-iters-warmup 10 \
    --num-iters 30 \
    --output-json results.json
```

**Advantages:**
- No API overhead
- Exact parameter control
- Standardized measurements
- Built-in warmup handling
- Reproducible results

**Limitations:**
- Requires vLLM installation
- Memory breakdown is estimated (due to PagedAttention)
- Must fit model in memory

### API-Based Method (Legacy)

Queries running vLLM API server and measures memory:

**Advantages:**
- Works with any running vLLM container
- No model reloading needed
- Quick for one-off tests

**Limitations:**
- Includes API server overhead
- Less precise parameter control
- Memory measurements less consistent

## Best Practices for Accuracy

1. **Use vLLM Bench Method**: More accurate than API-based profiling
2. **Specify Exact Parameters**: Match calculator inputs precisely
3. **Include Model Info**: Provide `--model-params`, `--num-layers`, `--hidden-size` for better estimates
4. **Warm-up Runs**: Default 10 warmup iterations help stabilize measurements
5. **Multiple Measurements**: 30 iterations provides statistical reliability
6. **Match Precision**: Ensure dtype matches calculator settings (fp16, int8, etc.)
7. **Known Models**: Start with well-documented models (Llama-2, GPT-2) for validation

## Comparing with Calculator

To validate calculator accuracy:

```bash
# 1. Profile with vLLM bench
python scripts/profile-vllm-bench.py \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 512 \
    --output-len 512 \
    --batch-size 4 \
    --dtype float16 \
    --model-params 6.7e9 \
    --output measured.json

# 2. Input same parameters into calculator:
#    - Model: Llama 2 7B (6.7B params)
#    - Quantization: FP16
#    - Batch size: 4
#    - Input length: 512
#    - Output length: 512

# 3. Compare results
# The total_measured_gb should match calculator's total within ±10%
```

### Expected Accuracy by Component

| Component | Expected Match | Variance |
|-----------|---------------|----------|
| Model Weights | 95-100% | ±2% |
| KV Cache | 80-95% | ±15% |
| Activations | 60-85% | ±30% |
| Framework Overhead | 50-80% | ±40% |
| **Total Memory** | **85-95%** | **±10%** |

## vLLM-Specific Features

### vLLM Support ✨

Both profilers support vLLM's unique features:

- **PagedAttention**: Pre-allocated KV cache management
- **Tensor Parallelism**: Multi-GPU support via `--tensor-parallel-size`
- **AMD ROCm GPUs**: MI300X, MI250X, etc.
- **NVIDIA GPUs**: A100, H100, etc.
- **Quantization**: AWQ, GPTQ, FP8, INT4/INT8

### Understanding vLLM Memory

| Component | Description | Typical % |
|-----------|-------------|-----------|
| **Model Weights** | Sharded across GPUs with tensor parallelism | 60-70% |
| **KV Cache** | Pre-allocated based on max_model_len | 15-25% |
| **Framework Overhead** | PagedAttention pools, scheduler, buffers | 10-15% |
| **Activations** | Temporary computation memory | 5-10% |

**Note**: Memory distribution varies significantly based on:
- Sequence length (`max_model_len`)
- Batch size
- Model architecture
- Tensor parallel size

### Reducing vLLM Memory Usage

If memory usage is too high:

1. **Use vLLM Bench Profiler**: More accurate baseline measurement
2. **Reduce sequence length**: Shorter `input-len` + `output-len` saves memory
3. **Lower batch size**: Fewer concurrent requests
4. **Enable quantization**: `--quantization awq` or `--quantization fp8`
5. **Adjust tensor parallelism**: Balance between speed and memory

## Troubleshooting

### vLLM Bench Method

**"ERROR: vllm command not found"**
- Install vLLM: `pip install vllm`
- Or run inside vLLM container: `docker exec vllm-container python /path/to/script.py`

**"CUDA out of memory"**
- Reduce `--input-len` or `--output-len`
- Reduce `--batch-size`
- Use `--quantization` for smaller memory footprint
- Increase `--tensor-parallel-size` to distribute across GPUs

**Memory breakdown seems off**
- Provide `--model-params` for accurate weight calculation
- Provide `--num-layers` and `--hidden-size` for KV cache accuracy
- Remember: breakdown is estimated; total is accurate

### API-Based Method

**"ERROR: Cannot connect to vLLM API"**
- Check if container is running: `docker ps`
- Verify port mapping: `docker port <container>`
- Test API directly: `curl http://localhost:8000/health`

**"WARNING: Could not detect GPU type"**
- Install rocm-smi: `apt install rocm-smi` (for AMD)
- Install nvidia-smi: (usually pre-installed with CUDA)

## Advanced: Per-Layer Profiling

For even more detailed analysis, use PyTorch's memory profiler:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA],
             profile_memory=True, record_shapes=True) as prof:
    output = model.generate(...)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
```

This shows per-operation memory usage but requires code changes.

## See Also

- `VLLM-PROFILING-RECOMMENDATIONS.md` - Detailed analysis and methodology
- `profile-vllm-bench.py` - New native vLLM benchmark profiler (recommended)
- `profile-vllm-model.py` - API-based vLLM profiler (legacy)
- `profile-docker-model.sh` - Wrapper script for quick profiling
- `compare-estimates.py` - Compare measured vs calculated memory

---

## Summary: Which Tool to Use?

| Use Case | Recommended Tool | Why |
|----------|-----------------|-----|
| **Calculator Validation** | `profile-vllm-bench.py` | Most accurate, exact parameters |
| **Quick Check** | `profile-docker-model.sh` | Fastest, works with running containers |
| **Production Profiling** | `profile-vllm-bench.py` | Reproducible, standardized |
| **Development/Debug** | `profile-vllm-model.py` | Flexible, API-based |
| **Batch Testing** | `profile-vllm-bench.py` | Scriptable, consistent |

**Default recommendation**: Start with `profile-vllm-bench.py` for best accuracy.
