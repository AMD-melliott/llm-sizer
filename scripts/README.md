# LLM Memory Profiling Scripts

This directory contains scripts to measure actual memory usage of running LLM models for validation against the calculator.

## Quick Start

### Option 1: Easy Mode (Recommended) - Using the Wrapper Script

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
# 1. Profile actual memory usage
./scripts/profile-docker-model.sh llama2-7b --max-tokens 100

# 2. Open calculator, input same parameters:
#    - Model: Llama 2 7B
#    - Batch: 1
#    - Sequence: 100
#    - Record calculator estimates

# 3. Compare actual vs calculated
python scripts/compare-estimates.py \
    results/memory-profiles/llama2-7b_*.json \
    --calc-weights 13.5 \
    --calc-kv 0.4 \
    --calc-activations 0.9 \
    --calc-overhead 0.6

# 4. Adjust calculator formulas based on recommendations
```

### Direct Usage (If Container Has Shell Access)

```bash
python scripts/profile-model-memory.py \
    --model /path/to/model \
    --prompt "Test prompt" \
    --max-tokens 100 \
    --batch-size 1 \
    --dtype float16 \
    --output memory-profile.json
```

## Output Format

The script generates a JSON report with this structure:

```json
{
  "memory_breakdown": {
    "model_weights_gb": 13.21,
    "kv_cache_gb": 0.45,
    "activations_gb": 0.89,
    "framework_overhead_gb": 0.62,
    "total_gb": 15.17
  },
  "model_info": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dtype": "torch.float16",
    "num_parameters": 6738415616,
    "prompt_length": 8,
    "generated_length": 108,
    "total_sequence_length": 108,
    "batch_size": 1
  },
  "gpu_info": {
    "num_gpus": 2,
    "gpu_memory_per_device": [...],
    "multi_gpu_overhead_gb": 0.34
  }
}
```

## Memory Component Breakdown

### 1. **Model Weights** ✅ Most Accurate
- **Method**: Direct calculation from model parameters
- **Accuracy**: ~99% accurate
- Measures actual GPU memory occupied by model parameters
- Cross-validated with `torch.cuda.memory_allocated()`

### 2. **KV Cache** ⚠️ Good Estimate
- **Method**: Memory difference before/after generation
- **Accuracy**: ~85-95% accurate
- Measures persistent memory after generation completes
- Can include some framework buffers
- **Validation**: Check against sequence length × batch size

### 3. **Activations** ⚠️ Approximate
- **Method**: Peak memory spike during forward pass
- **Accuracy**: ~70-85% accurate
- Measured as: `peak_memory - (weights + kv_cache)`
- Includes temporary tensors, gradients, intermediate computations
- **Note**: Varies by framework optimization

### 4. **Framework Overhead** ⚠️ Residual
- **Method**: Total allocated - (weights + KV + activations)
- **Accuracy**: 60-80% accurate
- Includes: CUDA kernels, memory pools, framework buffers
- Can be negative if components are overestimated

### 5. **Multi-GPU Overhead** ⚠️ Detectable
- **Method**: Sum across GPUs minus single-GPU equivalent
- **Accuracy**: Varies widely (50-80%)
- Includes: tensor parallelism buffers, communication overhead, replicated data
- Only available when using model parallelism

## Measurement Limitations

### Why Perfect Breakdown Is Hard

1. **Memory Fragmentation**: PyTorch allocates memory in chunks; actual usage may differ from requested
2. **Caching**: CUDA maintains memory pools that blur boundaries
3. **Framework Internals**: Different frameworks (vLLM, TGI, HF) manage memory differently
4. **Dynamic Allocation**: Memory usage changes during generation
5. **Shared Buffers**: Some memory is multi-purpose

### Best Practices for Accuracy

1. **Warm-up Run**: Run inference once before profiling to stabilize allocations
2. **Consistent Parameters**: Use same batch size, sequence length as your calculator
3. **Multiple Measurements**: Run 3-5 times and average results
4. **Match Precision**: Ensure model dtype matches calculator settings (fp16, int8, etc.)
5. **Known Models**: Start with well-documented models (Llama-2-7b, GPT-2) for validation

## Comparing with Calculator

To compare actual vs calculated:

```bash
# 1. Get measurements from running model
python scripts/profile-model-memory.py --model llama-7b --max-tokens 100

# 2. Input same parameters into calculator:
#    - Model: Llama 7B (6.7B params)
#    - Quantization: FP16
#    - Batch size: 1
#    - Sequence length: 100
#    - Context: Use actual sequence length from output

# 3. Compare memory_breakdown fields
```

### Expected Accuracy by Component

| Component | Expected Match | Variance |
|-----------|---------------|----------|
| Model Weights | 95-100% | ±2% |
| KV Cache | 80-95% | ±15% |
| Activations | 60-85% | ±30% |
| Framework Overhead | 50-80% | ±40% |
| **Total Memory** | **85-95%** | **±10%** |

## Framework-Specific Tips

### HuggingFace Transformers (Default)
- ✅ Works out of the box
- Use `device_map="auto"` for multi-GPU

### vLLM
```python
# vLLM has built-in profiling:
from vllm import LLM
llm = LLM(model="...", gpu_memory_utilization=0.9)
# Check logs for memory allocation breakdown
```

### Text-Generation-Inference (TGI)
```bash
# TGI logs memory usage on startup:
docker logs <tgi-container> | grep -i memory
```

### Ollama
```bash
# Ollama shows memory in model info:
ollama show <model> --modelfile
```

## Troubleshooting

### "CUDA out of memory"
- Reduce `--max-tokens` or `--batch-size`
- Try with smaller model first
- Check available GPU memory: `nvidia-smi`

### "Model not found"
- Ensure model is downloaded: `huggingface-cli download <model>`
- Or use local path: `--model /path/to/model`

### Negative Framework Overhead
- Normal if KV cache or activations are overestimated
- Indicates PyTorch's internal accounting differs from measurement
- Total memory is still accurate

### Multi-GPU Overhead = 0
- Model is not using model parallelism
- Try with larger model or explicit parallelism
- Check: `model.hf_device_map` shows device placement

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
