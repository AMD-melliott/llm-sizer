# LLM Profiling Guide

Complete guide for profiling LLM memory usage and performance using the enhanced vLLM benchmark-based approach.

## Quick Reference

### Single Model Profiling (Basic)
```bash
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 512 --output-len 512 --batch-size 8
```

### Single Model Profiling (High Confidence - Recommended)
```bash
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 512 --output-len 512 --batch-size 8 \
  --model-params 6.7e9 --num-layers 32 --num-heads 32 --hidden-size 4096 \
  --output results/memory-profiles/llama-7b.json
```

### Batch Profiling (From Config)
```bash
python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/my-model.yaml
```

### Quick Checks
```bash
# View latest profile
jq '.memory.total_gb, .latency_stats.mean_latency_ms' \
    $(ls -t results/memory-profiles/*.json | head -1)

# View all metadata
jq '.' profile.json | less

# Check component breakdown
jq '.memory_breakdown | {weights, kv_cache, activations, overhead, confidence}' profile.json
```

---

## Overview

The profiling system captures real-world memory and performance metrics from running LLM inference containers. It enables validation of calculator accuracy against ground truth data.

### What Gets Profiled

- **Memory**: Baseline, post-warmup, and peak measurements with per-GPU details
- **Components**: Model weights, KV cache, activations, framework overhead
- **Metadata**: Model architecture, quantization settings, dtype configuration, engine version
- **Performance**: Latency and throughput metrics from vLLM bench

### Key Features

✅ Three-point memory capture (baseline → post-warmup → peak)  
✅ Per-GPU memory statistics (sum, max, mean, stddev)  
✅ Per-component dtype tracking (weights, activations, KV cache)  
✅ vLLM engine version detection  
✅ Confidence-tracked component breakdown  
✅ Architecture-aware memory estimation  
✅ Schema versioning (v2.0)  

---

## Single Model Profiling

### Basic Usage

Simplest form - minimal parameters:

```bash
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 512 --output-len 512 \
  --batch-size 8
```

**Output**: JSON file in `results/memory-profiles/`

### High Confidence (Recommended)

Provide architecture parameters for accurate component breakdown:

```bash
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 512 --output-len 512 --batch-size 8 \
  --model-params 6.7e9 \
  --num-layers 32 \
  --num-heads 32 \
  --hidden-size 4096 \
  --output results/memory-profiles/llama-7b-detailed.json
```

**Benefit**: Confidence levels marked as "high" for components

### Multi-GPU

For tensor-parallel models:

```bash
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-70b-hf \
  --input-len 1024 --output-len 1024 --batch-size 4 \
  --tensor-parallel-size 4 \
  --model-params 70e9 \
  --num-layers 80 \
  --hidden-size 8192 \
  --output results/memory-profiles/llama-70b-tp4.json
```

### Quantized Models

With quantization:

```bash
python scripts/profile-vllm-bench-enhanced.py \
  --model TheBloke/Llama-2-13B-AWQ \
  --input-len 512 --output-len 512 --batch-size 8 \
  --quantization awq \
  --dtype float16 \
  --model-params 13e9 \
  --num-layers 40 \
  --hidden-size 5120 \
  --output results/memory-profiles/llama-13b-awq.json
```

### Mixed Precision KV Cache

Different precisions for KV cache:

```bash
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 1024 --output-len 1024 \
  --dtype float16 \
  --kv-cache-dtype fp8 \
  --model-params 6.7e9 \
  --num-layers 32 \
  --hidden-size 4096 \
  --output results/memory-profiles/llama-7b-fp8-kv.json
```

---

## Batch Profiling

Automate profiling across multiple models and configurations.

### Option 1: Configuration File (Recommended)

Create a YAML config file:

```yaml
# Save as: scripts/configs/my-models.yaml
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
    
  - model_id: "meta-llama/Llama-2-70b-hf"
    container_name: "vllm-inference"
    input_lengths: [512, 1024]
    output_lengths: [256, 512]
    batch_sizes: [1, 4, 8]
    dtype: "float16"
    tensor_parallel_size: 4
    model_params: 70e9
    num_layers: 80
    hidden_size: 8192
```

Run the batch profiler:

```bash
# Dry run to verify configuration
python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/my-models.yaml \
  --dry-run

# Actual profiling
python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/my-models.yaml
```

### Option 2: Command Line

Profile multiple configurations without a config file:

```bash
python scripts/batch-profile-bench-enhanced.py \
  --container vllm-inference \
  --model meta-llama/Llama-2-7b-hf \
  --model-params 6.7e9 --num-layers 32 --hidden-size 4096 \
  --input-len 256 512 1024 \
  --output-len 256 512 \
  --batch-size 1 8 16
```

### YAML Configuration Reference

**Core Fields** (required):

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | HuggingFace model identifier |
| `container_name` | string | Docker container name (must be running) |
| `input_lengths` | list | Input sequence lengths to test |
| `output_lengths` | list | Output lengths to test |
| `batch_sizes` | list | Batch sizes to test |

**Optional Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtype` | string | `float16` | Data type (float16, float32, etc.) |
| `tensor_parallel_size` | int | 1 | Number of GPUs for tensor parallelism |
| `quantization` | string | null | Quantization type (awq, gptq, fp8, etc.) |
| `kv_cache_dtype` | string | null | KV cache dtype (separate from weights) |
| `model_params` | float | null | Model parameter count (for accuracy) |
| `num_layers` | int | null | Number of layers |
| `num_heads` | int | null | Number of attention heads |
| `head_dim` | int | null | Dimension per head |
| `hidden_size` | int | null | Hidden layer size |
| `gpu_memory_utilization` | float | 0.9 | GPU memory fraction to use |
| `max_model_len` | int | null | Override max sequence length |

### Common Batch Profiling Patterns

**Test Batch Scaling**:
```yaml
input_lengths: [512]
output_lengths: [512]
batch_sizes: [1, 2, 4, 8, 16, 32]  # Vary batch only
```

**Test Context Length**:
```yaml
input_lengths: [128, 256, 512, 1024, 2048, 4096]  # Vary context
output_lengths: [128]
batch_sizes: [1]
```

**Production Simulation**:
```yaml
input_lengths: [512, 1024, 2048]
output_lengths: [256, 512]
batch_sizes: [8, 16]
```

**Multi-GPU Comparison**:
```yaml
models:
  - model_id: "meta-llama/Llama-3.1-70B-Instruct"
    container_name: "llama-70b-tp1"
    tensor_parallel_size: 1
    batch_sizes: [1, 4]
    # ...
    
  - model_id: "meta-llama/Llama-3.1-70B-Instruct"
    container_name: "llama-70b-tp2"
    tensor_parallel_size: 2
    batch_sizes: [1, 4]
    # ...
    
  - model_id: "meta-llama/Llama-3.1-70B-Instruct"
    container_name: "llama-70b-tp4"
    tensor_parallel_size: 4
    batch_sizes: [1, 4]
    # ...
```

---

## Understanding Profile Output

### Profile JSON Structure

Each profile contains:

```json
{
  "profiler_version": "2.0",
  "schema_version": "2.0",
  "metadata": { /* timestamp, run info */ },
  "model_info": {
    "name": "meta-llama/Llama-2-7b-hf",
    "num_parameters": 6.7e9,
    "num_layers": 32,
    "dtype": "float16"
  },
  "memory_measurements": {
    "baseline": { /* pre-run memory */ },
    "post_warmup": { /* after warmup */ },
    "peak": { /* maximum during run */ },
    "memory_increase_gb": 45.30
  },
  "memory_breakdown": {
    "total_measured_gb": 48.20,
    "model_weights_gb": 29.80,
    "kv_cache_gb": 9.20,
    "activations_gb": 5.50,
    "framework_overhead_gb": 1.25,
    "confidence_levels": {
      "overall": "high",
      "weights": "high",
      "kv_cache": "high",
      "activations": "medium"
    }
  },
  "latency_stats": {
    "mean_latency_ms": 125.4,
    "tokens_per_second": 256.8
  },
  "engine_metadata": {
    "engine_version": "0.5.4",
    "engine_type": "vllm"
  }
}
```

### Inspecting Results

**View all metadata**:
```bash
jq '.' profile.json | less
```

**Check version**:
```bash
jq '.profiler_version, .schema_version' profile.json
```

**Memory summary**:
```bash
jq '{
  baseline: .memory_measurements.baseline.total_gb,
  peak: .memory_measurements.peak.total_gb,
  increase: .memory_measurements.memory_increase_gb
}' profile.json
```

**Component breakdown**:
```bash
jq '.memory_breakdown | {
  weights: .model_weights_gb,
  kv_cache: .kv_cache_gb,
  activations: .activations_gb,
  overhead: .framework_overhead_gb,
  confidence: .confidence_levels.overall
}' profile.json
```

**Per-GPU details**:
```bash
jq '.memory_measurements.peak.per_gpu_details' profile.json
```

**Compare two profiles**:
```bash
jq -s '.[0].memory_breakdown.total_measured_gb as $a | 
       .[1].memory_breakdown.total_measured_gb as $b | 
       {profile1: $a, profile2: $b, diff_gb: ($b - $a), 
        diff_pct: (($b - $a) / $a * 100)}' \
  profile1.json profile2.json
```

---

## Docker Container Setup

### Start vLLM Container

```bash
# AMD GPUs
docker run -d --name vllm-inference --runtime=amd \
  -e AMD_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_TOKEN=${HF_TOKEN} \
  --shm-size=64g \
  -v ./scripts:/scripts \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  rocm/vllm:latest sleep infinity

# NVIDIA GPUs
docker run -d --name vllm-inference --gpus all \
  -e HF_TOKEN=${HF_TOKEN} \
  --shm-size=64g \
  -v ./scripts:/scripts \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:latest sleep infinity
```

### Copy Profiler to Container

```bash
docker cp scripts/profile-vllm-bench-enhanced.py \
  vllm-inference:/tmp/
```

### Run Inside Container

```bash
docker exec vllm-inference python /tmp/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 512 --output-len 512 \
  --model-params 6.7e9 --num-layers 32 --hidden-size 4096 \
  --output /tmp/profile.json
```

### Copy Results Out

```bash
docker cp vllm-inference:/tmp/profile.json \
  results/memory-profiles/
```

---

## Advanced Usage

### Common Model Architectures

**Llama-2-7B**:
```bash
--model-params 6.7e9 \
--num-layers 32 \
--num-heads 32 \
--hidden-size 4096 \
--head-dim 128
```

**Llama-2-13B**:
```bash
--model-params 13e9 \
--num-layers 40 \
--num-heads 40 \
--hidden-size 5120 \
--head-dim 128
```

**Llama-2-70B**:
```bash
--model-params 70e9 \
--num-layers 80 \
--num-heads 64 \
--hidden-size 8192 \
--head-dim 128
```

**Mistral-7B**:
```bash
--model-params 7.2e9 \
--num-layers 32 \
--num-heads 32 \
--hidden-size 4096 \
--head-dim 128
```

### Performance Monitoring

**Watch GPU memory during profiling**:
```bash
watch -n 1 rocm-smi  # AMD
watch -n 1 nvidia-smi  # NVIDIA
```

**Check container logs**:
```bash
docker logs -f vllm-inference
```

**Monitor disk usage**:
```bash
watch -n 5 df -h /path/to/results
```

### Incremental Profiling

Profile in stages instead of all at once:

```bash
# Stage 1: Small models
python scripts/batch-profile-bench-enhanced.py --config small-models.yaml

# Stage 2: Medium models
python scripts/batch-profile-bench-enhanced.py --config medium-models.yaml

# Stage 3: Large models (takes longest)
python scripts/batch-profile-bench-enhanced.py --config large-models.yaml
```

### Automation Workflow

Chain profiling with validation:

```bash
# 1. Profile
python scripts/batch-profile-bench-enhanced.py --config models.yaml

# 2. Validate against calculator
npm run batch-validate

# 3. Analyze results
python scripts/analyze-validation.py \
  results/memory-profiles/batch-bench-results-*.csv

# 4. Review recommendations
cat results/validation-reports/batch-summary-*.txt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Container not found** | `docker start vllm-inference` or restart with `docker run` |
| **vLLM not installed** | Use image with vLLM: `rocm/vllm:latest` or `nvcr.io/nvidia/vllm:latest` |
| **Model not found** | Check HF_TOKEN is set, model ID is correct, wait for download |
| **Out of memory** | Reduce batch_sizes, input/output lengths, or increase tensor_parallel_size |
| **Permission denied** | `chmod +x scripts/profile-vllm-bench-enhanced.py` |
| **Profiler times out** | Check container logs, increase timeout, use smaller model for testing |
| **GPU memory not clearing** | `docker kill` containers, wait 30s, check with `rocm-smi` |
| **Invalid JSON output** | Check container has vLLM properly installed, view logs |

---

## Schema Reference (v2.0)

### Version Compatibility

- **Profiler**: v2.0
- **Schema**: v2.0
- **Supports**: Python 3.8+

### Key Additions Over v1.0

- ✅ Baseline memory persistence (was only logged before)
- ✅ Post-warmup snapshot (detects lazy allocations)
- ✅ Per-GPU statistics (sum, max, mean, stddev)
- ✅ Dtype metadata (separate per component)
- ✅ Engine version detection
- ✅ Confidence-level tracking
- ✅ Architecture-aware estimation
- ✅ Schema versioning

### Checking Profile Version

```bash
jq -e '.profiler_version == "2.0"' profile.json && echo "v2.0" || echo "Not v2.0"
```

---

## File Organization

```
results/
├── memory-profiles/
│   ├── llama-7b-profile_20251103_120000.json
│   ├── llama-70b-profile_20251103_120030.json
│   └── batch-bench-report-20251103_120000.txt
└── validation-reports/
    └── batch-summary-*.txt

scripts/
├── profile-vllm-bench-enhanced.py      # Single model profiler
├── batch-profile-bench-enhanced.py     # Batch orchestrator
├── validate-calculator.ts              # Validation tool
└── configs/
    ├── models-enhanced.yaml            # Config template
    └── my-models.yaml                  # Your config
```

---

## Tips & Best Practices

✓ **Start small**: Test 1-2 configs before full batch  
✓ **Use dry-run**: Always verify config with `--dry-run` first  
✓ **Provide architecture**: Include model params for high-confidence estimates  
✓ **Monitor resources**: Watch GPU memory, disk space during runs  
✓ **Version your configs**: Keep configs in version control  
✓ **Match production**: Use exact production parameters in profiles  
✓ **Record everything**: Keep validation results for trend analysis  

---

## Integration with Calculator Validation

See [CALCULATOR-VALIDATION.md](CALCULATOR-VALIDATION.md) for how to validate profiling results against the LLM Sizer calculator.

## Next Steps

1. ✅ Set up vLLM container
2. ⏭️ Profile a single model to verify functionality
3. ⏭️ Create batch config for your models
4. ⏭️ Run batch profiling campaign
5. ⏭️ Validate against calculator
6. ⏭️ Analyze results and improve calculator

---

**Documentation Files**:
- [CALCULATOR-VALIDATION.md](CALCULATOR-VALIDATION.md) - Validation workflow
- [README.md](README.md) - Project overview
