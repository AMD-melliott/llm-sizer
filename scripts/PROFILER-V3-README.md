# vLLM Profiler v3.0 - Enhanced Internal Metrics Capture

## Overview

Version 3.0 of the vLLM profiler addresses critical gaps in memory measurement by capturing vLLM's internal memory allocation decisions and providing comprehensive GPU metrics.

## What's New in v3.0

### 🔴 Critical Enhancements

1. **vLLM Internal Metrics Capture**
   - Parses vLLM engine logs to extract the **source of truth** for memory allocation
   - Captures KV cache block statistics (number of blocks, block size, total KV cache)
   - Identifies attention backend (FlashAttention-2, xFormers, etc.)
   - Records vLLM's internal memory breakdown (weights, KV cache, activations)

2. **Real-Time Memory Monitoring**
   - Background thread monitors GPU memory every 100ms during execution
   - Captures full memory timeline (not just 3 snapshots)
   - Solves critical timing issue where vLLM process cleanup happens before measurement
   - Provides min, max, mean, median, P95 memory statistics

3. **Extended AMD GPU Metrics**
   - Upgraded from legacy `rocm-smi` to modern `amd-smi monitor`
   - Single call captures: memory, power, temperature, utilization, clocks
   - Process tracking - see which processes use GPU memory
   - Enables correlation analysis (memory vs power vs temperature)

### 🟢 Additional Improvements

4. **Multi-Phase Memory Capture**
   - Phase 0: System baseline (pre-vLLM)
   - Phase 1: Real-time monitoring starts
   - Phase 2: vLLM benchmark execution
   - Phase 3: Monitoring stops, statistics computed
   - Phase 4: vLLM log parsing
   - Phase 5: Extended GPU metrics (AMD only)
   - Phase 6: Final snapshot

5. **Enhanced Output Schema**
   - `vllm_internal_metrics`: vLLM's reported memory allocation
   - `memory_timeline`: Full timeline with extended metrics
   - `detailed_gpu_metrics`: Comprehensive AMD GPU data
   - `raw_logs`: vLLM stdout/stderr for future analysis

## Root Cause Analysis

### Problem (v2.1 and earlier)
- Only captured **external** GPU memory snapshots via `amd-smi`/`nvidia-smi`
- Missed vLLM's **internal** memory allocation decisions
- No visibility into KV cache block management
- Memory captured AFTER process cleanup (timing issue)
- Result: 0-14GB overhead discrepancies, failed validations

### Solution (v3.0)
- Captures vLLM's **internal view** of memory allocation
- Parses engine logs for KV cache block statistics
- Real-time monitoring captures memory DURING execution
- Comprehensive AMD GPU metrics for correlation analysis
- Result: High-confidence memory breakdown with vLLM's actual allocations

## Usage

### Basic Usage

```bash
# Auto-load model architecture from models.json
python scripts/profile-vllm-bench-v3.py \
    --hf-model-id meta-llama/Llama-3.2-1B-Instruct \
    --input-len 256 \
    --output-len 256 \
    --batch-size 8 \
    --output results.json
```

### Manual Architecture Specification

```bash
# Specify all parameters manually
python scripts/profile-vllm-bench-v3.py \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 512 \
    --output-len 512 \
    --batch-size 4 \
    --model-params 7e9 \
    --num-layers 32 \
    --num-heads 32 \
    --hidden-size 4096 \
    --output results.json
```

### Multi-GPU with Tensor Parallelism

```bash
python scripts/profile-vllm-bench-v3.py \
    --hf-model-id meta-llama/Llama-2-70b-hf \
    --input-len 1024 \
    --output-len 1024 \
    --batch-size 16 \
    --tensor-parallel-size 4 \
    --output results_70b_tp4.json
```

### Advanced Options

```bash
# Different KV cache dtype, custom quantization
python scripts/profile-vllm-bench-v3.py \
    --hf-model-id meta-llama/Llama-3.2-3B-Instruct \
    --input-len 1024 \
    --output-len 1024 \
    --batch-size 8 \
    --dtype float16 \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --output results_fp8_kv.json
```

## Output Schema (v3.0)

### vLLM Internal Metrics (NEW)

```json
{
  "vllm_internal_metrics": {
    "num_gpu_blocks": 7326,
    "num_cpu_blocks": 2048,
    "block_size": 16,
    "total_gpu_kv_cache_tokens": 117216,
    "total_gpu_kv_cache_gb": 7.28,
    "attention_backend": "FlashAttention-2",
    "model_weights_gb": 2.47,
    "activations_gb": 0.82,
    "total_allocated_gb": 10.57,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192,
    "cuda_graphs_enabled": true
  }
}
```

**Why This Matters:**
- `total_gpu_kv_cache_gb`: vLLM's actual KV cache allocation (source of truth!)
- `model_weights_gb`: vLLM's reported weight memory (not estimated)
- `num_gpu_blocks` × `block_size`: Actual KV cache capacity in tokens
- `attention_backend`: Affects memory footprint (FlashAttention vs xFormers)

### Memory Timeline (NEW)

```json
{
  "memory_timeline": {
    "min_gb": 18.42,
    "max_gb": 19.87,
    "mean_gb": 19.12,
    "median_gb": 19.15,
    "p95_gb": 19.65,
    "num_samples": 324,
    "timeline": [
      {"timestamp": 1699123456.123, "memory_gb": 18.45, "power_w": 285, "temp_c": 68},
      ...
    ],
    "extended": {
      "power": {"min_w": 250, "max_w": 320, "mean_w": 285},
      "temperature": {"min_c": 60, "max_c": 75, "mean_c": 68}
    }
  }
}
```

**Why This Matters:**
- Real-time monitoring captures peak memory during execution
- Extended metrics enable correlation analysis (memory vs power vs temp)
- Timeline reveals memory growth patterns during warmup/inference

### Detailed AMD GPU Metrics (NEW)

```json
{
  "detailed_gpu_metrics": {
    "tool": "amd-smi",
    "timestamp": 1699123456.789,
    "gpus": [
      {
        "gpu_id": 0,
        "vram_used_mb": 19712,
        "vram_free_mb": 176880,
        "vram_total_mb": 196592,
        "vram_percent": 10.0,
        "power_watts": 285,
        "hotspot_temp_c": 68,
        "memory_temp_c": 52,
        "gfx_utilization_pct": 87,
        "mem_utilization_pct": 42,
        "clock_gfx_mhz": 2100,
        "clock_mem_mhz": 1600,
        "processes": [
          {
            "process_info": "python (PID: 12345)",
            "memory_usage": {"value": 19200, "unit": "MB"}
          }
        ]
      }
    ]
  }
}
```

**Why This Matters:**
- Single call gets comprehensive GPU state
- Process tracking verifies vLLM is only GPU user
- Thermal/power data helps identify throttling issues
- Utilization shows if memory-bound vs compute-bound

### Memory Breakdown (Enhanced)

```json
{
  "memory_breakdown": {
    "total_measured_gb": 19.32,
    "baseline_gb": 2.15,
    "net_increase_gb": 17.17,
    "model_weights_gb": 2.47,
    "kv_cache_gb": 7.28,
    "activations_gb": 0.82,
    "framework_overhead_gb": 6.60,
    "overhead_percentage": 34.2,
    "estimation_method": "vllm_bench_enhanced_v3",
    "confidence_levels": {
      "overall": "high_from_vllm_logs",
      "weights": "high_from_vllm_logs",
      "kv_cache": "high_from_vllm_logs",
      "activations": "high_from_vllm_logs",
      "overhead": "calculated_residual"
    },
    "vllm_reported_values_used": {
      "weights": true,
      "kv_cache": true,
      "activations": true
    }
  }
}
```

**Confidence Levels:**
- `high_from_vllm_logs`: Value extracted from vLLM engine logs (highest confidence)
- `high_calculated`: Calculated using accurate architecture parameters
- `low_estimated`: Rough estimation (used when data unavailable)

## Interpreting Results

### Example Output Interpretation

```
🔍 vLLM Internal Metrics (Source of Truth):
  GPU Blocks: 7326
  Block Size: 16 tokens
  KV Cache (vLLM reported): 7.28 GB
  Model Weights (vLLM reported): 2.47 GB
  Activations (vLLM reported): 0.82 GB
  Attention Backend: FlashAttention-2

📈 Memory Timeline (Real-Time Monitoring):
  Samples Captured: 324
  Min: 18.42 GB
  Mean: 19.12 GB
  Peak: 19.87 GB
  P95: 19.65 GB

💾 Memory Breakdown:
  Total Measured: 19.32 GB
  ├─ Model Weights: 2.47 GB (12.8%)
  ├─ KV Cache: 7.28 GB (37.7%)
  ├─ Activations: 0.82 GB (4.2%)
  └─ Framework Overhead: 8.75 GB (45.3%)

  Estimation Confidence: HIGH_FROM_VLLM_LOGS
    Weights: high_from_vllm_logs (vLLM reported)
    KV Cache: high_from_vllm_logs (vLLM reported)
    Activations: high_from_vllm_logs (vLLM reported)
```

### What This Tells You

1. **KV Cache is accurately measured**: 7.28 GB from vLLM's block manager
   - 7326 blocks × 16 tokens/block = 117,216 tokens capacity
   - Validates calculator's KV cache formula

2. **Framework overhead is 8.75 GB (45%)**
   - Includes: vLLM engine, CUDA graphs, PagedAttention structures, ROCm driver
   - High confidence in other components means this residual is accurate

3. **Memory timeline shows stability**
   - Min-to-peak range: 1.45 GB (7% variation)
   - Indicates stable inference memory footprint
   - No unexpected memory spikes

4. **Extended metrics available** (AMD GPUs)
   - Power consumption, temperature, utilization tracked
   - Enables correlation analysis for performance optimization

## Comparison: v2.1 vs v3.0

### v2.1 Output (Before)
```
Memory Breakdown (Estimated):
  Total: 19.38 GB
  Weights: 2.47 GB  ← Calculated from params
  KV Cache: 0.13 GB  ← WAY OFF! (estimated)
  Activations: 2.33 GB  ← Estimated
  Overhead: 14.45 GB  ← Mystery overhead!

Confidence: LOW_ESTIMATED
```

**Problem**: KV cache drastically underestimated, leading to massive "overhead"

### v3.0 Output (After)
```
Memory Breakdown (vLLM Internal):
  Total: 19.32 GB
  Weights: 2.47 GB  ← From vLLM logs
  KV Cache: 7.28 GB  ← From vLLM logs! ✅
  Activations: 0.82 GB  ← From vLLM logs
  Overhead: 8.75 GB  ← Realistic overhead

Confidence: HIGH_FROM_VLLM_LOGS
```

**Solution**: All major components from vLLM's internal reporting

## Validation Workflow

### Step 1: Run Enhanced Profiler

```bash
python scripts/profile-vllm-bench-v3.py \
    --hf-model-id meta-llama/Llama-3.2-1B-Instruct \
    --input-len 256 \
    --output-len 256 \
    --batch-size 8 \
    --output results.json
```

### Step 2: Analyze vLLM Internal Metrics

```bash
# Extract KV cache from vLLM logs
jq '.vllm_internal_metrics.total_gpu_kv_cache_gb' results.json

# Compare with calculator estimate
# (Run calculator validation script)
```

### Step 3: Validate Calculator Formulas

Use the captured data to validate/improve calculator formulas:

```python
# From vLLM logs
vllm_kv_cache = 7.28  # GB
num_blocks = 7326
block_size = 16

# Calculate what calculator predicts
calculator_kv_cache = calculate_kv_cache(
    num_layers=32,
    hidden_size=2048,
    seq_len=512,
    batch_size=8
)

# Compare
error_pct = abs(calculator_kv_cache - vllm_kv_cache) / vllm_kv_cache * 100
```

### Step 4: Adjust Overhead Formulas

With accurate KV cache and weights, overhead becomes:

```python
overhead = total_measured - (weights + kv_cache + activations)
```

This is the **true** framework overhead, not inflated by KV cache underestimation.

## Troubleshooting

### Issue: amd-smi not found

**Symptom**: Warning about amd-smi failing, falls back to rocm-smi

**Solution**:
```bash
# Check ROCm version
rocm-smi --version

# Install/update amd-smi (ROCm 5.3+)
# amd-smi is included in modern ROCm installations
```

### Issue: No vLLM internal metrics captured

**Symptom**: `vllm_internal_metrics` fields are all `null`

**Possible Causes**:
1. vLLM version too old (< 0.4.0)
2. vLLM changed log format
3. Logs redirected elsewhere

**Solution**:
```bash
# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Manually inspect logs
python scripts/profile-vllm-bench-v3.py ... 2>&1 | tee vllm_logs.txt
grep -i "GPU blocks" vllm_logs.txt
```

### Issue: Memory timeline shows no samples

**Symptom**: `memory_timeline` is `null` or `num_samples: 0`

**Possible Causes**:
1. Benchmark completed too quickly (< 1 second)
2. GPU monitoring commands failing
3. Threading issue

**Solution**:
```bash
# Increase benchmark duration
python scripts/profile-vllm-bench-v3.py \
    --num-iters-warmup 20 \
    --num-iters 100 \
    ...

# Test GPU monitoring manually
amd-smi monitor -m --json
```

## Next Steps

### Priority 1: Validate Calculator Formulas
Use v3.0 profiling data to validate and improve calculator formulas:
- Compare calculator KV cache vs vLLM's actual allocation
- Analyze overhead patterns across different models
- Update overhead formulas based on real measurements

### Priority 2: Correlation Analysis
Analyze relationships between metrics:
- Memory vs Power consumption
- Memory vs Temperature
- KV cache utilization vs batch size
- Overhead vs model size

### Priority 3: Model-Specific Profiles
Build a database of overhead profiles for different model families:
- Llama overhead patterns
- Mistral overhead patterns
- DeepSeek overhead patterns
- Impact of quantization on overhead

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Memory Profiling Guide](https://docs.vllm.ai/en/latest/contributing/profiling.html)
- [AMD SMI Tool Documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [vLLM Block Manager Source](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager_v2.py)

## Version History

- **v3.0** (2025-01): Enhanced internal metrics capture
  - vLLM log parsing for internal allocation
  - Real-time memory monitoring
  - Extended AMD GPU metrics
  - Multi-phase memory capture

- **v2.1** (2024-12): Model database integration
  - Auto-load architecture from models.json
  - Single source of truth for model data

- **v2.0** (2024-11): Enhanced memory attribution
  - Multi-snapshot memory capture
  - Dtype-aware calculations
  - Calculator validation integration

- **v1.0** (2024-10): Initial release
  - Basic vLLM bench wrapper
  - Simple memory measurement
