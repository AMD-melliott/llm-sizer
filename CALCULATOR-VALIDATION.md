# Calculator Validation Guide

Complete guide for validating the LLM Sizer calculator against real-world memory measurements from production deployments.

## Quick Reference

### Single Model Validation (3 Steps)

#### Step 1: View Profile & Get Inputs
```bash
python scripts/validate-profile.py results/your-profile.json
```

#### Step 2: Input Values into Calculator
- Open LLM Sizer in browser
- Copy parameter values from validation report
- Record calculator's total memory estimate

#### Step 3: Validate & Record
```bash
python scripts/validate-profile.py results/your-profile.json \
    --calculator-total <GB> \
    --record validation-results.csv
```

### Batch Validation
```bash
npm run batch-validate
```

### View Summary
```bash
cat results/validation-reports/batch-summary-*.txt
```

---

## Overview

The validation system compares LLM Sizer calculator estimates against real-world profiling data to:
- Identify systematic calculation errors
- Measure component-level accuracy
- Guide calculator improvements
- Track accuracy improvements over time

### Workflow

```
1. Profile Models
   ↓
2. Run Calculator
   ↓
3. Compare Results
   ↓
4. Analyze Patterns
   ↓
5. Update Calculator
```

---

## Quick Start Validation

### 1. Profile a Model

```bash
# Profile with enhanced profiler (includes architecture details)
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 512 --output-len 512 --batch-size 8 \
  --model-params 6.7e9 --num-layers 32 --num-heads 32 --hidden-size 4096 \
  --output results/memory-profiles/llama-7b-profile.json
```

### 2. View Profile & Get Calculator Inputs

```bash
python scripts/validate-profile.py results/memory-profiles/llama-7b-profile.json
```

**Output shows:**
- Model configuration
- GPU configuration
- Profiled memory measurements (ground truth)
- **Exact inputs to copy into calculator**
- Target memory value to match

Example output:
```
═══════════════════════════════════════════════════════════════════════════════
LLM SIZER CALCULATOR VALIDATION
═══════════════════════════════════════════════════════════════════════════════

PROFILE: results/memory-profiles/llama-7b-profile.json

MODEL CONFIGURATION
───────────────────────────────────────────────────────────────────────────────
  Model: meta-llama/Llama-2-7b-hf
  Parameters: 6.7B
  Layers: 32, Heads: 32, Hidden: 4096

GPU CONFIGURATION
───────────────────────────────────────────────────────────────────────────────
  Type: AMD Instinct MI300X
  Count: 1
  VRAM: 206 GB

INFERENCE CONFIGURATION
───────────────────────────────────────────────────────────────────────────────
  Batch Size: 8
  Input Tokens: 512
  Output Tokens: 512
  Total Sequence: 1024
  Data Type: FP16
  Quantization: None

PROFILED MEMORY (GROUND TRUTH)
───────────────────────────────────────────────────────────────────────────────
  Total: 48.20 GB
  Model Weights: 13.40 GB
  KV Cache: 8.50 GB
  Activations: 5.60 GB
  Framework Overhead: 20.70 GB

► COPY THESE VALUES INTO CALCULATOR FORM, THEN RETURN HERE
```

### 3. Input Values into Calculator

1. Open LLM Sizer in browser
2. Set Model: `meta-llama/Llama-2-7b-hf` or `7B Custom`
3. Parameters: `6.7B`
4. GPU: `AMD Instinct MI300X`, `1 GPU`
5. Inference:
   - Batch Size: `8`
   - Input Tokens: `512`
   - Output Tokens: `512`
   - Data Type: `FP16`
6. Note the calculated total memory

### 4. Validate & Record

```bash
python scripts/validate-profile.py results/memory-profiles/llama-7b-profile.json \
    --calculator-total 45.30 \
    --record validation-results.csv
```

**Output shows:**
```
VALIDATION RESULT
───────────────────────────────────────────────────────────────────────────────
  Profiled Memory:   48.20 GB
  Calculated Memory: 45.30 GB
  Difference:        +2.90 GB (+6.0%)
  Quality:           Good ✓
  Status:            ✓ Recorded to validation-results.csv
```

---

## Understanding Validation Results

### Match Quality Indicators

| % Difference | Quality | Symbol | Interpretation |
|-------------|---------|--------|-----------------|
| ≤2% | Excellent | ✓✓ | Perfect match |
| 2-5% | Good | ✓ | Very accurate |
| 5-10% | Acceptable | ✓ | Within tolerance |
| 10-20% | Fair | ⚠ | Needs review |
| >20% | Poor | ✗ | Significant discrepancy |

### Sample Validation Report

```
═══════════════════════════════════════════════════════════════════════════════
LLM SIZER CALCULATOR VALIDATION REPORT
═══════════════════════════════════════════════════════════════════════════════

Profile: vllm-inference_20251103_120000.json

MODEL INFORMATION
───────────────────────────────────────────────────────────────────────────────
  Profiled:  meta-llama/Llama-2-7b-hf
  Matched:   Llama-2 7B
  Parameters: 6.7B (profiled) vs 6.7B (calculator)

GPU INFORMATION
───────────────────────────────────────────────────────────────────────────────
  Type:      AMD Instinct MI300X
  Count:     1x GPU
  VRAM:      206GB

CONFIGURATION
───────────────────────────────────────────────────────────────────────────────
  Batch Size:         8
  Input Tokens:       512
  Output Tokens:      512
  Inference Quant:    fp16
  KV Cache Quant:     fp16

MEMORY COMPARISON
═══════════════════════════════════════════════════════════════════════════════
Component                  Profiled Calculated   Diff      % Diff    Match
───────────────────────────────────────────────────────────────────────────────
Model Weights               13.40 GB   13.40 GB   +0.00 GB    +0.0%       ✓✓
KV Cache                     8.50 GB    8.45 GB   +0.05 GB    +0.6%       ✓✓
Activations                  5.60 GB    6.20 GB   -0.60 GB    -9.7%       ⚠
Framework Overhead          20.70 GB   17.10 GB   +3.60 GB   +17.3%       ⚠
Multi-GPU Overhead           0.00 GB    0.00 GB   +0.00 GB    +0.0%       ✓✓
───────────────────────────────────────────────────────────────────────────────
TOTAL                       48.20 GB   45.15 GB   +3.05 GB    +6.3%        ✓

SUMMARY
───────────────────────────────────────────────────────────────────────────────
  Overall Match:  GOOD (✓)
  Total Error:    +6.3%
  Absolute Error: 3.05 GB

RECOMMENDATIONS
───────────────────────────────────────────────────────────────────────────────
  ✓ Total memory within acceptable range (+6.3%)
  
  ⚠ Activations 9.7% lower than calculated
    → Check if framework uses activation checkpointing
    
  ⚠ Framework Overhead 17.3% higher than calculated
    → Current: 8% overhead calculation
    → Measured: 20.70 GB (14.3% of model size)
    → Recommendation: Increase overhead to 10-12%

═══════════════════════════════════════════════════════════════════════════════
```

---

## Batch Validation Workflow

### 1. Create Model List

Edit or create `configs/validation-campaign.yaml`:

```yaml
models:
  - model_id: "meta-llama/Llama-2-7b-hf"
    container_name: "vllm-inference"
    input_lengths: [256, 512, 1024]
    output_lengths: [256, 512]
    batch_sizes: [1, 8, 16]
    tensor_parallel_size: 1
    
  - model_id: "meta-llama/Llama-2-70b-hf"
    container_name: "vllm-inference"
    input_lengths: [512, 1024]
    output_lengths: [256, 512]
    batch_sizes: [1, 4, 8]
    tensor_parallel_size: 4
    
  - model_id: "meta-llama/Llama-3.1-8B-Instruct"
    container_name: "vllm-inference"
    input_lengths: [512, 1024, 2048]
    output_lengths: [256, 512, 1024]
    batch_sizes: [1, 8, 16]
    tensor_parallel_size: 1
```

### 2. Profile All Models

```bash
python scripts/batch-profile-bench-enhanced.py --config configs/validation-campaign.yaml
```

**Typical runtime**: 4-8 hours depending on model count and config complexity

### 3. Validate Each Profile

#### Option A: Manual Validation (for fewer profiles)

```bash
# For each profile, manually:
for profile in results/memory-profiles/*.json; do
    echo "=== Profile: $profile ==="
    python scripts/validate-profile.py "$profile"
    echo -n "Enter calculator total (GB): "
    read total
    python scripts/validate-profile.py "$profile" \
        --calculator-total "$total" \
        --record validation-results.csv
done
```

#### Option B: Batch Validation (automated)

```bash
npm run batch-validate
```

This automatically:
- Loads all profiles from `results/memory-profiles/`
- Auto-detects model and GPU
- Runs through calculator
- Generates individual reports
- Creates summary statistics

### 4. Analyze Results

```bash
# View results in terminal
column -t -s, validation-results.csv | less -S

# Or generate analysis report
python scripts/analyze-validation.py validation-results.csv --detailed

# Or open in spreadsheet
libreoffice validation-results.csv  # or Excel
```

---

## Advanced Validation

### Manual Model/GPU Matching

If auto-detection fails:

```bash
npm run validate-calculator -- profile.json \
  --model-id llama-2-7b \
  --gpu-id mi300x
```

### Override Configuration

Force specific settings:

```bash
npm run validate-calculator -- profile.json \
  --inference-quant fp8 \
  --kv-quant fp8_bf16 \
  --model-params 6.7e9
```

### Generate JSON Report

For scripting/automation:

```bash
npm run validate-calculator -- profile.json --json > report.json
```

### Programmatic Usage

Use validator in your own code:

```typescript
import { validateCalculator } from './scripts/validate-calculator';

const report = await validateCalculator('profile.json', {
  modelId: 'llama-2-7b',
  gpuId: 'mi300x',
  inferenceQuant: 'fp16',
  kvQuant: 'fp16_bf16',
});

console.log(`Match: ${report.summary.overall_match}`);
console.log(`Error: ${report.summary.total_percent_diff}%`);
```

---

## Analyzing Validation Results

### Common Discrepancy Patterns

#### 1. Framework Overhead Too Low

**Symptom**: Actual framework overhead much higher than calculated

```
Framework Overhead: 20.7 GB actual vs 9.5 GB calculated (+118%)
```

**Solution**: Adjust overhead percentage in calculator

```typescript
// Current: 8%
const overhead = (weights + kv_cache + activations) * 0.08;

// Increase to match measurements
const overhead = (weights + kv_cache + activations) * 0.12;  // 12%
```

**Root causes**:
- vLLM internal buffers and allocators
- GPU memory fragmentation
- Scheduler overhead
- Cache allocations

#### 2. KV Cache Compression

**Symptom**: Actual KV cache lower than calculated

```
KV Cache: 4.2 GB actual vs 8.5 GB calculated (-50%)
```

**Explanation**: Modern engines use:
- PagedAttention (vLLM)
- KV cache compression
- Smart memory reuse

**Solution**: Add compression factor

```typescript
const kvCompressionFactor = 0.6;  // 40% compression
const kvCache = (calculated_kv_size * kvCompressionFactor);
```

#### 3. Activation Checkpointing

**Symptom**: Activations much lower than calculated

```
Activations: 2.1 GB actual vs 8.3 GB calculated (-75%)
```

**Explanation**: Frameworks may use:
- Gradient checkpointing
- Activation recomputation
- Selective layer caching

**Solution**: Add checkpointing factor

```typescript
const activationFactor = 0.25;  // 75% reduction
const activations = (calculated_activations * activationFactor);
```

#### 4. Multi-GPU Overhead

**Symptom**: Multi-GPU overhead higher than expected

```
Multi-GPU Overhead: Tensor-parallel 4x GPU shows 15% overhead vs 8% expected
```

**Solution**: Adjust per-GPU overhead

```typescript
// Current: 2% per additional GPU
const multiGpuOverhead = numGpus > 1 
  ? baseMemory * 0.02 * (numGpus - 1) 
  : 0;

// Increase if needed
const multiGpuOverhead = numGpus > 1 
  ? baseMemory * 0.035 * (numGpus - 1)  // 3.5% per GPU
  : 0;
```

#### 5. Quantization Effects

**Symptom**: Weights much higher than quantization suggests

```
Model: AWQ-quantized 13B, but weights show FP16 size not INT4 size
```

**Root cause**: Quantization dtype mismatch in calculation

**Solution**: Verify quantization is applied

```typescript
// Check profiler data for actual dtype
const weights_gb = profile.memory_breakdown.model_weights_gb;
const expected_awq = (params * 0.5) / 1e9;  // INT4 is ~0.5B per param
if (weights_gb > expected_awq * 1.2) {
  console.warn("Quantization may not be applied");
}
```

### Validation by Model Size

Group results to identify size-specific issues:

```bash
# Small models (1-10B)
grep ",\(1\|2\|3\|4\|5\|6\|7\|8\|9\)\..*B," validation-results.csv

# Medium models (10-40B)
grep -E ",(1[0-9]|2[0-9]|3[0-9]|40)\..*B," validation-results.csv

# Large models (40B+)
grep -E ",(4[1-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9])\..*B," validation-results.csv
```

### Validation by GPU Count

Identify multi-GPU scaling issues:

```bash
# Single GPU results
grep ",1," validation-results.csv

# Multi-GPU results (2-8 GPUs)
grep ",\([2-8]\)," validation-results.csv
```

---

## Setting Up Priority Profiling

For models that need validation readiness:

1. **Create high-confidence profiles** with architecture details
2. **Verify profile metadata**:
   ```bash
   jq '.profiler_version, .memory_breakdown.confidence_levels.overall' profile.json
   ```

3. **Check for known issues**:
   - Peak memory equals baseline (lazy allocation failure)
   - Per-GPU imbalance in tensor parallelism
   - Unrealistic component ratios

4. **Ready status**:
   - ✅ All confidence levels "high"
   - ✅ Peak > baseline by > 40%
   - ✅ Per-GPU distribution reasonable
   - ✅ Component breakdown adds up correctly

---

## Data Dictionary

### CSV Output (validation-results.csv)

| Column | Description | Example |
|--------|-------------|---------|
| `profile_file` | Input profile filename | `llama-7b_20251103_120000.json` |
| `model_id` | HuggingFace model ID | `meta-llama/Llama-2-7b-hf` |
| `model_params_b` | Parameter count (billions) | `6.7` |
| `gpu_type` | GPU model | `AMD MI300X` |
| `gpu_count` | Number of GPUs | `1` |
| `batch_size` | Batch size used | `8` |
| `seq_length` | Total sequence length | `1024` |
| `dtype` | Data type | `float16` |
| `profiled_total_gb` | Measured total memory | `48.20` |
| `calculated_total_gb` | Calculator estimate | `45.15` |
| `difference_gb` | Profiled - Calculated | `3.05` |
| `difference_pct` | Percentage difference | `6.3` |
| `match_quality` | Match rating | `Good` |
| `profiled_weights_gb` | Measured model weights | `13.40` |
| `calculated_weights_gb` | Calculated weights | `13.40` |
| `weights_diff_pct` | Weights error % | `0.0` |
| `profiled_kv_gb` | Measured KV cache | `8.50` |
| `calculated_kv_gb` | Calculated KV cache | `8.45` |
| `kv_diff_pct` | KV cache error % | `0.6` |
| `profiled_activations_gb` | Measured activations | `5.60` |
| `calculated_activations_gb` | Calculated activations | `6.20` |
| `activations_diff_pct` | Activations error % | `-9.7` |
| `profiled_overhead_gb` | Measured overhead | `20.70` |
| `calculated_overhead_gb` | Calculated overhead | `17.10` |
| `overhead_diff_pct` | Overhead error % | `17.3` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Model not found** | Check if model exists in calculator's model list, use `--model-id` to override |
| **GPU not detected** | Verify GPU in profiler data, use `--gpu-id` to force match |
| **Calculator doesn't support config** | Check if GPU/model combination exists, try closest alternative |
| **Large discrepancies** | Check profiler confidence levels, verify quantization matches, review component breakdown |
| **Validation script fails** | Ensure profile JSON is valid: `jq empty profile.json` |

---

## Improving Calculator Accuracy

### Workflow

1. **Collect baseline profiles** - Profile diverse models/configs
2. **Validate each profile** - Compare against calculator
3. **Identify patterns** - Group errors by type/size/config
4. **Make targeted changes** - Fix formulas for problematic categories
5. **Re-validate** - Measure improvement
6. **Repeat** - Iterate until accuracy acceptable

### Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Average error | <5% | TBD |
| Good+ matches | >80% | TBD |
| Poor matches | <5% | TBD |
| Weights accuracy | <3% | TBD |
| KV cache accuracy | <5% | TBD |

---

## Integration with Profiling

See [PROFILING.md](PROFILING.md) for how to generate profiles for validation.

## Next Steps

1. ⏭️ Profile one model using high-confidence settings
2. ⏭️ Run through calculator and record result
3. ⏭️ Validate and identify accuracy issues
4. ⏭️ Create batch config for systematic validation
5. ⏭️ Analyze patterns across multiple models
6. ⏭️ Update calculator based on findings

---

**Documentation Files**:
- [PROFILING.md](PROFILING.md) - Complete profiling guide
- [README.md](README.md) - Project overview
