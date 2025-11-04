# LLM Sizer Remediation Plan
**Version:** 1.0  
**Date:** November 3, 2025  
**Status:** 🔴 Action Required

---

## Executive Summary

Based on validation analysis of 14 profiles across 5 models, we've identified critical gaps in calculator accuracy and areas needing additional data. This plan outlines:

1. **Critical calculator fixes** (framework overhead)
2. **Minor profiling fixes** (trust_remote_code flag)
3. **Additional data collection** (20+ new scenarios)
4. **Validation workflow improvements**

**Estimated Timeline:** 2-3 weeks  
**Priority:** 🔴 High - Calculator accuracy off by up to 500% for small models

---

## Phase 1: Critical Calculator Fixes (Week 1)

### 🔴 Priority 1: Framework Overhead Overhaul

**Problem:** Calculator estimates 8-10% overhead, actual ranges from 0-76%

**Root Cause:** Current formula uses fixed percentage
```typescript
// Current (WRONG)
const overhead = (weights + kvCache + activations) * 0.08;
```

**Solution:** Implement size-dependent baseline + proportional overhead

#### Implementation Steps

**Step 1.1: Add Framework Selection**
- [ ] Add dropdown for inference framework (vLLM, TGI, llama.cpp, TensorRT-LLM)
- [ ] Default to vLLM (most common)
- [ ] Each framework has different overhead characteristics

**Step 1.2: Implement vLLM Overhead Formula**
```typescript
// New vLLM overhead calculation
interface VLLMOverhead {
  baselineGB: number;        // Fixed overhead
  proportionalMultiplier: number;  // Scaling factor
  multiGPUMultiplier: number;     // Per-GPU overhead
  batchScaling: number;      // Batch size factor
}

const VLLM_OVERHEAD: VLLMOverhead = {
  baselineGB: 14.0,           // CUDA context + engine + memory pools
  proportionalMultiplier: 0.05,  // 5% of model memory
  multiGPUMultiplier: 0.15,   // 15% per additional GPU
  batchScaling: 0.002,        // 0.2% per batch size unit
};

function calculateVLLMOverhead(
  weights: number,
  kvCache: number,
  activations: number,
  numGPUs: number,
  batchSize: number
): number {
  const modelMemory = weights + kvCache + activations;
  
  // Base overhead (fixed)
  let overhead = VLLM_OVERHEAD.baselineGB;
  
  // Proportional to model memory
  overhead += modelMemory * VLLM_OVERHEAD.proportionalMultiplier;
  
  // Multi-GPU overhead (tensor parallelism communication)
  if (numGPUs > 1) {
    overhead += (weights / numGPUs) * VLLM_OVERHEAD.multiGPUMultiplier * (numGPUs - 1);
  }
  
  // Batch size scaling (larger batches = more internal buffers)
  if (batchSize > 1) {
    overhead += modelMemory * VLLM_OVERHEAD.batchScaling * Math.log2(batchSize);
  }
  
  return overhead;
}
```

**Step 1.3: Add Other Framework Profiles**
```typescript
const TGI_OVERHEAD = {
  baselineGB: 8.0,            // TGI is more efficient
  proportionalMultiplier: 0.08,
  multiGPUMultiplier: 0.12,
  batchScaling: 0.003,
};

const LLAMACPP_OVERHEAD = {
  baselineGB: 0.5,            // Very lightweight
  proportionalMultiplier: 0.03,
  multiGPUMultiplier: 0.0,    // No multi-GPU
  batchScaling: 0.001,
};

const TENSORRT_OVERHEAD = {
  baselineGB: 6.0,
  proportionalMultiplier: 0.06,
  multiGPUMultiplier: 0.10,
  batchScaling: 0.002,
};
```

**Step 1.4: Update UI**
- [ ] Add "Inference Framework" dropdown in calculator
- [ ] Add tooltip explaining overhead differences
- [ ] Show overhead breakdown in results
- [ ] Add "Advanced" toggle to show/hide overhead details

**Files to Modify:**
- `src/utils/memoryCalculations.ts` - Core calculation logic
- `src/components/Calculator.tsx` - Add framework selector
- `src/types/models.ts` - Add framework types
- `src/store/calculatorStore.ts` - Add framework state

**Testing:**
- [ ] Test with Llama-3.2-1B (expect 18.94 GB vs current ~3 GB)
- [ ] Test with Mistral-7B batch=1 (expect 20.28 GB)
- [ ] Test with Mistral-7B batch=8 (expect 41.27 GB)
- [ ] Test with GLM-4.5 TP=4 (expect 125 GB)

---

### 🟡 Priority 2: FP8 KV Cache Support

**Problem:** Calculator doesn't support FP8 KV cache (uses 50% less memory than FP16)

**Solution:** Add KV cache dtype selector

#### Implementation Steps

**Step 2.1: Add KV Cache Dtype Selector**
```typescript
type KVCacheDtype = 'fp32' | 'fp16' | 'bf16' | 'fp8_e4m3' | 'fp8_e5m2';

const KV_CACHE_MULTIPLIERS: Record<KVCacheDtype, number> = {
  fp32: 1.0,      // 4 bytes per element (baseline)
  fp16: 0.5,      // 2 bytes per element
  bf16: 0.5,      // 2 bytes per element
  fp8_e4m3: 0.25, // 1 byte per element
  fp8_e5m2: 0.25, // 1 byte per element
};

function calculateKVCache(
  numLayers: number,
  hiddenSize: number,
  seqLength: number,
  batchSize: number,
  kvCacheDtype: KVCacheDtype
): number {
  // Base calculation assumes fp32
  const baseKVCache = 2 * numLayers * hiddenSize * seqLength * batchSize * 4 / 1e9;
  
  // Apply dtype multiplier
  return baseKVCache * KV_CACHE_MULTIPLIERS[kvCacheDtype];
}
```

**Step 2.2: Update UI**
- [ ] Add "KV Cache Data Type" dropdown
- [ ] Default to same as model dtype
- [ ] Add info tooltip: "FP8 uses 50% less memory than FP16"
- [ ] Show memory savings in results

**Files to Modify:**
- `src/utils/memoryCalculations.ts` - Update KV cache calculation
- `src/components/InferenceConfig.tsx` - Add dtype selector
- `src/types/inference.ts` - Add KV cache dtype type

**Testing:**
- [ ] Llama-3.1-8B FP8 KV: expect 0.27 GB (batch=1)
- [ ] Qwen3-8B FP16 KV: expect 0.54 GB (batch=1) - exactly 2x
- [ ] Verify 2x ratio between FP8 and FP16

---

### 🟡 Priority 3: Batch Size Overhead Scaling

**Problem:** Calculator treats overhead as constant, but it scales with batch size

**Solution:** Already included in Step 1.2 above

**Additional Testing Needed:**
- [ ] Profile more batch sizes: 2, 4, 16, 32, 64
- [ ] Validate logarithmic scaling hypothesis
- [ ] Adjust `batchScaling` constant based on results

---

### 🟢 Priority 4: Multi-GPU Overhead

**Problem:** Tensor parallelism overhead not accurately modeled

**Solution:** Already included in Step 1.2 above

**Additional Validation:**
- [ ] Profile GLM-4.5 with TP=2, TP=4, TP=8
- [ ] Profile Llama-70B with TP=2, TP=4, TP=8
- [ ] Validate 15% per-GPU overhead factor

---

### 🟢 Priority 5: MoE Model Support

**Problem:** Mixture-of-Experts models not handled correctly

**Solution:** Add MoE architecture detection and adjusted weight calculation

```typescript
interface MoEConfig {
  isMoE: boolean;
  numExperts: number;
  expertsPerToken: number;  // Active experts
  sharedParams: number;     // Non-expert parameters
}

function calculateMoEWeights(
  totalParams: number,
  moeConfig: MoEConfig,
  dtype: ModelDtype
): number {
  if (!moeConfig.isMoE) {
    return totalParams * DTYPE_BYTES[dtype] / 1e9;
  }
  
  // Only count active parameters
  const activeParams = moeConfig.sharedParams + 
    (totalParams - moeConfig.sharedParams) * 
    (moeConfig.expertsPerToken / moeConfig.numExperts);
  
  return activeParams * DTYPE_BYTES[dtype] / 1e9;
}
```

**UI Changes:**
- [ ] Add "Model Architecture" dropdown: Standard, MoE, Hybrid
- [ ] Show MoE config fields when MoE selected
- [ ] Add tooltip explaining MoE memory characteristics

**Files to Modify:**
- `src/utils/memoryCalculations.ts` - Add MoE calculation
- `src/components/ModelConfig.tsx` - Add MoE fields
- `src/types/models.ts` - Add MoE types

**Testing:**
- [ ] GLM-4.5-Air: 54B params, 8 experts, 2 active
- [ ] Mixtral-8x7B: 46.7B params, 8 experts, 2 active
- [ ] Qwen2-57B-MoE: Add to profiling queue

---

## Phase 2: Profiling Infrastructure Fixes (Week 1)

### Fix 1: Add trust_remote_code Support

**Problem:** Kimi-K2 and other custom models fail without `trust_remote_code=True`

**Solution:** Update profiler to add flag for models with custom code

#### Implementation

**Step 1: Update batch profiler**
```python
# scripts/batch-profile-bench-enhanced.py

# Add to config schema
models:
  - model_id: "moonshotai/Kimi-K2-Instruct-0905"
    trust_remote_code: true  # NEW FIELD
    # ... other config
```

**Step 2: Update profiler command builder**
```python
def build_vllm_command(config: dict) -> list[str]:
    cmd = [
        "vllm", "bench", "latency",
        "--model", config["model_id"],
        # ... other args
    ]
    
    # Add trust_remote_code if specified
    if config.get("trust_remote_code", False):
        cmd.extend(["--trust-remote-code"])
    
    return cmd
```

**Step 3: Update quick.yaml**
```yaml
models:
  # ... existing models ...
  
  # Kimi K2 (Moonshot AI) - FIXED
  - model_id: "moonshotai/Kimi-K2-Instruct-0905"
    container_name: "vllm-inference"
    input_lengths: [512, 1024]
    output_lengths: [512, 1024]
    batch_sizes: [1, 4, 8]
    dtype: "bfloat16"
    tensor_parallel_size: 4
    trust_remote_code: true  # NEW
    model_params: 14.0e9
    num_layers: 40
    num_heads: 40
    hidden_size: 4096
```

**Files to Modify:**
- `scripts/batch-profile-bench-enhanced.py`
- `scripts/profile-vllm-bench-enhanced.py`
- `scripts/configs/quick.yaml`

**Testing:**
- [ ] Re-run Kimi-K2 profiles
- [ ] Verify successful profiling
- [ ] Add to validation dataset

---

### Fix 2: Add Model Registry with Metadata

**Solution:** Create centralized model registry with pre-filled metadata

```yaml
# scripts/configs/model-registry.yaml

models:
  "meta-llama/Llama-2-7b-hf":
    params: 6.7e9
    layers: 32
    heads: 32
    hidden: 4096
    head_dim: 128
    architecture: "llama"
    trust_remote_code: false
    
  "moonshotai/Kimi-K2-Instruct-0905":
    params: 14.0e9
    layers: 40
    heads: 40
    hidden: 4096
    head_dim: 102
    architecture: "custom"
    trust_remote_code: true  # Required!
    
  "zai-org/GLM-4.5-Air":
    params: 54.0e9
    layers: 40
    heads: 40
    hidden: 6144
    head_dim: 96
    architecture: "moe"
    moe_config:
      num_experts: 8
      experts_per_token: 2
    trust_remote_code: false
```

**Files to Create:**
- `scripts/configs/model-registry.yaml`
- `scripts/lib/model_registry.py` - Registry loader

---

### Fix 3: Improve Error Handling

```python
# scripts/batch-profile-bench-enhanced.py

def profile_model_safe(config: dict) -> dict:
    """Profile with comprehensive error handling"""
    try:
        return profile_model(config)
    except TrustRemoteCodeError as e:
        return {
            "error": "trust_remote_code_required",
            "message": "Add trust_remote_code: true to config",
            "model": config["model_id"]
        }
    except OOMError as e:
        return {
            "error": "out_of_memory",
            "message": f"Try smaller batch size or fewer GPUs",
            "model": config["model_id"]
        }
    except Exception as e:
        return {
            "error": "unknown",
            "message": str(e),
            "model": config["model_id"]
        }
```

---

## Phase 3: Additional Data Collection (Week 2-3)

### Data Gap Analysis

Based on current profiles, we need data for:

1. **More batch sizes** - Currently only have 1, 4, 8
2. **Longer sequences** - Currently max 1536 tokens
3. **Quantized models** - No AWQ/GPTQ profiles yet
4. **More GPU counts** - Need TP=2, TP=8 data
5. **Different frameworks** - All profiles are vLLM
6. **Edge cases** - Very small/large models

---

### Scenario 1: Batch Size Scaling Study

**Objective:** Understand overhead scaling with batch size

**Configuration:**
```yaml
# scripts/configs/batch-scaling-study.yaml

models:
  # Small model (1B)
  - model_id: "amd/Llama-3.2-1B-Instruct-FP8-KV"
    input_lengths: [1024]
    output_lengths: [512]
    batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128]  # Wide range
    dtype: "float16"
    tensor_parallel_size: 1
    
  # Medium model (7-8B)
  - model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    input_lengths: [1024]
    output_lengths: [512]
    batch_sizes: [1, 2, 4, 8, 16, 32, 64]
    dtype: "float16"
    tensor_parallel_size: 1
    
  # Large model (70B)
  - model_id: "meta-llama/Llama-2-70b-hf"
    input_lengths: [1024]
    output_lengths: [512]
    batch_sizes: [1, 2, 4, 8, 16]  # Fewer due to memory
    dtype: "float16"
    tensor_parallel_size: 4
```

**Expected Results:**
- Overhead vs batch size curve
- Identify knee point (optimal batch)
- Validate logarithmic scaling hypothesis

**Estimated Time:** 6-8 hours

---

### Scenario 2: Sequence Length Scaling Study

**Objective:** Validate KV cache grows linearly with sequence length

**Configuration:**
```yaml
# scripts/configs/sequence-scaling-study.yaml

models:
  - model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    input_lengths: [256, 512, 1024, 2048, 4096, 8192]  # 2x steps
    output_lengths: [256]  # Fixed output
    batch_sizes: [1, 8]
    dtype: "float16"
    tensor_parallel_size: 1
    
  - model_id: "amd/Llama-3.1-8B-Instruct-FP8-KV"
    input_lengths: [256, 512, 1024, 2048, 4096]
    output_lengths: [256]
    batch_sizes: [1, 8]
    dtype: "float16"
    kv_cache_dtype: "fp8"
    tensor_parallel_size: 1
```

**Expected Results:**
- KV cache = f(seq_length) should be perfectly linear
- Overhead should remain constant
- FP8 vs FP16 KV cache ratio validation

**Estimated Time:** 4-5 hours

---

### Scenario 3: Multi-GPU Scaling Study

**Objective:** Measure tensor parallelism overhead

**Configuration:**
```yaml
# scripts/configs/multi-gpu-study.yaml

models:
  # 13B model across different GPU counts
  - model_id: "meta-llama/Llama-2-13b-hf"
    input_lengths: [1024]
    output_lengths: [512]
    batch_sizes: [1, 8]
    dtype: "float16"
    tensor_parallel_size: 1  # Baseline
    
  - model_id: "meta-llama/Llama-2-13b-hf"
    tensor_parallel_size: 2
    # ... same other params
    
  - model_id: "meta-llama/Llama-2-13b-hf"
    tensor_parallel_size: 4
    
  - model_id: "meta-llama/Llama-2-13b-hf"
    tensor_parallel_size: 8
    
  # 70B model (requires multi-GPU)
  - model_id: "meta-llama/Llama-2-70b-hf"
    tensor_parallel_size: 2
    # ... params
    
  - model_id: "meta-llama/Llama-2-70b-hf"
    tensor_parallel_size: 4
    
  - model_id: "meta-llama/Llama-2-70b-hf"
    tensor_parallel_size: 8
```

**Expected Results:**
- Per-GPU overhead multiplier
- Communication overhead scaling
- Efficiency curve (speedup vs GPU count)

**Estimated Time:** 8-10 hours

---

### Scenario 4: Quantization Study

**Objective:** Measure memory reduction from quantization

**Configuration:**
```yaml
# scripts/configs/quantization-study.yaml

models:
  # Mistral-7B in different quantizations
  - model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    quantization: null  # FP16 baseline
    batch_sizes: [1, 8]
    
  - model_id: "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    quantization: "awq"
    batch_sizes: [1, 8]
    
  - model_id: "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    quantization: "gptq"
    batch_sizes: [1, 8]
    
  # Llama-3.1-8B quantized
  - model_id: "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    quantization: "awq"
    batch_sizes: [1, 8]
    
  - model_id: "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    quantization: "gptq"
    batch_sizes: [1, 8]
```

**Expected Results:**
- AWQ: ~4x weight reduction (16-bit → 4-bit)
- GPTQ: ~4x weight reduction
- FP8: ~2x weight reduction
- Overhead changes with quantization

**Estimated Time:** 6-8 hours

---

### Scenario 5: MoE Models Study

**Objective:** Understand sparse expert memory usage

**Configuration:**
```yaml
# scripts/configs/moe-study.yaml

models:
  # Mixtral 8x7B
  - model_id: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    batch_sizes: [1, 4, 8]
    tensor_parallel_size: 1
    model_params: 46.7e9
    num_experts: 8
    experts_per_token: 2
    
  - model_id: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tensor_parallel_size: 2
    
  - model_id: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tensor_parallel_size: 4
    
  # GLM-4.5-Air (larger MoE)
  - model_id: "zai-org/GLM-4.5-Air"
    batch_sizes: [1, 4, 8]
    tensor_parallel_size: 4
    model_params: 54.0e9
    num_experts: 8
    experts_per_token: 2
    
  # Qwen2-57B-MoE
  - model_id: "Qwen/Qwen2-57B-A14B-Instruct"
    batch_sizes: [1, 4, 8]
    tensor_parallel_size: 4
    model_params: 57.0e9
```

**Expected Results:**
- Active parameter count validation
- MoE overhead vs dense models
- Expert activation patterns

**Estimated Time:** 10-12 hours

---

### Scenario 6: Edge Cases Study

**Objective:** Test calculator at extremes

**Configuration:**
```yaml
# scripts/configs/edge-cases-study.yaml

models:
  # Very small models
  - model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    batch_sizes: [1, 8, 32]  # Small model = can handle large batches
    
  - model_id: "microsoft/phi-2"  # 2.7B
    batch_sizes: [1, 8, 32]
    
  # Very large models
  - model_id: "meta-llama/Llama-2-70b-hf"
    batch_sizes: [1, 2, 4, 8]
    tensor_parallel_size: 8
    
  - model_id: "codellama/CodeLlama-70b-Instruct-hf"
    batch_sizes: [1, 2, 4]
    tensor_parallel_size: 8
    
  # Extremely long sequences
  - model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    input_lengths: [16384, 32768]  # Very long context
    output_lengths: [512]
    batch_sizes: [1, 2]
    
  # Extreme batch sizes
  - model_id: "amd/Llama-3.2-1B-Instruct-FP8-KV"
    batch_sizes: [128, 256, 512]  # Massive batch
```

**Expected Results:**
- Identify calculator failure points
- Validate extreme scenarios
- Find practical limits

**Estimated Time:** 12-15 hours

---

### Scenario 7: Framework Comparison

**Objective:** Compare vLLM vs other inference engines

**Configuration:**
```yaml
# scripts/configs/framework-comparison.yaml

# NOTE: Requires setup of multiple frameworks

models:
  # Same model, different frameworks
  - model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    framework: "vllm"
    batch_sizes: [1, 8]
    
  - model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    framework: "tgi"  # Text Generation Inference
    batch_sizes: [1, 8]
    
  - model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    framework: "tensorrt-llm"
    batch_sizes: [1, 8]
```

**Expected Results:**
- Framework overhead comparison
- Efficiency differences
- Memory optimization strategies

**Estimated Time:** 15-20 hours (requires setup)

---

## Data Collection Priority Matrix

| Scenario | Priority | Impact | Effort | Time | Order |
|----------|----------|--------|--------|------|-------|
| Batch Size Scaling | 🔴 HIGH | High | Medium | 6-8h | 1 |
| Sequence Length | 🔴 HIGH | High | Low | 4-5h | 2 |
| Quantization | 🟡 MEDIUM | High | Medium | 6-8h | 3 |
| Multi-GPU Scaling | 🟡 MEDIUM | Medium | High | 8-10h | 4 |
| MoE Models | 🟡 MEDIUM | Medium | High | 10-12h | 5 |
| Edge Cases | 🟢 LOW | Low | High | 12-15h | 6 |
| Framework Comparison | 🟢 LOW | Medium | Very High | 15-20h | 7 |

**Total Estimated Time:** 61-85 hours of profiling

**Recommended Approach:**
1. Start with high-priority scenarios (1-3)
2. Update calculator with initial findings
3. Validate calculator improvements
4. Continue with remaining scenarios

---

## Phase 4: Validation Workflow Improvements (Week 2)

### Improvement 1: Automated Calculator Testing

**Create automated validation script:**

```typescript
// scripts/validate-calculator-automated.ts

interface ValidationTest {
  name: string;
  profilePath: string;
  calculatorInputs: CalculatorInputs;
  expectedMemory: number;
  tolerance: number;  // % tolerance
}

const VALIDATION_TESTS: ValidationTest[] = [
  {
    name: "Llama-3.2-1B Batch=1 FP8-KV",
    profilePath: "results/memory-profiles/vllm-inference_512in_512out_bs1_20251103_225123.json",
    calculatorInputs: {
      modelParams: 1.24e9,
      numLayers: 16,
      hiddenSize: 2048,
      numHeads: 32,
      batchSize: 1,
      seqLength: 1024,
      weightDtype: "fp16",
      kvCacheDtype: "fp8",
      framework: "vllm"
    },
    expectedMemory: 18.94,
    tolerance: 5  // ±5%
  },
  // ... more tests
];

async function runValidationSuite() {
  const results = [];
  
  for (const test of VALIDATION_TESTS) {
    const calculated = calculateMemory(test.calculatorInputs);
    const diff = Math.abs(calculated - test.expectedMemory);
    const diffPct = (diff / test.expectedMemory) * 100;
    
    results.push({
      name: test.name,
      expected: test.expectedMemory,
      calculated: calculated,
      diff: diff,
      diffPct: diffPct,
      passed: diffPct <= test.tolerance
    });
  }
  
  return results;
}
```

**Files to Create:**
- `scripts/validate-calculator-automated.ts`
- `scripts/validation-suite.json` - Test definitions

---

### Improvement 2: CI/CD Integration

**Add validation to CI pipeline:**

```yaml
# .github/workflows/validate-calculator.yml

name: Calculator Validation

on:
  pull_request:
    paths:
      - 'src/utils/memoryCalculations.ts'
      - 'src/components/**'
  push:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: 18
          
      - name: Install dependencies
        run: npm install
        
      - name: Run validation suite
        run: npm run validate:calculator
        
      - name: Check accuracy threshold
        run: |
          # Fail if any test has >10% error
          node scripts/check-validation-threshold.js --max-error 10
```

---

### Improvement 3: Validation Dashboard

**Create interactive validation dashboard:**

```typescript
// src/pages/ValidationDashboard.tsx

interface ValidationDashboardProps {
  profiles: Profile[];
  calculatorResults: CalculatorResult[];
}

export function ValidationDashboard({ profiles, calculatorResults }: ValidationDashboardProps) {
  return (
    <div className="validation-dashboard">
      <h1>Calculator Validation Dashboard</h1>
      
      {/* Summary Stats */}
      <div className="stats-grid">
        <StatCard title="Total Tests" value={profiles.length} />
        <StatCard title="Passing" value={passingTests} color="green" />
        <StatCard title="Failing" value={failingTests} color="red" />
        <StatCard title="Avg Error" value={`${avgError}%`} />
      </div>
      
      {/* Accuracy by Model Size */}
      <Chart 
        data={accuracyBySize}
        title="Accuracy by Model Size"
        xAxis="Model Size (B params)"
        yAxis="Error %"
      />
      
      {/* Accuracy by Batch Size */}
      <Chart 
        data={accuracyByBatch}
        title="Accuracy by Batch Size"
      />
      
      {/* Detailed Results Table */}
      <ValidationTable 
        results={validationResults}
        sortable
        filterable
      />
    </div>
  );
}
```

---

## Implementation Timeline

### Week 1: Critical Fixes
- **Days 1-2:** Implement framework overhead overhaul
- **Day 3:** Add FP8 KV cache support
- **Day 4:** Fix trust_remote_code issue
- **Day 5:** Testing and validation

### Week 2: Data Collection (High Priority)
- **Days 1-2:** Batch size scaling study (Scenario 1)
- **Days 2-3:** Sequence length study (Scenario 2)
- **Days 4-5:** Quantization study (Scenario 3)

### Week 3: Data Collection (Medium Priority) & Refinement
- **Days 1-2:** Multi-GPU study (Scenario 4)
- **Days 3-4:** MoE models study (Scenario 5)
- **Day 5:** Validation and documentation

---

## Success Metrics

### Calculator Accuracy Targets

| Model Size | Current Avg Error | Target Avg Error | Target Date |
|------------|-------------------|------------------|-------------|
| 1-2B | ~400% | <10% | Week 1 |
| 7-8B | ~20% | <5% | Week 1 |
| 13-30B | Unknown | <8% | Week 2 |
| 40-70B | Unknown | <10% | Week 3 |
| MoE Models | ~300% | <15% | Week 3 |

### Validation Coverage Targets

| Category | Current | Target Week 1 | Target Week 3 |
|----------|---------|---------------|---------------|
| Batch Sizes | 3 | 8 | 12 |
| Sequence Lengths | 3 | 6 | 10 |
| Model Sizes | 5 | 8 | 15 |
| GPU Configurations | 2 | 5 | 8 |
| Quantizations | 1 (FP8) | 3 | 5 |

---

## Rollout Plan

### Phase 1: Calculator Updates (Week 1)
1. Deploy framework overhead fix to staging
2. Run automated validation suite
3. If accuracy improves to <15% error: deploy to production
4. If not: iterate on overhead formula

### Phase 2: Extended Validation (Week 2-3)
1. Collect additional profiling data
2. Update calculator formulas based on findings
3. Re-run validation suite
4. Deploy incremental improvements

### Phase 3: Documentation (Ongoing)
1. Update calculator documentation
2. Add validation methodology doc
3. Create user guide for edge cases
4. Publish validation results

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overhead formula still inaccurate | Medium | High | Collect more data, iterate formula |
| Profiling takes longer than estimated | High | Medium | Prioritize high-impact scenarios |
| Calculator changes break existing features | Medium | High | Comprehensive unit tests, staging environment |
| Some models can't be profiled | Medium | Low | Document limitations, focus on common models |
| Multi-framework data too complex | High | Low | Start with vLLM only, expand later |

---

## Appendix A: Files to Modify

### Calculator Code
- `src/utils/memoryCalculations.ts` - Core calculation logic
- `src/components/Calculator.tsx` - Main calculator component
- `src/components/ModelConfig.tsx` - Model configuration inputs
- `src/components/InferenceConfig.tsx` - Inference settings
- `src/types/models.ts` - Type definitions
- `src/types/inference.ts` - Inference types
- `src/store/calculatorStore.ts` - State management

### Profiling Scripts
- `scripts/batch-profile-bench-enhanced.py` - Batch profiler
- `scripts/profile-vllm-bench-enhanced.py` - Single model profiler
- `scripts/configs/quick.yaml` - Quick test configuration
- `scripts/lib/model_registry.py` - Model metadata registry (NEW)

### Validation Scripts
- `scripts/analyze-profiles.py` - Profile analyzer
- `scripts/validate-calculator-automated.ts` - Automated testing (NEW)
- `scripts/check-validation-threshold.js` - CI validation check (NEW)

### Documentation
- `CALCULATOR-VALIDATION.md` - Validation guide
- `README.md` - Main documentation
- `REMEDIATION-PLAN.md` - This document

---

## Appendix B: Quick Commands Reference

```bash
# Run quick test suite
npm run validate:calculator

# Profile single model
python scripts/profile-vllm-bench-enhanced.py \
  --model meta-llama/Llama-2-7b-hf \
  --input-len 1024 --output-len 512 --batch-size 8

# Batch profile with config
python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/batch-scaling-study.yaml

# Analyze single profile
python scripts/analyze-profiles.py \
  results/memory-profiles/profile.json \
  --calc-total 20.5

# Run automated validation
npm run validate:automated

# Generate validation report
python scripts/generate-validation-report.py \
  --profiles results/memory-profiles/ \
  --output results/validation-report.html
```

---

## Questions for Stakeholders

1. **Priority Confirmation:** Do you agree with the prioritization of overhead fix > data collection?
2. **Timeline Flexibility:** Can we extend to 4 weeks if needed for comprehensive data?
3. **Framework Scope:** Start with vLLM only, or profile multiple frameworks in parallel?
4. **Accuracy Target:** Is ±10% error acceptable, or should we aim for ±5%?
5. **Breaking Changes:** Can we make breaking changes to calculator API if needed?

---

## Approval Sign-off

- [ ] Technical Lead Review
- [ ] Product Manager Approval
- [ ] Resource Allocation Confirmed
- [ ] Timeline Agreed
- [ ] Success Metrics Approved

**Document Version:** 1.0  
**Last Updated:** November 3, 2025  
**Next Review:** After Week 1 completion
