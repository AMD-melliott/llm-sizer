# PRD: LLM Sizer CLI Tool

**Status:** In Progress
**Date:** 2026-02-28
**Version:** 0.2

### Progress

| Phase | Task | Status |
|-------|------|--------|
| 1 | CLI build infrastructure (tsup, tsconfig.cli.json, entry point) | Done |
| 1 | Data loader & model resolver | Done |
| 1 | Output formatting utilities | Done |
| 1 | `calculate` command | Done |
| 1 | `models list\|show` commands | Done |
| 1 | `gpus list\|show\|compare` commands | Done |
| 2 | `docker run\|compose` commands | Not started |
| 2 | `serve vllm\|sglang` commands | Not started |
| 3 | `profile generate\|compare` commands | Not started |
| 3 | `benchmark generate\|analyze` commands | Not started |
| 4 | Integration tests, documentation, polish | Not started |

**Phase 1 completed 2026-02-28** — 9 commits, 55 new CLI tests (270 total), zero regressions.

## 1. Overview

Add a CLI component to the LLM Sizer tool that exposes the same calculation engine and data sets used by the web UI. The CLI enables scripted workflows for memory sizing, container configuration, inference server launch, benchmarking, profiling, and results analysis — all driven by the shared model/GPU databases and TypeScript calculation logic.

## 2. Motivation

The web UI is effective for interactive exploration, but several workflows are better served by a CLI:

- **Automation**: CI/CD pipelines, batch sizing across model matrices, scripted deployment
- **Reproducibility**: Version-controlled commands with deterministic outputs
- **Integration**: Pipe JSON output into deployment scripts, monitoring tools, or profiling harnesses
- **Profiling workflow**: Today, constructing profiler commands requires manually cross-referencing `models.json`, remembering flag names, and copying parameters — error-prone for studies that sweep across dozens of configurations
- **Gap coverage**: vLLM/SGLang bare-metal launch commands and benchmark generation have no UI equivalent today

## 3. Existing Infrastructure

### 3.1 Reusable Calculation Engine (Zero React Dependencies)

| Module | Location | Purpose |
|--------|----------|---------|
| `memoryCalculator.ts` | `src/utils/` | Generation model memory sizing |
| `embeddingCalculator.ts` | `src/utils/` | Embedding model memory sizing |
| `rerankingCalculator.ts` | `src/utils/` | Reranking model memory sizing |
| `performanceEstimator.ts` | `src/utils/` | Tokens/sec, latency estimation |
| `partitionCalculator.ts` | `src/utils/` | AMD GPU partition analysis |
| `dockerCommandBuilder.ts` | `src/utils/` | Docker run script generation |
| `dockerComposeBuilder.ts` | `src/utils/` | Docker Compose YAML generation |

All calculation functions are pure (inputs → results), with no UI or state management dependencies.

### 3.2 Shared Data Files

| File | Records | Purpose |
|------|---------|---------|
| `src/data/models.json` | 100+ | Generation model architectures |
| `src/data/embedding-models.json` | 30+ | Embedding model specs |
| `src/data/reranking-models.json` | 10+ | Reranking model specs |
| `src/data/gpus.json` | 57+ | GPU specs (AMD + NVIDIA) |
| `src/data/engine-parameters.json` | 1 (vLLM) | Inference engine parameter definitions |
| `src/data/container-images.json` | 4 | Container image registry |

### 3.3 Existing CLI Tools

| Script | Runner | Pattern |
|--------|--------|---------|
| `scripts/import-hf-model.ts` | `tsx` via `npm run import-model` | `commander` + `chalk` + shared `scripts/lib/` |
| `scripts/batch-import.ts` | `tsx` via `npm run batch-import` | Same pattern, batch wrapper |
| `scripts/validate-calculator.ts` | `tsx` via `npm run validate-calculator` | Compares calculator vs profiler output |
| `scripts/batch-validate.ts` | `tsx` via `npm run batch-validate` | Batch validation wrapper |

### 3.4 Profiling Infrastructure (Python)

| Script | Purpose |
|--------|---------|
| `scripts/profile-vllm-bench-v3.py` | Single-model vLLM profiling with internal metrics capture |
| `scripts/batch-profile-bench-v3.py` | Batch profiling from YAML configs |
| `scripts/compare-batch-estimates-v3.py` | Actual vs predicted comparison analysis |
| `scripts/lib/calculator_formulas.py` | Python port of calculator formulas for validation |
| `scripts/lib/model_loader.py` | Loads model params from `models.json` |

### 3.5 Lessons from Existing CLI

**What works well:**
- `commander` for argument parsing — clean flag definitions with type coercion
- Shared library pattern under `scripts/lib/` for reusable logic
- `--dry-run` and `--force` flags for safe scripting
- HuggingFace SDK integration for model validation

**What to improve:**
- No unified CLI namespace — each script is a standalone `tsx` invocation
- No machine-readable output control (`--output json|table|yaml`)
- Relative path resolution via `__dirname` is fragile across invocation contexts
- No `--quiet` mode for pipeline use
- Mixed TypeScript/Python without clear CLI-level orchestration

## 4. CLI Architecture

### 4.1 Entry Point

```
llm-sizer <command> [subcommand] [options]
```

Registered in `package.json` as:
```json
{
  "bin": { "llm-sizer": "./dist/cli.js" }
}
```

### 4.2 Global Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output`, `-o` | `json\|table\|yaml` | `table` (TTY) / `json` (pipe) | Output format |
| `--quiet`, `-q` | boolean | `false` | Suppress informational output, emit only result |
| `--no-color` | boolean | `false` | Disable chalk coloring |
| `--version`, `-V` | boolean | — | Print version and exit |
| `--help`, `-h` | boolean | — | Show help |

Auto-detect TTY: when stdout is a pipe, default to `json` output. When interactive, default to `table`.

### 4.3 Command Tree

```
llm-sizer
├── calculate         # Memory/performance calculations
├── docker            # Docker run / compose generation
│   ├── run           # Generate docker run command
│   └── compose       # Generate docker-compose.yml
├── serve             # Inference server launch commands
│   ├── vllm          # vLLM launch command
│   └── sglang        # SGLang launch command
├── benchmark         # Benchmark command generation
│   ├── generate      # Generate benchmark commands/configs
│   └── analyze       # Ingest and compare benchmark results
├── profile           # Memory profiling command generation
│   ├── generate      # Generate profiler commands or YAML configs
│   └── compare       # Compare profiling results vs calculator
├── models            # Model database operations
│   ├── list          # List/search models
│   ├── show          # Show model details
│   └── import        # Import from HuggingFace (wraps existing tool)
└── gpus              # GPU database operations
    ├── list          # List/search GPUs
    ├── show          # Show GPU details
    └── compare       # Side-by-side GPU comparison
```

## 5. Command Specifications

### 5.1 `llm-sizer calculate`

Run the memory calculation engine from the command line.

```bash
llm-sizer calculate \
  --model llama-3-70b \
  --gpu mi300x \
  --gpus 2 \
  --quant fp8 \
  --kv-quant fp8_bf16 \
  --batch-size 32 \
  --seq-length 4096 \
  --users 8 \
  --output json
```

**Flags:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--model`, `-m` | string | yes* | — | Model ID from database |
| `--custom-params` | number | — | — | Custom model: parameter count (billions) |
| `--custom-hidden` | number | — | — | Custom model: hidden size |
| `--custom-layers` | number | — | — | Custom model: number of layers |
| `--custom-heads` | number | — | — | Custom model: number of attention heads |
| `--gpu`, `-g` | string | yes | — | GPU ID from database |
| `--custom-vram` | number | — | — | Override GPU VRAM (GB) |
| `--gpus`, `-n` | number | no | 1 | Number of GPUs |
| `--quant` | `fp16\|fp8\|int8\|int4` | no | `fp16` | Inference quantization |
| `--kv-quant` | `fp16_bf16\|fp8_bf16\|int8` | no | `fp16_bf16` | KV cache quantization |
| `--type` | `generation\|embedding\|reranking` | no | `generation` | Model type |
| `--batch-size` | number | no | 1 | Batch size |
| `--seq-length` | number | no | 4096 | Sequence length |
| `--users` | number | no | 1 | Concurrent users |

*Required unless `--custom-params` is provided.

**JSON Output Schema:**
```json
{
  "status": "okay",
  "totalVRAM": 384.0,
  "usedVRAM": 142.3,
  "vramPercentage": 37.1,
  "memoryBreakdown": {
    "baseWeights": 70.0,
    "kvCache": 48.2,
    "activations": 12.8,
    "frameworkOverhead": 10.5,
    "multiGPUOverhead": 0.8
  },
  "performance": {
    "generationSpeed": 45,
    "totalThroughput": 360,
    "perUserSpeed": 45
  },
  "inputs": {
    "model": "llama-3-70b",
    "gpu": "mi300x",
    "numGPUs": 2,
    "quantization": "fp8"
  }
}
```

**Table Output:**
```
Model: Llama 3 70B | GPU: 2x MI300X | Quant: FP8

Memory Usage: 142.3 / 384.0 GB (37.1%)  [OK]

  Base Weights     70.0 GB  ████████████████████
  KV Cache         48.2 GB  █████████████
  Activations      12.8 GB  ███
  Framework OH     10.5 GB  ███
  Multi-GPU OH      0.8 GB  ▏

Performance: 360 tok/s total | 45 tok/s per user
```

### 5.2 `llm-sizer docker`

Generate container deployment configurations. Wraps existing `dockerCommandBuilder.ts` and `dockerComposeBuilder.ts`.

#### 5.2.1 `llm-sizer docker run`

```bash
llm-sizer docker run \
  --model llama-3-70b \
  --gpu mi300x \
  --gpus 4 \
  --engine vllm \
  --image rocm/vllm:latest \
  --hf-token-env HF_TOKEN \
  --port 8000 \
  --no-comments \
  > deploy.sh
```

**Additional Flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--engine` | `vllm` | `vllm` | Inference engine |
| `--image` | string | auto-select | Container image |
| `--hf-token-env` | string | `HF_TOKEN` | Env var name for HF token |
| `--port` | number | 8000 | Host port |
| `--container-toolkit` | boolean | `true` | Use AMD Container Toolkit |
| `--no-comments` | boolean | `false` | Omit comments from output |
| `--shm-size` | string | auto | Shared memory size |
| `--volume` | string[] | — | Additional volume mounts |
| `--restart` | `no\|always\|unless-stopped` | `unless-stopped` | Restart policy |

Output: Executable bash script to stdout.

#### 5.2.2 `llm-sizer docker compose`

```bash
llm-sizer docker compose \
  --model llama-3-70b \
  --gpu mi300x \
  --gpus 4 \
  > docker-compose.yml
```

Output: Docker Compose YAML to stdout, with optional `.env` template via `--env-file`.

### 5.3 `llm-sizer serve`

Generate bare-metal inference server launch commands (no Docker).

#### 5.3.1 `llm-sizer serve vllm`

```bash
llm-sizer serve vllm \
  --model meta-llama/Llama-3-70B \
  --gpu mi300x \
  --gpus 2 \
  --quant awq \
  --max-model-len 8192 \
  --port 8000
```

Output:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-70B \
  --tensor-parallel-size 2 \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype float16 \
  --port 8000
```

**Flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model`, `-m` | string | yes | HuggingFace model ID |
| `--gpu`, `-g` | string | yes | GPU ID (for memory calculation) |
| `--gpus`, `-n` | number | 1 | Number of GPUs → `--tensor-parallel-size` |
| `--quant` | string | — | Quantization method |
| `--kv-cache-dtype` | string | `auto` | KV cache data type |
| `--max-model-len` | number | model default | Max sequence length |
| `--port` | number | 8000 | API server port |
| `--mem-util` | number | 0.90 | `--gpu-memory-utilization` |
| `--enforce-eager` | boolean | `false` | Disable CUDA graph capture |
| `--trust-remote-code` | boolean | `false` | Trust remote code |
| `--extra-args` | string | — | Pass-through additional engine args |

Uses the calculator to validate the model fits in GPU memory at the specified configuration and warns if VRAM usage exceeds safe thresholds.

#### 5.3.2 `llm-sizer serve sglang`

```bash
llm-sizer serve sglang \
  --model meta-llama/Llama-3-70B \
  --gpu mi300x \
  --gpus 2 \
  --port 30000
```

Output:
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-70B \
  --tp 2 \
  --mem-fraction-static 0.88 \
  --port 30000
```

SGLang uses different flag names (`--tp` not `--tensor-parallel-size`, `--mem-fraction-static` not `--gpu-memory-utilization`, `--model-path` not `--model`). The CLI abstracts this behind a consistent interface and translates to engine-specific syntax.

### 5.4 `llm-sizer benchmark`

Generate benchmark test commands and configurations for vLLM/SGLang performance testing.

#### 5.4.1 `llm-sizer benchmark generate`

```bash
# Single benchmark command
llm-sizer benchmark generate \
  --model llama-3-70b \
  --gpu mi300x \
  --gpus 2 \
  --engine vllm \
  --suite throughput

# Sweep: generate matrix of configurations
llm-sizer benchmark generate \
  --model llama-3-70b \
  --gpu mi300x \
  --gpus 2 \
  --engine vllm \
  --suite sweep \
  --batch-sizes 1,4,16,64 \
  --seq-lengths 512,2048,8192 \
  --output json > bench-config.json
```

**Flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model`, `-m` | string | yes | Model ID |
| `--gpu`, `-g` | string | yes | GPU ID |
| `--gpus`, `-n` | number | 1 | Number of GPUs |
| `--engine` | `vllm\|sglang` | `vllm` | Target engine |
| `--suite` | `throughput\|latency\|sweep\|scaling` | `throughput` | Benchmark type |
| `--batch-sizes` | string | auto | Comma-separated batch sizes for sweep |
| `--seq-lengths` | string | auto | Comma-separated sequence lengths for sweep |
| `--num-prompts` | number | 1000 | Number of prompts |
| `--request-rate` | number | `inf` | Requests/sec (for latency suite) |
| `--warmup` | number | 3 | Warmup iterations |

**Suite Types:**
- `throughput`: Max throughput test (`benchmark_throughput.py`)
- `latency`: Fixed-rate latency test (`benchmark_serving.py` with `--request-rate`)
- `sweep`: Generate a matrix of (batch_size × seq_length) configurations
- `scaling`: Generate commands for 1x, 2x, 4x, 8x GPU counts

**Sweep JSON Output:**
```json
{
  "model": "llama-3-70b",
  "gpu": "mi300x",
  "engine": "vllm",
  "tests": [
    {
      "batch_size": 1,
      "seq_length": 512,
      "command": "python -m vllm.entrypoints.openai.api_server ...",
      "estimated_vram_pct": 28.5,
      "fits": true
    },
    {
      "batch_size": 64,
      "seq_length": 8192,
      "command": "...",
      "estimated_vram_pct": 94.2,
      "fits": false
    }
  ]
}
```

Each test entry includes a `fits` check from the calculator, so infeasible configurations are flagged before running.

#### 5.4.2 `llm-sizer benchmark analyze`

```bash
# Compare results against calculator predictions
llm-sizer benchmark analyze \
  --results bench-results.json \
  --model llama-3-70b \
  --gpu mi300x

# Compare two result sets (historical delta)
llm-sizer benchmark analyze \
  --results run-feb.json \
  --baseline run-jan.json

# Output as JSON for further processing
llm-sizer benchmark analyze \
  --results bench-results.json \
  --compare predicted \
  --output json
```

**Flags:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--results` | path | yes | Benchmark results file (JSON) |
| `--baseline` | path | no | Baseline results for comparison |
| `--compare` | `predicted\|baseline` | no | Comparison mode |
| `--model`, `-m` | string | no | Override model ID for prediction |
| `--gpu`, `-g` | string | no | Override GPU ID for prediction |
| `--threshold` | number | 10 | Delta percentage to flag as significant |

**Comparison Output (table):**
```
Model: Llama 3 70B | GPU: 2x MI300X

Metric                    Actual    Predicted    Delta
───────────────────────────────────────────────────────
Throughput (tok/s)          342        360       -5.0%
TTFT (ms)                    45         38      +18.4%  ⚠
Per-user speed (tok/s)       43         45       -4.4%
Peak VRAM (GB)            148.2      142.3       +4.1%
```

### 5.5 `llm-sizer profile`

Generate commands for the v3 memory profiling scripts and analyze their output against calculator predictions.

#### 5.5.1 `llm-sizer profile generate`

Generate single profiler commands or batch profiling YAML configurations.

```bash
# Generate a single profiler command
llm-sizer profile generate \
  --model llama-3-70b \
  --input-len 1024 \
  --output-len 512 \
  --batch-size 8 \
  --kv-cache-dtype fp8 \
  --results-dir results/llama-70b

# Generate a sweep across batch sizes
llm-sizer profile generate \
  --model llama-3-70b \
  --input-len 1024 \
  --output-len 512 \
  --batch-sizes 1,4,8,16,32 \
  --results-dir results/llama-70b-sweep

# Generate a YAML config for batch-profile-bench-v3.py
llm-sizer profile generate \
  --models llama-3-70b,mistral-7b,qwen2.5-7b \
  --input-lens 256,512,1024 \
  --output-lens 256,512 \
  --batch-sizes 1,8,16 \
  --format yaml \
  > profiling-campaign.yaml

# Generate YAML config with multi-GPU models
llm-sizer profile generate \
  --models llama-3-70b,deepseek-r1 \
  --input-lens 1024 \
  --output-lens 512 \
  --batch-sizes 1,4,8 \
  --gpus 4 \
  --format yaml \
  > multi-gpu-study.yaml
```

**Flags:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--model`, `-m` | string | yes† | — | Single model ID from database |
| `--models` | string | yes† | — | Comma-separated model IDs (for YAML/batch) |
| `--input-len` | number | yes† | — | Single input length |
| `--input-lens` | string | yes† | — | Comma-separated input lengths (for sweep) |
| `--output-len` | number | yes† | — | Single output length |
| `--output-lens` | string | yes† | — | Comma-separated output lengths (for sweep) |
| `--batch-size` | number | no† | 1 | Single batch size |
| `--batch-sizes` | string | no† | — | Comma-separated batch sizes (for sweep) |
| `--dtype` | `float16\|bfloat16` | no | `float16` | Model data type |
| `--kv-cache-dtype` | `fp8\|null` | no | `null` | KV cache data type |
| `--quant` | string | no | `null` | Quantization method |
| `--gpus`, `-n` | number | no | 1 | Tensor parallel size |
| `--container` | string | no | `vllm-inference` | Container name for profiler |
| `--trust-remote-code` | boolean | no | `false` | Trust remote code |
| `--enforce-eager` | boolean | no | `false` | Disable CUDA graphs |
| `--format` | `command\|yaml\|shell` | no | `command` | Output format |
| `--results-dir` | string | no | `results/` | Results output directory |

†Either singular or plural form required depending on format.

**Output Formats:**

`--format command` (default): Single profiler invocation
```bash
python scripts/profile-vllm-bench-v3.py \
  --hf-model-id meta-llama/Llama-3-70B \
  --input-len 1024 \
  --output-len 512 \
  --batch-size 8 \
  --dtype float16 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --output results/llama-70b/profile-llama-3-70b-in1024-out512-bs8.json
```

`--format shell`: Executable shell script with multiple invocations (for sweeps)
```bash
#!/bin/bash
set -euo pipefail
# Memory Profiling Study: llama-3-70b
# Generated: 2026-02-28T12:00:00Z
RESULTS_DIR="results/llama-70b-sweep"
mkdir -p "$RESULTS_DIR"

echo "=== Profiling llama-3-70b (batch_size=1) ==="
python scripts/profile-vllm-bench-v3.py \
  --hf-model-id meta-llama/Llama-3-70B \
  --input-len 1024 --output-len 512 --batch-size 1 \
  --dtype float16 \
  --output "$RESULTS_DIR/profile-bs1.json"

echo "=== Profiling llama-3-70b (batch_size=4) ==="
python scripts/profile-vllm-bench-v3.py \
  --hf-model-id meta-llama/Llama-3-70B \
  --input-len 1024 --output-len 512 --batch-size 4 \
  --dtype float16 \
  --output "$RESULTS_DIR/profile-bs4.json"
# ... etc
```

`--format yaml`: Batch profiler config for `batch-profile-bench-v3.py`
```yaml
models:
- hf_model_id: meta-llama/Llama-3-70B
  container_name: vllm-inference
  tensor_parallel_size: 1
  input_lengths: [256, 512, 1024]
  output_lengths: [256, 512]
  batch_sizes: [1, 8, 16]
  dtype: float16
  kv_cache_dtype: fp8
  quantization: null
  trust_remote_code: false
- hf_model_id: mistralai/Mistral-7B-Instruct-v0.3
  container_name: vllm-inference
  tensor_parallel_size: 1
  input_lengths: [256, 512, 1024]
  output_lengths: [256, 512]
  batch_sizes: [1, 8, 16]
  dtype: float16
  kv_cache_dtype: null
  quantization: null
  trust_remote_code: false
```

**Model ID Resolution:** The `--model` flag accepts llm-sizer model IDs (e.g., `llama-3-70b`) and resolves them to HuggingFace model IDs via `models.json` for the `--hf-model-id` flag in the profiler. It also resolves architecture parameters to determine appropriate `tensor_parallel_size` for large models (>65B parameters default to multi-GPU). If a model requires `trust_remote_code`, this is inherited from the model database entry if available.

#### 5.5.2 `llm-sizer profile compare`

Wraps the existing `compare-batch-estimates-v3.py` functionality with consistent CLI ergonomics.

```bash
# Compare profiling results against calculator estimates
llm-sizer profile compare \
  --results results/batch-bench-results-v3.csv

# With custom threshold
llm-sizer profile compare \
  --results results/batch-bench-results-v3.csv \
  --threshold 15

# Detailed component-level comparison
llm-sizer profile compare \
  --results results/profile-llama-70b.json \
  --detailed \
  --output json
```

**Flags:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--results` | path | yes | Profiling results (JSON or CSV) |
| `--threshold` | number | 10 | Delta percentage to flag as significant |
| `--detailed` | boolean | `false` | Show per-component breakdown |

**Output:**
```
Profiling Comparison: Calculator vs Actual (v3.0 profiler)

Model                     Component    Calculated  Actual   Delta
─────────────────────────────────────────────────────────────────
Llama-3.2-1B (bs=1)       Weights        2.47 GB   2.47 GB   0.0%
                          KV Cache       0.13 GB   7.28 GB  ⚠ +5500%
                          Activations    0.82 GB   0.82 GB   0.0%
                          Total         13.42 GB  19.32 GB  +44.0%
Llama-3.2-1B (bs=8)       ...
```

### 5.6 `llm-sizer models`

Query and manage the model database.

#### 5.6.1 `llm-sizer models list`

```bash
llm-sizer models list
llm-sizer models list --type generation --filter "70b"
llm-sizer models list --type embedding --output json
```

**Flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--type` | `generation\|embedding\|reranking` | all | Filter by model type |
| `--filter` | string | — | Case-insensitive substring match on name/ID |
| `--arch` | `transformer\|moe` | all | Filter by architecture |
| `--modality` | `text\|multimodal` | all | Filter by modality |

#### 5.6.2 `llm-sizer models show`

```bash
llm-sizer models show llama-3-70b
```

Output: Full model spec (parameters, hidden size, layers, heads, context length, HF model ID, architecture, modality).

#### 5.6.3 `llm-sizer models import`

Wraps existing `import-hf-model.ts` as a subcommand.

```bash
llm-sizer models import --model meta-llama/Llama-3.3-70B
llm-sizer models import --url https://huggingface.co/meta-llama/Llama-3.3-70B
llm-sizer models import --model meta-llama/Llama-3.3-70B --dry-run
```

Same flags as current `import-hf-model.ts` (`--model`, `--url`, `--file`, `--dry-run`, `--params`, `--context`, `--force`).

### 5.7 `llm-sizer gpus`

Query the GPU database.

#### 5.7.1 `llm-sizer gpus list`

```bash
llm-sizer gpus list
llm-sizer gpus list --vendor AMD --tier datacenter
llm-sizer gpus list --output json | jq '.[].id'
```

**Flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--vendor` | `AMD\|NVIDIA` | all | Filter by vendor |
| `--tier` | `datacenter\|professional\|consumer` | all | Filter by tier |
| `--min-vram` | number | — | Minimum VRAM (GB) |
| `--partitioning` | boolean | — | Only GPUs with partition support |

#### 5.7.2 `llm-sizer gpus show`

```bash
llm-sizer gpus show mi300x
```

Output: Full GPU spec including VRAM, bandwidth, compute, TDP, partition modes.

#### 5.7.3 `llm-sizer gpus compare`

```bash
llm-sizer gpus compare mi300x mi325x a100-80
```

Output: Side-by-side comparison table of key specs.

## 6. Implementation Plan

### Phase 1: Foundation & Core Commands

**Scope:** CLI framework, `calculate`, `models`, `gpus`

1. Create `src/cli/` directory structure:
   ```
   src/cli/
   ├── index.ts              # Entry point, commander setup
   ├── commands/
   │   ├── calculate.ts
   │   ├── models.ts
   │   └── gpus.ts
   ├── formatters/
   │   ├── table.ts          # chalk-based table output
   │   └── json.ts           # JSON serialization
   └── utils/
       ├── data-loader.ts    # Load JSON data files
       ├── model-resolver.ts # Resolve model IDs to full specs
       └── output.ts         # TTY detection, format selection
   ```
2. Configure `tsup` (or `esbuild`) for CLI-specific build, separate from Vite
3. Add `"bin"` entry and `"build:cli"` script to `package.json`
4. Implement `calculate` command wiring to existing `memoryCalculator.ts` / `performanceEstimator.ts`
5. Implement `models list|show|import` and `gpus list|show|compare`

**Tests:**
- Unit tests for `data-loader.ts` and `model-resolver.ts`
- Integration tests for `calculate` command with known model/GPU combinations (snapshot against expected output)
- Test JSON output can be parsed and contains required fields
- Test table output contains key values
- Test TTY auto-detection (JSON when piped, table when interactive)
- Test `--filter` and `--vendor`/`--tier` filtering for models/gpus

### Phase 2: Docker & Serve Commands

**Scope:** `docker run|compose`, `serve vllm|sglang`

1. Implement `docker run` command, wiring to existing `generateDockerRunCommand()`
2. Implement `docker compose` command, wiring to existing `generateDockerCompose()`
3. Build `serve` command framework with engine abstraction:
   ```
   src/cli/
   └── engines/
       ├── vllm.ts           # vLLM flag mapping
       └── sglang.ts         # SGLang flag mapping
   ```
4. Add SGLang parameter definitions (new data file or inline)
5. Implement memory-fit validation before generating serve commands

**Tests:**
- Snapshot tests for docker run output (with and without `--no-comments`)
- Snapshot tests for docker compose YAML output
- Verify vLLM serve output contains correct `--tensor-parallel-size`, `--quantization` flags
- Verify SGLang serve output uses `--tp`, `--mem-fraction-static` (correct flag names)
- Test memory-fit warnings trigger at appropriate thresholds
- Test `--extra-args` pass-through

### Phase 3: Benchmark & Profile Commands

**Scope:** `benchmark generate|analyze`, `profile generate|compare`

1. Implement `benchmark generate` with suite types (throughput, latency, sweep, scaling)
2. Implement `benchmark analyze` with results parsing for vLLM/SGLang JSON output formats
3. Implement `profile generate` with all three output formats (command, shell, yaml)
4. Implement `profile compare` wrapping `compare-batch-estimates-v3.py` logic
5. Build results parsers for profiler JSON and CSV output

**Tests:**
- Snapshot tests for generated benchmark commands per suite type
- Snapshot tests for generated profiler commands (command format)
- Verify generated YAML configs match `batch-profile-bench-v3.py` expected schema
- Verify generated shell scripts are valid bash (shellcheck if available)
- Test sweep matrix generation produces correct (batch × seq_length) cartesian product
- Test scaling suite generates correct GPU count progression
- Test profile compare output with sample profiling result fixtures
- Test model ID to HF model ID resolution for profiler commands
- Test multi-model YAML generation

### Phase 4: Polish & Integration

**Scope:** Cross-cutting concerns, CI, documentation

1. Add shell completions (bash/zsh) via commander's built-in support
2. Add `--dry-run` support to all write/generate commands
3. Error handling: consistent exit codes (0=success, 1=error, 2=validation warning)
4. Add man page generation or comprehensive `--help` text for all commands

## 7. Build & Distribution

### 7.1 Build Configuration

Separate build target from the Vite web build:

```json
{
  "scripts": {
    "build:cli": "tsup src/cli/index.ts --format esm --target node20 --dts",
    "dev:cli": "tsx src/cli/index.ts"
  }
}
```

`tsup` handles tree-shaking React/browser dependencies out of the CLI bundle.

### 7.2 Package.json Updates

```json
{
  "bin": {
    "llm-sizer": "./dist/cli.js"
  },
  "scripts": {
    "build": "tsc && vite build",
    "build:cli": "tsup src/cli/index.ts --format esm --target node20",
    "build:all": "npm run build && npm run build:cli",
    "dev:cli": "tsx src/cli/index.ts"
  }
}
```

### 7.3 Development Workflow

During development, use `tsx` for direct execution:
```bash
npx tsx src/cli/index.ts calculate --model llama-3-70b --gpu mi300x
```

After build:
```bash
npm link  # or npx . calculate --model llama-3-70b --gpu mi300x
```

## 8. Testing Strategy

### 8.1 Test Structure

```
tests/
├── cli/
│   ├── commands/
│   │   ├── calculate.test.ts
│   │   ├── docker.test.ts
│   │   ├── serve.test.ts
│   │   ├── benchmark.test.ts
│   │   ├── profile.test.ts
│   │   ├── models.test.ts
│   │   └── gpus.test.ts
│   ├── formatters/
│   │   ├── table.test.ts
│   │   └── json.test.ts
│   ├── utils/
│   │   ├── data-loader.test.ts
│   │   ├── model-resolver.test.ts
│   │   └── output.test.ts
│   └── integration/
│       ├── cli-e2e.test.ts       # End-to-end CLI invocation tests
│       └── fixtures/
│           ├── profiling-results-sample.json
│           ├── benchmark-results-sample.json
│           └── expected-outputs/
└── utils/                         # Existing calculator tests (unchanged)
```

### 8.2 Test Categories

**Unit Tests:**
- Data loading and model/GPU resolution
- Output formatting (table rendering, JSON serialization)
- Engine flag translation (vLLM ↔ SGLang parameter mapping)
- Sweep matrix generation
- YAML config generation
- Results parsing

**Integration Tests:**
- Full command execution via `commander.parseAsync()` with captured stdout
- Snapshot tests for command outputs across all formats
- Pipeline tests: `calculate --output json | jq .status` returns expected value

**Fixture-Based Tests:**
- Sample profiling result JSON files for `profile compare`
- Sample benchmark result files for `benchmark analyze`
- Expected YAML configs for `profile generate --format yaml`

### 8.3 Existing Test Updates

- No changes to existing calculator unit tests (`src/utils/` functions remain unchanged)
- Add test coverage for any new shared utilities created during CLI development
- Ensure `npm test` runs both existing and new CLI tests

## 9. Documentation

### 9.1 New Documentation

| Document | Location | Content |
|----------|----------|---------|
| CLI README | `docs/cli/README.md` | Installation, quick start, all command references |
| CLI examples | `docs/cli/examples.md` | Workflow examples: sizing → deploy → profile → analyze |
| Profiling workflow | `docs/cli/profiling-workflow.md` | End-to-end guide: generate configs → run profiler → compare results |

### 9.2 Documentation Updates

| Document | Update |
|----------|--------|
| Root `README.md` | Add CLI section with installation and basic usage |
| `CLAUDE.md` | Add CLI-specific development commands (`build:cli`, `dev:cli`), project structure updates for `src/cli/` |
| `scripts/README.md` | Note that standalone scripts are now also available via `llm-sizer` CLI |
| `scripts/PROFILER-V3-README.md` | Cross-reference `llm-sizer profile generate` as preferred way to create configs |

### 9.3 Inline Help

All commands provide comprehensive `--help` output with examples:
```
$ llm-sizer profile generate --help

Generate profiler commands or batch profiling YAML configs

Usage: llm-sizer profile generate [options]

Options:
  -m, --model <id>         Model ID from database
  --models <ids>           Comma-separated model IDs (for YAML/batch)
  --input-len <n>          Input sequence length
  ...

Examples:
  # Single profiler command
  $ llm-sizer profile generate --model llama-3-70b --input-len 1024 --output-len 512 --batch-size 8

  # Batch config YAML
  $ llm-sizer profile generate --models llama-3-70b,mistral-7b --format yaml > study.yaml

  # Sweep across batch sizes
  $ llm-sizer profile generate --model llama-3-70b --batch-sizes 1,4,8,16 --format shell > sweep.sh
```

## 10. Out of Scope (v1)

- npm package publishing (local/linked install only for now)
- Interactive/TUI mode (e.g., `llm-sizer interactive` with prompts)
- Direct profiler execution (the CLI generates commands; execution is manual or via existing scripts)
- SGLang benchmark result parsing (vLLM first, SGLang in a future version)
- Web UI changes (the CLI is additive, no UI modifications)

## 11. Success Criteria

1. `llm-sizer calculate` produces identical numerical results to the web UI for the same inputs
2. `llm-sizer docker run` output is functionally equivalent to the Container Config tab export
3. `llm-sizer profile generate --format yaml` produces configs that `batch-profile-bench-v3.py` accepts without modification
4. All commands support `--output json` and produce parseable, stable JSON schemas
5. `npm test` passes with >80% coverage on new CLI code
6. A user can go from "which model fits?" → "generate profiling config" → "compare results" entirely via CLI
