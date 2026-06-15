# CLI Tool Testing Walkthrough

Quick walkthrough of the Phase 1 CLI features. All commands use `npx tsx src/cli/index.ts` for dev mode (no build step needed).

## Prerequisites

```bash
cd /home/melliott/git/llm-sizer
npm install   # if not already done
```

## 1. Help & Discovery

```bash
# Top-level help
npx tsx src/cli/index.ts --help

# Subcommand help
npx tsx src/cli/index.ts calculate --help
npx tsx src/cli/index.ts models list --help
npx tsx src/cli/index.ts gpus compare --help
```

## 2. Memory Calculation

### Basic sizing — does a 70B model fit on a single MI300X?

```bash
npx tsx src/cli/index.ts calculate -m llama-3-70b -g mi300x --quant fp16
```

Expected: Shows a warning — fp16 weights alone are ~140GB on a 192GB card. High VRAM usage.

### Fix it with quantization and multi-GPU

```bash
npx tsx src/cli/index.ts calculate -m llama-3-70b -g mi300x -n 2 --quant fp8
```

Expected: ~25% VRAM usage on 2x MI300X with FP8. Status shows OK.

### Increase workload — batch size and concurrent users

```bash
npx tsx src/cli/index.ts calculate -m llama-3-70b -g mi300x -n 2 --quant fp8 --batch-size 32 --seq-length 8192 --users 8
```

Expected: KV cache grows significantly. Check if it still fits.

### JSON output for scripting

```bash
npx tsx src/cli/index.ts calculate -m llama-3-70b -g mi300x -n 2 --quant fp8 -o json
```

Expected: Pretty-printed JSON with `totalVRAM`, `usedVRAM`, `vramPercentage`, `memoryBreakdown`, `performance`, and `inputs` fields.

### Pipe to jq

```bash
npx tsx src/cli/index.ts calculate -m llama-3-70b -g mi300x -n 2 --quant fp8 -o json | jq '.memoryBreakdown'
```

## 3. Model Database Queries

### List all generation models

```bash
npx tsx src/cli/index.ts models list
```

### Filter by name

```bash
npx tsx src/cli/index.ts models list --filter "70b"
npx tsx src/cli/index.ts models list --filter "llama"
npx tsx src/cli/index.ts models list --filter "deepseek"
```

### Filter MoE models

```bash
npx tsx src/cli/index.ts models list --arch moe
```

### List embedding models

```bash
npx tsx src/cli/index.ts models list --type embedding
```

### List reranking models

```bash
npx tsx src/cli/index.ts models list --type reranking
```

### Show full model details

```bash
npx tsx src/cli/index.ts models show llama-3-70b
```

### JSON output for scripting — count models

```bash
npx tsx src/cli/index.ts models list -o json | jq 'length'
```

## 4. GPU Database Queries

### List all GPUs

```bash
npx tsx src/cli/index.ts gpus list
```

### Filter by vendor

```bash
npx tsx src/cli/index.ts gpus list --vendor AMD
npx tsx src/cli/index.ts gpus list --vendor NVIDIA
```

### Filter datacenter GPUs with 80GB+ VRAM

```bash
npx tsx src/cli/index.ts gpus list --tier datacenter --min-vram 80
```

### GPUs with partition support

```bash
npx tsx src/cli/index.ts gpus list --partitioning
```

### Show GPU details

```bash
npx tsx src/cli/index.ts gpus show mi300x
```

### Compare GPUs side by side

```bash
npx tsx src/cli/index.ts gpus compare mi300x a100-80gb h100-sxm
```

Expected: Side-by-side table with VRAM, bandwidth, compute, TDP, etc.

## 5. Built CLI Binary

### Build and test the compiled version

```bash
npm run build:cli
node dist/cli.js --version
node dist/cli.js calculate -m llama-3-70b -g mi300x -n 2 --quant fp8
```

## 6. Error Handling

### Unknown model

```bash
npx tsx src/cli/index.ts calculate -m fake-model -g mi300x
```

Expected: Error message "Model not found: fake-model", exit code 1.

### Unknown GPU

```bash
npx tsx src/cli/index.ts calculate -m llama-3-70b -g fake-gpu
```

Expected: Error message "GPU not found: fake-gpu", exit code 1.

### Missing required flags

```bash
npx tsx src/cli/index.ts calculate
```

Expected: Commander shows error about missing required option `--model`.

## 7. Run Automated Tests

```bash
# All tests (existing + new CLI tests)
npm test

# Just CLI tests
npx jest tests/cli/ --verbose

# Specific command tests
npx jest tests/cli/commands/calculate.test.ts --verbose
npx jest tests/cli/commands/models.test.ts --verbose
npx jest tests/cli/commands/gpus.test.ts --verbose
```

Expected: 270 tests, all passing.
