# CLI Testing Walkthrough Results

**Date:** 2026-03-02
**Branch:** feature/cli-tool
**Test Suite Version:** 270 tests across 15 suites

## Model Database Update

Before running the walkthrough, the model database was updated from 69 to 108 models (excluding custom). 38 new models were added:

### Imported via batch tool (22 models)
- **DeepSeek:** V3-0324, V3.2, V3.2-Exp
- **Qwen3:** 0.6B, 1.7B, 32B, 30B-A3B (MoE), 235B-A22B (MoE), Coder-480B-A35B (MoE)
- **Mistral:** Ministral-3 3B/8B/14B, Magistral-Small 24B, Devstral-Small-2 24B
- **NVIDIA:** Nemotron-Ultra 253B, Nemotron-Super 49B, Nemotron-Nano 8B, Nemotron-3-Nano 30B (MoE)
- **MiniMax:** M2 230B (MoE), M2.5 230B (MoE)
- **Allen AI:** OLMo-3 7B, OLMo-3 32B
- **GLM:** GLM-5 754B (MoE)

### Imported via tool with param overrides (4 models)
- **Microsoft:** Phi-4-mini 3.8B, Phi-4-reasoning 14.7B, Phi-4-reasoning-plus 14.7B, Phi-4-multimodal 5.6B

### Added manually (12 models)
- **InternVL:** InternVL3-8B, InternVL3-78B, InternVL3.5-38B, InternVL3.5-241B-A28B (all multimodal)
- **Mistral:** Mistral Large 3 675B (MoE)
- **GLM:** GLM-4.7 355B (MoE)
- **Google Gemma 3:** 4B IT, 12B IT (both multimodal)
- **Meta Llama 4:** Scout 109B 16E (MoE, multimodal), Maverick 400B 128E (MoE, multimodal)
- **Cohere:** Command A 111B

## Walkthrough Results

| # | Section | Status | Notes |
|---|---------|--------|-------|
| 1 | Help & Discovery | PASS | `--help`, `calculate --help`, `models list --help`, `gpus compare --help` all display correct usage |
| 2 | Memory Calculation | PASS | Basic sizing shows 90.8% VRAM warning for 70B fp16 on 1x MI300X. Multi-GPU + fp8 drops to 24.7%. Batch/users/seq-length scaling works. JSON output and jq piping work correctly |
| 3 | Model Database Queries | PASS | `models list` shows 108 models. Filters by name (`70b`, `llama`, `deepseek`), architecture (`moe`), type (`embedding`, `reranking`), and modality all work. `models show` and JSON output work |
| 4 | GPU Database Queries | PASS | `gpus list` shows all GPUs. Vendor, tier, min-vram, and partitioning filters work. `gpus show` displays full details including partition modes. `gpus compare` shows side-by-side table |
| 5 | Built CLI Binary | PASS | `npm run build:cli` succeeds. `node dist/cli.js --version` returns `1.0.0`. Calculate command produces identical output to dev mode |
| 6 | Error Handling | PASS | Unknown model: `Error: Model not found: "fake-model"` (exit 1). Unknown GPU: `Error: GPU not found: "fake-gpu"` (exit 1). Missing `--model`: Commander shows required option error (exit 1) |
| 7 | Automated Tests | PASS | 270 tests, 15 suites, all passing in ~1s |

## Issues Found & Fixed

### Fixed: Invalid GPU ID in walkthrough doc
- **File:** `docs/plans/cli-testing-walkthrough.md`
- **Issue:** `gpus compare` example used `h100-80` which is not a valid GPU ID
- **Fix:** Changed to `h100-sxm` (the correct ID for NVIDIA H100 SXM5 80GB)
- **Note:** The `gpus compare` command silently skips unknown GPU IDs rather than erroring — could be improved with a warning

## Test Suite Breakdown

```
PASS tests/cli/commands/gpus.test.ts       (15 tests)
PASS tests/cli/commands/models.test.ts     (14 tests)
PASS tests/cli/commands/calculate.test.ts  ( 6 tests)
PASS tests/cli/utils/model-resolver.test.ts (10 tests)
PASS tests/cli/utils/data-loader.test.ts   ( 4 tests)
PASS tests/cli/utils/output.test.ts        ( 5 tests)
PASS tests/generation.test.ts
PASS tests/multimodal.test.ts
PASS tests/embedding.test.ts
PASS tests/reranking.test.ts
PASS tests/edge-cases.test.ts
PASS tests/partitioning.test.ts
PASS tests/dockerComposeBuilder.test.ts
PASS tests/dockerCommandBuilder.test.ts
PASS tests/containerValidation.test.ts

Test Suites: 15 passed, 15 total
Tests:       270 passed, 270 total
```
