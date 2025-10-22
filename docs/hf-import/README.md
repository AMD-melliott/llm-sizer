# HuggingFace Model Import Script

This script automates the process of importing model specifications from HuggingFace into the LLM Inference Calculator.

**Now using the official [@huggingface/hub](https://www.npmjs.com/package/@huggingface/hub) SDK for improved reliability and authentication support!**

## Features

- ✅ **Official HuggingFace SDK integration** for reliable API access
- ✅ **Authentication support** for gated models (Llama, etc.)
- ✅ Fetch model configurations from HuggingFace
- ✅ Auto-detect parameter counts, architecture, and specifications
- ✅ Validate model data against schema
- ✅ Automatic backup before modifying models.json
- ✅ Sorted output by parameter count
- ✅ Support for both Transformer and MoE architectures
- ✅ Support for multiple config naming conventions (standard & GPT-2 style)
- ✅ Dry-run mode for previewing changes

## Usage

### Basic Import

Import a model using its HuggingFace URL:

```bash
npm run import-model -- --url https://huggingface.co/meta-llama/Llama-3.3-70B
```

Or using the model ID directly:

```bash
npm run import-model -- --model meta-llama/Llama-3.3-70B
```

### With Manual Overrides

Override specific values if auto-detection fails:

```bash
npm run import-model -- --model meta-llama/Llama-3.3-70B \
  --params 70 \
  --context 128000
```

### Dry Run

Preview changes without modifying the file:

```bash
npm run import-model -- --model meta-llama/Llama-3.3-70B --dry-run
```

### Skip Confirmation

Use `--force` to skip the confirmation prompt:

```bash
npm run import-model -- --model meta-llama/Llama-3.3-70B --force
```

### Gated Models (Authentication)

For gated models like Llama, set your HuggingFace token:

```bash
export HF_TOKEN=hf_your_token_here
npm run import-model -- --model meta-llama/Llama-3.2-1B
```

Get your token at: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Options

| Option | Alias | Description |
|--------|-------|-------------|
| `--url <url>` | `-u` | HuggingFace model URL |
| `--model <id>` | `-m` | HuggingFace model ID (e.g., `meta-llama/Llama-3.3-70B`) |
| `--dry-run` | | Preview changes without writing to file |
| `--params <number>` | | Override parameter count (in billions) |
| `--context <number>` | | Override context length |
| `--force` | | Skip confirmation prompts |

## How It Works

1. **Parse Model ID**: Extracts the model ID from URL or uses direct ID
2. **Validate Model**: Checks that the model exists on HuggingFace
3. **Fetch Config**: Downloads `config.json` from the model repository
4. **Fetch Parameters**: Attempts to get parameter count from API
5. **Transform Data**: Converts HuggingFace config to our schema
6. **Validate**: Ensures all required fields are present and valid
7. **Backup**: Creates timestamped backup of `models.json`
8. **Update**: Adds new model and sorts by parameter count

## Supported Model Types

### Transformer Models
- Llama, Mistral, Qwen, Falcon, Yi, etc.
- Auto-detects standard transformer architectures

### MoE (Mixture of Experts) Models
- Mixtral and other MoE architectures
- Captures expert count and experts-per-token

## Field Mapping

The script maps HuggingFace config fields to our schema, supporting multiple naming conventions:

| HuggingFace Config | Alternative | Our Schema |
|-------------------|-------------|------------|
| `hidden_size` | `n_embd` | `hidden_size` |
| `num_hidden_layers` | `n_layer` | `num_layers` |
| `num_attention_heads` | `n_head` | `num_heads` |
| `num_key_value_heads` | - | `num_kv_heads` |
| `max_position_embeddings` | `n_positions`, `n_ctx` | `default_context_length` |
| `intermediate_size` | - | `intermediate_size` |
| `vocab_size` | - | `vocab_size` |
| `num_local_experts`, `num_experts` | - | `num_experts` |
| `num_experts_per_tok` | - | `experts_per_token` |

**Note**: Standard naming (left column) is used by Llama, Mistral, etc. Alternative naming is used by GPT-2 and similar models.

## Examples

### Import Llama 3.3 70B

```bash
npm run import-model -- --model meta-llama/Llama-3.3-70B
```

### Import Mistral 7B with override

```bash
npm run import-model -- --model mistralai/Mistral-7B-v0.1 --context 32768
```

### Import Mixtral (MoE model)

```bash
npm run import-model -- --model mistralai/Mixtral-8x7B-v0.1
```

## Troubleshooting

### Parameter count not found

If the script can't auto-detect the parameter count, provide it manually:

```bash
npm run import-model -- --model <model-id> --params <count-in-billions>
```

### Missing config.json

Some models may not have a `config.json` file. In this case, you'll need to manually provide the values or use a different model source.

### Validation errors

If validation fails, check the error messages. Common issues:
- Missing required fields
- Values out of valid range
- Duplicate model ID

## File Structure

```
scripts/
├── import-hf-model.ts      # Main CLI entry point
├── lib/
│   ├── types.ts           # TypeScript interfaces
│   ├── hf-client.ts       # HuggingFace API client
│   ├── model-parser.ts    # Data transformation logic
│   ├── validator.ts       # Validation logic
│   └── file-handler.ts    # File I/O operations
└── schemas/
    └── model.schema.json   # JSON schema for validation
```

## Batch Import

Import multiple models at once using the batch import script:

### Create a Model List

Create a file `models-to-import.txt`:
```text
# Comments start with #
microsoft/Phi-3.5-mini-instruct
Qwen/Qwen2.5-1.5B-Instruct
google/gemma-2b-it
```

### Run Batch Import

```bash
npm run batch-import -- --file models-to-import.txt
```

Options:
- `--dry-run` - Preview all imports
- `--continue-on-error` - Don't stop on failures
- `--log <path>` - Custom log file path

## Common Issues

### Gated Models (403 Error)

For models like Llama that require access approval:
1. Visit the model page on HuggingFace
2. Click "Agree and access repository"
3. Accept terms and conditions
4. Re-run import after approval

### Multimodal Models

Vision-language models are not supported. The tool will detect and reject these models with a clear error message.

### Model Not Found

Check if:
- Organization name is correct (e.g., `CohereLabs` not `CohereForAI`)
- Model ID matches exactly what's on HuggingFace
- Model hasn't been renamed or moved

## Import Success Statistics

Based on recent imports (October 2025):
- **Success Rate:** 89% for text-only models
- **Common Issues:**
  - 30% - Gated access required
  - 20% - Multimodal models (not supported)
  - 10% - Incorrect model IDs

## Future Enhancements

- [x] Batch import from file
- [x] Better error handling for gated models
- [x] Multimodal model detection
- [ ] MCP server integration for enhanced model discovery
- [ ] Automatic model discovery (trending models)
- [ ] Support for GGUF and quantized formats
- [ ] Performance benchmark import
