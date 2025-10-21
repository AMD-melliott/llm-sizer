# Quick Start Guide - HuggingFace Model Import

## Installation

If you're pulling the project fresh or after updates, make sure dependencies are installed:

```bash
npm install
```

The script uses the official [@huggingface/hub](https://www.npmjs.com/package/@huggingface/hub) SDK and is now ready to use!

## Basic Usage

```bash
# Import by model ID (recommended)
npm run import-model -- --model Qwen/Qwen2.5-7B

# Import by URL
npm run import-model -- --url https://huggingface.co/Qwen/Qwen2.5-7B

# Dry run (preview only)
npm run import-model -- --model Qwen/Qwen2.5-7B --dry-run

# Skip confirmation
npm run import-model -- --model Qwen/Qwen2.5-7B --force
```

## Common Scenarios

### 1. Import a new model

```bash
npm run import-model -- --model microsoft/Phi-3-mini-4k-instruct
```

### 2. Preview before importing

```bash
npm run import-model -- --model microsoft/Phi-3-mini-4k-instruct --dry-run
```

### 3. Model with gated access (requires auth)

For models that require authentication (like Llama), set your HuggingFace token first:

```bash
export HF_TOKEN=hf_your_token_here
npm run import-model -- --model meta-llama/Llama-3.2-1B
```

**Get your token**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> **Note**: If you can't authenticate, you can still provide parameters manually as a fallback:
> ```bash
> npm run import-model -- --model meta-llama/Llama-3.1-8B --params 8 --context 128000
> ```

### 4. Override auto-detected values

```bash
npm run import-model -- --model mistralai/Mistral-7B-v0.3 --context 32768
```

## What Gets Imported

The script automatically extracts:

✅ Model name and ID  
✅ Parameter count (in billions)  
✅ Hidden size/dimension  
✅ Number of layers  
✅ Number of attention heads  
✅ Context length  
✅ Architecture type (transformer/moe)  
✅ Optional: KV heads, intermediate size, vocab size, expert config  

## Example Output

```
📦 Importing model: Qwen/Qwen2.5-7B

Model ID: Qwen/Qwen2.5-7B

Validating model exists...
✓ Model found

Fetching model configuration...
✓ Config fetched

Fetching parameter count...
✓ Found 7B parameters

Extracted Model Information:
  ID:              qwen2-5-7b
  Name:            Qwen2.5 7B
  Parameters:      7B
  Hidden Size:     3584
  Layers:          28
  Attention Heads: 28
  Context Length:  131072
  Architecture:    transformer
  KV Heads:        4
  Intermediate:    18944
  Vocab Size:      152064

Validating model data...
✓ Validation passed

Changes to models.json:

+ Added models:
  + qwen2-5-7b (Qwen2.5 7B) - 7B params

Total models: 14 → 15

Proceed with import? (y/N)
```

## Troubleshooting

### "Model not found"
The model doesn't exist or is private. Check the model ID.

### "Could not determine parameter count"
Provide it manually: `--params <number>`

### "HTTP 401: Unauthorized" or "Authentication required"
Model requires authentication (gated model). Set your HuggingFace token:

```bash
export HF_TOKEN=hf_your_token_here
npm run import-model -- --model meta-llama/Llama-3.2-1B
```

Get your token at: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Alternative**: Provide values manually with `--params` and `--context`, or use a public model instead.

### "Validation failed"
Check the error message. Usually missing required fields. Use manual overrides.

## Need Help?

See the full documentation in `scripts/README.md`
