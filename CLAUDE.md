# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Inference Calculator - A web application for estimating LLM memory requirements and performance characteristics for inference workloads on AMD and NVIDIA GPUs.

## Architecture

### Technology Stack (TypeScript/React)
- **Frontend Framework**: React 18+ with TypeScript
- **UI Components**: Tailwind CSS + shadcn/ui or Chakra UI
- **Build Tool**: Vite
- **State Management**: React Context or Zustand
- **Charts**: Recharts or Chart.js

### Core Calculation Engine
The application centers around a memory calculation engine that computes:
- Base model weights memory based on parameter count and quantization
- KV cache requirements based on batch size, sequence length, and users
- Activation memory for inference operations
- Framework and multi-GPU overhead

### Key Data Structures
- **GPU configurations**: Store in `data/gpus.json` with vendor, VRAM, bandwidth, and compute specs
- **Model presets**: Store in `data/models.json` with parameter counts and context lengths
- **Application state**: Managed via React Context/Zustand following the AppState interface from PRD

## Development Commands

### Initial Setup
```bash
# Create project structure
npm create vite@latest . -- --template react-ts
npm install

# Install core dependencies
npm install tailwindcss @tailwindcss/forms recharts
npm install -D @types/react @types/node

# Initialize Tailwind
npx tailwindcss init -p
```

### Development
```bash
npm run dev          # Start development server on http://localhost:5173
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

### Testing
```bash
npm test             # Run all tests
npm run test:watch   # Run tests in watch mode
npm run test:coverage # Generate coverage report
```

## Project Structure

```
llm-sizer/
├── src/
│   ├── components/      # React components
│   │   ├── ModelSelector.tsx
│   │   ├── GPUSelector.tsx
│   │   ├── QuantizationOptions.tsx
│   │   ├── InferenceParameters.tsx
│   │   ├── ResultsDisplay.tsx
│   │   └── MemoryVisualization.tsx
│   ├── hooks/          # Custom React hooks
│   │   └── useMemoryCalculation.ts
│   ├── utils/          # Calculation logic
│   │   ├── memoryCalculator.ts
│   │   └── performanceEstimator.ts
│   ├── data/           # Configuration files
│   │   ├── gpus.json
│   │   └── models.json
│   ├── types/          # TypeScript definitions
│   │   └── index.ts
│   ├── App.tsx
│   └── main.tsx
├── public/
└── package.json
```

## Implementation Guidelines

### Memory Calculation Formula
```typescript
// Base weights calculation
baseWeights = (modelParams * bitsPerParam) / 8 / 1e9  // GB

// KV cache per token (GQA-aware, matches vLLM's kv_cache_interface.py)
// num_kv_heads defaults to num_heads for MHA models
headSize = hiddenSize / numHeads
kvCachePerToken = 2 * numLayers * numKVHeads * headSize * kvBytesPerElement

// Total KV cache
totalKVCache = kvCachePerToken * batchSize * sequenceLength * concurrentUsers / 1e9  // GB

// Activation memory (peak ~one layer, layers processed sequentially)
// With FlashAttention, attention scores aren't materialized
intermediateSize = model.intermediate_size ?? hiddenSize * 4
activations = (batchSize * sequenceLength * intermediateSize * 2) / numGPUs / 1e9

// Framework overhead (baseline + proportional, calibrated from vLLM profiling)
frameworkOverhead = 1.5 + (baseWeights + totalKVCache + activations) * 0.05
                  + (numGPUs > 1 ? 0.5 * (numGPUs - 1) : 0)  // NCCL buffers

// Multi-GPU overhead (TP communication buffers, scales with weights only)
multiGPUOverhead = numGPUs > 1 ? baseWeights * 0.01 * (numGPUs - 1) : 0
```

### State Management Pattern
- Use React Context or Zustand for global state
- Keep calculation logic separate from UI components
- Update results reactively when any input changes
- Debounce slider inputs to prevent excessive recalculation

### Performance Optimization
- Memoize expensive calculations with useMemo
- Virtualize long GPU/model lists if needed
- Lazy load visualization components
- Use React.memo for pure components

## Key Features to Implement

1. **Model Selection**: Dropdown with popular models + custom parameter input
2. **Quantization Options**: Both inference and KV cache quantization with tooltips
3. **GPU Configuration**: Presets for AMD/NVIDIA GPUs + custom VRAM option
4. **Multi-GPU Support**: Scale from 1-32+ GPUs with overhead calculation
5. **Inference Parameters**: Batch size, sequence length, concurrent users (log scale sliders)
6. **Memory Visualization**: Horizontal stacked bar chart showing memory breakdown
7. **Performance Metrics**: Tokens/sec estimates based on GPU compute and bandwidth
8. **Responsive Design**: Mobile-first with breakpoints for tablet/desktop

## External Data Files

### gpus.json Structure
```json
{
  "gpus": [
    {
      "id": "mi300x",
      "vendor": "AMD",
      "name": "AMD Instinct MI300X",
      "category": "enterprise",
      "vram_gb": 192,
      "memory_bandwidth_gbps": 5300,
      "compute_tflops_fp16": 1300,
      "compute_tflops_fp8": 2600
    }
  ]
}
```

### models.json Structure
```json
{
  "models": [
    {
      "id": "deepseek-r1-70b",
      "name": "DeepSeek-R1 70B",
      "parameters_billions": 70,
      "hidden_size": 8192,
      "num_layers": 80,
      "num_heads": 64,
      "num_kv_heads": 8,
      "intermediate_size": 28672,
      "default_context_length": 32768
    }
  ]
}
```