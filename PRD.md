# Product Requirements Document (PRD)
## LLM Inference Calculator Web Application

---

## 1. Executive Summary

### 1.1 Overview
A lightweight, modern web application that helps users estimate LLM (Large Language Model) memory requirements and performance characteristics for inference workloads. The tool will support both AMD and NVIDIA GPUs with preset configurations and custom GPU parameters.

### 1.2 Objectives
- Provide accurate memory and performance estimates for LLM inference
- Support popular enterprise and consumer GPUs from AMD and NVIDIA
- Enable custom GPU configuration for flexibility
- Deliver a modern, responsive user interface
- Maintain extensibility through external GPU configuration files

---

## 2. Product Scope

### 2.1 In Scope
- LLM inference memory calculation
- GPU preset configurations (AMD & NVIDIA)
- Custom GPU parameter input
- Quantization options (FP16, INT8, etc.)
- KV cache quantization settings
- Multi-GPU support
- Batch size and sequence length configuration
- Concurrent user calculations
- Memory allocation visualization
- Performance metrics (tokens/sec)
- Offloading options (CPU/RAM/NVMe)

### 2.2 Out of Scope (v1)
- Fine-tuning calculations (future enhancement)
- Model training estimates
- Cost calculations
- Cloud provider integration
- User accounts/authentication
- Saved configurations

---

## 3. User Stories

### 3.1 Primary Users
- ML Engineers evaluating hardware requirements
- DevOps teams planning infrastructure
- Researchers comparing GPU options
- Product managers estimating deployment costs

### 3.2 Key User Stories
1. As an ML engineer, I want to select a popular LLM model and see memory requirements across different GPUs
2. As a DevOps engineer, I want to estimate how many concurrent users a GPU configuration can support
3. As a researcher, I want to compare AMD and NVIDIA GPUs for my inference workload
4. As a product manager, I want to understand the impact of quantization on memory usage

---

## 4. Functional Requirements

### 4.1 Model Selection
- Dropdown with popular LLM models (DeepSeek-R1, Llama, Mistral, etc.)
- Display model parameter count
- Support for custom model parameters

### 4.2 Quantization Options
- **Inference Quantization**: FP16, FP8, INT8, INT4
- **KV Cache Quantization**: FP16/BF16, FP8/BF16, INT8
- Tooltips explaining trade-offs

### 4.3 Hardware Configuration
- GPU preset selector (AMD MI300X, MI250, H100, A100, etc.)
- Custom VRAM input
- Number of GPUs selector (1-32+)
- Display total available VRAM

### 4.4 Inference Parameters
- Batch size slider (1-32, log scale)
- Sequence length slider (2K-128K tokens)
- Concurrent users slider (1-32+, log scale)
- Offloading toggle (CPU/RAM/NVMe)

### 4.5 Results Display
- **VRAM Usage Gauge**: Visual percentage indicator
- **Memory Breakdown**:
  - Base model weights
  - Activations
  - KV cache
  - Framework overhead
  - Multi-GPU overhead
- **Performance Metrics**:
  - Generation speed (tok/sec)
  - Total throughput (tok/sec)
  - Per-user speed (tok/sec)
- **Configuration Summary**: Display selected settings

### 4.6 Memory Allocation Visualization
- Horizontal bar chart showing memory distribution
- Color-coded segments with percentages
- Absolute values in GB

---

## 5. Technical Requirements

### 5.1 Technology Stack

#### Option A: TypeScript (Recommended)
- **Frontend**: React 18+ with TypeScript
- **UI Framework**: Tailwind CSS + shadcn/ui or Chakra UI
- **Charts**: Recharts or Chart.js
- **Build Tool**: Vite
- **State Management**: React Context or Zustand

#### Option B: Python
- **Frontend**: Streamlit or Gradio
- **Alternative**: FastAPI + React frontend

### 5.2 Data Storage
- GPU configurations in JSON/YAML file
- Model presets in JSON/YAML file
- No backend database required (static data)

### 5.3 Performance
- Initial load time < 2 seconds
- Calculation updates < 100ms
- Responsive design (mobile, tablet, desktop)

### 5.4 Browser Support
- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)

---

## 6. Data Model

### 6.1 GPU Configuration Schema
```yaml
gpus:
  - id: "mi300x"
    vendor: "AMD"
    name: "AMD Instinct MI300X"
    category: "enterprise"
    vram_gb: 192
    memory_bandwidth_gbps: 5300
    compute_tflops_fp16: 1300
    compute_tflops_fp8: 2600
    
  - id: "h100-sxm"
    vendor: "NVIDIA"
    name: "NVIDIA H100 SXM"
    category: "enterprise"
    vram_gb: 80
    memory_bandwidth_gbps: 3350
    compute_tflops_fp16: 989
    compute_tflops_fp8: 1979
    
  - id: "rtx-4090"
    vendor: "NVIDIA"
    name: "NVIDIA RTX 4090"
    category: "consumer"
    vram_gb: 24
    memory_bandwidth_gbps: 1008
    compute_tflops_fp16: 82.6
```

### 6.2 Model Configuration Schema
```yaml
models:
  - id: "deepseek-r1-70b"
    name: "DeepSeek-R1 70B"
    parameters_billions: 70
    default_context_length: 32768
    architecture: "transformer"
    
  - id: "llama-3-70b"
    name: "Llama 3 70B"
    parameters_billions: 70
    default_context_length: 8192
    architecture: "transformer"
```

### 6.3 Application State Model
```typescript
interface AppState {
  // Model Configuration
  selectedModel: string;
  customModelParams?: number;
  
  // Quantization
  inferenceQuantization: 'fp16' | 'fp8' | 'int8' | 'int4';
  kvCacheQuantization: 'fp16_bf16' | 'fp8_bf16' | 'int8';
  
  // Hardware
  selectedGPU: string;
  customVRAM?: number;
  numGPUs: number;
  
  // Inference Parameters
  batchSize: number;
  sequenceLength: number;
  concurrentUsers: number;
  enableOffloading: boolean;
  
  // Computed Results
  results: {
    totalVRAM: number;
    usedVRAM: number;
    vramPercentage: number;
    memoryBreakdown: {
      baseWeights: number;
      activations: number;
      kvCache: number;
      frameworkOverhead: number;
      multiGPUOverhead: number;
    };
    performance: {
      generationSpeed: number;
      totalThroughput: number;
      perUserSpeed: number;
    };
    status: 'okay' | 'warning' | 'error';
  };
}
```

---

## 7. UI/UX Requirements

### 7.1 Layout
- Two-column layout (desktop)
- Left: Configuration inputs
- Right: Results and visualizations
- Single column (mobile/tablet)

### 7.2 Design Principles
- Clean, modern aesthetic
- Clear visual hierarchy
- Immediate feedback on input changes
- Helpful tooltips and explanations
- Accessible (WCAG 2.1 AA)

### 7.3 Color Coding
- Green: Optimal/Safe memory usage
- Yellow