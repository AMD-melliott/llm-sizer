# LLM Inference Calculator

A modern web application for estimating memory requirements and performance characteristics for Large Language Models (LLMs), embedding models, and reranking models on AMD and NVIDIA GPUs. Built with React, TypeScript, and Tailwind CSS.

## Features

### Model Support
- **Text Generation Models**: 100+ models from 1B to 600B+ parameters (Llama, Mistral, Mixtral, DeepSeek-R1, Qwen, Gemma, and more)
- **Multimodal (Vision-Language) Models**: Support for LLaVA, Phi-3-Vision, Florence-2, Qwen-VL, and other vision-language models
- **Embedding Models**: 30+ embedding models for semantic search and RAG applications (OpenAI, Cohere, BGE, E5, Snowflake Arctic, etc.)
- **Reranking Models**: Cross-encoder and late-interaction models for result refinement (Cohere Rerank, BGE-Reranker, etc.)

### Hardware Configuration
- **AMD GPUs**: Instinct MI300X/A, MI325X, MI350X, MI355X, MI250X, MI210, Radeon RX 7000 series, Strix Halo
- **NVIDIA GPUs**: B200, H200, H100, A100, L40S, L4, RTX 4090/4080, RTX 3090, and more
- **Custom GPU Support**: Define your own GPU specifications
- **Multi-GPU Scaling**: Calculate requirements for 1-32+ GPUs with automatic overhead estimation

### Quantization Options
- **Inference Quantization**: FP16, FP8, INT8, INT4 (with accuracy/memory tradeoffs)
- **KV Cache Quantization**: FP16/BF16, FP8/BF16, INT8 options
- **Per-Component Control**: Different quantization for model weights and KV cache

### Performance & Memory Analysis
- **Detailed Memory Breakdown**: Base weights, activations, KV cache, framework overhead, multi-GPU overhead
- **Multimodal Memory**: Vision encoder weights, image preprocessing, vision activations, projector layers
- **Interactive Visualizations**: Real-time memory allocation charts
- **Performance Metrics**: Tokens/second, throughput, latency estimates based on hardware capabilities
- **Offloading Detection**: Automatic warnings when VRAM is exceeded

### Documentation System
- **Interactive Documentation**: Built-in comprehensive documentation with tabbed interface
- **Formula Explanations**: Detailed breakdowns of all memory calculations
- **Example Calculations**: Step-by-step walkthroughs for different model types
- **Glossary**: Complete reference for technical terms
- **Best Practices**: Optimization guides for different use cases

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AMD-melliott/llm-sizer.git
cd llm-sizer
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The production build will be in the `dist` directory.

## Usage

1. **Select Model Type**: Choose between Text Generation, Embedding, or Reranking models
2. **Select a Model**: Browse pre-configured models or enter custom parameters
   - Text generation: Llama, Mistral, DeepSeek, Qwen, Gemma, etc.
   - Multimodal: LLaVA, Phi-3-Vision, Florence-2, Qwen-VL
   - Embedding: BGE, E5, Snowflake Arctic, Cohere, OpenAI
   - Reranking: Cohere Rerank, BGE-Reranker
3. **Configure Hardware**: Select your GPU(s) and specify the number of units
4. **Set Quantization**: Choose inference and KV cache quantization levels
5. **Adjust Parameters**:
   - **Generation**: Batch size, sequence length, concurrent users, image count (for multimodal)
   - **Embedding**: Batch size, documents per batch, average document size, chunking parameters
   - **Reranking**: Batch size, number of queries, documents per query, max lengths
6. **View Results**: See real-time memory usage, performance metrics, and interactive visualizations
7. **Read Documentation**: Access comprehensive guides, formulas, and examples in the Documentation tab

## Key Calculations

### Text Generation Models

#### Memory Components
- **Base Model Weights**: `(parameters × bits_per_param) / 8 / 1e9 GB`
- **KV Cache**: `2 × layers × hidden_size × kv_bits × batch × seq_len × users / 8 / 1e9 GB`
- **Activations**: Dynamic based on batch size and model dimensions
- **Framework Overhead**: ~8% of base memory usage
- **Multi-GPU Overhead**: 2% per additional GPU

#### Multimodal Extensions
- **Vision Encoder Weights**: Based on vision model parameters and quantization
- **Vision Activations**: Image processing memory for batch × num_images
- **Projector Weights**: MLP/linear layers connecting vision and language models
- **Image Token KV Cache**: Additional KV cache for image tokens in context

### Embedding Models

#### Memory Components
- **Model Weights**: Similar to generation models but typically 100M-1B parameters
- **Batch Input Memory**: `batch_size × avg_doc_size × hidden_size × 4 / 1e9 GB`
- **Attention Memory**: `batch_size × seq_length² × num_heads × 4 / 1e9 GB` (per layer)
- **Embedding Storage**: `batch_size × embedding_dimension × 4 / 1e9 GB`
- **Activation Memory**: Intermediate layer computations
- **Framework Overhead**: ~10% for CUDA kernels and buffers

### Reranking Models

#### Memory Components
- **Model Weights**: Cross-encoder models typically 300M-1B parameters
- **Pair Batch Memory**: `effective_batch × pair_length × hidden_size × 4 / 1e9 GB`
- **Attention Memory**: Self-attention over query+document concatenations
- **Scoring Memory**: Relevance scores for each query-document pair
- **Activation Memory**: FFN layers, layer norms, and residual connections

### Performance Metrics

#### Generation Models
- Tokens/second based on memory bandwidth and compute capability
- Total throughput across all concurrent users
- Per-user performance metrics
- First token latency estimates

#### Embedding Models
- Documents per second processing rate
- Tokens per second throughput
- Embeddings generated per second

#### Reranking Models
- Query-document pairs processed per second
- Queries per second throughput
- Average latency per query

## Technology Stack

- **Frontend**: React 19 with TypeScript
- **Styling**: Tailwind CSS 4
- **State Management**: Zustand
- **Charts**: Recharts for interactive visualizations
- **Icons**: Lucide React
- **Build Tool**: Vite 7
- **Testing**: Jest with ts-jest

## Project Structure

```
llm-sizer/
├── src/
│   ├── components/         # React components
│   │   ├── Documentation/  # Built-in documentation system
│   │   └── Tabs/          # Tabbed interface components
│   ├── hooks/             # Custom React hooks
│   ├── utils/             # Calculation utilities
│   │   ├── memoryCalculator.ts      # Text generation calculations
│   │   ├── embeddingCalculator.ts   # Embedding model calculations
│   │   ├── rerankingCalculator.ts   # Reranking model calculations
│   │   └── performanceEstimator.ts  # Performance metrics
│   ├── data/              # Model and GPU configurations
│   │   ├── models.json            # 100+ text/multimodal models
│   │   ├── embedding-models.json  # 30+ embedding models
│   │   ├── reranking-models.json  # Reranking models
│   │   └── gpus.json              # AMD & NVIDIA GPU specs
│   ├── content/           # Documentation content
│   ├── types/             # TypeScript definitions
│   ├── store/             # Zustand state management
│   └── App.tsx            # Main application component
├── scripts/               # Import utilities
│   ├── import-hf-model.ts # Hugging Face model import
│   └── batch-import.ts    # Batch model import
├── tests/                 # Jest test suite
├── docs/                  # Architecture documentation
└── public/
```

## Supported Hardware

### AMD GPUs

#### Datacenter (Instinct)
- MI355X (288GB HBM3e) - 2025
- MI350X (288GB HBM3e) - 2025
- MI325X (256GB HBM3e) - 2024
- MI300X (192GB HBM3) - 2023
- MI300A (128GB HBM3) - 2023
- MI250X (128GB HBM2e) - 2021
- MI210 (64GB HBM2e) - 2021

#### Professional (Radeon PRO)
- W7900 (48GB GDDR6)
- W7800 (32GB GDDR6)
- W7700 (16GB GDDR6)

#### Consumer (Radeon RX & APU)
- RX 7900 XTX (24GB GDDR6)
- RX 7900 XT (20GB GDDR6)
- RX 7900 GRE (16GB GDDR6)
- RX 7800 XT (16GB GDDR6)
- RX 7700 XT (12GB GDDR6)
- Ryzen AI Max+ 395 - Strix Halo (96GB LPDDR5X-8000)
- Ryzen AI Max PRO 390 - Strix Halo (96GB LPDDR5X-8000)

### NVIDIA GPUs

#### Datacenter (Hopper & Ampere)
- B200 (192GB HBM3e) - 2025
- H200 (141GB HBM3e) - 2024
- H100 SXM5 (80GB HBM3) - 2022
- H100 PCIe (80GB HBM3) - 2022
- H100 NVL (94GB HBM3) - 2023
- A100 (80GB / 40GB HBM2e) - 2020/2021
- A40 (48GB GDDR6) - 2020
- A30 (24GB HBM2) - 2021
- L40S (48GB GDDR6) - 2023
- L4 (24GB GDDR6) - 2023

#### Professional (RTX Ada & Ampere)
- RTX 6000 Ada (48GB GDDR6)
- RTX A6000 (48GB GDDR6)
- RTX A5000 (24GB GDDR6)
- RTX A4000 (16GB GDDR6)

#### Consumer (RTX 40/30 Series)
- RTX 4090 (24GB GDDR6X)
- RTX 4080 Super (16GB GDDR6X)
- RTX 4080 (16GB GDDR6X)
- RTX 4070 Ti Super (16GB GDDR6X)
- RTX 4070 Ti (12GB GDDR6X)
- RTX 3090 Ti (24GB GDDR6X)
- RTX 3090 (24GB GDDR6X)
- RTX 3080 Ti (12GB GDDR6X)
- RTX 3080 (10GB GDDR6X)

## Advanced Features

### Hugging Face Model Import
Import models directly from Hugging Face Hub with automatic parameter detection:

```bash
# Import a single model
npm run import-model -- --model "meta-llama/Llama-3.2-1B-Instruct"

# Batch import multiple models
npm run batch-import -- --file models-to-import.txt
```

The import scripts automatically:
- Fetch model configuration from Hugging Face
- Extract architecture parameters (layers, hidden size, attention heads)
- Detect multimodal capabilities and vision configurations
- Validate against schema and add to the model database

See `docs/hf-import/` for detailed documentation.

### Testing
Comprehensive test suite covering:
- Text generation memory calculations
- Multimodal model calculations
- Embedding model calculations
- Reranking model calculations
- Edge cases and validation

```bash
npm test
```

### Custom GPU Configuration
Define custom GPU specifications for hardware not in the database, including VRAM, memory bandwidth, and compute capabilities.

## Model Categories

### Text Generation Models (100+)
- **Small Models (1B-3B)**: Gemma, Llama 3.2, Qwen2.5, Phi-3
- **Medium Models (7B-13B)**: Llama 3.1, Mistral, Qwen, DeepSeek
- **Large Models (30B-70B)**: Llama 3.1 70B, Mixtral 8x22B, Qwen2.5 72B
- **Extra Large (100B+)**: DeepSeek-R1, Llama 3.1 405B, Arctic 480B

### Multimodal Models (20+)
- **LLaVA Family**: LLaVA-1.5, LLaVA-NeXT
- **Phi-3 Vision**: Microsoft's vision-language models
- **Qwen-VL**: Alibaba's multimodal models
- **Florence-2**: Microsoft's unified vision model
- **InternVL**: Various scales of vision-language models

### Embedding Models (30+)
- **OpenAI**: text-embedding-3-large, text-embedding-3-small
- **Cohere**: Embed v3 (English/Multilingual)
- **BGE (BAAI)**: Large, Base, Small variants
- **E5**: Mistral-based embedding models
- **Snowflake Arctic**: High-performance embedding models
- **Jina AI**: v2/v3 multilingual embeddings

### Reranking Models (10+)
- **Cohere**: Rerank v3 (English/Multilingual)
- **BGE Reranker**: v2-m3, Large, Base variants
- **Jina**: Reranker v2
- **MixedBread**: mxbai-rerank models

## Use Cases

### 1. RAG System Planning
- Calculate embedding model requirements for document encoding
- Estimate reranking model overhead for result refinement
- Size generation models for answer synthesis
- Plan multi-GPU deployments for large document collections

### 2. Vision-Language Applications
- Estimate memory for multimodal chat applications
- Calculate image processing overhead in VLMs
- Plan hardware for visual question answering systems
- Size deployments for image captioning services

### 3. Production Inference Optimization
- Compare quantization strategies (FP16 vs INT8 vs INT4)
- Evaluate single vs multi-GPU configurations
- Optimize batch sizes for throughput vs latency
- Plan scaling for concurrent user loads

### 4. Hardware Procurement
- Compare AMD vs NVIDIA GPU options
- Calculate cost-per-token for different hardware
- Determine minimum viable GPU configurations
- Plan datacenter capacity for model serving

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Areas for Contribution
- Additional model configurations
- New GPU specifications
- Enhanced performance estimations
- Documentation improvements
- Test coverage expansion

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Model configurations based on official Hugging Face model cards
- GPU specifications from AMD and NVIDIA vendor documentation
- Memory calculation formulas adapted from industry best practices and research papers
- Community feedback and contributions

## Links

- **GitHub Repository**: [AMD-melliott/llm-sizer](https://github.com/AMD-melliott/llm-sizer)
- **Documentation**: Built-in documentation tab in the application
- **Issues & Feature Requests**: GitHub Issues
- **Hugging Face Import Guide**: `docs/hf-import/QUICKSTART.md`