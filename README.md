# LLM Inference Calculator

A modern web application for estimating Large Language Model (LLM) memory requirements and performance characteristics for inference workloads on AMD and NVIDIA GPUs.

## Features

- **Model Selection**: Choose from popular models including DeepSeek-R1, Llama, Mistral, Mixtral, and more
- **GPU Configuration**: Support for enterprise and consumer GPUs from AMD and NVIDIA
- **Quantization Options**: Both inference (FP16, FP8, INT8, INT4) and KV cache quantization
- **Multi-GPU Support**: Scale from 1 to 32+ GPUs with overhead calculations
- **Performance Estimation**: Get tokens/second estimates based on hardware capabilities
- **Memory Visualization**: Interactive charts showing memory allocation breakdown
- **Offloading Support**: Enable CPU/RAM/NVMe offloading when VRAM is exceeded

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-sizer.git
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

1. **Select a Model**: Choose from pre-configured models or enter custom parameters
2. **Configure Hardware**: Select your GPU(s) and specify the number of units
3. **Set Quantization**: Choose inference and KV cache quantization levels
4. **Adjust Parameters**: Set batch size, sequence length, and concurrent users
5. **View Results**: See real-time memory usage, performance metrics, and visualizations

## Key Calculations

### Memory Components

- **Base Model Weights**: `(parameters × bits_per_param) / 8 / 1e9 GB`
- **KV Cache**: `2 × layers × hidden_size × kv_bits × batch × seq_len × users / 8 / 1e9 GB`
- **Activations**: Dynamic based on batch size and model dimensions
- **Framework Overhead**: ~8% of base memory usage
- **Multi-GPU Overhead**: 2% per additional GPU

### Performance Metrics

- Generation speed (tokens/second)
- Total throughput across all users
- Per-user performance
- First token latency estimates

## Technology Stack

- **Frontend**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Charts**: Recharts
- **Build Tool**: Vite

## Project Structure

```
llm-sizer/
├── src/
│   ├── components/      # React components
│   ├── hooks/          # Custom React hooks
│   ├── utils/          # Calculation utilities
│   ├── data/           # GPU and model configurations
│   ├── types/          # TypeScript definitions
│   ├── store/          # Zustand state management
│   └── App.tsx         # Main application component
├── public/
└── package.json
```

## Supported Hardware

### AMD GPUs
- MI300X (192GB)
- MI250X (128GB)
- MI210 (64GB)

### NVIDIA GPUs
- H200 (141GB)
- H100 SXM/PCIe (80GB)
- A100 (40/80GB)
- L40S (48GB)
- RTX 4090 (24GB)
- RTX 4080 (16GB)
- RTX 3090 (24GB)
- RTX 3080 (10GB)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Model configurations based on official model cards
- GPU specifications from vendor documentation
- Memory calculation formulas adapted from industry best practices