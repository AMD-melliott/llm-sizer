import { DocSection } from '../components/Documentation';
import { FormulaBlock } from '../components/Documentation/FormulaBlock';
import { ExampleCalculation } from '../components/Documentation/ExampleCalculation';
import { Glossary, GlossaryTerm } from '../components/Documentation/Glossary';

// Glossary Terms
export const glossaryTerms: GlossaryTerm[] = [
  {
    term: 'Parameters',
    definition: 'The trainable weights in a neural network. For LLMs, this typically refers to billions of floating-point numbers that define the model\'s behavior. Common sizes include 7B (7 billion), 70B (70 billion), and 175B (175 billion) parameters.',
    category: 'model'
  },
  {
    term: 'Tokens',
    definition: 'The basic units of text that LLMs process. A token can be a word, part of a word, or punctuation. On average, 1 token ≈ 0.75 words in English.',
    category: 'model'
  },
  {
    term: 'Sequence Length',
    definition: 'The maximum number of tokens that can be processed in a single inference pass. Also called context window or context length.',
    category: 'model'
  },
  {
    term: 'Batch Size',
    definition: 'The number of independent sequences processed simultaneously. Larger batch sizes increase throughput but require more memory.',
    category: 'performance'
  },
  {
    term: 'KV Cache',
    definition: 'Key-Value cache stores previously computed attention keys and values to avoid redundant calculations during autoregressive generation. This significantly speeds up inference but requires additional memory.',
    category: 'model'
  },
  {
    term: 'Attention Heads',
    definition: 'Parallel attention mechanisms in transformer models. Multi-head attention allows the model to focus on different aspects of the input simultaneously.',
    category: 'model'
  },
  {
    term: 'Hidden Size',
    definition: 'The dimensionality of the model\'s internal representations (also called embedding dimension or model dimension). Typical values range from 2048 to 12288.',
    category: 'model'
  },
  {
    term: 'FP32 (Float32)',
    definition: '32-bit floating point representation. The standard precision for training but rarely used for inference due to high memory requirements.',
    category: 'quantization'
  },
  {
    term: 'FP16 (Float16)',
    definition: '16-bit floating point representation. Provides a good balance between memory efficiency and accuracy, commonly used for inference.',
    category: 'quantization'
  },
  {
    term: 'BF16 (BFloat16)',
    definition: 'Brain Float 16 - An alternative 16-bit format with the same exponent range as FP32 but reduced precision. More stable than FP16 for some models.',
    category: 'quantization'
  },
  {
    term: 'INT8',
    definition: '8-bit integer quantization. Reduces memory by 75% compared to FP32 with minimal accuracy loss. Requires calibration.',
    category: 'quantization'
  },
  {
    term: 'INT4',
    definition: '4-bit integer quantization. Aggressive compression reducing memory by 87.5% compared to FP32. May have noticeable accuracy degradation for some models.',
    category: 'quantization'
  },
  {
    term: 'Quantization',
    definition: 'The process of reducing the precision of model weights and activations to lower bit representations, significantly reducing memory requirements and improving inference speed.',
    category: 'quantization'
  },
  {
    term: 'VRAM',
    definition: 'Video RAM - The dedicated memory on GPUs used for storing model weights, activations, and intermediate computations.',
    category: 'hardware'
  },
  {
    term: 'Memory Bandwidth',
    definition: 'The rate at which data can be read from or written to GPU memory, measured in GB/s. Critical for inference performance, especially for large models.',
    category: 'hardware'
  },
  {
    term: 'Tensor Cores',
    definition: 'Specialized hardware units in NVIDIA GPUs designed for accelerating matrix operations, particularly beneficial for lower precision formats (FP16, INT8).',
    category: 'hardware'
  },
  {
    term: 'Activations',
    definition: 'Intermediate values computed during the forward pass through the neural network. These need to be stored in memory during inference.',
    category: 'model'
  },
  {
    term: 'Throughput',
    definition: 'The number of tokens processed per second. Higher throughput means faster generation but may require more memory (via larger batch sizes).',
    category: 'performance'
  },
  {
    term: 'Latency',
    definition: 'The time delay between input and output. For LLMs, this includes time to first token (TTFT) and per-token generation time.',
    category: 'performance'
  },
  {
    term: 'Multi-GPU',
    definition: 'Using multiple GPUs to distribute model weights and computations. Required for models that don\'t fit on a single GPU.',
    category: 'hardware'
  },
  {
    term: 'Tensor Parallelism',
    definition: 'A technique for splitting individual model layers across multiple GPUs, enabling parallel computation of matrix operations.',
    category: 'hardware'
  },
  {
    term: 'Pipeline Parallelism',
    definition: 'A technique for splitting model layers across multiple GPUs sequentially, with different GPUs processing different stages of the forward/backward pass.',
    category: 'hardware'
  },
  {
    term: 'GQA (Grouped Query Attention)',
    definition: 'An attention variant where multiple query heads share a smaller number of key-value heads. This reduces KV cache memory proportionally (e.g., 64 query heads with 8 KV heads = 8x KV cache reduction). Used by most modern LLMs including Llama 3, Mistral, and Gemma.',
    category: 'model'
  },
  {
    term: 'MHA (Multi-Head Attention)',
    definition: 'The standard attention mechanism where each query head has its own key-value head (num_kv_heads = num_heads). All KV heads are independent, resulting in higher KV cache memory compared to GQA.',
    category: 'model'
  },
  {
    term: 'Head Size',
    definition: 'The dimension of each attention head, calculated as hidden_size / num_heads. Common values are 64, 128, or 256. Used in the KV cache formula: KV per token per layer = 2 × num_kv_heads × head_size × dtype_bytes.',
    category: 'model'
  },
  {
    term: 'Intermediate Size',
    definition: 'The hidden dimension of the feed-forward network (FFN) layers within each transformer block. Typically 4x the hidden_size. Determines peak activation memory since FFN intermediates are the largest buffers during a forward pass.',
    category: 'model'
  },
  {
    term: 'FP8 (Float8)',
    definition: 'An 8-bit floating point format that halves memory compared to FP16. Supported as FP8 E4M3 (4-bit exponent, 3-bit mantissa, better precision) and FP8 E5M2 (5-bit exponent, 2-bit mantissa, wider range). Can be used for both model weights and KV cache independently.',
    category: 'quantization'
  },
  {
    term: 'FlashAttention',
    definition: 'An optimized attention algorithm that avoids materializing the full attention score matrix in GPU memory. This significantly reduces activation memory, making the FFN intermediate buffer the dominant activation cost instead of the quadratic attention matrix.',
    category: 'performance'
  },
  {
    term: 'NCCL',
    definition: 'NVIDIA Collective Communications Library. Handles GPU-to-GPU communication in multi-GPU setups. Requires dedicated memory buffers (~0.5 GB per additional GPU) for all-reduce and other collective operations.',
    category: 'hardware'
  },
  {
    term: 'num_kv_heads',
    definition: 'The number of key-value attention heads in a model. For MHA models this equals num_heads. For GQA models this is smaller (e.g., 8 KV heads with 64 query heads). Directly determines KV cache size per token.',
    category: 'model'
  },
];

// Documentation Sections
export const documentationSections: DocSection[] = [
  {
    id: 'overview',
    title: 'Overview',
    content: (
      <div className="space-y-4">
        <p>
          Welcome to the LLM Inference Calculator documentation. This guide explains how we calculate
          memory requirements and performance metrics for Large Language Model (LLM) inference.
        </p>
        <p>
          Understanding memory requirements is crucial for:
        </p>
        <ul className="list-disc pl-6 space-y-2">
          <li><strong>Hardware Selection</strong>: Choosing the right GPU(s) for your use case</li>
          <li><strong>Cost Optimization</strong>: Balancing performance and infrastructure costs</li>
          <li><strong>Deployment Planning</strong>: Estimating capacity and scaling requirements</li>
          <li><strong>Performance Tuning</strong>: Understanding trade-offs between different configurations</li>
        </ul>
        <p>
          Memory requirements for LLM inference depend on several factors including model size,
          quantization level, batch size, sequence length, and the number of concurrent users.
          This calculator helps you estimate these requirements accurately.
        </p>
      </div>
    ),
  },
  {
    id: 'methodology',
    title: 'Calculation Methodology',
    content: (
      <div className="space-y-4">
        <p>
          The total memory required for LLM inference consists of five main components:
        </p>
        <ol className="list-decimal pl-6 space-y-2">
          <li><strong>Model Weights Memory</strong>: Storage for the model parameters</li>
          <li><strong>KV Cache Memory</strong>: Storage for attention mechanism cache</li>
          <li><strong>Activation Memory</strong>: Storage for intermediate computations</li>
          <li><strong>Framework Overhead</strong>: Additional memory for CUDA kernels and buffers</li>
          <li><strong>Multi-GPU Overhead</strong>: Communication overhead when using multiple GPUs</li>
        </ol>
        <p>
          Each component scales differently with model configuration and inference parameters.
          Let's examine each in detail.
        </p>
      </div>
    ),
    subsections: [
      {
        id: 'model-weights',
        title: '1. Model Weights Memory',
        content: (
          <div className="space-y-4">
            <p>
              The model weights memory is determined by the number of parameters and the bits per parameter
              (which depends on the quantization level).
            </p>
            
            <FormulaBlock
              title="Model Weights Memory Formula"
              formula="memory_weights = (num_parameters × bits_per_param) / 8 / 10^9 GB"
              explanation="Convert bits to bytes (÷8) and bytes to gigabytes (÷10^9)"
              variables={[
                { symbol: 'num_parameters', description: 'Total number of model parameters (e.g., 7B, 70B)' },
                { symbol: 'bits_per_param', description: 'Precision: FP32=32, FP16=16, INT8=8, INT4=4' },
              ]}
            />

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 mb-2">Quantization Impact:</h5>
              <ul className="space-y-1 text-sm">
                <li>• <strong>FP32</strong>: 7B model = 28 GB</li>
                <li>• <strong>FP16</strong>: 7B model = 14 GB (50% reduction)</li>
                <li>• <strong>INT8</strong>: 7B model = 7 GB (75% reduction)</li>
                <li>• <strong>INT4</strong>: 7B model = 3.5 GB (87.5% reduction)</li>
              </ul>
            </div>
          </div>
        ),
      },
      {
        id: 'kv-cache',
        title: '2. KV Cache Memory',
        content: (
          <div className="space-y-4">
            <p>
              The KV (Key-Value) cache stores previously computed attention keys and values during
              autoregressive generation. This is essential for efficient inference but requires
              significant memory, especially for long sequences.
            </p>

            <FormulaBlock
              title="KV Cache Memory Formula (GQA-aware)"
              formula="memory_kv = 2 × num_layers × num_kv_heads × head_size × kv_bytes × batch_size × seq_len / 10^9 GB"
              explanation="Factor of 2 accounts for both keys and values. For GQA models, num_kv_heads is less than num_heads, significantly reducing KV cache. For MHA models (num_kv_heads = num_heads), num_kv_heads × head_size = hidden_size. This matches vLLM's internal KV cache allocation formula."
              variables={[
                { symbol: 'num_layers', description: 'Number of transformer layers in the model' },
                { symbol: 'num_kv_heads', description: 'Number of key-value heads (may be less than query heads in GQA models)' },
                { symbol: 'head_size', description: 'Dimension per head (hidden_size / num_heads)' },
                { symbol: 'kv_bytes', description: 'Bytes per element: FP16=2, FP8=1, INT8=1' },
                { symbol: 'batch_size', description: 'Number of sequences processed in parallel' },
                { symbol: 'seq_len', description: 'Maximum sequence length (context window)' },
              ]}
            />

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 mb-2">Important Notes:</h5>
              <ul className="space-y-1 text-sm">
                <li>• KV cache memory scales <strong>linearly</strong> with sequence length</li>
                <li>• Doubling batch size doubles KV cache memory</li>
                <li>• KV cache can be quantized independently from model weights (e.g., FP8 KV with FP16 weights)</li>
                <li>• For long contexts (32K+ tokens), KV cache often exceeds model weights memory</li>
                <li>• <strong>GQA models</strong> (Llama 3, Mistral, Gemma, etc.) use fewer KV heads than query heads, reducing KV cache by up to 8x compared to MHA models</li>
              </ul>
            </div>
          </div>
        ),
      },
      {
        id: 'activations',
        title: '3. Activation Memory',
        content: (
          <div className="space-y-4">
            <p>
              Activation memory stores intermediate values computed during the forward pass.
              Transformer layers are processed sequentially, so only one layer's activations
              are held at a time -- the peak is determined by the FFN intermediate buffer,
              not the sum across all layers.
            </p>

            <FormulaBlock
              title="Activation Memory Formula"
              formula="memory_activation = batch_size × seq_len × intermediate_size × 2 / num_gpus / 10^9 GB"
              explanation="The intermediate_size is the FFN hidden dimension (typically 4x hidden_size). The factor of 2 accounts for the input tensor and intermediate buffer coexisting during FFN computation. With FlashAttention (used by vLLM and other modern engines), attention scores are not fully materialized."
              variables={[
                { symbol: 'batch_size', description: 'Number of sequences processed in parallel' },
                { symbol: 'seq_len', description: 'Sequence length being processed' },
                { symbol: 'intermediate_size', description: 'FFN intermediate dimension (defaults to hidden_size × 4)' },
                { symbol: 'num_gpus', description: 'Number of GPUs (activations are distributed in tensor parallelism)' },
              ]}
            />

            <p className="text-sm text-gray-700">
              Activation memory is generally much smaller than model weights or KV cache, since only
              one layer's intermediates are live at any time. It scales with batch size and sequence
              length but is independent of the number of layers.
            </p>
          </div>
        ),
      },
      {
        id: 'framework-overhead',
        title: '4. Framework Overhead',
        content: (
          <div className="space-y-4">
            <p>
              Inference frameworks (vLLM, TGI, etc.) require additional memory for the CUDA context,
              PyTorch runtime, engine initialization, memory pools, and internal buffers. This overhead
              has both a fixed and proportional component.
            </p>

            <FormulaBlock
              title="Framework Overhead Formula"
              formula="memory_overhead = 1.5 GB + (memory_weights + memory_kv + memory_activation) × 0.05 + nccl_overhead"
              explanation="The 1.5 GB fixed baseline covers the CUDA context, PyTorch runtime, and engine initialization. The 5% proportional term covers internal buffers and compilation artifacts. For multi-GPU setups, NCCL communication buffers add ~0.5 GB per additional GPU. This model is calibrated from vLLM profiling data, where small models see high relative overhead (fixed costs dominate) and large models see lower relative overhead."
              variables={[
                { symbol: '1.5 GB', description: 'Fixed baseline: CUDA context + PyTorch runtime + engine' },
                { symbol: '0.05', description: 'Proportional factor for internal buffers (~5% of model memory)' },
                { symbol: 'nccl_overhead', description: '0.5 GB × (num_gpus - 1) for NCCL communication buffers' },
              ]}
            />

            <p className="text-sm text-gray-700">
              This overhead is important to account for in production deployments. For small models
              (1-3B parameters), the fixed 1.5 GB baseline can represent 10-50% of total memory usage.
              For large models (70B+), the proportional term is more significant but the overall
              ratio is lower (~6-7%).
            </p>
          </div>
        ),
      },
      {
        id: 'multi-gpu',
        title: '5. Multi-GPU Overhead',
        content: (
          <div className="space-y-4">
            <p>
              When using multiple GPUs with tensor parallelism, additional memory is required for
              all-reduce communication buffers and activation synchronization. The fixed NCCL
              buffer cost is included in the framework overhead above; this term covers the
              remaining proportional overhead that scales with model weights.
            </p>

            <FormulaBlock
              title="Multi-GPU Overhead Formula"
              formula="memory_multi_gpu = memory_weights × 0.01 × (num_gpus - 1)"
              explanation="Approximately 1% of model weights per additional GPU for tensor parallelism communication buffers. This scales with weights only, not KV cache or activations, since those don't require cross-GPU synchronization beyond what NCCL provides."
              variables={[
                { symbol: 'memory_weights', description: 'Model weights memory (in GB)' },
                { symbol: 'num_gpus', description: 'Total number of GPUs' },
              ]}
            />

            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 mb-2">Multi-GPU Strategies:</h5>
              <ul className="space-y-1 text-sm">
                <li>• <strong>Tensor Parallelism</strong>: Split layers across GPUs (better for latency)</li>
                <li>• <strong>Pipeline Parallelism</strong>: Split model depth-wise (better for throughput)</li>
                <li>• <strong>Data Parallelism</strong>: Replicate model, split batch (best for high throughput)</li>
              </ul>
            </div>
          </div>
        ),
      },
    ],
  },
  {
    id: 'embedding-models',
    title: 'Embedding Model Calculations',
    content: (
      <div className="space-y-4">
        <p>
          Embedding models generate dense vector representations of text for semantic search,
          clustering, and retrieval tasks. Unlike generation models, they don't use KV caching
          but process documents in batches to produce fixed-size embeddings.
        </p>
        <p>
          Memory requirements for embedding models consist of:
        </p>
        <ol className="list-decimal pl-6 space-y-2">
          <li><strong>Model Weights Memory</strong>: Storage for the model parameters</li>
          <li><strong>Batch Input Memory</strong>: Storage for the current batch of tokens</li>
          <li><strong>Attention Memory</strong>: Storage for self-attention computations</li>
          <li><strong>Embedding Storage</strong>: Storage for output embeddings</li>
          <li><strong>Activation Memory</strong>: Storage for intermediate computations</li>
          <li><strong>Framework Overhead</strong>: Additional memory for buffers (10%)</li>
          <li><strong>Multi-GPU Overhead</strong>: Communication overhead for multi-GPU setups (2% per GPU)</li>
        </ol>
      </div>
    ),
    subsections: [
      {
        id: 'embedding-weights',
        title: '1. Model Weights',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Embedding Model Weights Formula"
              formula="memory_weights = (parameters_millions × 10^6 × bits_per_param) / 8 / 10^9 GB"
              explanation="Same as generation models, but embedding models are typically smaller (100M-1B parameters)"
              variables={[
                { symbol: 'parameters_millions', description: 'Model size in millions (e.g., 335M for bge-large)' },
                { symbol: 'bits_per_param', description: 'Precision: FP16=16, INT8=8, INT4=4' },
              ]}
            />
          </div>
        ),
      },
      {
        id: 'embedding-batch-input',
        title: '2. Batch Input Memory',
        content: (
          <div className="space-y-4">
            <p>
              Memory required to store the input tokens being processed in the current batch.
            </p>
            <FormulaBlock
              title="Batch Input Memory Formula"
              formula="memory_batch = batch_size × avg_document_size × hidden_size × 4 / 10^9 GB"
              explanation="Stores FP32 token embeddings for all documents in the batch"
              variables={[
                { symbol: 'batch_size', description: 'Number of documents processed simultaneously' },
                { symbol: 'avg_document_size', description: 'Average document length in tokens' },
                { symbol: 'hidden_size', description: 'Model embedding dimension' },
              ]}
            />
          </div>
        ),
      },
      {
        id: 'embedding-attention',
        title: '3. Attention Memory',
        content: (
          <div className="space-y-4">
            <p>
              Self-attention matrices for the current layer being processed. Unlike generation models,
              attention memory is typically calculated for one layer at a time since intermediate
              results don't need to be cached.
            </p>
            <FormulaBlock
              title="Attention Memory Formula (Per Layer)"
              formula="memory_attention = batch_size × seq_length^2 × num_heads × 4 / 10^9 GB"
              explanation="Attention scores stored as FP32. For multiple layers processing in parallel, multiply by number of concurrent layers."
              variables={[
                { symbol: 'seq_length', description: 'Sequence length (capped at model max_tokens)' },
                { symbol: 'num_heads', description: 'Number of attention heads in the model' },
              ]}
            />
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm">
              <strong>Note:</strong> Our implementation uses a conservative estimate that accounts for
              potential parallel layer processing during optimization passes. This may overestimate
              memory but ensures safety margins.
            </div>
          </div>
        ),
      },
      {
        id: 'embedding-storage',
        title: '4. Embedding Output Storage',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Embedding Storage Formula"
              formula="memory_embeddings = batch_size × embedding_dimension × 4 / 10^9 GB"
              explanation="Output embeddings stored as FP32 vectors"
              variables={[
                { symbol: 'embedding_dimension', description: 'Output embedding size (e.g., 768, 1024)' },
              ]}
            />
          </div>
        ),
      },
      {
        id: 'embedding-activations',
        title: '5. Activation Memory',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Activation Memory Formula"
              formula="memory_activation = batch_size × seq_length × hidden_size × num_layers × 2 × 4 / 10^9 GB"
              explanation="Intermediate activations in FFN layers, layer norms, etc. Factor of 2 accounts for bidirectional processing."
              variables={[
                { symbol: 'num_layers', description: 'Number of transformer layers' },
              ]}
            />
          </div>
        ),
      },
      {
        id: 'embedding-overhead',
        title: '6. Framework & Multi-GPU Overhead',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Framework Overhead Formula"
              formula="memory_overhead = (memory_weights + memory_activation) × 0.10"
              explanation="10% overhead for CUDA kernels, temporary buffers, and internal data structures"
            />
            <FormulaBlock
              title="Multi-GPU Overhead Formula"
              formula="memory_multi_gpu = base_memory × 0.02 × (num_gpus - 1)"
              explanation="2% overhead per additional GPU for communication buffers"
            />
          </div>
        ),
      },
    ],
  },
  {
    id: 'reranking-models',
    title: 'Reranking Model Calculations',
    content: (
      <div className="space-y-4">
        <p>
          Reranking models (cross-encoders) process query-document pairs by concatenating them
          and running through a transformer to produce relevance scores. They're more accurate
          than bi-encoders but computationally intensive.
        </p>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h5 className="font-semibold text-gray-900 mb-2">Key Insight:</h5>
          <p className="text-sm">
            Memory scales with <strong>batch size</strong>, not total query-document pairs.
            If you have 10 queries × 100 documents = 1,000 pairs but use batch_size=32,
            only 32 pairs are in memory at once.
          </p>
        </div>
        <p>
          Memory requirements for reranking models consist of:
        </p>
        <ol className="list-decimal pl-6 space-y-2">
          <li><strong>Model Weights Memory</strong>: Storage for the model parameters</li>
          <li><strong>Pair Batch Memory</strong>: Storage for query-document pairs in current batch</li>
          <li><strong>Attention Memory</strong>: Storage for self-attention (one layer at a time)</li>
          <li><strong>Scoring Memory</strong>: Storage for relevance scores</li>
          <li><strong>Activation Memory</strong>: Storage for intermediate computations</li>
          <li><strong>Framework Overhead</strong>: Additional memory for buffers (10%)</li>
          <li><strong>Multi-GPU Overhead</strong>: Communication overhead (2% per GPU)</li>
        </ol>
      </div>
    ),
    subsections: [
      {
        id: 'reranking-weights',
        title: '1. Model Weights',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Reranking Model Weights Formula"
              formula="memory_weights = (parameters_millions × 10^6 × bits_per_param) / 8 / 10^9 GB"
              explanation="Reranking models are typically 300M-1B parameters"
              variables={[
                { symbol: 'parameters_millions', description: 'Model size in millions (e.g., 560M for bge-reranker-large)' },
              ]}
            />
          </div>
        ),
      },
      {
        id: 'reranking-batch',
        title: '2. Pair Batch Memory',
        content: (
          <div className="space-y-4">
            <p>
              Memory for the current batch of query-document pairs being processed.
            </p>
            <FormulaBlock
              title="Pair Batch Memory Formula"
              formula="memory_batch = effective_batch_size × pair_length × hidden_size × 4 / 10^9 GB"
              explanation="Where effective_batch_size = min(batch_size, total_pairs) and pair_length = min(query_len + doc_len, model_max_length)"
              variables={[
                { symbol: 'effective_batch_size', description: 'Actual number of pairs processed simultaneously' },
                { symbol: 'pair_length', description: 'Combined query + document length (capped by model limit)' },
              ]}
            />
          </div>
        ),
      },
      {
        id: 'reranking-attention',
        title: '3. Attention Memory',
        content: (
          <div className="space-y-4">
            <p>
              Cross-attention over concatenated query-document sequences. Only one layer's
              attention is stored at a time during the forward pass.
            </p>
            <FormulaBlock
              title="Attention Memory Formula"
              formula="memory_attention = effective_batch_size × pair_length^2 × num_heads × 4 / 10^9 GB"
              explanation="Self-attention over concatenated sequences, stored as FP32"
            />
          </div>
        ),
      },
      {
        id: 'reranking-scoring',
        title: '4. Scoring Memory',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Scoring Memory Formula"
              formula="memory_scoring = effective_batch_size × 4 / 10^9 GB"
              explanation="Stores FP32 relevance scores for each query-document pair"
            />
          </div>
        ),
      },
      {
        id: 'reranking-activations',
        title: '5. Activation Memory',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Activation Memory Formula"
              formula="memory_activation = effective_batch_size × pair_length × hidden_size × 4 × 4 / 10^9 GB"
              explanation="4× multiplier accounts for: attention output, FFN up projection, FFN down projection, residual connections"
            />
          </div>
        ),
      },
      {
        id: 'reranking-overhead',
        title: '6. Framework & Multi-GPU Overhead',
        content: (
          <div className="space-y-4">
            <FormulaBlock
              title="Framework Overhead Formula"
              formula="memory_overhead = (memory_weights + memory_activation) × 0.10"
              explanation="10% overhead for framework operations"
            />
            <FormulaBlock
              title="Multi-GPU Overhead Formula"
              formula="memory_multi_gpu = base_memory × 0.02 × (num_gpus - 1)"
              explanation="2% overhead per additional GPU"
            />
          </div>
        ),
      },
    ],
  },
  {
    id: 'examples',
    title: 'Example Calculations',
    content: (
      <div className="space-y-4">
        <p>
          Let's walk through complete memory calculations for different model sizes and configurations.
        </p>

        <ExampleCalculation
          title="Example 1: Small Model (7B) on Consumer GPU"
          description="Llama-2-7B with FP16 quantization, single user, moderate sequence length. This model uses MHA (num_kv_heads = num_heads = 32)."
          parameters={[
            { label: 'Model', value: '7B parameters (MHA)' },
            { label: 'Quantization', value: 'FP16 (16 bits)' },
            { label: 'Batch Size', value: '1' },
            { label: 'Sequence Length', value: '2048 tokens' },
            { label: 'Hidden Size', value: '4096' },
            { label: 'Num Layers', value: '32' },
            { label: 'Heads / KV Heads', value: '32 / 32 (MHA)' },
            { label: 'GPUs', value: '1' },
          ]}
          steps={[
            {
              label: 'Model Weights',
              calculation: '(7 × 10^9 × 16) / 8 / 10^9',
              result: '14.00 GB'
            },
            {
              label: 'KV Cache',
              calculation: '2 × 32 × 32 × 128 × 2 × 1 × 2048 / 10^9',
              result: '1.07 GB'
            },
            {
              label: 'Activations',
              calculation: '1 × 2048 × 16384 × 2 / 10^9',
              result: '0.07 GB'
            },
            {
              label: 'Framework Overhead',
              calculation: '1.5 + (14.00 + 1.07 + 0.07) × 0.05',
              result: '2.26 GB'
            },
          ]}
          totalMemory="17.40 GB"
          notes={[
            'Fits comfortably on RTX 4090 (24GB) or RTX 3090 (24GB)',
            'Framework overhead includes 1.5 GB fixed baseline for CUDA/engine',
            'Batch size can be increased to 4-8 for higher throughput',
          ]}
        />

        <ExampleCalculation
          title="Example 2: Large Model (70B) with GQA on Enterprise GPU"
          description="Llama-2-70B with INT8 quantization and Grouped Query Attention (GQA). GQA reduces KV cache by 8x compared to standard MHA."
          parameters={[
            { label: 'Model', value: '70B parameters (GQA)' },
            { label: 'Quantization', value: 'INT8 (8 bits) weights, FP16 KV cache' },
            { label: 'Batch Size', value: '4' },
            { label: 'Sequence Length', value: '4096 tokens' },
            { label: 'Hidden Size', value: '8192' },
            { label: 'Num Layers', value: '80' },
            { label: 'Heads / KV Heads', value: '64 / 8 (GQA, 8x reduction)' },
            { label: 'GPUs', value: '1' },
          ]}
          steps={[
            {
              label: 'Model Weights',
              calculation: '(70 × 10^9 × 8) / 8 / 10^9',
              result: '70.00 GB'
            },
            {
              label: 'KV Cache (GQA: 8 KV heads × 128 head_size)',
              calculation: '2 × 80 × 8 × 128 × 2 × 4 × 4096 / 10^9',
              result: '5.37 GB'
            },
            {
              label: 'Activations',
              calculation: '4 × 4096 × 32768 × 2 / 10^9',
              result: '1.07 GB'
            },
            {
              label: 'Framework Overhead',
              calculation: '1.5 + (70.00 + 5.37 + 1.07) × 0.05',
              result: '5.32 GB'
            },
          ]}
          totalMemory="81.76 GB"
          notes={[
            'GQA reduces KV cache from ~43 GB (MHA) to ~5.4 GB -- a major savings',
            'Fits on a single A100 80GB or H100 80GB, though tightly',
            'INT8 weight quantization halves the 140 GB FP16 weight footprint',
            'Without GQA this configuration would require ~120+ GB',
          ]}
        />

        <ExampleCalculation
          title="Example 3: Very Large Model (175B) on Multi-GPU Setup"
          description="GPT-3 scale model with FP16 (MHA), distributed across 8 GPUs with tensor parallelism"
          parameters={[
            { label: 'Model', value: '175B parameters (MHA)' },
            { label: 'Quantization', value: 'FP16 (16 bits)' },
            { label: 'Batch Size', value: '8' },
            { label: 'Sequence Length', value: '2048 tokens' },
            { label: 'Hidden Size', value: '12288' },
            { label: 'Num Layers', value: '96' },
            { label: 'Heads / KV Heads', value: '96 / 96 (MHA)' },
            { label: 'GPUs', value: '8' },
          ]}
          steps={[
            {
              label: 'Model Weights',
              calculation: '(175 × 10^9 × 16) / 8 / 10^9',
              result: '350.00 GB'
            },
            {
              label: 'KV Cache (MHA)',
              calculation: '2 × 96 × 96 × 128 × 2 × 8 × 2048 / 10^9',
              result: '77.31 GB'
            },
            {
              label: 'Activations (÷8 GPUs)',
              calculation: '8 × 2048 × 49152 × 2 / 8 / 10^9',
              result: '0.20 GB'
            },
            {
              label: 'Framework Overhead (incl. NCCL)',
              calculation: '1.5 + (350 + 77.31 + 0.20) × 0.05 + 0.5 × 7',
              result: '26.38 GB'
            },
            {
              label: 'Multi-GPU Overhead (1% of weights per extra GPU)',
              calculation: '350 × 0.01 × (8 - 1)',
              result: '24.50 GB'
            },
          ]}
          totalMemory="478.39 GB (59.80 GB per GPU)"
          notes={[
            'Requires 8x H100 80GB GPUs with high-speed interconnect (NVLink)',
            'Tensor parallelism distributes weights across GPUs',
            'Each GPU needs ~60 GB, fitting on an H100 80GB',
            'Framework overhead includes 3.5 GB for NCCL buffers (0.5 GB x 7 extra GPUs)',
            'Activations divided by 8 GPUs in tensor parallelism',
          ]}
        />
      </div>
    ),
  },
  {
    id: 'best-practices',
    title: 'Best Practices',
    content: (
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Optimization Strategies</h3>
          <div className="space-y-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="font-semibold text-gray-900 mb-2">1. Choose the Right Quantization</h4>
              <ul className="list-disc pl-6 space-y-1 text-sm">
                <li><strong>FP16</strong>: Best balance for most use cases. Minimal accuracy loss, 50% memory reduction.</li>
                <li><strong>INT8</strong>: Excellent for production. 75% memory reduction with careful calibration.</li>
                <li><strong>INT4</strong>: Aggressive compression for resource-constrained scenarios. Test accuracy carefully.</li>
                <li><strong>KV Cache Quantization</strong>: Can often use INT8 for KV cache even with FP16 weights.</li>
              </ul>
            </div>

            <div className="border-l-4 border-green-500 pl-4">
              <h4 className="font-semibold text-gray-900 mb-2">2. Optimize Batch Size</h4>
              <ul className="list-disc pl-6 space-y-1 text-sm">
                <li>Start with batch size = 1 and gradually increase until GPU utilization is high</li>
                <li>Larger batches improve throughput but increase latency and memory usage</li>
                <li>Monitor GPU memory utilization (target 80-90% for efficiency)</li>
                <li>Use dynamic batching to automatically group requests</li>
              </ul>
            </div>

            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="font-semibold text-gray-900 mb-2">3. Manage Sequence Length</h4>
              <ul className="list-disc pl-6 space-y-1 text-sm">
                <li>Longer sequences dramatically increase KV cache memory</li>
                <li>Use sliding window attention for very long documents</li>
                <li>Consider chunking long inputs if full context isn't needed</li>
                <li>Monitor actual usage - most queries use far less than max context</li>
              </ul>
            </div>

            <div className="border-l-4 border-orange-500 pl-4">
              <h4 className="font-semibold text-gray-900 mb-2">4. Multi-GPU Efficiency</h4>
              <ul className="list-disc pl-6 space-y-1 text-sm">
                <li>Use tensor parallelism for latency-sensitive applications</li>
                <li>Use pipeline parallelism for throughput-focused workloads</li>
                <li>Ensure high-speed GPU interconnect (NVLink, InfiniBand)</li>
                <li>Consider cost vs. benefit - more GPUs = more overhead</li>
              </ul>
            </div>
          </div>
        </div>



        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">💡 Pro Tips</h3>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>Always add 10-20% buffer</strong> to calculated memory requirements for safety</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>Monitor actual usage</strong> in production - theoretical calculations are estimates</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>Test different quantization levels</strong> with your specific use case</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>Consider memory bandwidth</strong>, not just capacity - faster memory = better performance</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>Profile your workload</strong> before scaling - understand your actual patterns</span>
            </li>
          </ul>
        </div>
      </div>
    ),
  },
  {
    id: 'performance',
    title: 'Performance Metrics',
    content: (
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Understanding Throughput</h3>
          <p className="mb-4">
            Throughput measures how many tokens the system can process per second. It's influenced by
            model size, quantization, batch size, and hardware capabilities.
          </p>

          <FormulaBlock
            title="Approximate Throughput"
            formula="throughput ≈ (memory_bandwidth / bytes_per_token) × batch_size"
            explanation="Memory bandwidth is often the bottleneck for LLM inference, especially for large models."
            variables={[
              { symbol: 'memory_bandwidth', description: 'GPU memory bandwidth (GB/s)' },
              { symbol: 'bytes_per_token', description: 'Memory accessed per token (depends on model size)' },
              { symbol: 'batch_size', description: 'Number of concurrent sequences' },
            ]}
          />

          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Memory-Bound Scenario</h4>
              <p className="text-sm text-gray-700">
                Most LLM inference is <strong>memory-bound</strong>, meaning throughput is limited by
                how fast data can be loaded from GPU memory, not by compute speed.
              </p>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Compute-Bound Scenario</h4>
              <p className="text-sm text-gray-700">
                Very large batch sizes or small models may become <strong>compute-bound</strong>,
                where GPU compute is the limiting factor.
              </p>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Latency Considerations</h3>
          <div className="space-y-4">
            <div className="border-l-4 border-orange-500 pl-4">
              <h4 className="font-semibold text-gray-900 mb-2">Time to First Token (TTFT)</h4>
              <p className="text-sm text-gray-700">
                The time between sending a prompt and receiving the first generated token. Affected by:
              </p>
              <ul className="list-disc pl-6 mt-2 text-sm space-y-1">
                <li>Prompt length (longer prompts = longer TTFT)</li>
                <li>Model size (larger models = slower processing)</li>
                <li>GPU compute capability</li>
                <li>Current batch size (more concurrent requests = higher latency)</li>
              </ul>
            </div>

            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="font-semibold text-gray-900 mb-2">Inter-Token Latency</h4>
              <p className="text-sm text-gray-700">
                The time between successive generated tokens. Generally more consistent than TTFT:
              </p>
              <ul className="list-disc pl-6 mt-2 text-sm space-y-1">
                <li>Primarily determined by memory bandwidth</li>
                <li>Less affected by prompt length (KV cache is already populated)</li>
                <li>Can be optimized with speculative decoding</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-300">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">Model Size</th>
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">GPU</th>
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">Typical TTFT</th>
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">Throughput</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                <tr>
                  <td className="px-4 py-2 text-sm">7B FP16</td>
                  <td className="px-4 py-2 text-sm">RTX 4090</td>
                  <td className="px-4 py-2 text-sm">50-100ms</td>
                  <td className="px-4 py-2 text-sm">80-120 tok/s</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-sm">13B FP16</td>
                  <td className="px-4 py-2 text-sm">A100 40GB</td>
                  <td className="px-4 py-2 text-sm">100-150ms</td>
                  <td className="px-4 py-2 text-sm">50-80 tok/s</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-sm">70B INT8</td>
                  <td className="px-4 py-2 text-sm">A100 80GB</td>
                  <td className="px-4 py-2 text-sm">200-300ms</td>
                  <td className="px-4 py-2 text-sm">20-40 tok/s</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-sm">70B FP16</td>
                  <td className="px-4 py-2 text-sm">2× H100 80GB</td>
                  <td className="px-4 py-2 text-sm">150-250ms</td>
                  <td className="px-4 py-2 text-sm">40-70 tok/s</td>
                </tr>
              </tbody>
            </table>
            <p className="text-xs text-gray-600 mt-2">
              * Values are approximate and depend on specific implementation, batch size, and prompt length
            </p>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 'glossary',
    title: 'Glossary',
    content: (
      <div>
        <p className="mb-6 text-gray-700">
          Comprehensive list of technical terms used in LLM inference and memory calculation.
          Use the search and filter options to find specific terms.
        </p>
        <Glossary terms={glossaryTerms} />
      </div>
    ),
  },
];
