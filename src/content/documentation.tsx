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
              title="KV Cache Memory Formula"
              formula="memory_kv = 2 × num_layers × seq_len × batch_size × hidden_size × kv_bits / 8 / 10^9 GB"
              explanation="Factor of 2 accounts for both keys and values. KV cache scales linearly with sequence length and batch size."
              variables={[
                { symbol: 'num_layers', description: 'Number of transformer layers in the model' },
                { symbol: 'seq_len', description: 'Maximum sequence length (context window)' },
                { symbol: 'batch_size', description: 'Number of sequences processed in parallel' },
                { symbol: 'hidden_size', description: 'Model hidden dimension' },
                { symbol: 'kv_bits', description: 'Precision for KV cache (often FP16 = 16 bits)' },
              ]}
            />

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 mb-2">Important Notes:</h5>
              <ul className="space-y-1 text-sm">
                <li>• KV cache memory scales <strong>linearly</strong> with sequence length</li>
                <li>• Doubling batch size doubles KV cache memory</li>
                <li>• KV cache can be quantized independently from model weights</li>
                <li>• For long contexts (32K+ tokens), KV cache often exceeds model weights memory</li>
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
              Activation memory stores intermediate values computed during the forward pass through
              the network. This includes outputs from each layer that are needed for subsequent computations.
            </p>
            
            <FormulaBlock
              title="Activation Memory Formula"
              formula="memory_activation = batch_size × seq_len × hidden_size × 4 / 10^9 GB"
              explanation="The factor of 4 is an approximation accounting for multiple activation tensors per layer. Activations are typically stored in FP32."
              variables={[
                { symbol: 'batch_size', description: 'Number of sequences processed in parallel' },
                { symbol: 'seq_len', description: 'Sequence length being processed' },
                { symbol: 'hidden_size', description: 'Model hidden dimension' },
              ]}
            />

            <p className="text-sm text-gray-700">
              Activation memory is generally smaller than KV cache memory but scales similarly with
              batch size and sequence length.
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
              Deep learning frameworks (PyTorch, TensorFlow, etc.) require additional memory for
              CUDA kernels, temporary buffers, and internal data structures.
            </p>
            
            <FormulaBlock
              title="Framework Overhead Formula"
              formula="memory_overhead = (memory_weights + memory_kv + memory_activation) × 0.08"
              explanation="Typically 5-10% of the base memory requirements. We use 8% for accuracy based on empirical measurements."
            />

            <p className="text-sm text-gray-700">
              This overhead is relatively small but important to account for in production deployments
              to avoid out-of-memory errors.
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
              When using multiple GPUs, additional memory is required for communication buffers,
              gradient synchronization, and maintaining consistency across devices.
            </p>
            
            <FormulaBlock
              title="Multi-GPU Overhead Formula"
              formula="memory_multi_gpu = base_memory × 0.02 × (num_gpus - 1)"
              explanation="Approximately 2% overhead per additional GPU for communication buffers and synchronization when using tensor parallelism."
              variables={[
                { symbol: 'base_memory', description: 'Sum of weights, KV cache, activation, and framework overhead' },
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
          description="Llama-2-7B with FP16 quantization, single user, moderate sequence length"
          parameters={[
            { label: 'Model', value: '7B parameters' },
            { label: 'Quantization', value: 'FP16 (16 bits)' },
            { label: 'Batch Size', value: '1' },
            { label: 'Sequence Length', value: '2048 tokens' },
            { label: 'Hidden Size', value: '4096' },
            { label: 'Num Layers', value: '32' },
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
              calculation: '2 × 32 × 2048 × 1 × 4096 × 16 / 8 / 10^9',
              result: '1.07 GB'
            },
            {
              label: 'Activations',
              calculation: '1 × 2048 × 4096 × 4 / 10^9',
              result: '0.03 GB'
            },
            {
              label: 'Framework Overhead (8%)',
              calculation: '(14.00 + 1.07 + 0.03) × 0.08',
              result: '1.21 GB'
            },
          ]}
          totalMemory="16.31 GB"
          notes={[
            'Fits comfortably on RTX 4090 (24GB) or RTX 3090 (24GB)',
            'Can handle up to 4K context with careful optimization',
            'Batch size can be increased to 4-8 for higher throughput',
          ]}
        />

        <ExampleCalculation
          title="Example 2: Medium Model (70B) on Enterprise GPU"
          description="Llama-2-70B with INT8 quantization, higher batch size"
          parameters={[
            { label: 'Model', value: '70B parameters' },
            { label: 'Quantization', value: 'INT8 (8 bits)' },
            { label: 'Batch Size', value: '4' },
            { label: 'Sequence Length', value: '4096 tokens' },
            { label: 'Hidden Size', value: '8192' },
            { label: 'Num Layers', value: '80' },
            { label: 'GPUs', value: '1' },
          ]}
          steps={[
            {
              label: 'Model Weights',
              calculation: '(70 × 10^9 × 8) / 8 / 10^9',
              result: '70.00 GB'
            },
            {
              label: 'KV Cache',
              calculation: '2 × 80 × 4096 × 4 × 8192 × 16 / 8 / 10^9',
              result: '34.36 GB'
            },
            {
              label: 'Activations',
              calculation: '4 × 4096 × 8192 × 4 / 10^9',
              result: '0.54 GB'
            },
            {
              label: 'Framework Overhead (8%)',
              calculation: '(70.00 + 34.36 + 0.54) × 0.08',
              result: '8.39 GB'
            },
          ]}
          totalMemory="113.29 GB"
          notes={[
            'Requires A100 80GB or H100 80GB',
            'INT8 quantization makes this possible on single GPU',
            'KV cache is ~50% of total memory due to high batch size and long context',
            'Can serve 4 concurrent users with 4K context each',
          ]}
        />

        <ExampleCalculation
          title="Example 3: Large Model (175B) on Multi-GPU Setup"
          description="GPT-3 scale model with FP16, distributed across multiple GPUs"
          parameters={[
            { label: 'Model', value: '175B parameters' },
            { label: 'Quantization', value: 'FP16 (16 bits)' },
            { label: 'Batch Size', value: '8' },
            { label: 'Sequence Length', value: '2048 tokens' },
            { label: 'Hidden Size', value: '12288' },
            { label: 'Num Layers', value: '96' },
            { label: 'GPUs', value: '8' },
          ]}
          steps={[
            {
              label: 'Model Weights',
              calculation: '(175 × 10^9 × 16) / 8 / 10^9',
              result: '350.00 GB'
            },
            {
              label: 'KV Cache',
              calculation: '2 × 96 × 2048 × 8 × 12288 × 16 / 8 / 10^9',
              result: '38.65 GB'
            },
            {
              label: 'Activations',
              calculation: '8 × 2048 × 12288 × 4 / 10^9',
              result: '0.80 GB'
            },
            {
              label: 'Framework Overhead (8%)',
              calculation: '(350.00 + 38.65 + 0.80) × 0.08',
              result: '31.16 GB'
            },
            {
              label: 'Multi-GPU Overhead (2% per extra GPU)',
              calculation: '(350.00 + 38.65 + 0.80 + 31.16) × 0.02 × (8 - 1)',
              result: '58.89 GB'
            },
          ]}
          totalMemory="479.50 GB (59.94 GB per GPU)"
          notes={[
            'Requires 8× H100 80GB GPUs',
            'Tensor parallelism distributes weights across GPUs',
            'Each GPU needs ~60GB, fitting on an H100 80GB',
            'Can support 8 concurrent users with 2K context',
            'Multi-GPU overhead is significant in large clusters',
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
