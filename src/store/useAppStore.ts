import { create } from 'zustand';
import { AppStore, InferenceQuantization, KVCacheQuantization } from '../types';

const useAppStore = create<AppStore>((set) => ({
  // Initial state
  modelType: 'generation',
  selectedModel: 'llama-3-70b',
  customModelParams: undefined,
  customHiddenSize: undefined,
  customNumLayers: undefined,
  customNumHeads: undefined,
  inferenceQuantization: 'fp16',
  kvCacheQuantization: 'fp16_bf16',
  selectedGPU: 'mi300x',
  customVRAM: undefined,
  numGPUs: 1,
  batchSize: 1,
  sequenceLength: 4096,
  concurrentUsers: 1,
  numImages: 1,
  imageResolution: 336,
  embeddingBatchSize: 256,
  documentsPerBatch: 64,
  avgDocumentSize: 512,
  chunkSize: 512,
  chunkOverlap: 20,
  rerankingBatchSize: 32,
  numQueries: 1,
  docsPerQuery: 50,
  maxQueryLength: 256,
  maxDocLength: 512,
  results: null,

  // Partitioning state
  partitioningGPU: null,
  partitioningMode: 'SPX',
  partitioningModelType: 'generation',
  partitioningShowOnlyFits: false,

  // Actions
  setModelType: (type) => set((state) => {
    // Set appropriate default model when switching types
    let newModel = state.selectedModel;
    if (type === 'embedding') {
      newModel = 'bge-large-en-v1.5';
    } else if (type === 'reranking') {
      newModel = 'bge-reranker-large';
    } else if (type === 'generation') {
      newModel = 'llama-3-70b';
    }
    return { modelType: type, selectedModel: newModel, results: null };
  }),
  setSelectedModel: (modelId: string) => set({ selectedModel: modelId }),
  setCustomModelParams: (params: number | undefined) => set({ customModelParams: params }),
  setInferenceQuantization: (quant: InferenceQuantization) => set({ inferenceQuantization: quant }),
  setKVCacheQuantization: (quant: KVCacheQuantization) => set({ kvCacheQuantization: quant }),
  setSelectedGPU: (gpuId: string) => set({ selectedGPU: gpuId }),
  setCustomVRAM: (vram: number | undefined) => set({ customVRAM: vram }),
  setNumGPUs: (num: number) => set({ numGPUs: num }),
  setBatchSize: (size: number) => set({ batchSize: size }),
  setSequenceLength: (length: number) => set({ sequenceLength: length }),
  setConcurrentUsers: (users: number) => set({ concurrentUsers: users }),
  setNumImages: (num: number) => set({ numImages: num }),
  setImageResolution: (resolution: number) => set({ imageResolution: resolution }),
  setEmbeddingBatchSize: (size: number) => set({ embeddingBatchSize: size }),
  setDocumentsPerBatch: (count: number) => set({ documentsPerBatch: count }),
  setAvgDocumentSize: (size: number) => set({ avgDocumentSize: size }),
  setChunkSize: (size: number) => set({ chunkSize: size }),
  setChunkOverlap: (overlap: number) => set({ chunkOverlap: overlap }),
  setRerankingBatchSize: (size: number) => set({ rerankingBatchSize: size }),
  setNumQueries: (count: number) => set({ numQueries: count }),
  setDocsPerQuery: (count: number) => set({ docsPerQuery: count }),
  setMaxQueryLength: (length: number) => set({ maxQueryLength: length }),
  setMaxDocLength: (length: number) => set({ maxDocLength: length }),
  calculateResults: () => {
    // This is handled by the useMemoryCalculation hook
    console.log('Calculating results...');
  },
  setResults: (results) => set({ results }),

  // Partitioning actions
  setPartitioningGPU: (gpuId: string | null) => set({ partitioningGPU: gpuId }),
  setPartitioningMode: (mode) => set({ partitioningMode: mode }),
  setPartitioningModelType: (type) => set({ partitioningModelType: type }),
  setPartitioningShowOnlyFits: (show: boolean) => set({ partitioningShowOnlyFits: show }),
}));

export default useAppStore;