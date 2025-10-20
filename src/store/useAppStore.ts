import { create } from 'zustand';
import { AppStore, InferenceQuantization, KVCacheQuantization } from '../types';

const useAppStore = create<AppStore>((set) => ({
  // Initial state
  selectedModel: 'llama-3-70b',
  customModelParams: undefined,
  customHiddenSize: undefined,
  customNumLayers: undefined,
  customNumHeads: undefined,
  inferenceQuantization: 'fp16',
  kvCacheQuantization: 'fp16_bf16',
  selectedGPU: 'h100-sxm',
  customVRAM: undefined,
  numGPUs: 1,
  batchSize: 1,
  sequenceLength: 4096,
  concurrentUsers: 1,
  enableOffloading: false,
  results: null,

  // Actions
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
  setEnableOffloading: (enable: boolean) => set({ enableOffloading: enable }),
  calculateResults: () => {
    // This is handled by the useMemoryCalculation hook
    console.log('Calculating results...');
  },
}));

export default useAppStore;