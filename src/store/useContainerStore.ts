import { create } from 'zustand';
import type {
  ContainerStore,
  ContainerState,
  VolumeMount,
  EnvironmentVariable,
  PortMapping,
  ContainerConfig,
  ConfigValidationResult,
  EngineParametersData,
  ContainerImagesData,
} from '../types';
import useAppStore from './useAppStore';
import engineParametersData from '../data/engine-parameters.json';
import containerImagesData from '../data/container-images.json';
import { validateContainerConfig, calculateSharedMemory } from '../utils/configValidator';

const initialState: ContainerState = {
  selectedEngineId: 'vllm',
  selectedImageId: 'rocm/vllm:latest',
  useContainerToolkit: true,
  containerName: 'vllm-inference',

  modelPath: './models',
  mountModelPath: true,
  mountHFCache: true,
  customVolumes: [],
  
  customEnvironment: [],
  
  apiPort: 8000,
  customPorts: [],
  
  useHostNetwork: false,
  customShmSize: undefined,
  trustRemoteCode: false,
  enableHealthcheck: true,
  autoRemoveContainer: true,
  
  customEngineParams: new Map(),
  
  outputFormat: 'docker-run',
  
  generatedConfig: null,
  validationResult: null,
};

export const useContainerStore = create<ContainerStore>((set, get) => ({
  ...initialState,

  // Engine and Image Selection
  setSelectedEngineId: (engineId: string) => set({ selectedEngineId: engineId }),
  
  setSelectedImageId: (imageId: string) => set({ selectedImageId: imageId }),
  
  // Container Runtime
  setUseContainerToolkit: (use: boolean) => set({ useContainerToolkit: use }),

  setContainerName: (name: string) => {
    // Validate Docker container naming rules:
    // - Must start with alphanumeric character
    // - Can contain a-z, A-Z, 0-9, _, -, .
    // - Cannot be empty
    // - Must be between 1 and 253 characters
    const dockerNameRegex = /^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/;

    if (!name) {
      console.warn('Container name cannot be empty');
      return;
    }

    if (name.length > 253) {
      console.warn('Container name too long (max 253 characters)');
      return;
    }

    if (!dockerNameRegex.test(name)) {
      console.warn(
        'Invalid container name. Must start with alphanumeric and contain only a-z, A-Z, 0-9, _, -, .'
      );
      return;
    }

    set({ containerName: name });
  },
  
  // Volumes
  setModelPath: (path: string) => set({ modelPath: path }),

  setMountModelPath: (mount: boolean) => set({ mountModelPath: mount }),

  setMountHFCache: (mount: boolean) => set({ mountHFCache: mount }),
  
  addCustomVolume: (volume: VolumeMount) =>
    set((state) => ({
      customVolumes: [...state.customVolumes, volume],
    })),
  
  removeCustomVolume: (index: number) =>
    set((state) => ({
      customVolumes: state.customVolumes.filter((_, i) => i !== index),
    })),
  
  // Environment
  addEnvironmentVariable: (env: EnvironmentVariable) =>
    set((state) => ({
      customEnvironment: [...state.customEnvironment, env],
    })),
  
  removeEnvironmentVariable: (index: number) =>
    set((state) => ({
      customEnvironment: state.customEnvironment.filter((_, i) => i !== index),
    })),
  
  // Ports
  setApiPort: (port: number) => set({ apiPort: port }),
  
  addCustomPort: (port: PortMapping) =>
    set((state) => ({
      customPorts: [...state.customPorts, port],
    })),
  
  removeCustomPort: (index: number) =>
    set((state) => ({
      customPorts: state.customPorts.filter((_, i) => i !== index),
    })),
  
  // Advanced Options
  setUseHostNetwork: (use: boolean) => set({ useHostNetwork: use }),
  
  setCustomShmSize: (size: string | undefined) => set({ customShmSize: size }),
  
  setTrustRemoteCode: (trust: boolean) => set({ trustRemoteCode: trust }),
  
  setEnableHealthcheck: (enable: boolean) => set({ enableHealthcheck: enable }),
  
  setAutoRemoveContainer: (autoRemove: boolean) => set({ autoRemoveContainer: autoRemove }),
  
  // Engine Parameters
  setCustomEngineParam: (flag: string, value: string | number | boolean | null) =>
    set((state) => {
      const newParams = new Map(state.customEngineParams);
      if (value === null) {
        newParams.delete(flag);
      } else {
        newParams.set(flag, value);
      }
      return { customEngineParams: newParams };
    }),
  
  clearCustomEngineParam: (flag: string) =>
    set((state) => {
      const newParams = new Map(state.customEngineParams);
      newParams.delete(flag);
      return { customEngineParams: newParams };
    }),
  
  // Output
  setOutputFormat: (format) => set({ outputFormat: format }),
  
  // Generation
  generateConfig: async () => {
    try {
      const state = get();
      const appState = useAppStore.getState();

      // Get selected engine
      const engine = (engineParametersData as EngineParametersData).engines.find(
        (e) => e.id === state.selectedEngineId
      );
      if (!engine) {
        console.error('Engine not found:', state.selectedEngineId);
        set({
          generatedConfig: null,
          validationResult: {
            valid: false,
            messages: [
              {
                level: 'error',
                message: `Engine "${state.selectedEngineId}" not found`,
                suggestion: 'Please select a valid inference engine',
              },
            ],
            securityIssues: [],
            recommendations: [],
          },
        });
        return;
      }

      // Get selected image
      const image = (containerImagesData as ContainerImagesData).images.find(
        (img) => img.fullImage === state.selectedImageId
      );
      if (!image) {
        console.error('Image not found:', state.selectedImageId);
        set({
          generatedConfig: null,
          validationResult: {
            valid: false,
            messages: [
              {
                level: 'error',
                message: `Container image "${state.selectedImageId}" not found`,
                suggestion: 'Please select a valid container image',
              },
            ],
            securityIssues: [],
            recommendations: [],
          },
        });
        return;
      }

      // Get model information from app store
      const modelId = appState.selectedModel;
      const gpuId = appState.selectedGPU;
      const numGPUs = appState.numGPUs;

      // Import GPU and model data with proper error handling
      const [gpusData, modelsData] = await Promise.all([
        import('../data/gpus.json'),
        import('../data/models.json'),
      ]);

      const gpu = gpusData.gpus.find((g) => g.id === gpuId);
      const model = modelsData.models.find((m) => m.id === modelId);

      if (!gpu) {
        console.error('GPU not found:', gpuId);
        set({
          generatedConfig: null,
          validationResult: {
            valid: false,
            messages: [
              {
                level: 'error',
                message: `GPU "${gpuId}" not found`,
                suggestion: 'Please select a valid GPU from the Calculator tab',
              },
            ],
            securityIssues: [],
            recommendations: [],
          },
        });
        return;
      }

      if (!model) {
        console.error('Model not found:', modelId);
        set({
          generatedConfig: null,
          validationResult: {
            valid: false,
            messages: [
              {
                level: 'error',
                message: `Model "${modelId}" not found`,
                suggestion: 'Please select a valid model from the Calculator tab',
              },
            ],
            securityIssues: [],
            recommendations: [],
          },
        });
        return;
      }

      // Calculate shared memory size
      const shmSize =
        state.customShmSize || calculateShmSize(numGPUs, model.parameters_billions);

      // Build GPU IDs array
      const gpuIds = Array.from({ length: numGPUs }, (_, i) => i.toString());

      // Build volumes
      const volumes: VolumeMount[] = [];

      if (state.mountModelPath) {
        volumes.push({
          hostPath: state.modelPath,
          containerPath: '/models',
          readOnly: false,
          description: 'Model storage directory',
        });
      }

      if (state.mountHFCache) {
        volumes.push({
          hostPath: '~/.cache/huggingface',
          containerPath: '/root/.cache/huggingface',
          readOnly: false,
          description: 'HuggingFace cache',
        });
      }

      volumes.push(...state.customVolumes);

      // Build environment variables
      const environment: EnvironmentVariable[] = [];

      if (state.useContainerToolkit) {
        // Use "all" for 8 or more GPUs, otherwise list specific IDs
        const gpuValue = numGPUs >= 8 ? 'all' : gpuIds.join(',');
        environment.push({
          key: 'AMD_VISIBLE_DEVICES',
          value: gpuValue,
          description: 'AMD GPUs visible to container',
        });
      } else {
        environment.push({
          key: 'ROCR_VISIBLE_DEVICES',
          value: gpuIds.join(','),
          description: 'ROCm visible devices',
        });
      }

      // Add HF_TOKEN if needed
      environment.push({
        key: 'HF_TOKEN',
        value: '${HF_TOKEN}',
        sensitive: true,
        description: 'HuggingFace API token for gated models',
      });

      environment.push(...state.customEnvironment);

      // Build ports
      const ports: PortMapping[] = [
        {
          host: state.apiPort,
          container: 8000,
          protocol: 'tcp',
          description: 'vLLM API endpoint',
        },
        ...state.customPorts,
      ];

      // Build engine parameters
      const engineParams = buildEngineParams(
        engine,
        appState,
        state,
        model,
        numGPUs
      );

      // Create config
      const config: ContainerConfig = {
        engine,
        image,
        containerName: state.containerName,
        useContainerToolkit: state.useContainerToolkit,
        autoRemove: state.autoRemoveContainer,
        gpuIds,
        gpuCount: numGPUs,
        shmSize,
        volumes,
        environment,
        ports,
        useHostNetwork: state.useHostNetwork,
        engineParams,
        model: {
          id: model.id,
          name: model.name,
          parameters: model.parameters_billions,
        },
        gpus: Array.from({ length: numGPUs }, () => ({
          id: gpu.id,
          name: gpu.name,
          vram: gpu.vram_gb,
        })),
        memoryUsage: {
          estimated: appState.results?.usedVRAM || 0,
          available: gpu.vram_gb * numGPUs,
          percentage: appState.results?.vramPercentage || 0,
        },
      };

      // Validate config
      const validationResult = get().validateConfig(config);

      set({
        generatedConfig: config,
        validationResult,
      });
    } catch (error) {
      console.error('Failed to generate container configuration:', error);
      set({
        generatedConfig: null,
        validationResult: {
          valid: false,
          messages: [
            {
              level: 'error',
              message: 'Failed to generate configuration',
              suggestion:
                error instanceof Error
                  ? error.message
                  : 'An unexpected error occurred. Please try again.',
            },
          ],
          securityIssues: [],
          recommendations: [],
        },
      });
    }
  },
  
  validateConfig: (config: ContainerConfig): ConfigValidationResult => {
    // Use the centralized validator
    return validateContainerConfig(config);
  },
  
  // Reset
  resetToDefaults: () => set(initialState),
}));

// Helper Functions
function calculateShmSize(gpuCount: number, modelParams: number): string {
  const calculation = calculateSharedMemory(gpuCount, modelParams);
  return `${calculation.recommended}g`;
}

function buildEngineParams(
  _engine: any,
  appState: any,
  containerState: ContainerState,
  model: any,
  numGPUs: number
): Array<{ flag: string; value: string | number | boolean }> {
  const params: Array<{ flag: string; value: string | number | boolean }> = [];
  
  // Helper to check if custom override exists
  const hasCustom = (flag: string) => containerState.customEngineParams.has(flag);
  const getCustom = (flag: string) => containerState.customEngineParams.get(flag);
  
  // Model (always required, but can be overridden)
  // Use hf_model_id if available (includes org prefix), otherwise fallback to id
  const modelIdentifier = model.hf_model_id || model.id;
  params.push({
    flag: '--model',
    value: hasCustom('--model') ? (getCustom('--model') as string) : modelIdentifier,
  });
  
  // Tensor parallelism (can be overridden for advanced use cases)
  params.push({
    flag: '--tensor-parallel-size',
    value: hasCustom('--tensor-parallel-size') ? (getCustom('--tensor-parallel-size') as number) : numGPUs,
  });
  
  // Dtype based on quantization (can be overridden)
  if (!hasCustom('--dtype')) {
    const quantMap: Record<string, string> = {
      fp16: 'float16',
      fp8: 'float16',
      int8: 'float16',
      int4: 'float16',
    };
    params.push({
      flag: '--dtype',
      value: quantMap[appState.inferenceQuantization] || 'auto',
    });
  } else {
    params.push({
      flag: '--dtype',
      value: getCustom('--dtype') as string,
    });
  }
  
  // Quantization (can be overridden)
  if (!hasCustom('--quantization')) {
    if (appState.inferenceQuantization === 'int8' || appState.inferenceQuantization === 'int4') {
      params.push({
        flag: '--quantization',
        value: 'gptq',
      });
    }
  } else if (getCustom('--quantization')) {
    params.push({
      flag: '--quantization',
      value: getCustom('--quantization') as string,
    });
  }
  
  // Max model length (can be overridden)
  if (hasCustom('--max-model-len')) {
    params.push({
      flag: '--max-model-len',
      value: getCustom('--max-model-len') as number,
    });
  } else if (appState.sequenceLength) {
    params.push({
      flag: '--max-model-len',
      value: appState.sequenceLength,
    });
  }
  
  // KV cache dtype (can be overridden)
  if (hasCustom('--kv-cache-dtype')) {
    params.push({
      flag: '--kv-cache-dtype',
      value: getCustom('--kv-cache-dtype') as string,
    });
  } else if (appState.kvCacheQuantization === 'fp8_bf16') {
    params.push({
      flag: '--kv-cache-dtype',
      value: 'fp8',
    });
  }
  
  // Trust remote code (can be overridden)
  if (hasCustom('--trust-remote-code')) {
    if (getCustom('--trust-remote-code')) {
      params.push({
        flag: '--trust-remote-code',
        value: true,
      });
    }
  } else if (containerState.trustRemoteCode) {
    params.push({
      flag: '--trust-remote-code',
      value: true,
    });
  }
  
  // Add any additional custom parameters not covered above
  containerState.customEngineParams.forEach((value, flag) => {
    // Skip if already handled
    if (params.some(p => p.flag === flag)) {
      return;
    }
    params.push({ flag, value });
  });
  
  return params;
}
