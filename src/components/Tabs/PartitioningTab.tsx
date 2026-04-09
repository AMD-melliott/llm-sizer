import { useEffect, useMemo } from 'react';
import useAppStore from '../../store/useAppStore';
import {
  GPU,
  PartitionConfiguration,
  PartitionAnalysisResults,
  Model,
  EmbeddingModel,
  RerankingModel,
} from '../../types';
import {
  getPartitionCapableGPUs,
  getPartitionMode,
  analyzePartitionConfiguration,
} from '../../utils/partitionCalculator';
import PartitionModeSelector from '../PartitionModeSelector';
import PartitionVisualization from '../PartitionVisualization';
import PartitionResults from '../PartitionResults';

import gpusData from '../../data/gpus.json';
import modelsData from '../../data/models.json';
import embeddingModelsData from '../../data/embedding-models.json';
import rerankingModelsData from '../../data/reranking-models.json';

export default function PartitioningTab() {
  const {
    partitioningGPU,
    partitioningMode,
    partitioningModelType,
    partitioningShowOnlyFits,
    inferenceQuantization,
    kvCacheQuantization,
    batchSize,
    sequenceLength,
    concurrentUsers,
    setPartitioningGPU,
    setPartitioningMode,
    setPartitioningModelType,
    setPartitioningShowOnlyFits,
  } = useAppStore();

  const gpus = gpusData.gpus as GPU[];
  const partitionCapableGPUs = useMemo(() => getPartitionCapableGPUs(gpus), [gpus]);

  useEffect(() => {
    if (!partitioningGPU && partitionCapableGPUs.length > 0) {
      setPartitioningGPU(partitionCapableGPUs[0].id);
    }
  }, [partitioningGPU, partitionCapableGPUs, setPartitioningGPU]);

  const selectedGPU = useMemo(
    () => partitionCapableGPUs.find((g) => g.id === partitioningGPU),
    [partitionCapableGPUs, partitioningGPU]
  );

  const availableModes = useMemo(
    () => selectedGPU?.partitioning?.modes || [],
    [selectedGPU]
  );

  const currentMode = useMemo(
    () => selectedGPU ? getPartitionMode(selectedGPU, partitioningMode) : undefined,
    [selectedGPU, partitioningMode]
  );

  const analysisResults = useMemo<PartitionAnalysisResults | null>(() => {
    if (!selectedGPU || !currentMode) return null;

    const configuration: PartitionConfiguration = {
      gpu: selectedGPU,
      mode: currentMode,
      inferenceQuantization,
      kvCacheQuantization,
      batchSize,
      sequenceLength,
      concurrentUsers,
    };

    const generationModels = modelsData.models as Model[];
    const embeddingModels = embeddingModelsData.models as EmbeddingModel[];
    const rerankingModels = (rerankingModelsData as any).models as RerankingModel[];

    let modelsToAnalyze: (Model | EmbeddingModel | RerankingModel)[];
    if (partitioningModelType === 'generation') {
      modelsToAnalyze = generationModels;
    } else if (partitioningModelType === 'embedding') {
      modelsToAnalyze = embeddingModels;
    } else if (partitioningModelType === 'reranking') {
      modelsToAnalyze = rerankingModels;
    } else {
      modelsToAnalyze = [...generationModels, ...embeddingModels, ...rerankingModels];
    }

    return analyzePartitionConfiguration(
      configuration,
      partitioningModelType,
      modelsToAnalyze
    );
  }, [
    selectedGPU,
    currentMode,
    inferenceQuantization,
    kvCacheQuantization,
    batchSize,
    sequenceLength,
    concurrentUsers,
    partitioningModelType,
  ]);

  if (partitionCapableGPUs.length === 0) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--warning-light)', border: '1px solid var(--warning)' }}>
          <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--warning-text)' }}>
            No Partition-Capable GPUs Available
          </h3>
          <p style={{ color: 'var(--text-secondary)' }}>
            GPU partitioning is currently only supported on AMD Instinct datacenter GPUs.
            Please check back later for expanded support.
          </p>
        </div>
      </div>
    );
  }

  const selectStyle = {
    backgroundColor: 'var(--surface)',
    borderColor: 'var(--border-strong)',
    color: 'var(--text-primary)',
  };

  const inputStyle = {
    backgroundColor: 'var(--surface)',
    borderColor: 'var(--border-strong)',
    color: 'var(--text-primary)',
  };

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
            GPU Partitioning Mode Visualization
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Analyze which models fit within AMD GPU partition configurations (SPX, DPX, CPX)
          </p>
        </div>

        {/* GPU Selection and Inference Parameters */}
        <div
          className="rounded-xl p-6"
          style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border-strong)', boxShadow: '0 1px 3px var(--shadow-color)' }}
        >
          <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* GPU Selection */}
            <div className="md:col-span-2">
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                GPU
              </label>
              <select
                value={partitioningGPU || ''}
                onChange={(e) => setPartitioningGPU(e.target.value)}
                className="w-full px-3 py-2 rounded-xl focus:outline-none focus:ring-2"
                style={selectStyle}
              >
                {partitionCapableGPUs.map((gpu) => (
                  <option key={gpu.id} value={gpu.id}>
                    {gpu.name} ({gpu.vram_gb}GB)
                  </option>
                ))}
              </select>
            </div>
            {/* Model Type Filter */}
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                Model Category
              </label>
              <select
                value={partitioningModelType}
                onChange={(e) => setPartitioningModelType(e.target.value as any)}
                className="w-full px-3 py-2 rounded-xl focus:outline-none focus:ring-2"
                style={selectStyle}
              >
                <option value="generation">Generation</option>
                <option value="embedding">Embedding</option>
                <option value="reranking">Reranking</option>
                <option value="all">All</option>
              </select>
            </div>
            {/* Inference Parameters */}
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                Quantization
              </label>
              <select
                value={inferenceQuantization}
                onChange={(e) => useAppStore.getState().setInferenceQuantization(e.target.value as any)}
                className="w-full px-3 py-2 rounded-xl focus:outline-none focus:ring-2"
                style={selectStyle}
              >
                <option value="fp16">FP16</option>
                <option value="fp8">FP8</option>
                <option value="int8">INT8</option>
                <option value="int4">INT4</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                Batch Size
              </label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => useAppStore.getState().setBatchSize(parseInt(e.target.value))}
                min="1"
                max="128"
                className="w-full px-2 py-2 rounded-xl focus:outline-none focus:ring-2 font-mono tabular-nums"
                style={inputStyle}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                Seq Length
              </label>
              <input
                type="number"
                value={sequenceLength}
                onChange={(e) => useAppStore.getState().setSequenceLength(parseInt(e.target.value))}
                min="512"
                max="32768"
                step="512"
                className="w-full px-2 py-2 rounded-xl focus:outline-none focus:ring-2 font-mono tabular-nums"
                style={inputStyle}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                Users
              </label>
              <input
                type="number"
                value={concurrentUsers}
                onChange={(e) => useAppStore.getState().setConcurrentUsers(parseInt(e.target.value))}
                min="1"
                max="100"
                className="w-full px-2 py-2 rounded-xl focus:outline-none focus:ring-2 font-mono tabular-nums"
                style={inputStyle}
              />
            </div>
          </div>
        </div>

        {/* Partition Visualization */}
        {selectedGPU && currentMode && (
          <PartitionVisualization mode={currentMode} gpuName={selectedGPU.name} />
        )}

        {/* Partition Mode Selection */}
        {availableModes.length > 0 && (
          <PartitionModeSelector
            modes={availableModes}
            selectedMode={partitioningMode}
            onModeChange={setPartitioningMode}
          />
        )}

        {/* Results */}
        {analysisResults && (
          <PartitionResults
            results={analysisResults}
            showOnlyFits={partitioningShowOnlyFits}
            onToggleShowOnlyFits={setPartitioningShowOnlyFits}
          />
        )}
      </div>
    </main>
  );
}
