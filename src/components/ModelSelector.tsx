import React, { useMemo } from 'react';
import { Info } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import { Model, EmbeddingModel, RerankingModel } from '../types';

interface ModelSelectorProps {
  models: Model[];
  embeddingModels?: EmbeddingModel[];
  rerankingModels?: RerankingModel[];
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ models, embeddingModels = [], rerankingModels = [] }) => {
  const {
    modelType,
    selectedModel,
    setSelectedModel,
  } = useAppStore();

  // Sort models by name
  const sortedModels = useMemo(() => [...models].sort((a, b) => a.name.localeCompare(b.name)), [models]);
  const sortedEmbeddingModels = useMemo(() => [...embeddingModels].sort((a, b) => a.name.localeCompare(b.name)), [embeddingModels]);
  const sortedRerankingModels = useMemo(() => [...rerankingModels].sort((a, b) => a.name.localeCompare(b.name)), [rerankingModels]);

  const currentModel = sortedModels.find(m => m.id === selectedModel);
  const currentEmbeddingModel = sortedEmbeddingModels.find(m => m.id === selectedModel);
  const currentRerankingModel = sortedRerankingModels.find(m => m.id === selectedModel);

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-2">
          Model Selection
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
        >
          {modelType === 'embedding' && sortedEmbeddingModels.map(model => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.parameters_millions}M params)
            </option>
          ))}
          {modelType === 'reranking' && sortedRerankingModels.map(model => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.parameters_millions}M params)
            </option>
          ))}
          {modelType === 'generation' && sortedModels.map(model => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.parameters_billions}B params)
            </option>
          ))}
        </select>
      </div>

      {currentEmbeddingModel && (
        <div className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center space-x-2">
            <Info className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-700">Model Details</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">Dimensions:</span>
              <span className="ml-2 font-medium">{currentEmbeddingModel.dimensions}d</span>
            </div>
            <div>
              <span className="text-gray-500">Max Tokens:</span>
              <span className="ml-2 font-medium">{currentEmbeddingModel.max_tokens}</span>
            </div>
          </div>
        </div>
      )}

      {currentRerankingModel && (
        <div className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center space-x-2">
            <Info className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-700">Model Details</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">Max Query:</span>
              <span className="ml-2 font-medium">{currentRerankingModel.max_query_length} tokens</span>
            </div>
            <div>
              <span className="text-gray-500">Max Doc:</span>
              <span className="ml-2 font-medium">{currentRerankingModel.max_doc_length} tokens</span>
            </div>
          </div>
        </div>
      )}

      {currentModel && modelType === 'generation' && (
        <div className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center space-x-2">
            <Info className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-700">Model Details</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">Hidden Size:</span>
              <span className="ml-2 font-medium">{currentModel.hidden_size}</span>
            </div>
            <div>
              <span className="text-gray-500">Layers:</span>
              <span className="ml-2 font-medium">{currentModel.num_layers}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;
