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

  const sortedModels = useMemo(() => [...models].sort((a, b) => a.name.localeCompare(b.name)), [models]);
  const sortedEmbeddingModels = useMemo(() => [...embeddingModels].sort((a, b) => a.name.localeCompare(b.name)), [embeddingModels]);
  const sortedRerankingModels = useMemo(() => [...rerankingModels].sort((a, b) => a.name.localeCompare(b.name)), [rerankingModels]);

  const currentModel = sortedModels.find(m => m.id === selectedModel);
  const currentEmbeddingModel = sortedEmbeddingModels.find(m => m.id === selectedModel);
  const currentRerankingModel = sortedRerankingModels.find(m => m.id === selectedModel);

  const renderModelDetails = (details: { label: string; value: string | number }[]) => (
    <div className="rounded-lg p-3 space-y-2" style={{ backgroundColor: 'var(--surface-inset)' }}>
      <div className="flex items-center space-x-2">
        <Info className="w-4 h-4" style={{ color: 'var(--accent)' }} />
        <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Model Details</span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        {details.map(({ label, value }) => (
          <div key={label}>
            <span style={{ color: 'var(--text-muted)' }}>{label}:</span>
            <span className="ml-2 font-medium font-mono" style={{ color: 'var(--text-primary)' }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="model-select" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Model Selection
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full rounded-lg shadow-sm focus:outline-none focus:ring-2"
          style={{ borderColor: 'var(--border-strong)', color: 'var(--text-primary)', backgroundColor: 'var(--surface)' }}
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

      {currentEmbeddingModel && renderModelDetails([
        { label: 'Dimensions', value: `${currentEmbeddingModel.dimensions}d` },
        { label: 'Max Tokens', value: currentEmbeddingModel.max_tokens },
      ])}

      {currentRerankingModel && renderModelDetails([
        { label: 'Max Query', value: `${currentRerankingModel.max_query_length} tokens` },
        { label: 'Max Doc', value: `${currentRerankingModel.max_doc_length} tokens` },
      ])}

      {currentModel && modelType === 'generation' && renderModelDetails([
        { label: 'Hidden Size', value: currentModel.hidden_size },
        { label: 'Layers', value: currentModel.num_layers },
      ])}
    </div>
  );
};

export default ModelSelector;
