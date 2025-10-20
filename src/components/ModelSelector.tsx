import React from 'react';
import { Info } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import { Model } from '../types';

interface ModelSelectorProps {
  models: Model[];
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ models }) => {
  const {
    selectedModel,
    setSelectedModel,
    customModelParams,
    setCustomModelParams,
    customHiddenSize,
    customNumLayers,
    customNumHeads,
  } = useAppStore();

  const currentModel = models.find(m => m.id === selectedModel);
  const isCustom = selectedModel === 'custom';

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
          <optgroup label="DeepSeek Models">
            {models.filter(m => m.id.startsWith('deepseek')).map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.parameters_billions}B params)
              </option>
            ))}
          </optgroup>
          <optgroup label="Llama Models">
            {models.filter(m => m.id.startsWith('llama')).map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.parameters_billions}B params)
              </option>
            ))}
          </optgroup>
          <optgroup label="Mistral Models">
            {models.filter(m => m.id.startsWith('mistral') || m.id.startsWith('mixtral')).map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.parameters_billions}B params)
              </option>
            ))}
          </optgroup>
          <optgroup label="Other Models">
            {models.filter(m =>
              !m.id.startsWith('deepseek') &&
              !m.id.startsWith('llama') &&
              !m.id.startsWith('mistral') &&
              !m.id.startsWith('mixtral') &&
              m.id !== 'custom'
            ).map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.parameters_billions}B params)
              </option>
            ))}
          </optgroup>
          <optgroup label="Custom">
            <option value="custom">Custom Model</option>
          </optgroup>
        </select>
      </div>

      {currentModel && (
        <div className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center space-x-2">
            <Info className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-700">Model Details</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">Parameters:</span>
              <span className="ml-2 font-medium">
                {isCustom && customModelParams ? customModelParams : currentModel.parameters_billions}B
              </span>
            </div>
            <div>
              <span className="text-gray-500">Hidden Size:</span>
              <span className="ml-2 font-medium">
                {isCustom && customHiddenSize ? customHiddenSize : currentModel.hidden_size}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Layers:</span>
              <span className="ml-2 font-medium">
                {isCustom && customNumLayers ? customNumLayers : currentModel.num_layers}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Attention Heads:</span>
              <span className="ml-2 font-medium">
                {isCustom && customNumHeads ? customNumHeads : currentModel.num_heads}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Context Length:</span>
              <span className="ml-2 font-medium">{currentModel.default_context_length.toLocaleString()} tokens</span>
            </div>
            <div>
              <span className="text-gray-500">Architecture:</span>
              <span className="ml-2 font-medium capitalize">{currentModel.architecture}</span>
            </div>
          </div>
        </div>
      )}

      {isCustom && (
        <div className="space-y-3 border-t pt-3">
          <div>
            <label htmlFor="custom-params" className="block text-sm font-medium text-gray-700 mb-1">
              Custom Parameters (Billions)
            </label>
            <input
              id="custom-params"
              type="number"
              min="0.1"
              max="1000"
              step="0.1"
              value={customModelParams || ''}
              onChange={(e) => setCustomModelParams(e.target.value ? parseFloat(e.target.value) : undefined)}
              placeholder="e.g., 7"
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="custom-hidden" className="block text-sm font-medium text-gray-700 mb-1">
                Hidden Size
              </label>
              <input
                id="custom-hidden"
                type="number"
                min="128"
                max="32768"
                step="128"
                value={customHiddenSize || ''}
                onChange={(e) => useAppStore.setState({
                  customHiddenSize: e.target.value ? parseInt(e.target.value) : undefined
                })}
                placeholder="e.g., 4096"
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <div>
              <label htmlFor="custom-layers" className="block text-sm font-medium text-gray-700 mb-1">
                Layers
              </label>
              <input
                id="custom-layers"
                type="number"
                min="1"
                max="200"
                value={customNumLayers || ''}
                onChange={(e) => useAppStore.setState({
                  customNumLayers: e.target.value ? parseInt(e.target.value) : undefined
                })}
                placeholder="e.g., 32"
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;