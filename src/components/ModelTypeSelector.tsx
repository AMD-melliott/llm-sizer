import { Brain, Database, GitCompare } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import { ModelType } from '../types';

const modelTypes: { value: ModelType; label: string; icon: typeof Brain; description: string }[] = [
  {
    value: 'generation',
    label: 'Text Generation',
    icon: Brain,
    description: 'Large Language Models for text generation and chat',
  },
  {
    value: 'embedding',
    label: 'Text Embedding',
    icon: Database,
    description: 'Models for semantic search and vector representations',
  },
  {
    value: 'reranking',
    label: 'Reranking',
    icon: GitCompare,
    description: 'Cross-encoders for relevance scoring and reranking',
  },
];

export default function ModelTypeSelector() {
  const { modelType, setModelType } = useAppStore();

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        Model Type
      </label>
      <div className="grid grid-cols-1 gap-3">
        {modelTypes.map((type) => {
          const Icon = type.icon;
          const isSelected = modelType === type.value;

          return (
            <button
              key={type.value}
              onClick={() => setModelType(type.value)}
              className={`
                flex items-start p-4 rounded-lg border-2 transition-all text-left
                ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }
              `}
            >
              <div
                className={`
                  flex-shrink-0 p-2 rounded-lg mr-3
                  ${isSelected ? 'bg-blue-100' : 'bg-gray-100'}
                `}
              >
                <Icon
                  className={`w-5 h-5 ${isSelected ? 'text-blue-600' : 'text-gray-600'}`}
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center">
                  <span
                    className={`
                      text-sm font-semibold
                      ${isSelected ? 'text-blue-900' : 'text-gray-900'}
                    `}
                  >
                    {type.label}
                  </span>
                  {isSelected && (
                    <span className="ml-2 flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-blue-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                    </span>
                  )}
                </div>
                <p
                  className={`
                    text-xs mt-1
                    ${isSelected ? 'text-blue-700' : 'text-gray-600'}
                  `}
                >
                  {type.description}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
