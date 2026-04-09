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
      <label className="block text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
        Model Type
      </label>
      <div className="grid grid-cols-1 gap-2">
        {modelTypes.map((type) => {
          const Icon = type.icon;
          const isSelected = modelType === type.value;

          return (
            <button
              key={type.value}
              onClick={() => setModelType(type.value)}
              className="flex items-start p-3.5 rounded-lg transition-all text-left"
              style={{
                backgroundColor: isSelected ? 'var(--accent-light)' : 'var(--surface)',
                border: `2px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
              }}
            >
              <div
                className="flex-shrink-0 p-2 rounded-lg mr-3"
                style={{ backgroundColor: isSelected ? 'var(--accent-muted)' : 'var(--surface-inset)' }}
              >
                <Icon
                  className="w-5 h-5"
                  style={{ color: isSelected ? 'var(--accent)' : 'var(--text-muted)' }}
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center">
                  <span
                    className="text-sm font-semibold"
                    style={{ color: isSelected ? 'var(--accent-text)' : 'var(--text-primary)' }}
                  >
                    {type.label}
                  </span>
                  {isSelected && (
                    <span className="ml-2 flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full opacity-75" style={{ backgroundColor: 'var(--accent)' }} />
                      <span className="relative inline-flex rounded-full h-2 w-2" style={{ backgroundColor: 'var(--accent)' }} />
                    </span>
                  )}
                </div>
                <p className="text-xs mt-0.5" style={{ color: isSelected ? 'var(--accent)' : 'var(--text-muted)' }}>
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
