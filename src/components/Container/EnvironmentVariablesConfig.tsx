import React, { useState } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import type { EnvironmentVariable } from '../../types';

// Common presets for vLLM and container environments
const ENV_PRESETS = [
  {
    key: 'HF_TOKEN',
    value: '${HF_TOKEN}',
    description: 'HuggingFace API token for gated models',
    sensitive: true,
  },
  {
    key: 'VLLM_LOGGING_LEVEL',
    value: 'INFO',
    description: 'vLLM logging verbosity (DEBUG, INFO, WARNING, ERROR)',
    sensitive: false,
  },
  {
    key: 'VLLM_WORKER_MULTIPROC_METHOD',
    value: 'spawn',
    description: 'Multiprocessing method for vLLM workers',
    sensitive: false,
  },
  {
    key: 'HF_HUB_OFFLINE',
    value: '0',
    description: 'Set to 1 to use offline mode for HuggingFace Hub',
    sensitive: false,
  },
  {
    key: 'NCCL_DEBUG',
    value: 'INFO',
    description: 'NCCL debugging level for multi-GPU communication',
    sensitive: false,
  },
  {
    key: 'PYTORCH_ROCM_ARCH',
    value: 'gfx942;gfx950',
    description: 'Target ROCm GPU architectures: gfx90a (MI210/MI250), gfx942 (MI300X/MI325X), gfx950 (MI350X/MI355X)',
    sensitive: false,
  },
  {
    key: 'HF_HOME',
    value: '/models/.cache',
    description: 'Override HuggingFace cache directory',
    sensitive: false,
  },
  {
    key: 'VLLM_USE_MODELSCOPE',
    value: 'False',
    description: 'Disable ModelScope hub (use HuggingFace)',
    sensitive: false,
  },
];

export const EnvironmentVariablesConfig: React.FC = () => {
  const { customEnvironment, addEnvironmentVariable, removeEnvironmentVariable } =
    useContainerStore();

  const [showAddForm, setShowAddForm] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [newEnvVar, setNewEnvVar] = useState<EnvironmentVariable>({
    key: '',
    value: '',
    description: '',
    sensitive: false,
  });

  const handleAdd = () => {
    if (newEnvVar.key && newEnvVar.value) {
      addEnvironmentVariable(newEnvVar);
      setNewEnvVar({ key: '', value: '', description: '', sensitive: false });
      setShowAddForm(false);
    }
  };

  const handleAddPreset = (preset: EnvironmentVariable) => {
    // Check if already exists
    const exists = customEnvironment.some((env) => env.key === preset.key);
    if (!exists) {
      addEnvironmentVariable(preset);
    }
    setShowPresets(false);
  };

  const formatValue = (env: EnvironmentVariable): string => {
    if (env.sensitive && env.value.startsWith('${')) {
      return env.value; // Show placeholder syntax
    }
    if (env.sensitive) {
      return '••••••••'; // Mask sensitive values
    }
    return env.value;
  };

  return (
    <div className="rounded-lg" style={{ backgroundColor: 'var(--surface-inset)', border: '1px solid var(--border)' }}>
      {/* Header - Clickable to toggle */}
      <div className="w-full flex items-center justify-between p-4 rounded-t-lg transition-colors" onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-alt)'} onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}>
        <div className="flex items-center cursor-pointer flex-1" onClick={() => setIsExpanded(!isExpanded)}>
          <svg
            className="w-5 h-5 mr-2"
            style={{ color: 'var(--accent)' }}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          <h4 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Environment Variables
          </h4>
          {customEnvironment.length > 0 && (
            <span className="ml-3 px-2 py-1 text-xs font-medium rounded" style={{ backgroundColor: 'var(--success-light)', color: 'var(--success-text)' }}>
              {customEnvironment.length} Added
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowPresets(!showPresets)}
            className="text-xs hover:underline"
            style={{ color: 'var(--accent)' }}
          >
            {showPresets ? 'Hide' : 'Presets'}
          </button>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="text-xs hover:underline"
            style={{ color: 'var(--accent)' }}
          >
            {showAddForm ? 'Cancel' : '+ Add Variable'}
          </button>
          <svg
            className={`w-5 h-5 transition-transform cursor-pointer ${isExpanded ? 'transform rotate-180' : ''}`}
            style={{ color: 'var(--text-muted)' }}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Content - Collapsible */}
      {isExpanded && (
        <div className="p-4 pt-0">

      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
        Additional environment variables to pass to the container. Use ${'{'}VAR{'}'} syntax for
        host environment variable substitution.
      </p>

      {/* Presets List */}
      {showPresets && (
        <div className="mb-3 rounded p-3" style={{ backgroundColor: 'var(--accent-light)', border: '1px solid var(--accent)' }}>
          <h5 className="text-xs font-semibold mb-2" style={{ color: 'var(--accent-text)' }}>
            Common Presets
          </h5>
          <div className="space-y-2">
            {ENV_PRESETS.map((preset) => (
              <button
                key={preset.key}
                onClick={() => handleAddPreset(preset)}
                disabled={customEnvironment.some((env) => env.key === preset.key)}
                className="w-full text-left p-2 rounded transition-colors"
                style={{
                  backgroundColor: 'var(--surface)',
                  border: '1px solid var(--accent)',
                  cursor: customEnvironment.some((env) => env.key === preset.key) ? 'not-allowed' : 'pointer',
                  opacity: customEnvironment.some((env) => env.key === preset.key) ? 0.5 : 1
                }}
                onMouseEnter={(e) => !customEnvironment.some((env) => env.key === preset.key) && (e.currentTarget.style.backgroundColor = 'var(--accent-light)')}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface)'}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center">
                      <span className="text-xs font-medium font-mono" style={{ color: 'var(--text-primary)' }}>
                        {preset.key}
                      </span>
                      {preset.sensitive && (
                        <span className="ml-2 px-1.5 py-0.5 text-xs rounded" style={{ backgroundColor: 'var(--warning-light)', color: 'var(--warning-text)' }}>
                          sensitive
                        </span>
                      )}
                    </div>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                      {preset.description}
                    </p>
                  </div>
                  {customEnvironment.some((env) => env.key === preset.key) && (
                    <span className="text-xs ml-2" style={{ color: 'var(--success-text)' }}>✓ Added</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Add Form */}
      {showAddForm && (
        <div className="mb-3 rounded p-3 space-y-2" style={{ backgroundColor: 'var(--accent-light)', border: '1px solid var(--accent)' }}>
          <div>
            <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
              Variable Name
            </label>
            <input
              type="text"
              value={newEnvVar.key}
              onChange={(e) => setNewEnvVar({ ...newEnvVar, key: e.target.value.toUpperCase() })}
              placeholder="MY_VARIABLE"
              className="w-full px-2 py-1.5 text-sm font-mono rounded focus:ring-2 focus:border-transparent"
              style={{
                backgroundColor: 'var(--surface)',
                border: '1px solid var(--border-strong)',
                color: 'var(--text-primary)'
              }}
              onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
              onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
              Value
            </label>
            <input
              type={newEnvVar.sensitive ? 'password' : 'text'}
              value={newEnvVar.value}
              onChange={(e) => setNewEnvVar({ ...newEnvVar, value: e.target.value })}
              placeholder="${MY_VARIABLE} or hardcoded value"
              className="w-full px-2 py-1.5 text-sm font-mono rounded focus:ring-2 focus:border-transparent"
              style={{
                backgroundColor: 'var(--surface)',
                border: '1px solid var(--border-strong)',
                color: 'var(--text-primary)'
              }}
              onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
              onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
              Description (optional)
            </label>
            <input
              type="text"
              value={newEnvVar.description}
              onChange={(e) => setNewEnvVar({ ...newEnvVar, description: e.target.value })}
              placeholder="Brief description"
              className="w-full px-2 py-1.5 text-sm rounded focus:ring-2 focus:border-transparent"
              style={{
                backgroundColor: 'var(--surface)',
                border: '1px solid var(--border-strong)',
                color: 'var(--text-primary)'
              }}
              onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
              onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
            />
          </div>
          <div className="flex items-center space-x-2">
            <input
              id="new-env-sensitive"
              type="checkbox"
              checked={newEnvVar.sensitive}
              onChange={(e) => setNewEnvVar({ ...newEnvVar, sensitive: e.target.checked })}
              className="h-4 w-4 rounded"
              style={{ accentColor: 'var(--warning-text)' }}
            />
            <label htmlFor="new-env-sensitive" className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Mark as sensitive (will use ${'{'}...{'}'} in output)
            </label>
          </div>
          <button
            onClick={handleAdd}
            disabled={!newEnvVar.key || !newEnvVar.value}
            className="w-full px-3 py-2 text-white text-sm font-medium rounded transition-colors"
            style={{
              backgroundColor: (!newEnvVar.key || !newEnvVar.value) ? 'var(--text-muted)' : 'var(--accent)',
              cursor: (!newEnvVar.key || !newEnvVar.value) ? 'not-allowed' : 'pointer'
            }}
            onMouseEnter={(e) => {
              if (newEnvVar.key && newEnvVar.value) {
                e.currentTarget.style.opacity = '0.9';
              }
            }}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          >
            Add Variable
          </button>
        </div>
      )}

      {/* Variables List */}
      {customEnvironment.length > 0 ? (
        <div className="space-y-2">
          {customEnvironment.map((env, idx) => (
            <div
              key={idx}
              className="flex items-start justify-between p-2 rounded"
              style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
            >
              <div className="flex-1 min-w-0 mr-2">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                    {env.key}
                  </span>
                  {env.sensitive && (
                    <span className="px-1.5 py-0.5 text-xs rounded" style={{ backgroundColor: 'var(--warning-light)', color: 'var(--warning-text)' }}>
                      sensitive
                    </span>
                  )}
                </div>
                <div className="text-xs mt-1 font-mono" style={{ color: 'var(--text-muted)' }}>
                  {formatValue(env)}
                </div>
                {env.description && (
                  <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{env.description}</p>
                )}
              </div>
              <button
                onClick={() => removeEnvironmentVariable(idx)}
                className="flex-shrink-0"
                style={{ color: 'var(--danger-text)' }}
                onMouseEnter={(e) => e.currentTarget.style.opacity = '0.7'}
                onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
                title="Remove variable"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-4 text-xs" style={{ color: 'var(--text-muted)' }}>
          No custom environment variables added. Default variables (AMD_VISIBLE_DEVICES, HF_TOKEN)
          are automatically included.
        </div>
      )}
        </div>
      )}
    </div>
  );
};
