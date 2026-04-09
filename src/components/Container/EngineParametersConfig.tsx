import React, { useState, useMemo } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import useAppStore from '../../store/useAppStore';
import engineParametersData from '../../data/engine-parameters.json';
import type { EngineParameter, EngineParametersData, InferenceEngine } from '../../types';

export const EngineParametersConfig: React.FC = () => {
  const { selectedEngineId, customEngineParams, setCustomEngineParam, clearCustomEngineParam } =
    useContainerStore();
  const { numGPUs } = useAppStore();

  const [showAdvanced, setShowAdvanced] = useState(false);

  const engine = useMemo((): InferenceEngine | undefined => {
    return (engineParametersData as EngineParametersData).engines.find(
      (e) => e.id === selectedEngineId
    );
  }, [selectedEngineId]);

  if (!engine) {
    return null;
  }

  // Filter parameters - show manual adjustable ones in advanced section
  const manualParameters = engine.parameters.filter(
    (p: EngineParameter) => p.source === 'manual' || !p.source
  );

  // Parameters that can be overridden from calculator values
  const overridableParameters = engine.parameters.filter(
    (p: EngineParameter) => p.source === 'calculator' && !p.required
  );

  const handleParamChange = (flag: string, value: string | number | boolean | null) => {
    if (value === null || value === '' || value === undefined) {
      clearCustomEngineParam(flag);
    } else {
      setCustomEngineParam(flag, value);
    }
  };

  const getParamValue = (param: EngineParameter): string | number | boolean => {
    const customValue = customEngineParams.get(param.flag);
    if (customValue !== undefined) {
      return customValue;
    }
    if (param.default !== undefined && param.default !== null) {
      return param.default;
    }
    return '';
  };

  const isParamCustomized = (flag: string): boolean => {
    return customEngineParams.has(flag);
  };

  const renderParameterInput = (param: EngineParameter) => {
    const value = getParamValue(param);
    const isCustom = isParamCustomized(param.flag);

    // Validation for tensor-parallel-size
    const isTensorParallel = param.flag === '--tensor-parallel-size';
    const tensorParallelMismatch =
      isTensorParallel &&
      isCustom &&
      typeof value === 'number' &&
      value !== numGPUs;

    switch (param.type) {
      case 'boolean':
        return (
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={!!value}
              onChange={(e) => handleParamChange(param.flag, e.target.checked)}
              className="h-4 w-4 rounded"
              style={{ accentColor: 'var(--accent)' }}
            />
            <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              {param.description}
            </label>
            {param.security_warning && isCustom && value && (
              <span className="text-xs ml-2" style={{ color: 'var(--warning-text)' }}>
                ⚠️ {param.security_warning}
              </span>
            )}
          </div>
        );

      case 'select':
        return (
          <div className="space-y-1">
            <select
              value={String(value)}
              onChange={(e) => handleParamChange(param.flag, e.target.value || null)}
              className="w-full px-3 py-2 text-sm rounded focus:ring-2 focus:border-transparent"
              style={{
                border: '1px solid var(--border-strong)',
                backgroundColor: 'var(--surface)',
                color: 'var(--text-primary)'
              }}
              onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
              onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
            >
              <option value="">Auto / Default</option>
              {param.options?.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            {param.options && value && (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {param.options.find((o) => o.value === value)?.description}
              </p>
            )}
          </div>
        );

      case 'number':
        return (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <input
                type="number"
                value={value as number}
                onChange={(e) =>
                  handleParamChange(
                    param.flag,
                    e.target.value ? parseFloat(e.target.value) : null
                  )
                }
                min={param.validation?.min}
                max={param.validation?.max}
                step={param.type === 'number' && param.validation?.min === 0 ? 0.1 : 1}
                placeholder={param.default?.toString() || ''}
                className="flex-1 px-3 py-2 text-sm rounded font-mono focus:ring-2 focus:border-transparent"
                style={{
                  border: '1px solid var(--border-strong)',
                  backgroundColor: 'var(--surface)',
                  color: 'var(--text-primary)'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
                onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
              />
              {isCustom && (
                <button
                  onClick={() => handleParamChange(param.flag, null)}
                  className="px-2 py-1 text-xs rounded transition-colors"
                  style={{
                    backgroundColor: 'var(--surface-alt)',
                    color: 'var(--text-secondary)'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--border)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-alt)'}
                  title="Reset to default"
                >
                  Reset
                </button>
              )}
            </div>
            {tensorParallelMismatch && (
              <p className="text-xs" style={{ color: 'var(--danger-text)' }}>
                ⚠️ Tensor parallel size ({value}) doesn't match GPU count ({numGPUs}). This will
                cause errors!
              </p>
            )}
            {param.validation && (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {param.validation.min !== undefined && `Min: ${param.validation.min}`}
                {param.validation.min !== undefined && param.validation.max !== undefined && ', '}
                {param.validation.max !== undefined && `Max: ${param.validation.max}`}
              </p>
            )}
          </div>
        );

      default:
        return (
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={value as string}
              onChange={(e) => handleParamChange(param.flag, e.target.value || null)}
              placeholder={param.default?.toString() || ''}
              className="flex-1 px-3 py-2 text-sm rounded focus:ring-2 focus:border-transparent"
              style={{
                border: '1px solid var(--border-strong)',
                backgroundColor: 'var(--surface)',
                color: 'var(--text-primary)'
              }}
              onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
              onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
            />
            {isCustom && (
              <button
                onClick={() => handleParamChange(param.flag, null)}
                className="px-2 py-1 text-xs rounded transition-colors"
                style={{
                  backgroundColor: 'var(--surface-alt)',
                  color: 'var(--text-secondary)'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--border)'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-alt)'}
                title="Reset to default"
              >
                Reset
              </button>
            )}
          </div>
        );
    }
  };

  const [isExpanded, setIsExpanded] = useState(false);

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
              d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
            />
          </svg>
          <h4 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Engine Parameters
          </h4>
          {customEngineParams.size > 0 && (
            <span className="ml-3 px-2 py-1 text-xs font-medium rounded" style={{ backgroundColor: 'var(--accent-light)', color: 'var(--accent-text)' }}>
              {customEngineParams.size} Custom
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs hover:underline"
            style={{ color: 'var(--accent)' }}
          >
            {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
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

      {/* Info */}
      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
        Parameters are automatically configured from calculator settings. Override only if needed.
      </p>

      {/* Overridable Calculator Parameters */}
      {overridableParameters.length > 0 && showAdvanced && (
        <div className="mb-4">
          <h5 className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>
            Override Calculator Settings
          </h5>
          <div className="space-y-3">
            {overridableParameters.map((param: EngineParameter) => (
              <div key={param.flag}>
                <div className="flex items-start justify-between mb-1">
                  <label className="block text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
                    {param.flag}
                    {isParamCustomized(param.flag) && (
                      <span className="ml-1" style={{ color: 'var(--accent)' }}>(custom)</span>
                    )}
                  </label>
                  {param.validation?.mustMatchGpuCount && (
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      Must match GPU count
                    </span>
                  )}
                </div>
                <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
                  {param.description}
                </p>
                {renderParameterInput(param)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Manual Parameters */}
      {manualParameters.length > 0 && showAdvanced && (
        <div>
          <h5 className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>
            Additional Parameters
          </h5>
          <div className="space-y-3">
            {manualParameters.map((param: EngineParameter) => (
              <div key={param.flag}>
                <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
                  {param.flag}
                  {param.required && <span style={{ color: 'var(--danger-text)' }} className="ml-1">*</span>}
                  {isParamCustomized(param.flag) && (
                    <span className="ml-1" style={{ color: 'var(--accent)' }}>(custom)</span>
                  )}
                </label>
                <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
                  {param.description}
                </p>
                {renderParameterInput(param)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary of customizations */}
      {customEngineParams.size > 0 && (
        <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="flex items-center justify-between">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {customEngineParams.size} custom parameter{customEngineParams.size > 1 ? 's' : ''}{' '}
              set
            </span>
            <button
              onClick={() => {
                customEngineParams.forEach((_, flag) => clearCustomEngineParam(flag));
              }}
              className="text-xs hover:underline"
              style={{ color: 'var(--danger-text)' }}
            >
              Clear All
            </button>
          </div>
        </div>
      )}
        </div>
      )}
    </div>
  );
};
