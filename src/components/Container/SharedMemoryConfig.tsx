import React, { useState, useMemo } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import { calculateSharedMemory } from '../../utils/configValidator';

export const SharedMemoryConfig: React.FC = () => {
  const { generatedConfig, customShmSize, setCustomShmSize } = useContainerStore();
  const [showDetails, setShowDetails] = useState(false);
  const [overrideValue, setOverrideValue] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  if (!generatedConfig) {
    return null;
  }

  // Memoize calculation to avoid recalculating on every render
  const calculation = useMemo(
    () => calculateSharedMemory(
      generatedConfig.gpuCount,
      generatedConfig.model.parameters
    ),
    [generatedConfig.gpuCount, generatedConfig.model.parameters]
  );

  const currentShmGB = parseInt(generatedConfig.shmSize);
  const isCustom = !!customShmSize;
  const isInsufficient = currentShmGB < calculation.recommended;

  const handleOverride = () => {
    if (overrideValue) {
      const value = parseInt(overrideValue);
      // Validate: must be positive, reasonable (1-512GB range)
      if (!isNaN(value) && value > 0 && value <= 512) {
        setCustomShmSize(`${value}g`);
      } else if (value > 512) {
        alert('Shared memory cannot exceed 512GB. Please enter a reasonable value.');
      } else {
        alert('Please enter a valid positive number.');
      }
    }
  };

  const handleReset = () => {
    setCustomShmSize(undefined);
    setOverrideValue('');
  };

  return (
    <div className="rounded-lg" style={{ backgroundColor: 'var(--surface-inset)', border: '1px solid var(--border)' }}>
      {/* Header - Clickable to toggle */}
      <div className="w-full flex items-center justify-between p-4 rounded-t-lg transition-colors" style={{ cursor: 'pointer' }} onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-alt)'} onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}>
        <div className="flex items-center flex-1" onClick={() => setIsExpanded(!isExpanded)}>
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
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
          <h4 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Shared Memory Configuration
          </h4>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-xs hover:underline"
            style={{ color: 'var(--accent)' }}
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
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

      {/* Current Status */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Current Setting:</span>
          <span
            className={`text-sm font-semibold font-mono`}
            style={{
              color: isInsufficient ? 'var(--warning-text)' : 'var(--success-text)'
            }}
          >
            {generatedConfig.shmSize} {isCustom && '(custom)'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Recommended:</span>
          <span className="text-sm font-semibold font-mono" style={{ color: 'var(--accent)' }}>
            {calculation.recommended}g
          </span>
        </div>
      </div>

      {/* Warning if insufficient */}
      {isInsufficient && (
        <div className="rounded p-3 mb-3" style={{ backgroundColor: 'var(--warning-light)', border: '1px solid var(--warning-text)' }}>
          <div className="flex items-start">
            <svg
              className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5"
              style={{ color: 'var(--warning-text)' }}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <div>
              <p className="text-xs font-medium" style={{ color: 'var(--warning-text)' }}>
                Shared memory may be insufficient
              </p>
              <p className="text-xs mt-1" style={{ color: 'var(--warning-text)' }}>
                This may cause IPC issues or OOM during multi-GPU operations.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Breakdown */}
      {showDetails && (
        <div className="rounded p-3 mb-3 space-y-2" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            <div className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              Calculation Breakdown:
            </div>

            {/* GPU-based */}
            <div className="flex items-start mb-2">
              <div className="w-1 h-1 rounded-full mt-1.5 mr-2 flex-shrink-0" style={{ backgroundColor: 'var(--accent)' }} />
              <div>
                <div className="font-medium" style={{ color: 'var(--text-primary)' }}>
                  GPU-based: {calculation.gpuBased}GB
                </div>
                <div style={{ color: 'var(--text-muted)' }}>
                  {calculation.reasoning.gpuReasoning}
                </div>
              </div>
            </div>

            {/* Model-based */}
            <div className="flex items-start mb-2">
              <div className="w-1 h-1 rounded-full mt-1.5 mr-2 flex-shrink-0" style={{ backgroundColor: 'var(--accent)' }} />
              <div>
                <div className="font-medium" style={{ color: 'var(--text-primary)' }}>
                  Model-based: {calculation.modelBased}GB
                </div>
                <div style={{ color: 'var(--text-muted)' }}>
                  {calculation.reasoning.modelReasoning}
                </div>
              </div>
            </div>

            {/* Final recommendation */}
            <div className="flex items-start pt-2" style={{ borderTop: '1px solid var(--border)' }}>
              <div className="w-1 h-1 rounded-full mt-1.5 mr-2 flex-shrink-0" style={{ backgroundColor: 'var(--success-text)' }} />
              <div>
                <div className="font-medium" style={{ color: 'var(--text-primary)' }}>
                  Recommendation: {calculation.recommended}GB
                </div>
                <div style={{ color: 'var(--text-muted)' }}>
                  {calculation.reasoning.finalReasoning}
                </div>
              </div>
            </div>
          </div>

          <div className="pt-2" style={{ borderTop: '1px solid var(--border)' }}>
            <p className="text-xs italic" style={{ color: 'var(--text-muted)' }}>
              Shared memory is used for inter-process communication between GPUs and temporary
              tensor operations. Insufficient memory can cause performance degradation or failures.
            </p>
          </div>
        </div>
      )}

      {/* Override Controls */}
      <div className="pt-3" style={{ borderTop: '1px solid var(--border)' }}>
        <label className="block text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Override Shared Memory
        </label>
        <div className="flex items-center space-x-2">
          <input
            type="number"
            min="1"
            max="512"
            value={overrideValue}
            onChange={(e) => setOverrideValue(e.target.value)}
            placeholder={`${calculation.recommended}`}
            className="flex-1 px-3 py-1.5 text-sm rounded focus:ring-2 focus:ring-offset-0 font-mono"
            style={{
              border: '1px solid var(--border-strong)',
              backgroundColor: 'var(--surface)',
              color: 'var(--text-primary)'
            }}
            onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
            onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
          />
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>GB</span>
          <button
            onClick={handleOverride}
            disabled={!overrideValue}
            className="px-3 py-1.5 text-xs font-medium text-white rounded transition-colors"
            style={{
              backgroundColor: overrideValue ? 'var(--accent)' : 'var(--text-muted)',
              cursor: overrideValue ? 'pointer' : 'not-allowed'
            }}
            onMouseEnter={(e) => overrideValue && (e.currentTarget.style.opacity = '0.9')}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          >
            Apply
          </button>
          {isCustom && (
            <button
              onClick={handleReset}
              className="px-3 py-1.5 text-xs font-medium rounded transition-colors"
              style={{
                backgroundColor: 'var(--surface-alt)',
                color: 'var(--text-secondary)'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--border)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-alt)'}
            >
              Reset
            </button>
          )}
        </div>
        {overrideValue && parseInt(overrideValue) < calculation.recommended && (
          <p className="text-xs mt-2" style={{ color: 'var(--warning-text)' }}>
            ⚠️ Warning: This value is below the recommended {calculation.recommended}GB
          </p>
        )}
      </div>
        </div>
      )}
    </div>
  );
};
