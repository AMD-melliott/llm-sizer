import React, { useState, useMemo } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import { calculateSharedMemory } from '../../utils/configValidator';

export const SharedMemoryConfig: React.FC = () => {
  const { generatedConfig, customShmSize, setCustomShmSize } = useContainerStore();
  const [showDetails, setShowDetails] = useState(false);
  const [overrideValue, setOverrideValue] = useState('');

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
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center">
          <svg
            className="w-5 h-5 mr-2 text-blue-500"
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
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Shared Memory Configuration
          </h4>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {/* Current Status */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-gray-600 dark:text-gray-400">Current Setting:</span>
          <span
            className={`text-sm font-semibold ${
              isInsufficient
                ? 'text-orange-600 dark:text-orange-400'
                : 'text-green-600 dark:text-green-400'
            }`}
          >
            {generatedConfig.shmSize} {isCustom && '(custom)'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">Recommended:</span>
          <span className="text-sm font-semibold text-blue-600 dark:text-blue-400">
            {calculation.recommended}g
          </span>
        </div>
      </div>

      {/* Warning if insufficient */}
      {isInsufficient && (
        <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded p-3 mb-3">
          <div className="flex items-start">
            <svg
              className="w-5 h-5 text-orange-500 mr-2 flex-shrink-0 mt-0.5"
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
              <p className="text-xs font-medium text-orange-900 dark:text-orange-200">
                Shared memory may be insufficient
              </p>
              <p className="text-xs text-orange-700 dark:text-orange-300 mt-1">
                This may cause IPC issues or OOM during multi-GPU operations.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Breakdown */}
      {showDetails && (
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded p-3 mb-3 space-y-2">
          <div className="text-xs text-gray-700 dark:text-gray-300">
            <div className="font-semibold mb-2 text-gray-900 dark:text-gray-100">
              Calculation Breakdown:
            </div>

            {/* GPU-based */}
            <div className="flex items-start mb-2">
              <div className="w-1 h-1 rounded-full bg-blue-500 mt-1.5 mr-2 flex-shrink-0" />
              <div>
                <div className="font-medium text-gray-800 dark:text-gray-200">
                  GPU-based: {calculation.gpuBased}GB
                </div>
                <div className="text-gray-600 dark:text-gray-400">
                  {calculation.reasoning.gpuReasoning}
                </div>
              </div>
            </div>

            {/* Model-based */}
            <div className="flex items-start mb-2">
              <div className="w-1 h-1 rounded-full bg-purple-500 mt-1.5 mr-2 flex-shrink-0" />
              <div>
                <div className="font-medium text-gray-800 dark:text-gray-200">
                  Model-based: {calculation.modelBased}GB
                </div>
                <div className="text-gray-600 dark:text-gray-400">
                  {calculation.reasoning.modelReasoning}
                </div>
              </div>
            </div>

            {/* Final recommendation */}
            <div className="flex items-start pt-2 border-t border-gray-200 dark:border-gray-700">
              <div className="w-1 h-1 rounded-full bg-green-500 mt-1.5 mr-2 flex-shrink-0" />
              <div>
                <div className="font-medium text-gray-800 dark:text-gray-200">
                  Recommendation: {calculation.recommended}GB
                </div>
                <div className="text-gray-600 dark:text-gray-400">
                  {calculation.reasoning.finalReasoning}
                </div>
              </div>
            </div>
          </div>

          <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs text-gray-600 dark:text-gray-400 italic">
              Shared memory is used for inter-process communication between GPUs and temporary
              tensor operations. Insufficient memory can cause performance degradation or failures.
            </p>
          </div>
        </div>
      )}

      {/* Override Controls */}
      <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
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
            className="flex-1 px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400">GB</span>
          <button
            onClick={handleOverride}
            disabled={!overrideValue}
            className="px-3 py-1.5 text-xs font-medium bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-300 disabled:dark:bg-gray-700 disabled:cursor-not-allowed transition-colors"
          >
            Apply
          </button>
          {isCustom && (
            <button
              onClick={handleReset}
              className="px-3 py-1.5 text-xs font-medium bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              Reset
            </button>
          )}
        </div>
        {overrideValue && parseInt(overrideValue) < calculation.recommended && (
          <p className="text-xs text-orange-600 dark:text-orange-400 mt-2">
            ⚠️ Warning: This value is below the recommended {calculation.recommended}GB
          </p>
        )}
      </div>
    </div>
  );
};
