import React, { useEffect } from 'react';
import useAppStore from '../../store/useAppStore';
import { useContainerStore } from '../../store/useContainerStore';
import { EngineSelector } from './EngineSelector';
import { ImageSelector } from './ImageSelector';
import { VolumeConfiguration } from './VolumeConfiguration';
import { ConfigOutput } from './ConfigOutput';
import { ValidationDisplay } from './ValidationDisplay';
import { SharedMemoryConfig } from './SharedMemoryConfig';
import { EngineParametersConfig } from './EngineParametersConfig';
import { EnvironmentVariablesConfig } from './EnvironmentVariablesConfig';

export const ContainerConfigTab: React.FC = () => {
  const appState = useAppStore();
  const { generateConfig, generatedConfig, validationResult } = useContainerStore();

  // Auto-generate container config when calculator results are available
  useEffect(() => {
    if (appState.results !== null) {
      generateConfig();
    }
  }, [
    appState.selectedModel,
    appState.selectedGPU,
    appState.numGPUs,
    appState.inferenceQuantization,
    appState.kvCacheQuantization,
    appState.sequenceLength,
    appState.results,
  ]);
  
  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Container Configuration Generator
        </h2>
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
          Generate production-ready Docker configurations for deploying your LLM inference workload on AMD GPUs
        </p>
      </div>
      
      {/* Main Layout: Two Column */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left Panel: Configuration Options */}
        <div className="lg:col-span-2 space-y-4">
          {/* Configuration Card */}
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-6 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Container Configuration
              </h3>
              
              <div className="space-y-4">
                <EngineSelector />
                <ImageSelector />
              </div>
            </div>
            
            <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Volume & Runtime Options
              </h3>
              <VolumeConfiguration />
            </div>
          </div>
          
          {/* Shared Memory Configuration */}
          {generatedConfig && (
            <SharedMemoryConfig />
          )}
          
          {/* Engine Parameters Configuration */}
          {generatedConfig && (
            <EngineParametersConfig />
          )}
          
          {/* Environment Variables Configuration */}
          <EnvironmentVariablesConfig />
          
          {/* From Calculator Card */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 p-4">
            <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-200 mb-3">
              From Calculator
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">Model:</span>
                <span className="font-medium text-blue-900 dark:text-blue-100">
                  {appState.selectedModel.split('/').pop() || appState.selectedModel}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">GPUs:</span>
                <span className="font-medium text-blue-900 dark:text-blue-100">
                  {appState.numGPUs}x {appState.selectedGPU.split('-').slice(-2).join(' ')}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">Quantization:</span>
                <span className="font-medium text-blue-900 dark:text-blue-100">
                  {appState.inferenceQuantization.toUpperCase()}
                </span>
              </div>
              {appState.results && (
                <div className="flex justify-between">
                  <span className="text-blue-700 dark:text-blue-300">Memory:</span>
                  <span className="font-medium text-blue-900 dark:text-blue-100">
                    {appState.results.usedVRAM.toFixed(1)}GB / {appState.results.totalVRAM.toFixed(1)}GB
                    ({appState.results.vramPercentage.toFixed(1)}%)
                  </span>
                </div>
              )}
            </div>
          </div>
          
          {/* Generate Button */}
          <button
            onClick={generateConfig}
            className="w-full px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition-colors shadow-sm"
          >
            🔄 Regenerate Configuration
          </button>
        </div>
        
        {/* Right Panel: Output & Validation */}
        <div className="lg:col-span-3 space-y-4">
          {/* Validation Display */}
          {validationResult && (
            <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Validation Status
              </h3>
              <ValidationDisplay />
            </div>
          )}
          
          {/* Generated Configuration Output */}
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Generated Configuration
            </h3>
            <ConfigOutput />
          </div>
          
          {/* Usage Instructions */}
          {generatedConfig && (
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center">
                <svg className="w-5 h-5 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Quick Start
              </h4>
              <ol className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex">
                  <span className="font-semibold mr-2 text-blue-600 dark:text-blue-400">1.</span>
                  <span>Download or copy the generated configuration</span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2 text-blue-600 dark:text-blue-400">2.</span>
                  <span>
                    {generatedConfig.useContainerToolkit
                      ? 'Ensure AMD Container Toolkit is installed on your system'
                      : 'Ensure ROCm drivers are installed on your system'}
                  </span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2 text-blue-600 dark:text-blue-400">3.</span>
                  <span>Set required environment variables (e.g., HF_TOKEN for gated models)</span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2 text-blue-600 dark:text-blue-400">4.</span>
                  <span>Run the command or docker-compose up -d</span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2 text-blue-600 dark:text-blue-400">5.</span>
                  <span>Access the API at http://localhost:{generatedConfig.ports[0]?.host || 8000}</span>
                </li>
              </ol>
              
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  📚 For more information, see the{' '}
                  <a
                    href="https://rocm.docs.amd.com/projects/container-toolkit"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    AMD Container Toolkit Documentation
                  </a>
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
