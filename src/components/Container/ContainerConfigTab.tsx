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
        <h2 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          Container Configuration Generator
        </h2>
        <p className="mt-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
          Generate production-ready Docker configurations for deploying your LLM inference workload on AMD GPUs
        </p>
      </div>
      
      {/* Main Layout: Two Column */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left Panel: Configuration Options */}
        <div className="lg:col-span-2 space-y-4">
          {/* Configuration Card */}
          <div className="rounded-xl p-6 space-y-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 4px 6px var(--shadow-color)' }}>
            <div>
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
                Container Configuration
              </h3>

              <div className="space-y-4">
                <EngineSelector />
                <ImageSelector />
              </div>
            </div>

            <div className="pt-6" style={{ borderTop: '1px solid var(--border)' }}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
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
          <div className="rounded-lg p-4" style={{ backgroundColor: 'var(--accent-light)', border: '1px solid var(--accent)' }}>
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--accent-text)' }}>
              From Calculator
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span style={{ color: 'var(--accent)' }}>Model:</span>
                <span className="font-medium" style={{ color: 'var(--accent-text)' }}>
                  {appState.selectedModel.split('/').pop() || appState.selectedModel}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--accent)' }}>GPUs:</span>
                <span className="font-medium" style={{ color: 'var(--accent-text)' }}>
                  {appState.numGPUs}x {appState.selectedGPU.split('-').slice(-2).join(' ')}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--accent)' }}>Quantization:</span>
                <span className="font-medium" style={{ color: 'var(--accent-text)' }}>
                  {appState.inferenceQuantization.toUpperCase()}
                </span>
              </div>
              {appState.results && (
                <div className="flex justify-between">
                  <span style={{ color: 'var(--accent)' }}>Memory:</span>
                  <span className="font-medium font-mono" style={{ color: 'var(--accent-text)' }}>
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
            className="w-full px-6 py-3 text-white font-medium rounded-md transition-colors"
            style={{ backgroundColor: 'var(--accent)', boxShadow: '0 1px 3px var(--shadow-color)' }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '0.9'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          >
            🔄 Regenerate Configuration
          </button>
        </div>
        
        {/* Right Panel: Output & Validation */}
        <div className="lg:col-span-3 space-y-4">
          {/* Validation Display */}
          {validationResult && (
            <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 4px 6px var(--shadow-color)' }}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
                Validation Status
              </h3>
              <ValidationDisplay />
            </div>
          )}

          {/* Generated Configuration Output */}
          <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 4px 6px var(--shadow-color)' }}>
            <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              Generated Configuration
            </h3>
            <ConfigOutput />
          </div>
          
          {/* Usage Instructions */}
          {generatedConfig && (
            <div className="rounded-lg p-6" style={{ backgroundColor: 'var(--surface-inset)', border: '1px solid var(--border)' }}>
              <h4 className="text-sm font-semibold mb-3 flex items-center" style={{ color: 'var(--text-primary)' }}>
                <svg className="w-5 h-5 mr-2" style={{ color: 'var(--accent)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Quick Start
              </h4>
              <ol className="space-y-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
                <li className="flex">
                  <span className="font-semibold mr-2" style={{ color: 'var(--accent)' }}>1.</span>
                  <span>Download or copy the generated configuration</span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2" style={{ color: 'var(--accent)' }}>2.</span>
                  <span>
                    {generatedConfig.useContainerToolkit
                      ? 'Ensure AMD Container Toolkit is installed on your system'
                      : 'Ensure ROCm drivers are installed on your system'}
                  </span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2" style={{ color: 'var(--accent)' }}>3.</span>
                  <span>Set required environment variables (e.g., HF_TOKEN for gated models)</span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2" style={{ color: 'var(--accent)' }}>4.</span>
                  <span>Run the command or docker-compose up -d</span>
                </li>
                <li className="flex">
                  <span className="font-semibold mr-2" style={{ color: 'var(--accent)' }}>5.</span>
                  <span>Access the API at http://localhost:{generatedConfig.ports[0]?.host || 8000}</span>
                </li>
              </ol>

              <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  📚 For more information, see the{' '}
                  <a
                    href="https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#amd-rocm"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline"
                    style={{ color: 'var(--accent)' }}
                  >
                    vLLM AMD ROCm Installation Guide
                  </a>
                  {' '}and the{' '}
                  <a
                    href="https://rocm.docs.amd.com/projects/container-toolkit"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline"
                    style={{ color: 'var(--accent)' }}
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
