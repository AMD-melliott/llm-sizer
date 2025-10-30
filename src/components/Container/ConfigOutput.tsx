import React, { useState } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import { generateDockerRunCommand } from '../../utils/dockerCommandBuilder';
import { generateDockerCompose } from '../../utils/dockerComposeBuilder';

export const ConfigOutput: React.FC = () => {
  const { generatedConfig, outputFormat, setOutputFormat, enableHealthcheck } = useContainerStore();
  const [copySuccess, setCopySuccess] = useState(false);
  
  if (!generatedConfig) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 dark:bg-gray-800 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600">
        <div className="text-center">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Configure your container and click "Generate" to see the output
          </p>
        </div>
      </div>
    );
  }
  
  const output = outputFormat === 'docker-run'
    ? generateDockerRunCommand(generatedConfig, { includeComments: true, includeManagementCommands: true })
    : generateDockerCompose(generatedConfig, { includeComments: true, enableHealthcheck });
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(output);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };
  
  const handleDownload = () => {
    const filename = outputFormat === 'docker-run' 
      ? `run-${generatedConfig.containerName}.sh`
      : `docker-compose.yml`;
    
    const blob = new Blob([output], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  return (
    <div className="space-y-3">
      {/* Format Toggle */}
      <div className="flex items-center justify-between">
        <div className="flex space-x-2">
          <button
            onClick={() => setOutputFormat('docker-run')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              outputFormat === 'docker-run'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            Docker Run
          </button>
          <button
            onClick={() => setOutputFormat('docker-compose')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              outputFormat === 'docker-compose'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            Docker Compose
          </button>
        </div>
        
        {/* Action Buttons */}
        <div className="flex space-x-2">
          <button
            onClick={handleCopy}
            className="flex items-center px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white text-sm rounded-md transition-colors"
            title="Copy to clipboard"
          >
            {copySuccess ? (
              <>
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy
              </>
            )}
          </button>
          
          <button
            onClick={handleDownload}
            className="flex items-center px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-md transition-colors"
            title="Download file"
          >
            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download
          </button>
        </div>
      </div>
      
      {/* Code Output */}
      <div className="relative">
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm leading-relaxed font-mono border border-gray-700">
          <code>{output}</code>
        </pre>
      </div>
      
      {/* Configuration Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-md text-sm">
        <div>
          <p className="text-gray-500 dark:text-gray-400 text-xs">Model</p>
          <p className="text-gray-900 dark:text-gray-100 font-medium truncate">
            {generatedConfig.model.name}
          </p>
        </div>
        <div>
          <p className="text-gray-500 dark:text-gray-400 text-xs">GPUs</p>
          <p className="text-gray-900 dark:text-gray-100 font-medium">
            {generatedConfig.gpuCount}x {generatedConfig.gpus[0]?.name.split(' ').slice(-1)[0] || 'GPU'}
          </p>
        </div>
        <div>
          <p className="text-gray-500 dark:text-gray-400 text-xs">Memory Usage</p>
          <p className="text-gray-900 dark:text-gray-100 font-medium">
            {generatedConfig.memoryUsage.percentage.toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-gray-500 dark:text-gray-400 text-xs">Container</p>
          <p className="text-gray-900 dark:text-gray-100 font-medium truncate">
            {generatedConfig.containerName}
          </p>
        </div>
      </div>
    </div>
  );
};
