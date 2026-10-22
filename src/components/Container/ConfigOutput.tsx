import React, { useState } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import { generateDockerRunCommand, generateDockerCommandOnly } from '../../utils/dockerCommandBuilder';
import { generateDockerCompose } from '../../utils/dockerComposeBuilder';

export const ConfigOutput: React.FC = () => {
  const { generatedConfig, outputFormat, setOutputFormat, enableHealthcheck } = useContainerStore();
  const [copySuccess, setCopySuccess] = useState(false);
  const [copyPlainSuccess, setCopyPlainSuccess] = useState(false);

  if (!generatedConfig) {
    return (
      <div
        className="flex items-center justify-center h-64 rounded-xl"
        style={{ backgroundColor: 'var(--surface-inset)', border: '2px dashed var(--border-strong)' }}
      >
        <div className="text-center">
          <svg className="mx-auto h-12 w-12" style={{ color: 'var(--text-muted)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="mt-2 text-sm" style={{ color: 'var(--text-muted)' }}>
            Configure your container and click "Generate" to see the output
          </p>
        </div>
      </div>
    );
  }

  const output = outputFormat === 'docker-run'
    ? generateDockerRunCommand(generatedConfig, { includeComments: true, includeManagementCommands: true })
    : generateDockerCompose(generatedConfig, { includeComments: true, enableHealthcheck });

  const plainCommand = outputFormat === 'docker-run'
    ? generateDockerCommandOnly(generatedConfig)
    : null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(output);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleCopyPlain = async () => {
    if (!plainCommand) return;
    try {
      await navigator.clipboard.writeText(plainCommand);
      setCopyPlainSuccess(true);
      setTimeout(() => setCopyPlainSuccess(false), 2000);
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
      {/* Plain Docker Run Command (for docker-run mode only) */}
      {outputFormat === 'docker-run' && plainCommand && (
        <div
          className="rounded-xl p-4"
          style={{ backgroundColor: 'var(--accent-light)', border: '1px solid var(--accent-muted)' }}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <svg className="w-5 h-5 mr-2" style={{ color: 'var(--accent)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                Docker Run Command
              </h3>
            </div>
            <button
              onClick={handleCopyPlain}
              className="flex items-center px-3 py-1.5 text-xs rounded-lg transition-colors"
              style={{ backgroundColor: 'var(--accent)', color: 'var(--text-inverse)' }}
              title="Copy command to clipboard"
            >
              {copyPlainSuccess ? (
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
          </div>
          <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
            Copy this command to paste directly into your server terminal. No bash scripting needed.
          </p>
          <pre
            className="p-3 rounded-lg overflow-x-auto text-xs leading-relaxed font-mono"
            style={{ backgroundColor: '#0f172a', color: '#e2e8f0', border: '1px solid var(--border)' }}
          >
            <code>{plainCommand}</code>
          </pre>
        </div>
      )}

      {/* Format Toggle */}
      <div className="flex items-center justify-between">
        <div className="flex space-x-2">
          <button
            onClick={() => setOutputFormat('docker-run')}
            className="px-4 py-2 text-sm font-medium rounded-lg transition-colors"
            style={{
              backgroundColor: outputFormat === 'docker-run' ? 'var(--accent)' : 'var(--surface-inset)',
              color: outputFormat === 'docker-run' ? 'var(--text-inverse)' : 'var(--text-secondary)',
              border: outputFormat === 'docker-run' ? 'none' : '1px solid var(--border)',
            }}
          >
            Docker Run
          </button>
          <button
            onClick={() => setOutputFormat('docker-compose')}
            className="px-4 py-2 text-sm font-medium rounded-lg transition-colors"
            style={{
              backgroundColor: outputFormat === 'docker-compose' ? 'var(--accent)' : 'var(--surface-inset)',
              color: outputFormat === 'docker-compose' ? 'var(--text-inverse)' : 'var(--text-secondary)',
              border: outputFormat === 'docker-compose' ? 'none' : '1px solid var(--border)',
            }}
          >
            Docker Compose
          </button>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-2">
          <button
            onClick={handleCopy}
            className="flex items-center px-3 py-2 text-sm rounded-lg transition-colors"
            style={{ backgroundColor: 'var(--surface-inset)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
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
            className="flex items-center px-3 py-2 text-sm rounded-lg transition-colors"
            style={{ backgroundColor: 'var(--accent)', color: 'var(--text-inverse)' }}
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
        <pre
          className="p-4 rounded-xl overflow-x-auto text-sm leading-relaxed font-mono"
          style={{ backgroundColor: '#0f172a', color: '#e2e8f0', border: '1px solid var(--border)' }}
        >
          <code>{output}</code>
        </pre>
      </div>

      {/* Configuration Summary */}
      <div
        className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 rounded-xl text-sm"
        style={{ backgroundColor: 'var(--surface-inset)', border: '1px solid var(--border)' }}
      >
        <div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Model</p>
          <p className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>
            {generatedConfig.model.name}
          </p>
        </div>
        <div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>GPUs</p>
          <p className="font-medium" style={{ color: 'var(--text-primary)' }}>
            {generatedConfig.gpuCount}x {generatedConfig.gpus[0]?.name.split(' ').slice(-1)[0] || 'GPU'}
          </p>
        </div>
        <div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Memory Usage</p>
          <p className="font-medium" style={{ color: 'var(--text-primary)' }}>
            {generatedConfig.memoryUsage.percentage.toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Container</p>
          <p className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>
            {generatedConfig.containerName}
          </p>
        </div>
      </div>
    </div>
  );
};
