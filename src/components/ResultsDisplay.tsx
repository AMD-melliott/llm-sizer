import React from 'react';
import { AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { CalculationResults } from '../types';
import { formatMemorySize } from '../utils/memoryCalculator';
import useAppStore from '../store/useAppStore';

interface ResultsDisplayProps {
  results: CalculationResults | null;
  isCalculating: boolean;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, isCalculating }) => {
  if (isCalculating) {
    return (
      <div
        className="flex items-center justify-center h-64 rounded-xl"
        style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
      >
        <div className="animate-spin rounded-full h-10 w-10 border-2 border-transparent" style={{ borderTopColor: 'var(--accent)', borderRightColor: 'var(--accent)' }} />
      </div>
    );
  }

  if (!results) {
    return (
      <div
        className="flex items-center justify-center h-64 rounded-xl"
        style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
      >
        <p>Configure parameters to see results</p>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (results.status) {
      case 'okay':
        return <CheckCircle className="w-5 h-5" style={{ color: 'var(--success)' }} />;
      case 'warning':
        return <AlertCircle className="w-5 h-5" style={{ color: 'var(--warning)' }} />;
      case 'error':
        return <XCircle className="w-5 h-5" style={{ color: 'var(--danger)' }} />;
    }
  };

  const getStatusStyle = () => {
    switch (results.status) {
      case 'okay':
        return { backgroundColor: 'var(--success-light)', borderColor: 'var(--success)', color: 'var(--success-text)' };
      case 'warning':
        return { backgroundColor: 'var(--warning-light)', borderColor: 'var(--warning)', color: 'var(--warning-text)' };
      case 'error':
        return { backgroundColor: 'var(--danger-light)', borderColor: 'var(--danger)', color: 'var(--danger-text)' };
    }
  };

  // Build gradient stops for the gauge
  const getGaugeGradient = () => {
    const pct = Math.min(results.vramPercentage, 100);
    if (pct <= 60) return 'linear-gradient(90deg, #10b981, #34d399)';
    if (pct <= 80) return 'linear-gradient(90deg, #10b981, #3b82f6)';
    if (pct <= 90) return 'linear-gradient(90deg, #10b981, #3b82f6, #f59e0b)';
    return 'linear-gradient(90deg, #10b981, #f59e0b, #ef4444)';
  };

  const pct = results.vramPercentage;

  const {
    inferenceQuantization,
    kvCacheQuantization,
    batchSize,
    sequenceLength,
    concurrentUsers,
    numGPUs,
  } = useAppStore();

  const quantLabels: Record<string, string> = {
    fp16: 'FP16',
    fp8: 'FP8',
    int8: 'INT8',
    int4: 'INT4',
    fp16_bf16: 'FP16',
    fp8_bf16: 'FP8',
  };

  return (
    <div className="space-y-4">
      {/* Inference Parameters Summary */}
      <div
        className="rounded-xl p-6"
        style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 1px 3px var(--shadow-color)' }}
      >
        <div className="flex items-baseline justify-between mb-4">
          <h3 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>Inference Parameters</h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Adjust inference parameters below</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div>
            <span className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Quantization</span>
            <span className="text-sm font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>{quantLabels[inferenceQuantization]}</span>
          </div>
          <div>
            <span className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>KV Cache</span>
            <span className="text-sm font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>{quantLabels[kvCacheQuantization]}</span>
          </div>
          <div>
            <span className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Batch Size</span>
            <span className="text-sm font-semibold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{batchSize}</span>
          </div>
          <div>
            <span className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Sequence Length</span>
            <span className="text-sm font-semibold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{sequenceLength.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Concurrent Users</span>
            <span className="text-sm font-semibold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{concurrentUsers}</span>
          </div>
          {numGPUs > 1 && (
            <div>
              <span className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>GPU Count</span>
              <span className="text-sm font-semibold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{numGPUs}</span>
            </div>
          )}
        </div>
      </div>

      {/* VRAM Usage Gauge */}
      <div
        className="rounded-xl p-6"
        style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 1px 3px var(--shadow-color)' }}
      >
        <div className="flex items-baseline justify-between mb-4">
          <h3 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>VRAM Usage</h3>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold font-mono tabular-nums" style={{ color: pct > 100 ? 'var(--danger)' : pct > 90 ? 'var(--warning)' : 'var(--accent)' }}>
              {pct.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Enhanced gauge */}
        <div className="relative mb-2">
          {/* Track */}
          <div className="w-full h-10 rounded-lg overflow-hidden" style={{ backgroundColor: 'var(--gauge-track)' }}>
            {/* Fill */}
            <div
              className="h-full rounded-lg transition-all duration-700 ease-out flex items-center justify-end pr-3"
              style={{
                width: `${Math.min(pct, 100)}%`,
                background: getGaugeGradient(),
                minWidth: pct > 0 ? '2rem' : '0',
              }}
            >
              {pct > 15 && (
                <span className="text-xs font-semibold text-white drop-shadow-sm font-mono">
                  {formatMemorySize(results.usedVRAM)}
                </span>
              )}
            </div>
          </div>

          {/* 100% threshold marker */}
          {pct > 80 && pct <= 100 && (
            <div
              className="absolute top-0 h-full w-px"
              style={{ left: '100%', backgroundColor: 'var(--danger)', opacity: 0.5 }}
            />
          )}
        </div>

        {/* Labels */}
        <div className="flex justify-between text-xs" style={{ color: 'var(--text-muted)' }}>
          <span>0 GB</span>
          <span className="font-mono">{formatMemorySize(results.totalVRAM)} total</span>
        </div>

        {pct > 100 && (
          <div className="mt-3 text-xs font-medium" style={{ color: 'var(--danger)' }}>
            Exceeds capacity by {formatMemorySize(results.usedVRAM - results.totalVRAM)}
          </div>
        )}

        {/* Status Message */}
        {results.message && (
          <div
            className="mt-4 p-3 rounded-lg flex items-start space-x-2 text-sm"
            style={{
              ...getStatusStyle(),
              border: '1px solid',
            }}
          >
            {getStatusIcon()}
            <p>{results.message}</p>
          </div>
        )}
      </div>

      {/* Memory Breakdown */}
      <div
        className="rounded-xl p-6"
        style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 1px 3px var(--shadow-color)' }}
      >
        <h3 className="text-base font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Memory Breakdown</h3>

        <div className="space-y-2.5">
          {[
            { label: 'Base Model Weights', value: results.memoryBreakdown.baseWeights },
            { label: 'KV Cache', value: results.memoryBreakdown.kvCache },
            { label: 'Activations', value: results.memoryBreakdown.activations },
            { label: 'Framework Overhead', value: results.memoryBreakdown.frameworkOverhead },
            ...(results.memoryBreakdown.multiGPUOverhead > 0
              ? [{ label: 'Multi-GPU Overhead', value: results.memoryBreakdown.multiGPUOverhead }]
              : []),
          ].map((item) => (
            <div key={item.label} className="flex justify-between items-center">
              <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
              <span className="text-sm font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>
                {formatMemorySize(item.value)}
              </span>
            </div>
          ))}

          <div className="pt-2.5 mt-1" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="flex justify-between items-center">
              <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Total</span>
              <span className="text-sm font-bold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>
                {formatMemorySize(results.usedVRAM)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
