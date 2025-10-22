import React from 'react';
import { AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { CalculationResults } from '../types';
import { formatMemorySize } from '../utils/memoryCalculator';

interface ResultsDisplayProps {
  results: CalculationResults | null;
  isCalculating: boolean;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, isCalculating }) => {
  if (isCalculating) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <p>Configure parameters to see results</p>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (results.status) {
      case 'okay':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
    }
  };

  const getStatusColor = () => {
    switch (results.status) {
      case 'okay':
        return 'bg-green-50 border-green-200';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200';
      case 'error':
        return 'bg-red-50 border-red-200';
    }
  };

  const getGaugeColor = () => {
    if (results.vramPercentage > 100) return 'bg-red-500';
    if (results.vramPercentage > 90) return 'bg-orange-500';
    if (results.vramPercentage > 80) return 'bg-yellow-500';
    if (results.vramPercentage > 60) return 'bg-blue-500';
    return 'bg-green-500';
  };

  return (
    <div className="space-y-6">
      {/* VRAM Usage Gauge */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">VRAM Usage</h3>

        <div className="mb-4">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-gray-600">Memory Utilization</span>
            <span className="text-sm font-medium">
              {formatMemorySize(results.usedVRAM)} / {formatMemorySize(results.totalVRAM)}
            </span>
          </div>

          <div className="w-full bg-gray-200 rounded-full h-8 overflow-hidden">
            <div
              className={`h-full ${getGaugeColor()} transition-all duration-500 flex items-center justify-center text-white text-sm font-medium`}
              style={{ width: `${Math.min(results.vramPercentage, 100)}%` }}
            >
              {results.vramPercentage.toFixed(1)}%
            </div>
          </div>

          {results.vramPercentage > 100 && (
            <div className="mt-2 text-xs text-red-600">
              Exceeds capacity by {formatMemorySize(results.usedVRAM - results.totalVRAM)}
            </div>
          )}
        </div>

        {/* Status Message */}
        {results.message && (
          <div className={`p-3 rounded-lg border ${getStatusColor()} flex items-start space-x-2`}>
            {getStatusIcon()}
            <p className="text-sm">{results.message}</p>
          </div>
        )}
      </div>

      {/* Memory Breakdown */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Memory Breakdown</h3>

        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Base Model Weights</span>
            <span className="text-sm font-medium">{formatMemorySize(results.memoryBreakdown.baseWeights)}</span>
          </div>

          <div className="flex justify-between">
            <span className="text-sm text-gray-600">KV Cache</span>
            <span className="text-sm font-medium">{formatMemorySize(results.memoryBreakdown.kvCache)}</span>
          </div>

          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Activations</span>
            <span className="text-sm font-medium">{formatMemorySize(results.memoryBreakdown.activations)}</span>
          </div>

          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Framework Overhead</span>
            <span className="text-sm font-medium">{formatMemorySize(results.memoryBreakdown.frameworkOverhead)}</span>
          </div>

          {results.memoryBreakdown.multiGPUOverhead > 0 && (
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Multi-GPU Overhead</span>
              <span className="text-sm font-medium">{formatMemorySize(results.memoryBreakdown.multiGPUOverhead)}</span>
            </div>
          )}

          <div className="pt-3 border-t">
            <div className="flex justify-between">
              <span className="text-sm font-semibold">Total</span>
              <span className="text-sm font-bold">{formatMemorySize(results.usedVRAM)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics - Hidden per PRD section 2.5 */}
      {/* Performance metrics temporarily hidden to avoid displaying misleading information
          until we have proper data to support these calculations. This section will be
          re-enabled in a future update once accurate performance estimation is available. */}
    </div>
  );
};

export default ResultsDisplay;