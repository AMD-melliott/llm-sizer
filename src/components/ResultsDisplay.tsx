import React from 'react';
import { AlertCircle, CheckCircle, XCircle, Activity, Zap, Clock } from 'lucide-react';
import { CalculationResults } from '../types';
import { formatMemorySize } from '../utils/memoryCalculator';
import { getPerformanceRating, estimateLatency } from '../utils/performanceEstimator';

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

  const performanceRating = getPerformanceRating(results.performance.perUserSpeed);
  const latency = estimateLatency(results.performance.perUserSpeed);

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

      {/* Performance Metrics */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>

        <div className="grid grid-cols-2 gap-4 mb-4">
          {/* Embedding-specific metrics */}
          {results.performance.documentsPerSecond !== undefined && (
            <>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Activity className="w-4 h-4 text-blue-500" />
                  <span className="text-xs text-gray-600">Documents/sec</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.documentsPerSecond.toFixed(1)}</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Zap className="w-4 h-4 text-green-500" />
                  <span className="text-xs text-gray-600">Tokens/sec</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.tokensPerSecond?.toFixed(1)}</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Zap className="w-4 h-4 text-purple-500" />
                  <span className="text-xs text-gray-600">Embeddings/sec</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.embeddingsPerSecond?.toFixed(1)}</p>
              </div>
            </>
          )}

          {/* Reranking-specific metrics */}
          {results.performance.queryDocPairsPerSecond !== undefined && (
            <>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Activity className="w-4 h-4 text-blue-500" />
                  <span className="text-xs text-gray-600">Query-Doc Pairs/sec</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.queryDocPairsPerSecond.toFixed(1)}</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Zap className="w-4 h-4 text-green-500" />
                  <span className="text-xs text-gray-600">Queries/sec</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.queriesPerSecond?.toFixed(2)}</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Clock className="w-4 h-4 text-purple-500" />
                  <span className="text-xs text-gray-600">Avg Latency</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.avgLatencyMs?.toFixed(1)} ms</p>
              </div>
            </>
          )}

          {/* Generation-specific metrics */}
          {results.performance.generationSpeed && results.performance.documentsPerSecond === undefined && results.performance.queryDocPairsPerSecond === undefined && (
            <>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Zap className="w-4 h-4 text-blue-500" />
                  <span className="text-xs text-gray-600">Generation Speed</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.generationSpeed} tok/s</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Activity className="w-4 h-4 text-green-500" />
                  <span className="text-xs text-gray-600">Total Throughput</span>
                </div>
                <p className="text-lg font-semibold">{results.performance.totalThroughput} tok/s</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Clock className="w-4 h-4 text-purple-500" />
                  <span className="text-xs text-gray-600">First Token Latency</span>
                </div>
                <p className="text-lg font-semibold">{latency.firstToken} ms</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Clock className="w-4 h-4 text-orange-500" />
                  <span className="text-xs text-gray-600">100 Token Response</span>
                </div>
                <p className="text-lg font-semibold">{(latency.fullResponse / 1000).toFixed(1)} s</p>
              </div>
            </>
          )}
        </div>

        <div className={`p-3 rounded-lg bg-gray-50 border-l-4 ${
          performanceRating.color === 'text-green-600' ? 'border-green-500' :
          performanceRating.color === 'text-blue-600' ? 'border-blue-500' :
          performanceRating.color === 'text-yellow-600' ? 'border-yellow-500' :
          performanceRating.color === 'text-orange-600' ? 'border-orange-500' :
          'border-red-500'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <span className={`text-sm font-semibold ${performanceRating.color}`}>
                {performanceRating.rating} Performance
              </span>
              <p className="text-xs text-gray-600 mt-1">{performanceRating.description}</p>
            </div>
            <span className="text-2xl font-bold text-gray-700">
              {results.performance.perUserSpeed} tok/s
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;