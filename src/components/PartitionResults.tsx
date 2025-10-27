import { PartitionAnalysisResults, Model, EmbeddingModel, RerankingModel, MemoryBreakdown } from '../types';
import { getModelDisplayName, formatMemorySize } from '../utils/partitionCalculator';
import { Check, AlertTriangle, X, ArrowUpDown, ArrowUp, ArrowDown, ChevronDown, ChevronRight } from 'lucide-react';
import { useState, useMemo } from 'react';

interface PartitionResultsProps {
  results: PartitionAnalysisResults;
  showOnlyFits: boolean;
  onToggleShowOnlyFits: (show: boolean) => void;
}

type SortField = 'status' | 'model' | 'type' | 'memoryUsed' | 'usage' | 'recommended';
type SortDirection = 'asc' | 'desc';

export default function PartitionResults({
  results,
  showOnlyFits,
  onToggleShowOnlyFits,
}: PartitionResultsProps) {
  const { compatibleModels, configuration } = results;
  const [sortField, setSortField] = useState<SortField>('memoryUsed');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  
  const toggleRowExpanded = (index: number) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };
  
  // Filter models based on showOnlyFits
  const filteredModels = showOnlyFits
    ? compatibleModels.filter((m) => m.fits)
    : compatibleModels;

  // Sort models
  const displayedModels = useMemo(() => {
    const sorted = [...filteredModels];
    
    sorted.sort((a, b) => {
      let comparison = 0;
      
      switch (sortField) {
        case 'status':
          const statusOrder = { 'fits': 0, 'tight': 1, 'no-fit': 2 };
          comparison = statusOrder[a.status] - statusOrder[b.status];
          break;
        case 'model':
          comparison = getModelDisplayName(a.model).localeCompare(getModelDisplayName(b.model));
          break;
        case 'type':
          const typeA = 'parameters_billions' in a.model ? 'generation' : 'embedding';
          const typeB = 'parameters_billions' in b.model ? 'generation' : 'embedding';
          comparison = typeA.localeCompare(typeB);
          break;
        case 'memoryUsed':
          comparison = a.memoryUsed - b.memoryUsed;
          break;
        case 'usage':
          comparison = a.percentUsed - b.percentUsed;
          break;
        case 'recommended':
          // Sort by recommended quantization quality (fp16 highest -> int4 lowest)
          const order: Record<string, number> = { fp16: 0, fp8: 1, int8: 2, int4: 3 };
          comparison = (order[a.recommendedQuantization || 'zz'] ?? 99) - (order[b.recommendedQuantization || 'zz'] ?? 99);
          break;
      }
      
      return sortDirection === 'asc' ? comparison : -comparison;
    });
    
    return sorted;
  }, [filteredModels, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-4 h-4 inline ml-1 opacity-40" />;
    }
    return sortDirection === 'asc' 
      ? <ArrowUp className="w-4 h-4 inline ml-1" />
      : <ArrowDown className="w-4 h-4 inline ml-1" />;
  };

  // Categorize models
  const fitsModels = compatibleModels.filter((m) => m.status === 'fits');
  const tightModels = compatibleModels.filter((m) => m.status === 'tight');
  const noFitModels = compatibleModels.filter((m) => m.status === 'no-fit');

  const getStatusBadge = (status: 'fits' | 'tight' | 'no-fit', percentUsed: number) => {
    const baseClasses = 'px-2 py-1 rounded text-xs font-medium';
    switch (status) {
      case 'fits':
        return (
          <span className={`${baseClasses} bg-green-100 text-green-700`}>
            ✓ Fits ({percentUsed.toFixed(0)}%)
          </span>
        );
      case 'tight':
        return (
          <span className={`${baseClasses} bg-yellow-100 text-yellow-700`}>
            ⚠ Tight ({percentUsed.toFixed(0)}%)
          </span>
        );
      case 'no-fit':
        return (
          <span className={`${baseClasses} bg-red-100 text-red-700`}>
            ✗ Too Large
          </span>
        );
    }
  };

  const getModelType = (model: Model | EmbeddingModel | RerankingModel): string => {
    if ('parameters_billions' in model) {
      return 'Text Gen';
    } else if ('dimensions' in model) {
      return 'Embedding';
    } else if ('max_query_length' in model) {
      return 'Reranking';
    }
    return 'Unknown';
  };

  const renderMemoryBreakdown = (breakdown: MemoryBreakdown) => {
    const entries: { label: string; value: number; show: boolean }[] = [
      { label: 'Base Weights', value: breakdown.baseWeights, show: true },
      { label: 'Activations', value: breakdown.activations, show: true },
      { label: 'KV Cache', value: breakdown.kvCache, show: true },
      { label: 'Framework Overhead', value: breakdown.frameworkOverhead, show: true },
      { label: 'Multi-GPU Overhead', value: breakdown.multiGPUOverhead, show: breakdown.multiGPUOverhead > 0 },
      { label: 'Safety Margin', value: breakdown.safetyMargin || 0, show: (breakdown.safetyMargin || 0) > 0 },
      // Multimodal components
      { label: 'Vision Weights', value: breakdown.visionWeights || 0, show: (breakdown.visionWeights || 0) > 0 },
      { label: 'Vision Activations', value: breakdown.visionActivations || 0, show: (breakdown.visionActivations || 0) > 0 },
      { label: 'Projector Weights', value: breakdown.projectorWeights || 0, show: (breakdown.projectorWeights || 0) > 0 },
      { label: 'Image Preprocessing', value: breakdown.imagePreprocessing || 0, show: (breakdown.imagePreprocessing || 0) > 0 },
      { label: 'Image Tokens KV', value: breakdown.imageTokensKV || 0, show: (breakdown.imageTokensKV || 0) > 0 },
      // Embedding components
      { label: 'Batch Input Memory', value: breakdown.batchInputMemory || 0, show: (breakdown.batchInputMemory || 0) > 0 },
      { label: 'Attention Memory', value: breakdown.attentionMemory || 0, show: (breakdown.attentionMemory || 0) > 0 },
      { label: 'Embedding Storage', value: breakdown.embeddingStorage || 0, show: (breakdown.embeddingStorage || 0) > 0 },
      // Reranking components
      { label: 'Pair Batch Memory', value: breakdown.pairBatchMemory || 0, show: (breakdown.pairBatchMemory || 0) > 0 },
      { label: 'Scoring Memory', value: breakdown.scoringMemory || 0, show: (breakdown.scoringMemory || 0) > 0 },
    ];

    const visibleEntries = entries.filter(e => e.show && e.value > 0);
    const total = visibleEntries.reduce((sum, e) => sum + e.value, 0);

    return (
      <div className="px-4 py-3 bg-gray-50">
        <div className="text-sm font-semibold text-gray-700 mb-2">Memory Breakdown</div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          {visibleEntries.map((entry) => (
            <div key={entry.label} className="flex justify-between">
              <span className="text-gray-600">{entry.label}:</span>
              <span className="text-gray-900 font-medium">{formatMemorySize(entry.value)}</span>
            </div>
          ))}
          <div className="flex justify-between pt-2 border-t border-gray-300 col-span-2 font-semibold">
            <span className="text-gray-700">Total:</span>
            <span className="text-gray-900">{formatMemorySize(total)}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          Compatible Models
        </h3>
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showOnlyFits}
            onChange={(e) => onToggleShowOnlyFits(e.target.checked)}
            className="w-4 h-4 rounded border-gray-300 bg-white text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
          />
          <span className="text-sm text-gray-700">Show only models that fit</span>
        </label>
      </div>

      {/* Models List */}
      <div className="bg-white rounded-lg overflow-hidden border border-gray-300 shadow-sm">
        <div className="max-h-[600px] overflow-y-auto">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0">
              <tr className="text-left text-sm text-gray-700 font-medium">
                <th className="px-4 py-3 w-10"></th>
                <th 
                  className="px-4 py-3 cursor-pointer hover:bg-gray-100 select-none"
                  onClick={() => handleSort('model')}
                >
                  Model <SortIcon field="model" />
                </th>
                <th 
                  className="px-4 py-3 cursor-pointer hover:bg-gray-100 select-none"
                  onClick={() => handleSort('type')}
                >
                  Type <SortIcon field="type" />
                </th>
                <th 
                  className="px-4 py-3 text-right cursor-pointer hover:bg-gray-100 select-none"
                  onClick={() => handleSort('memoryUsed')}
                >
                  Memory Used <SortIcon field="memoryUsed" />
                </th>
                <th 
                  className="px-4 py-3 text-right cursor-pointer hover:bg-gray-100 select-none"
                  onClick={() => handleSort('usage')}
                >
                  Usage <SortIcon field="usage" />
                </th>
                <th 
                  className="px-4 py-3 text-right cursor-pointer hover:bg-gray-100 select-none"
                  onClick={() => handleSort('recommended')}
                >
                  Recommended Quant <SortIcon field="recommended" />
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {displayedModels.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-gray-600">
                    No models match the current filter
                  </td>
                </tr>
              ) : (
                displayedModels.map((result, index) => {
                  const isExpanded = expandedRows.has(index);
                  return (
                    <>
                      <tr
                        key={`model-${index}`}
                        className={`hover:bg-gray-50 ${
                          !result.fits ? 'opacity-60' : ''
                        }`}
                      >
                        <td className="px-4 py-3">
                          <button
                            onClick={() => toggleRowExpanded(index)}
                            className="text-gray-500 hover:text-gray-700 focus:outline-none"
                            aria-label="Toggle memory breakdown"
                          >
                            {isExpanded ? (
                              <ChevronDown className="w-4 h-4" />
                            ) : (
                              <ChevronRight className="w-4 h-4" />
                            )}
                          </button>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-gray-900 font-medium">
                            {getModelDisplayName(result.model)}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-700">
                            {getModelType(result.model)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900">
                          <div className="space-y-1">
                            <div>{formatMemorySize(result.memoryUsed)}</div>
                            <div className="h-2 w-full bg-gray-200 rounded overflow-hidden">
                              <div
                                className={`h-full rounded ${
                                  result.percentUsed > 90
                                    ? 'bg-red-500'
                                    : result.percentUsed > 80
                                    ? 'bg-yellow-500'
                                    : 'bg-green-500'
                                }`}
                                style={{ width: `${Math.min(result.percentUsed, 100)}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-right">
                          {getStatusBadge(result.status, result.percentUsed)}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900">
                          {result.recommendedQuantization ? (
                            <span className="text-xs px-2 py-1 rounded bg-blue-100 text-blue-700 font-medium uppercase">
                              {result.recommendedQuantization}
                            </span>
                          ) : (
                            <span className="text-xs text-gray-500">—</span>
                          )}
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr key={`breakdown-${index}`}>
                          <td colSpan={6} className="p-0">
                            {renderMemoryBreakdown(result.memoryBreakdown)}
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-1">
            <Check className="w-5 h-5 text-green-600" />
            <span className="text-green-700 font-semibold">Fits</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">{fitsModels.length}</div>
          <div className="text-xs text-gray-600">models fit comfortably</div>
        </div>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-1">
            <AlertTriangle className="w-5 h-5 text-yellow-600" />
            <span className="text-yellow-700 font-semibold">Tight Fit</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">{tightModels.length}</div>
          <div className="text-xs text-gray-600">&gt;80% VRAM usage</div>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-1">
            <X className="w-5 h-5 text-red-600" />
            <span className="text-red-700 font-semibold">Too Large</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">{noFitModels.length}</div>
          <div className="text-xs text-gray-600">exceeds partition VRAM</div>
        </div>
      </div>

      {/* Configuration Info */}
      <div className="bg-white rounded-lg p-4 border border-gray-300 shadow-sm">
        <div className="text-sm text-gray-700 mb-2 font-medium">Configuration:</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Per Partition:</span>{' '}
            <span className="text-gray-900 font-medium">
              {formatMemorySize(configuration.mode.vramPerPartition)}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Quantization:</span>{' '}
            <span className="text-gray-900 font-medium uppercase">
              {configuration.inferenceQuantization}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Batch Size:</span>{' '}
            <span className="text-gray-900 font-medium">{configuration.batchSize}</span>
          </div>
          <div>
            <span className="text-gray-600">Seq Length:</span>{' '}
            <span className="text-gray-900 font-medium">{configuration.sequenceLength}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
