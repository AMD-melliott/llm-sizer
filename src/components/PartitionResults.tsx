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
    const baseClasses = 'px-2 py-1 rounded text-xs font-medium font-mono tabular-nums';
    switch (status) {
      case 'fits':
        return (
          <span className={baseClasses} style={{ backgroundColor: 'var(--success-light)', color: 'var(--success-text)' }}>
            ✓ Fits ({percentUsed.toFixed(0)}%)
          </span>
        );
      case 'tight':
        return (
          <span className={baseClasses} style={{ backgroundColor: 'var(--warning-light)', color: 'var(--warning-text)' }}>
            ⚠ Tight ({percentUsed.toFixed(0)}%)
          </span>
        );
      case 'no-fit':
        return (
          <span className={baseClasses} style={{ backgroundColor: 'var(--danger-light)', color: 'var(--danger-text)' }}>
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
      <div className="px-4 py-3" style={{ backgroundColor: 'var(--surface-inset)' }}>
        <div className="text-sm font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>Memory Breakdown</div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          {visibleEntries.map((entry) => (
            <div key={entry.label} className="flex justify-between">
              <span style={{ color: 'var(--text-secondary)' }}>{entry.label}:</span>
              <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{formatMemorySize(entry.value)}</span>
            </div>
          ))}
          <div className="flex justify-between pt-2 border-t col-span-2 font-semibold" style={{ borderColor: 'var(--border-strong)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Total:</span>
            <span className="font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{formatMemorySize(total)}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
          Compatible Models
        </h3>
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showOnlyFits}
            onChange={(e) => onToggleShowOnlyFits(e.target.checked)}
            className="w-4 h-4 rounded focus:ring-offset-0"
            style={{
              borderColor: 'var(--border-strong)',
              backgroundColor: 'var(--surface)',
              color: 'var(--accent)'
            }}
          />
          <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Show only models that fit</span>
        </label>
      </div>

      {/* Models List */}
      <div className="rounded-xl overflow-hidden" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border-strong)', boxShadow: '0 1px 3px var(--shadow-color)' }}>
        <div className="max-h-[600px] overflow-y-auto">
          <table className="w-full">
            <thead className="sticky top-0" style={{ backgroundColor: 'var(--surface-inset)' }}>
              <tr className="text-left text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                <th className="px-4 py-3 w-10"></th>
                <th
                  className="px-4 py-3 cursor-pointer select-none"
                  onClick={() => handleSort('model')}
                  style={{ backgroundColor: 'var(--surface-inset)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-inset)'}
                >
                  Model <SortIcon field="model" />
                </th>
                <th
                  className="px-4 py-3 cursor-pointer select-none"
                  onClick={() => handleSort('type')}
                  style={{ backgroundColor: 'var(--surface-inset)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-inset)'}
                >
                  Type <SortIcon field="type" />
                </th>
                <th
                  className="px-4 py-3 text-right cursor-pointer select-none"
                  onClick={() => handleSort('memoryUsed')}
                  style={{ backgroundColor: 'var(--surface-inset)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-inset)'}
                >
                  Memory Used <SortIcon field="memoryUsed" />
                </th>
                <th
                  className="px-4 py-3 text-right cursor-pointer select-none"
                  onClick={() => handleSort('usage')}
                  style={{ backgroundColor: 'var(--surface-inset)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-inset)'}
                >
                  Usage <SortIcon field="usage" />
                </th>
                <th
                  className="px-4 py-3 text-right cursor-pointer select-none"
                  onClick={() => handleSort('recommended')}
                  style={{ backgroundColor: 'var(--surface-inset)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-inset)'}
                >
                  Recommended Quant <SortIcon field="recommended" />
                </th>
              </tr>
            </thead>
            <tbody style={{ borderTop: '1px solid var(--border)' }}>
              {displayedModels.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center" style={{ color: 'var(--text-secondary)' }}>
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
                        className={!result.fits ? 'opacity-60' : ''}
                        style={{ borderTop: index > 0 ? '1px solid var(--border)' : undefined }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--surface-inset)'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                      >
                        <td className="px-4 py-3">
                          <button
                            onClick={() => toggleRowExpanded(index)}
                            className="focus:outline-none"
                            style={{ color: 'var(--text-muted)' }}
                            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
                            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
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
                          <div className="font-medium" style={{ color: 'var(--text-primary)' }}>
                            {getModelDisplayName(result.model)}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs px-2 py-1 rounded" style={{ backgroundColor: 'var(--surface-inset)', color: 'var(--text-secondary)' }}>
                            {getModelType(result.model)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right" style={{ color: 'var(--text-primary)' }}>
                          <div className="space-y-1">
                            <div className="font-mono tabular-nums">{formatMemorySize(result.memoryUsed)}</div>
                            <div className="h-2 w-full rounded overflow-hidden" style={{ backgroundColor: 'var(--surface-inset)' }}>
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
                        <td className="px-4 py-3 text-right" style={{ color: 'var(--text-primary)' }}>
                          {result.recommendedQuantization ? (
                            <span className="text-xs px-2 py-1 rounded font-medium uppercase font-mono" style={{ backgroundColor: 'var(--accent-light)', color: 'var(--accent-text)' }}>
                              {result.recommendedQuantization}
                            </span>
                          ) : (
                            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>—</span>
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
        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--success-light)', border: '1px solid var(--border)' }}>
          <div className="flex items-center space-x-2 mb-1">
            <Check className="w-5 h-5" style={{ color: 'var(--success-text)' }} />
            <span className="font-semibold" style={{ color: 'var(--success-text)' }}>Fits</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{fitsModels.length}</div>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>models fit comfortably</div>
        </div>
        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--warning-light)', border: '1px solid var(--border)' }}>
          <div className="flex items-center space-x-2 mb-1">
            <AlertTriangle className="w-5 h-5" style={{ color: 'var(--warning-text)' }} />
            <span className="font-semibold" style={{ color: 'var(--warning-text)' }}>Tight Fit</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{tightModels.length}</div>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>&gt;80% VRAM usage</div>
        </div>
        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--danger-light)', border: '1px solid var(--border)' }}>
          <div className="flex items-center space-x-2 mb-1">
            <X className="w-5 h-5" style={{ color: 'var(--danger-text)' }} />
            <span className="font-semibold" style={{ color: 'var(--danger-text)' }}>Too Large</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{noFitModels.length}</div>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>exceeds partition VRAM</div>
        </div>
      </div>

      {/* Configuration Info */}
      <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border-strong)', boxShadow: '0 1px 3px var(--shadow-color)' }}>
        <div className="text-sm mb-2 font-medium" style={{ color: 'var(--text-secondary)' }}>Configuration:</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span style={{ color: 'var(--text-secondary)' }}>Per Partition:</span>{' '}
            <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>
              {formatMemorySize(configuration.mode.vramPerPartition)}
            </span>
          </div>
          <div>
            <span style={{ color: 'var(--text-secondary)' }}>Quantization:</span>{' '}
            <span className="font-medium uppercase font-mono" style={{ color: 'var(--text-primary)' }}>
              {configuration.inferenceQuantization}
            </span>
          </div>
          <div>
            <span style={{ color: 'var(--text-secondary)' }}>Batch Size:</span>{' '}
            <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{configuration.batchSize}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-secondary)' }}>Seq Length:</span>{' '}
            <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>{configuration.sequenceLength}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
