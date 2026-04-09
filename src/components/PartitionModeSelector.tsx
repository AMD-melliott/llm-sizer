import { PartitionMode, PartitionModeType } from '../types';
import { formatMemorySize, formatCompute } from '../utils/partitionCalculator';

interface PartitionModeSelectorProps {
  modes: PartitionMode[];
  selectedMode: PartitionModeType;
  onModeChange: (mode: PartitionModeType) => void;
}

export default function PartitionModeSelector({
  modes,
  selectedMode,
  onModeChange,
}: PartitionModeSelectorProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Partitioning Mode</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {modes.map((mode) => {
          const isSelected = mode.mode === selectedMode;
          return (
            <button
              key={mode.mode}
              onClick={() => onModeChange(mode.mode)}
              className="p-4 rounded-xl border-2 transition-all text-left"
              style={{
                borderColor: isSelected ? 'var(--accent)' : 'var(--border-strong)',
                backgroundColor: isSelected ? 'var(--accent-light)' : 'var(--surface)',
                boxShadow: isSelected ? '0 1px 3px var(--shadow-color)' : undefined,
              }}
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>{mode.mode}</span>
                  <span className="text-sm font-mono tabular-nums" style={{ color: 'var(--text-secondary)' }}>
                    {mode.partitionCount}x
                  </span>
                </div>
                <div className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                  {mode.name}
                </div>
                <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  {mode.description}
                </div>
                <div className="pt-2 space-y-1 border-t" style={{ borderColor: 'var(--border-strong)' }}>
                  <div className="flex justify-between text-xs">
                    <span style={{ color: 'var(--text-secondary)' }}>VRAM/Partition:</span>
                    <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>
                      {formatMemorySize(mode.vramPerPartition)}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span style={{ color: 'var(--text-secondary)' }}>Bandwidth:</span>
                    <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>
                      {mode.bandwidthPerPartition} GB/s
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span style={{ color: 'var(--text-secondary)' }}>Compute (FP16):</span>
                    <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>
                      {formatCompute(mode.computeFP16PerPartition)}
                    </span>
                  </div>
                  {mode.computeFP8PerPartition && (
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-secondary)' }}>Compute (FP8):</span>
                      <span className="font-medium font-mono tabular-nums" style={{ color: 'var(--text-primary)' }}>
                        {formatCompute(mode.computeFP8PerPartition)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
