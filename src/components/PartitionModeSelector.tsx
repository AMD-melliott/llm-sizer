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
      <h3 className="text-lg font-semibold text-gray-900">Partitioning Mode</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {modes.map((mode) => {
          const isSelected = mode.mode === selectedMode;
          return (
            <button
              key={mode.mode}
              onClick={() => onModeChange(mode.mode)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                isSelected
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 bg-white hover:border-gray-400 hover:bg-gray-50'
              }`}
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xl font-bold text-gray-900">{mode.mode}</span>
                  <span className="text-sm text-gray-600">
                    {mode.partitionCount}x
                  </span>
                </div>
                <div className="text-sm font-medium text-gray-700">
                  {mode.name}
                </div>
                <div className="text-xs text-gray-600">
                  {mode.description}
                </div>
                <div className="pt-2 space-y-1 border-t border-gray-300">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">VRAM/Partition:</span>
                    <span className="text-gray-900 font-medium">
                      {formatMemorySize(mode.vramPerPartition)}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">Bandwidth:</span>
                    <span className="text-gray-900 font-medium">
                      {mode.bandwidthPerPartition} GB/s
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">Compute (FP16):</span>
                    <span className="text-gray-900 font-medium">
                      {formatCompute(mode.computeFP16PerPartition)}
                    </span>
                  </div>
                  {mode.computeFP8PerPartition && (
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Compute (FP8):</span>
                      <span className="text-gray-900 font-medium">
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
