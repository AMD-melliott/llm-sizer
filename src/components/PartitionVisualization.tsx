import { PartitionMode } from '../types';
import { formatMemorySize, formatBandwidth } from '../utils/partitionCalculator';

interface PartitionVisualizationProps {
  mode: PartitionMode;
  gpuName: string;
}

export default function PartitionVisualization({
  mode,
  gpuName,
}: PartitionVisualizationProps) {
  // Generate partition colors
  const getPartitionColor = (index: number) => {
    const colors = [
      'bg-blue-500',
      'bg-green-500',
      'bg-purple-500',
      'bg-yellow-500',
      'bg-red-500',
      'bg-indigo-500',
      'bg-pink-500',
      'bg-teal-500',
    ];
    return colors[index % colors.length];
  };

  // Calculate grid layout
  const gridCols = mode.partitionCount === 1 ? 1 : mode.partitionCount === 2 ? 2 : 4;

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">
        {gpuName} Partition Layout
      </h3>
      <div className="bg-white rounded-lg p-6 border border-gray-300 shadow-sm">
        <div
          className={`grid gap-3`}
          style={{ gridTemplateColumns: `repeat(${gridCols}, 1fr)` }}
        >
          {Array.from({ length: mode.partitionCount }, (_, index) => (
            <div
              key={index}
              className={`${getPartitionColor(index)} rounded-lg p-4 min-h-[120px] flex flex-col justify-between`}
            >
              <div className="text-white font-bold text-lg">
                Partition {index + 1}
              </div>
              <div className="space-y-1 text-white text-sm">
                <div className="flex justify-between">
                  <span className="opacity-90">VRAM:</span>
                  <span className="font-semibold">
                    {formatMemorySize(mode.vramPerPartition)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="opacity-90">BW:</span>
                  <span className="font-semibold">
                    {formatBandwidth(mode.bandwidthPerPartition)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="opacity-90">FP16:</span>
                  <span className="font-semibold">
                    {mode.computeFP16PerPartition.toFixed(1)} TF
                  </span>
                </div>
                {mode.computeFP8PerPartition !== undefined && (
                  <div className="flex justify-between">
                    <span className="opacity-90">FP8:</span>
                    <span className="font-semibold">
                      {mode.computeFP8PerPartition!.toFixed(1)} TF
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-4 border-t border-gray-300">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Mode:</span>{' '}
              <span className="text-gray-900 font-medium">{mode.mode} - {mode.name}</span>
            </div>
            <div>
              <span className="text-gray-600">Total Partitions:</span>{' '}
              <span className="text-gray-900 font-medium">{mode.partitionCount}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
