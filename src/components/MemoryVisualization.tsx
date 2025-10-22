import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { MemoryBreakdown } from '../types';
import { formatMemorySize } from '../utils/memoryCalculator';

interface MemoryVisualizationProps {
  memoryBreakdown: MemoryBreakdown;
  totalVRAM: number;
}

const MemoryVisualization: React.FC<MemoryVisualizationProps> = ({ memoryBreakdown, totalVRAM }) => {
  // Prepare data for the bar chart
  const data = [
    {
      name: 'Model Weights',
      value: memoryBreakdown.baseWeights,
      percentage: ((memoryBreakdown.baseWeights / totalVRAM) * 100).toFixed(1),
      color: '#3B82F6', // blue
    },
    {
      name: 'KV Cache',
      value: memoryBreakdown.kvCache,
      percentage: ((memoryBreakdown.kvCache / totalVRAM) * 100).toFixed(1),
      color: '#10B981', // green
    },
    {
      name: 'Activations',
      value: memoryBreakdown.activations,
      percentage: ((memoryBreakdown.activations / totalVRAM) * 100).toFixed(1),
      color: '#F59E0B', // yellow
    },
    {
      name: 'Framework',
      value: memoryBreakdown.frameworkOverhead,
      percentage: ((memoryBreakdown.frameworkOverhead / totalVRAM) * 100).toFixed(1),
      color: '#8B5CF6', // purple
    },
  ];

  // Add multimodal components if present
  if (memoryBreakdown.visionWeights && memoryBreakdown.visionWeights > 0) {
    data.push({
      name: 'Vision Encoder',
      value: memoryBreakdown.visionWeights,
      percentage: ((memoryBreakdown.visionWeights / totalVRAM) * 100).toFixed(1),
      color: '#EC4899', // pink
    });
  }

  if (memoryBreakdown.visionActivations && memoryBreakdown.visionActivations > 0) {
    data.push({
      name: 'Vision Activations',
      value: memoryBreakdown.visionActivations,
      percentage: ((memoryBreakdown.visionActivations / totalVRAM) * 100).toFixed(1),
      color: '#F472B6', // lighter pink
    });
  }

  if (memoryBreakdown.projectorWeights && memoryBreakdown.projectorWeights > 0) {
    data.push({
      name: 'Projector',
      value: memoryBreakdown.projectorWeights,
      percentage: ((memoryBreakdown.projectorWeights / totalVRAM) * 100).toFixed(1),
      color: '#A855F7', // violet
    });
  }

  if (memoryBreakdown.imageTokensKV && memoryBreakdown.imageTokensKV > 0) {
    data.push({
      name: 'Image Tokens KV',
      value: memoryBreakdown.imageTokensKV,
      percentage: ((memoryBreakdown.imageTokensKV / totalVRAM) * 100).toFixed(1),
      color: '#14B8A6', // teal
    });
  }

  if (memoryBreakdown.multiGPUOverhead && memoryBreakdown.multiGPUOverhead > 0) {
    data.push({
      name: 'Multi-GPU',
      value: memoryBreakdown.multiGPUOverhead,
      percentage: ((memoryBreakdown.multiGPUOverhead / totalVRAM) * 100).toFixed(1),
      color: '#EF4444', // red
    });
  }

  const totalUsed = data.reduce((sum, item) => sum + item.value, 0);
  const remainingVRAM = Math.max(0, totalVRAM - totalUsed);

  // Custom tooltip component
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload[0]) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-sm">{data.name}</p>
          <p className="text-sm text-gray-600">
            {formatMemorySize(data.value)} ({data.percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  // Stacked horizontal bar visualization
  const stackedData = data.map((item, index) => {
    const prevSum = data.slice(0, index).reduce((sum, d) => sum + d.value, 0);
    return {
      ...item,
      start: (prevSum / totalVRAM) * 100,
      width: (item.value / totalVRAM) * 100,
    };
  });

  return (
    <div className="bg-white rounded-lg p-6 shadow-sm border">
      <h3 className="text-lg font-semibold mb-4">Memory Allocation Visualization</h3>

      {/* Stacked Horizontal Bar */}
      <div className="mb-6">
        <div className="relative h-12 bg-gray-100 rounded-lg overflow-hidden">
          {stackedData.map((item, index) => (
            <div
              key={index}
              className="absolute h-full transition-all duration-500"
              style={{
                left: `${item.start}%`,
                width: `${item.width}%`,
                backgroundColor: item.color,
              }}
            >
              {item.width > 5 && (
                <div className="h-full flex items-center justify-center text-white text-xs font-medium">
                  {item.percentage}%
                </div>
              )}
            </div>
          ))}
          {remainingVRAM > 0 && (
            <div
              className="absolute h-full bg-gray-200"
              style={{
                left: `${(totalUsed / totalVRAM) * 100}%`,
                width: `${(remainingVRAM / totalVRAM) * 100}%`,
              }}
            >
              <div className="h-full flex items-center justify-center text-gray-600 text-xs">
                Free
              </div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 mt-3">
          {data.map((item, index) => (
            <div key={index} className="flex items-center space-x-1">
              <div
                className="w-3 h-3 rounded"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-xs text-gray-600">
                {item.name}: {formatMemorySize(item.value)}
              </span>
            </div>
          ))}
          {remainingVRAM > 0 && (
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 rounded bg-gray-200" />
              <span className="text-xs text-gray-600">
                Free: {formatMemorySize(remainingVRAM)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Bar Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              label={{ value: 'Memory (GB)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t">
        <div className="text-center">
          <p className="text-xs text-gray-500">Total Used</p>
          <p className="text-lg font-semibold">{formatMemorySize(totalUsed)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">Available</p>
          <p className="text-lg font-semibold">{formatMemorySize(totalVRAM)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">Remaining</p>
          <p className={`text-lg font-semibold ${remainingVRAM < 0 ? 'text-red-600' : 'text-green-600'}`}>
            {formatMemorySize(remainingVRAM)}
          </p>
        </div>
      </div>
    </div>
  );
};

export default MemoryVisualization;