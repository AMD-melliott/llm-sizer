import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { MemoryBreakdown } from '../types';
import { formatMemorySize } from '../utils/memoryCalculator';

interface MemoryVisualizationProps {
  memoryBreakdown: MemoryBreakdown;
  totalVRAM: number;
}

const MemoryVisualization: React.FC<MemoryVisualizationProps> = ({ memoryBreakdown, totalVRAM }) => {
  const data = [
    { name: 'Model Weights', value: memoryBreakdown.baseWeights, percentage: ((memoryBreakdown.baseWeights / totalVRAM) * 100).toFixed(1), color: '#3B82F6' },
    { name: 'KV Cache', value: memoryBreakdown.kvCache, percentage: ((memoryBreakdown.kvCache / totalVRAM) * 100).toFixed(1), color: '#10B981' },
    { name: 'Activations', value: memoryBreakdown.activations, percentage: ((memoryBreakdown.activations / totalVRAM) * 100).toFixed(1), color: '#F59E0B' },
    { name: 'Framework', value: memoryBreakdown.frameworkOverhead, percentage: ((memoryBreakdown.frameworkOverhead / totalVRAM) * 100).toFixed(1), color: '#8B5CF6' },
  ];

  if (memoryBreakdown.visionWeights && memoryBreakdown.visionWeights > 0) {
    data.push({ name: 'Vision Encoder', value: memoryBreakdown.visionWeights, percentage: ((memoryBreakdown.visionWeights / totalVRAM) * 100).toFixed(1), color: '#EC4899' });
  }
  if (memoryBreakdown.visionActivations && memoryBreakdown.visionActivations > 0) {
    data.push({ name: 'Vision Activations', value: memoryBreakdown.visionActivations, percentage: ((memoryBreakdown.visionActivations / totalVRAM) * 100).toFixed(1), color: '#F472B6' });
  }
  if (memoryBreakdown.projectorWeights && memoryBreakdown.projectorWeights > 0) {
    data.push({ name: 'Projector', value: memoryBreakdown.projectorWeights, percentage: ((memoryBreakdown.projectorWeights / totalVRAM) * 100).toFixed(1), color: '#A855F7' });
  }
  if (memoryBreakdown.imageTokensKV && memoryBreakdown.imageTokensKV > 0) {
    data.push({ name: 'Image Tokens KV', value: memoryBreakdown.imageTokensKV, percentage: ((memoryBreakdown.imageTokensKV / totalVRAM) * 100).toFixed(1), color: '#14B8A6' });
  }
  if (memoryBreakdown.multiGPUOverhead && memoryBreakdown.multiGPUOverhead > 0) {
    data.push({ name: 'Multi-GPU', value: memoryBreakdown.multiGPUOverhead, percentage: ((memoryBreakdown.multiGPUOverhead / totalVRAM) * 100).toFixed(1), color: '#EF4444' });
  }

  const totalUsed = data.reduce((sum, item) => sum + item.value, 0);
  const remainingVRAM = Math.max(0, totalVRAM - totalUsed);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload[0]) {
      const d = payload[0].payload;
      return (
        <div
          className="p-3 rounded-lg shadow-lg text-sm"
          style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>{d.name}</p>
          <p style={{ color: 'var(--text-secondary)' }}>
            {formatMemorySize(d.value)} ({d.percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  const stackedData = data.map((item, index) => {
    const prevSum = data.slice(0, index).reduce((sum, d) => sum + d.value, 0);
    return {
      ...item,
      start: (prevSum / totalVRAM) * 100,
      width: (item.value / totalVRAM) * 100,
    };
  });

  return (
    <div
      className="rounded-xl p-6"
      style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 1px 3px var(--shadow-color)' }}
    >
      <h3 className="text-base font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Memory Allocation</h3>

      {/* Stacked Horizontal Bar */}
      <div className="mb-6">
        <div className="relative h-10 rounded-lg overflow-hidden" style={{ backgroundColor: 'var(--gauge-track)' }}>
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
                <div className="h-full flex items-center justify-center text-white text-xs font-medium font-mono drop-shadow-sm">
                  {item.percentage}%
                </div>
              )}
            </div>
          ))}
          {remainingVRAM > 0 && (
            <div
              className="absolute h-full"
              style={{
                left: `${(totalUsed / totalVRAM) * 100}%`,
                width: `${(remainingVRAM / totalVRAM) * 100}%`,
                backgroundColor: 'var(--surface-inset)',
              }}
            >
              <div className="h-full flex items-center justify-center text-xs" style={{ color: 'var(--text-muted)' }}>
                Free
              </div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 mt-3">
          {data.map((item, index) => (
            <div key={index} className="flex items-center space-x-1.5">
              <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: item.color }} />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {item.name}: <span className="font-mono">{formatMemorySize(item.value)}</span>
              </span>
            </div>
          ))}
          {remainingVRAM > 0 && (
            <div className="flex items-center space-x-1.5">
              <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: 'var(--surface-inset)', border: '1px solid var(--border)' }} />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Free: <span className="font-mono">{formatMemorySize(remainingVRAM)}</span>
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Bar Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} angle={-45} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} label={{ value: 'Memory (GB)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-muted)' } }} />
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
      <div className="grid grid-cols-3 gap-4 mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="text-center">
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Total Used</p>
          <p className="text-base font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>{formatMemorySize(totalUsed)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Available</p>
          <p className="text-base font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>{formatMemorySize(totalVRAM)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Remaining</p>
          <p className="text-base font-semibold font-mono" style={{ color: remainingVRAM < 0 ? 'var(--danger)' : 'var(--success)' }}>
            {formatMemorySize(remainingVRAM)}
          </p>
        </div>
      </div>
    </div>
  );
};

export default MemoryVisualization;
