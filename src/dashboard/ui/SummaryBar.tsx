import type { DashboardSnapshot } from '../server/types';

interface SummaryBarProps {
  summary: DashboardSnapshot['summary'] | null;
  warnings: string[];
}

function formatVram(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(0)} GB`;
  return `${mb} MB`;
}

export function SummaryBar({ summary, warnings }: SummaryBarProps) {
  if (!summary) {
    return (
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-center animate-pulse">
            <div className="h-3 bg-gray-700 rounded w-20 mx-auto mb-2" />
            <div className="h-6 bg-gray-700 rounded w-12 mx-auto" />
          </div>
        ))}
      </div>
    );
  }

  const stats = [
    { label: 'Instances', value: summary.instanceCount, color: 'text-blue-400' },
    { label: 'Total VRAM', value: formatVram(summary.totalVramMb), color: 'text-green-400' },
    { label: 'VRAM Used', value: formatVram(summary.usedVramMb), color: 'text-orange-400' },
    { label: 'Active Requests', value: summary.totalActiveRequests, color: 'text-purple-400' },
  ];

  return (
    <div>
      <div className="grid grid-cols-4 gap-3 mb-6">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-center"
          >
            <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">
              {stat.label}
            </div>
            <div className={`text-2xl font-bold ${stat.color}`}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>
      {warnings.length > 0 && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 mb-4 text-yellow-300 text-sm">
          {warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
