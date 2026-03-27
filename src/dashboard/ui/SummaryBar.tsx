import type { DashboardSnapshot } from '../server/types';

interface SummaryBarProps {
  summary: DashboardSnapshot['summary'] | null;
  warnings: string[];
  theme?: 'dark' | 'light';
}

function formatVram(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(0)} GB`;
  return `${mb} MB`;
}

export function SummaryBar({ summary, warnings, theme = 'dark' }: SummaryBarProps) {
  const isDark = theme === 'dark';

  const skeletonCard = isDark
    ? 'bg-gray-800 border border-gray-700'
    : 'bg-white border border-gray-200 shadow-sm';
  const skeletonBlock = isDark ? 'bg-gray-700' : 'bg-gray-200';

  const cardClass = isDark
    ? 'bg-gray-800 border border-gray-700'
    : 'bg-white border border-gray-200 shadow-sm';
  const labelClass = isDark ? 'text-gray-400' : 'text-gray-500';
  const warningClass = isDark
    ? 'bg-yellow-900/30 border border-yellow-700 text-yellow-300'
    : 'bg-yellow-50 border border-yellow-300 text-yellow-700';

  if (!summary) {
    return (
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className={`${skeletonCard} rounded-lg p-4 text-center animate-pulse`}>
            <div className={`h-3 ${skeletonBlock} rounded w-20 mx-auto mb-2`} />
            <div className={`h-6 ${skeletonBlock} rounded w-12 mx-auto`} />
          </div>
        ))}
      </div>
    );
  }

  const stats = [
    {
      label: 'Instances',
      value: summary.instanceCount,
      color: isDark ? 'text-blue-400' : 'text-blue-600',
    },
    {
      label: 'Total VRAM',
      value: formatVram(summary.totalVramMb),
      color: isDark ? 'text-green-400' : 'text-green-600',
    },
    {
      label: 'VRAM Used',
      value: formatVram(summary.usedVramMb),
      color: isDark ? 'text-orange-400' : 'text-orange-500',
    },
    {
      label: 'Active Requests',
      value: summary.totalActiveRequests,
      color: isDark ? 'text-purple-400' : 'text-purple-600',
    },
  ];

  return (
    <div>
      <div className="grid grid-cols-4 gap-3 mb-6">
        {stats.map((stat) => (
          <div key={stat.label} className={`${cardClass} rounded-lg p-4 text-center`}>
            <div className={`${labelClass} text-xs uppercase tracking-wide mb-1`}>
              {stat.label}
            </div>
            <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
          </div>
        ))}
      </div>
      {warnings.length > 0 && (
        <div className={`${warningClass} rounded-lg p-3 mb-4 text-sm`}>
          {warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
