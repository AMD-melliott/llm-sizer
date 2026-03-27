import { useDashboardData } from '../../dashboard/ui/hooks/useDashboardData';
import { SummaryBar } from '../../dashboard/ui/SummaryBar';
import { InstanceGrid } from '../../dashboard/ui/InstanceGrid';

interface DashboardTabProps {
  apiBase: string;
}

export function DashboardTab({ apiBase }: DashboardTabProps) {
  const { status, instances, gpus, loading, error } = useDashboardData(apiBase);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Server Dashboard</h2>
          <p className="text-sm text-gray-500">Live vLLM instances and GPU metrics</p>
        </div>
        {status && (
          <span className="text-xs text-gray-400">
            Updated {new Date(status.timestamp).toLocaleTimeString()}
          </span>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      {loading && !error && (
        <div className="text-gray-400 text-center py-12">
          Connecting to dashboard backend...
        </div>
      )}

      {!loading && (
        <>
          <SummaryBar
            summary={status?.summary ?? null}
            warnings={status?.warnings ?? []}
            theme="light"
          />
          <InstanceGrid
            instances={instances}
            gpuDevices={gpus?.devices ?? []}
            apiBase={apiBase}
            theme="light"
          />
        </>
      )}
    </div>
  );
}
