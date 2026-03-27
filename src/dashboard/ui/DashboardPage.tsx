import { SummaryBar } from './SummaryBar';
import { InstanceGrid } from './InstanceGrid';
import { useDashboardData } from './hooks/useDashboardData';

interface DashboardPageProps {
  apiBase?: string;
}

export function DashboardPage({ apiBase = 'http://localhost:3001' }: DashboardPageProps) {
  const { status, instances, gpus, loading, error } = useDashboardData(apiBase);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">vLLM Dashboard</h1>
          {status && (
            <div className="text-gray-500 text-xs">
              Last updated: {new Date(status.timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm">
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
              theme="dark"
            />
            <InstanceGrid instances={instances} gpuDevices={gpus?.devices ?? []} apiBase={apiBase} theme="dark" />
          </>
        )}
      </div>
    </div>
  );
}
