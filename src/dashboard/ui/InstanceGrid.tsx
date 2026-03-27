import type { VllmInstance, GpuDevice } from '../server/types';
import { InstanceCard } from './InstanceCard';

interface InstanceGridProps {
  instances: VllmInstance[];
  gpuDevices: GpuDevice[];
  apiBase: string;
  theme?: 'dark' | 'light';
}

export function InstanceGrid({ instances, gpuDevices, apiBase, theme = 'dark' }: InstanceGridProps) {
  const isDark = theme === 'dark';

  if (instances.length === 0) {
    const emptyClass = isDark
      ? 'bg-gray-800 border-gray-700 text-gray-400'
      : 'bg-gray-50 border-gray-200 text-gray-600';
    const emptySubClass = isDark ? 'text-gray-500' : 'text-gray-500';

    return (
      <div className={`${emptyClass} border rounded-lg p-8 text-center`}>
        <div className="text-lg mb-2">No vLLM instances found</div>
        <div className={`${emptySubClass} text-sm`}>
          Make sure vLLM containers are running and use a recognized image
          (vllm/*, rocm/vllm*).
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {instances.map((instance) => (
        <InstanceCard
          key={instance.containerId}
          instance={instance}
          gpuDevices={gpuDevices}
          apiBase={apiBase}
          theme={theme}
        />
      ))}
    </div>
  );
}
