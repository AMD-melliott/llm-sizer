import type { VllmInstance, GpuDevice } from '../server/types';
import { InstanceCard } from './InstanceCard';

interface InstanceGridProps {
  instances: VllmInstance[];
  gpuDevices: GpuDevice[];
  apiBase: string;
}

export function InstanceGrid({ instances, gpuDevices, apiBase }: InstanceGridProps) {
  if (instances.length === 0) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-8 text-center">
        <div className="text-gray-400 text-lg mb-2">No vLLM instances found</div>
        <div className="text-gray-500 text-sm">
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
        />
      ))}
    </div>
  );
}
