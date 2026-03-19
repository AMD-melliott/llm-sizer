import { useState } from 'react';
import type { VllmInstance, GpuDevice } from '../server/types';
import { LogViewer } from './LogViewer';

interface InstanceCardProps {
  instance: VllmInstance;
  gpuDevices: GpuDevice[];
  apiBase: string;
}

const STATUS_COLORS: Record<VllmInstance['status'], { bg: string; text: string }> = {
  running: { bg: 'bg-green-900/50', text: 'text-green-400' },
  starting: { bg: 'bg-yellow-900/50', text: 'text-yellow-400' },
  stopped: { bg: 'bg-gray-700/50', text: 'text-gray-400' },
  error: { bg: 'bg-red-900/50', text: 'text-red-400' },
};

function vramBarColor(percent: number): string {
  if (percent >= 90) return 'bg-red-500';
  if (percent >= 70) return 'bg-orange-500';
  return 'bg-blue-500';
}

function formatVram(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(0)} GB`;
  return `${mb} MB`;
}

export function InstanceCard({ instance, gpuDevices, apiBase }: InstanceCardProps) {
  const [showLaunchArgs, setShowLaunchArgs] = useState(false);
  const [showLogs, setShowLogs] = useState(false);

  const vramPercent =
    instance.vramTotalMb > 0
      ? (instance.vramUsedMb / instance.vramTotalMb) * 100
      : 0;

  const statusStyle = STATUS_COLORS[instance.status];
  const gpuLabel = instance.gpuIds.length > 0
    ? instance.gpuIds.map((id) => {
        const device = gpuDevices.find((d) => d.id === `gpu-${id}` || String(d.physicalId) === id);
        return device ? `GPU ${id} (${device.name})` : `GPU ${id}`;
      }).join(', ')
    : 'No GPU assigned';

  const infoItems = [
    gpuLabel,
    instance.partitionInfo,
    instance.quantization ? instance.quantization.toUpperCase() : null,
    instance.tensorParallelSize > 1 ? `TP=${instance.tensorParallelSize}` : null,
  ].filter(Boolean);

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      <div className="flex justify-between items-center mb-2">
        <div>
          <span className="text-white font-semibold text-sm">{instance.modelName}</span>
          <span className="text-gray-500 text-xs ml-2">{instance.containerName}</span>
        </div>
        <span className={`${statusStyle.bg} ${statusStyle.text} px-2 py-0.5 rounded text-xs font-medium uppercase`}>
          {instance.status}
        </span>
      </div>

      <div className="text-gray-400 text-xs mb-3">
        {infoItems.join(' · ')}
      </div>

      {instance.vramTotalMb > 0 && (
        <div className="mb-3">
          <div className="bg-gray-900 rounded-full h-2 mb-1">
            <div
              className={`h-full rounded-full transition-all ${vramBarColor(vramPercent)}`}
              style={{ width: `${Math.min(vramPercent, 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-gray-400 text-xs">
            <span>VRAM: {formatVram(instance.vramUsedMb)} / {formatVram(instance.vramTotalMb)}</span>
            <span>{vramPercent.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {instance.kvCachePercent !== undefined && (
        <div className="flex gap-4 text-xs text-gray-400 mb-3">
          <span>KV Cache: <span className="text-white">{instance.kvCachePercent.toFixed(0)}%</span></span>
          <span>Running: <span className="text-white">{instance.runningRequests ?? 0}</span></span>
          <span>Waiting: <span className="text-white">{instance.waitingRequests ?? 0}</span></span>
        </div>
      )}

      <div className="flex gap-2 text-xs">
        <button
          onClick={() => setShowLaunchArgs(!showLaunchArgs)}
          className="text-blue-400 hover:text-blue-300"
        >
          {showLaunchArgs ? 'Hide' : 'Show'} launch args
        </button>
        <button
          onClick={() => setShowLogs(!showLogs)}
          className="text-blue-400 hover:text-blue-300"
        >
          {showLogs ? 'Hide' : 'Show'} logs
        </button>
      </div>

      {showLaunchArgs && (
        <div className="mt-3 bg-gray-900 rounded p-3 text-xs font-mono text-gray-300 overflow-x-auto">
          <div className="text-gray-500 mb-1">Command:</div>
          <div>{instance.launchArgs.join(' ')}</div>
          {Object.keys(instance.envVars).length > 0 && (
            <>
              <div className="text-gray-500 mt-2 mb-1">Environment:</div>
              {Object.entries(instance.envVars).map(([k, v]) => (
                <div key={k}>{k}={v}</div>
              ))}
            </>
          )}
        </div>
      )}

      {showLogs && (
        <div className="mt-3">
          <LogViewer
            containerId={instance.containerId}
            apiBase={apiBase}
          />
        </div>
      )}
    </div>
  );
}
