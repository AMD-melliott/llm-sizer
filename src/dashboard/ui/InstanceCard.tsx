import { useState } from 'react';
import type { VllmInstance, GpuDevice } from '../server/types';
import { LogViewer } from './LogViewer';

interface InstanceCardProps {
  instance: VllmInstance;
  gpuDevices: GpuDevice[];
  apiBase: string;
  theme?: 'dark' | 'light';
}

const STATUS_COLORS_DARK: Record<VllmInstance['status'], { bg: string; text: string }> = {
  running: { bg: 'bg-green-900/50', text: 'text-green-400' },
  starting: { bg: 'bg-yellow-900/50', text: 'text-yellow-400' },
  stopped: { bg: 'bg-gray-700/50', text: 'text-gray-400' },
  error: { bg: 'bg-red-900/50', text: 'text-red-400' },
};

const STATUS_COLORS_LIGHT: Record<VllmInstance['status'], { bg: string; text: string }> = {
  running: { bg: 'bg-green-50', text: 'text-green-700' },
  starting: { bg: 'bg-yellow-50', text: 'text-yellow-700' },
  stopped: { bg: 'bg-gray-100', text: 'text-gray-500' },
  error: { bg: 'bg-red-50', text: 'text-red-700' },
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

export function InstanceCard({ instance, gpuDevices, apiBase, theme = 'dark' }: InstanceCardProps) {
  const [showLaunchArgs, setShowLaunchArgs] = useState(false);
  const [showLogs, setShowLogs] = useState(false);

  const isDark = theme === 'dark';

  const vramPercent =
    instance.vramTotalMb > 0
      ? (instance.vramUsedMb / instance.vramTotalMb) * 100
      : 0;

  const statusStyle = (isDark ? STATUS_COLORS_DARK : STATUS_COLORS_LIGHT)[instance.status];
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

  const cardClass = isDark
    ? 'bg-gray-800 border-gray-700'
    : 'bg-white border-gray-200 shadow-sm';
  const nameClass = isDark ? 'text-white' : 'text-gray-900';
  const metaClass = 'text-gray-500';
  const vramBarBgClass = isDark ? 'bg-gray-900' : 'bg-gray-200';
  const vramTextClass = isDark ? 'text-gray-400' : 'text-gray-500';
  const kvValueClass = isDark ? 'text-white' : 'text-gray-900';
  const buttonClass = isDark ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700';
  const codeBlockClass = isDark
    ? 'bg-gray-900 text-gray-300'
    : 'bg-gray-50 text-gray-700';
  const codeLabelClass = isDark ? 'text-gray-500' : 'text-gray-400';

  return (
    <div className={`${cardClass} border rounded-lg p-4`}>
      <div className="flex justify-between items-center mb-2">
        <div>
          <span className={`${nameClass} font-semibold text-sm`}>{instance.modelName}</span>
          <span className={`${metaClass} text-xs ml-2`}>{instance.containerName}</span>
        </div>
        <span className={`${statusStyle.bg} ${statusStyle.text} px-2 py-0.5 rounded text-xs font-medium uppercase`}>
          {instance.status}
        </span>
      </div>

      <div className={`${metaClass} text-xs mb-3`}>
        {infoItems.join(' · ')}
      </div>

      {instance.vramTotalMb > 0 && (
        <div className="mb-3">
          <div className={`${vramBarBgClass} rounded-full h-2 mb-1`}>
            <div
              className={`h-full rounded-full transition-all ${vramBarColor(vramPercent)}`}
              style={{ width: `${Math.min(vramPercent, 100)}%` }}
            />
          </div>
          <div className={`flex justify-between ${vramTextClass} text-xs`}>
            <span>VRAM: {formatVram(instance.vramUsedMb)} / {formatVram(instance.vramTotalMb)}</span>
            <span>{vramPercent.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {instance.kvCachePercent !== undefined && (
        <div className={`flex gap-4 text-xs ${vramTextClass} mb-3`}>
          <span>KV Cache: <span className={kvValueClass}>{instance.kvCachePercent.toFixed(0)}%</span></span>
          <span>Running: <span className={kvValueClass}>{instance.runningRequests ?? 0}</span></span>
          <span>Waiting: <span className={kvValueClass}>{instance.waitingRequests ?? 0}</span></span>
        </div>
      )}

      <div className="flex gap-2 text-xs">
        <button
          onClick={() => setShowLaunchArgs(!showLaunchArgs)}
          className={buttonClass}
        >
          {showLaunchArgs ? 'Hide' : 'Show'} launch args
        </button>
        <button
          onClick={() => setShowLogs(!showLogs)}
          className={buttonClass}
        >
          {showLogs ? 'Hide' : 'Show'} logs
        </button>
      </div>

      {showLaunchArgs && (
        <div className={`mt-3 ${codeBlockClass} rounded p-3 text-xs font-mono overflow-x-auto`}>
          <div className={`${codeLabelClass} mb-1`}>Command:</div>
          <div>{instance.launchArgs.join(' ')}</div>
          {Object.keys(instance.envVars).length > 0 && (
            <>
              <div className={`${codeLabelClass} mt-2 mb-1`}>Environment:</div>
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
