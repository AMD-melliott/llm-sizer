import React, { useState, useMemo } from 'react';
import { Cpu, Zap, Server, Briefcase, Home, Settings } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import { GPU } from '../types';

interface GPUSelectorProps {
  gpus: GPU[];
}

type GPUTier = 'datacenter' | 'professional' | 'consumer' | 'custom';

interface PresetConfig {
  id: string;
  label: string;
  gpuId: string;
  count: number;
  totalVRAM: number;
}

const GPUSelector: React.FC<GPUSelectorProps> = ({ gpus }) => {
  const {
    selectedGPU,
    setSelectedGPU,
    customVRAM,
    setCustomVRAM,
    numGPUs,
    setNumGPUs,
  } = useAppStore();

  const [activeTier, setActiveTier] = useState<GPUTier>('datacenter');

  const currentGPU = gpus.find(g => g.id === selectedGPU);
  const isCustom = selectedGPU === 'custom';

  const totalVRAM = currentGPU
    ? (isCustom && customVRAM ? customVRAM : currentGPU.vram_gb) * numGPUs
    : 0;

  // Group GPUs by tier
  const gpusByTier = useMemo(() => {
    return {
      datacenter: gpus.filter(g => g.tier === 'datacenter'),
      professional: gpus.filter(g => g.tier === 'professional'),
      consumer: gpus.filter(g => g.tier === 'consumer'),
      custom: gpus.filter(g => g.tier === 'custom'),
    };
  }, [gpus]);

  // Multi-GPU presets for datacenter GPUs
  const datacenterPresets: PresetConfig[] = useMemo(() => [
    { id: 'preset-mi355x-4', label: '4x MI355X', gpuId: 'mi355x', count: 4, totalVRAM: 1152 },
    { id: 'preset-mi355x-8', label: '8x MI355X', gpuId: 'mi355x', count: 8, totalVRAM: 2304 },
    { id: 'preset-mi325x-4', label: '4x MI325X', gpuId: 'mi325x', count: 4, totalVRAM: 1024 },
    { id: 'preset-mi325x-8', label: '8x MI325X', gpuId: 'mi325x', count: 8, totalVRAM: 2048 },
    { id: 'preset-mi300x-4', label: '4x MI300X', gpuId: 'mi300x', count: 4, totalVRAM: 768 },
    { id: 'preset-mi300x-8', label: '8x MI300X', gpuId: 'mi300x', count: 8, totalVRAM: 1536 },
  ], []);

  const handlePresetClick = (preset: PresetConfig) => {
    setSelectedGPU(preset.gpuId);
    setNumGPUs(preset.count);
  };

  const handleTierChange = (tier: GPUTier) => {
    setActiveTier(tier);
    const currentTierGPUs = gpusByTier[tier];
    if (currentTierGPUs.length > 0 && !currentTierGPUs.find(g => g.id === selectedGPU)) {
      setSelectedGPU(currentTierGPUs[0].id);
    }
  };

  const getTierIcon = (tier: GPUTier) => {
    switch (tier) {
      case 'datacenter': return <Server className="w-4 h-4" />;
      case 'professional': return <Briefcase className="w-4 h-4" />;
      case 'consumer': return <Home className="w-4 h-4" />;
      case 'custom': return <Settings className="w-4 h-4" />;
    }
  };

  const getTierBadge = (tier: GPUTier) => {
    const badges: Record<GPUTier, { label: string; icon: typeof Server }> = {
      datacenter: { label: 'Data Center', icon: Server },
      professional: { label: 'Professional', icon: Briefcase },
      consumer: { label: 'Consumer', icon: Home },
      custom: { label: 'Custom', icon: Settings },
    };
    return badges[tier];
  };

  const currentTierBadge = currentGPU ? getTierBadge(currentGPU.tier) : null;

  return (
    <div className="space-y-4">
      {/* Tier Tabs */}
      <div className="space-y-1">
        {(['datacenter', 'professional', 'consumer', 'custom'] as GPUTier[]).map((tier) => (
          <button
            key={tier}
            onClick={() => handleTierChange(tier)}
            className="w-full flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all"
            style={{
              backgroundColor: activeTier === tier ? 'var(--accent)' : 'var(--surface-inset)',
              color: activeTier === tier ? 'var(--text-inverse)' : 'var(--text-secondary)',
              boxShadow: activeTier === tier ? '0 1px 3px var(--shadow-color)' : 'none',
            }}
          >
            {getTierIcon(tier)}
            <span className="capitalize">{tier}</span>
          </button>
        ))}
      </div>

      {/* Multi-GPU Presets for Datacenter */}
      {activeTier === 'datacenter' && (
        <div className="rounded-lg p-3" style={{ backgroundColor: 'var(--accent-light)' }}>
          <h4 className="text-xs font-semibold mb-2" style={{ color: 'var(--accent-text)' }}>Quick Presets:</h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {datacenterPresets.map((preset) => {
              const isActive = selectedGPU === preset.gpuId && numGPUs === preset.count;
              return (
                <button
                  key={preset.id}
                  onClick={() => handlePresetClick(preset)}
                  className="px-3 py-2 text-xs font-medium rounded-md transition-all"
                  style={{
                    backgroundColor: isActive ? 'var(--accent)' : 'var(--surface)',
                    color: isActive ? 'var(--text-inverse)' : 'var(--accent)',
                    border: isActive ? 'none' : '1px solid var(--border)',
                    boxShadow: isActive ? '0 2px 4px var(--shadow-color)' : 'none',
                  }}
                >
                  <div>{preset.label}</div>
                  <div className="text-[10px] opacity-75">{preset.totalVRAM} GB</div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* GPU Selection Dropdown */}
      <div>
        <label htmlFor="gpu-select" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Select GPU
        </label>
        <select
          id="gpu-select"
          value={selectedGPU}
          onChange={(e) => setSelectedGPU(e.target.value)}
          className="w-full rounded-lg shadow-sm focus:outline-none focus:ring-2"
          style={{ borderColor: 'var(--border-strong)', color: 'var(--text-primary)', backgroundColor: 'var(--surface)' }}
        >
          {gpusByTier[activeTier].map(gpu => (
            <option key={gpu.id} value={gpu.id}>
              {gpu.name} ({gpu.vram_gb}GB {gpu.memory_type})
            </option>
          ))}
        </select>
      </div>

      {/* GPU Specifications Card */}
      {currentGPU && (
        <div className="rounded-lg p-3 space-y-2" style={{ backgroundColor: 'var(--surface-inset)' }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4" style={{ color: 'var(--accent)' }} />
              <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>GPU Specifications</span>
            </div>
            {currentTierBadge && (
              <span className="text-xs px-2 py-0.5 rounded-full flex items-center gap-1" style={{ backgroundColor: 'var(--accent-light)', color: 'var(--accent-text)' }}>
                {currentTierBadge.label}
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span style={{ color: 'var(--text-muted)' }}>VRAM:</span>
              <span className="ml-2 font-medium font-mono" style={{ color: 'var(--text-primary)' }}>
                {isCustom && customVRAM ? customVRAM : currentGPU.vram_gb} GB
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Memory:</span>
              <span className="ml-2 font-medium" style={{ color: 'var(--text-primary)' }}>{currentGPU.memory_type}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Bandwidth:</span>
              <span className="ml-2 font-medium font-mono" style={{ color: 'var(--text-primary)' }}>{currentGPU.memory_bandwidth_gbps.toLocaleString()} GB/s</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>PCIe:</span>
              <span className="ml-2 font-medium" style={{ color: 'var(--text-primary)' }}>Gen {currentGPU.pcie_gen}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>FP16:</span>
              <span className="ml-2 font-medium font-mono" style={{ color: 'var(--text-primary)' }}>{currentGPU.compute_tflops_fp16.toLocaleString()} TFLOPS</span>
            </div>
            {currentGPU.compute_tflops_fp8 && (
              <div>
                <span style={{ color: 'var(--text-muted)' }}>FP8:</span>
                <span className="ml-2 font-medium font-mono" style={{ color: 'var(--text-primary)' }}>{currentGPU.compute_tflops_fp8.toLocaleString()} TFLOPS</span>
              </div>
            )}
            <div>
              <span style={{ color: 'var(--text-muted)' }}>TDP:</span>
              <span className="ml-2 font-medium font-mono" style={{ color: 'var(--text-primary)' }}>{currentGPU.tdp_watts}W</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Year:</span>
              <span className="ml-2 font-medium" style={{ color: 'var(--text-primary)' }}>{currentGPU.release_year}</span>
            </div>
            {currentGPU.nvlink_bandwidth_gbps && (
              <div className="col-span-2">
                <span style={{ color: 'var(--text-muted)' }}>Interconnect:</span>
                <span className="ml-2 font-medium font-mono" style={{ color: 'var(--text-primary)' }}>{currentGPU.nvlink_bandwidth_gbps} GB/s</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Custom VRAM Input */}
      {isCustom && (
        <div className="pt-3" style={{ borderTop: '1px solid var(--border)' }}>
          <label htmlFor="custom-vram" className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
            Custom VRAM (GB)
          </label>
          <input
            id="custom-vram"
            type="number"
            min="1"
            max="1000"
            value={customVRAM || ''}
            onChange={(e) => setCustomVRAM(e.target.value ? parseFloat(e.target.value) : undefined)}
            placeholder="e.g., 24"
            className="w-full rounded-lg shadow-sm focus:outline-none focus:ring-2"
            style={{ borderColor: 'var(--border-strong)', color: 'var(--text-primary)', backgroundColor: 'var(--surface)' }}
          />
        </div>
      )}

      {/* Number of GPUs */}
      <div>
        <label htmlFor="num-gpus" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Number of GPUs
        </label>
        <div className="flex items-center space-x-4">
          <input
            id="num-gpus"
            type="range"
            min="1"
            max="32"
            value={numGPUs}
            onChange={(e) => setNumGPUs(parseInt(e.target.value))}
            className="flex-1"
          />
          <div className="w-20 text-center">
            <span className="text-lg font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>{numGPUs}</span>
            <span className="text-sm block" style={{ color: 'var(--text-muted)' }}>GPU{numGPUs > 1 ? 's' : ''}</span>
          </div>
        </div>
        {/* Quick GPU count buttons */}
        {currentGPU?.multi_gpu_capable && (
          <div className="flex space-x-2 mt-2">
            {[1, 2, 4, 8, 16].map(count => (
              <button
                key={count}
                onClick={() => setNumGPUs(count)}
                className="px-3 py-1 text-xs rounded-md font-medium transition-all"
                style={{
                  backgroundColor: numGPUs === count ? 'var(--accent)' : 'var(--surface-inset)',
                  color: numGPUs === count ? 'var(--text-inverse)' : 'var(--text-secondary)',
                }}
              >
                {count}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Total VRAM Summary */}
      {totalVRAM > 0 && (
        <div className="rounded-lg p-3 flex items-center justify-between" style={{ backgroundColor: 'var(--accent-light)' }}>
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4" style={{ color: 'var(--accent)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--accent-text)' }}>Total Available VRAM</span>
          </div>
          <span className="text-lg font-bold font-mono" style={{ color: 'var(--accent-text)' }}>{totalVRAM.toLocaleString()} GB</span>
        </div>
      )}

      {/* Multi-GPU Info */}
      {numGPUs > 1 && currentGPU?.multi_gpu_capable && (
        <div className="rounded-lg p-3 text-xs" style={{ backgroundColor: 'var(--warning-light)', color: 'var(--warning-text)' }}>
          <strong>Multi-GPU Configuration:</strong> Using tensor parallelism across {numGPUs} GPUs.
          {currentGPU.nvlink_bandwidth_gbps && (
            <span> Interconnect bandwidth: {(currentGPU.nvlink_bandwidth_gbps * (numGPUs - 1)).toLocaleString()} GB/s total.</span>
          )}
        </div>
      )}

      {numGPUs > 1 && !currentGPU?.multi_gpu_capable && (
        <div className="rounded-lg p-3 text-xs" style={{ backgroundColor: 'var(--danger-light)', color: 'var(--danger-text)' }}>
          <strong>Warning:</strong> This GPU is not optimized for multi-GPU configurations.
        </div>
      )}
    </div>
  );
};

export default GPUSelector;
