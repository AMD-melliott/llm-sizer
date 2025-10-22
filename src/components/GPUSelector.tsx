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

  // Multi-GPU presets for datacenter GPUs (emphasizing AMD as per PRD)
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
    // Auto-select first GPU in the tier if current selection is not in this tier
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
    const badges = {
      datacenter: { label: '🏢 Data Center', color: 'bg-blue-100 text-blue-800' },
      professional: { label: '💼 Professional', color: 'bg-green-100 text-green-800' },
      consumer: { label: '🏠 Consumer', color: 'bg-gray-100 text-gray-800' },
      custom: { label: '⚙️ Custom', color: 'bg-orange-100 text-orange-800' },
    };
    return badges[tier];
  };

  const currentTierBadge = currentGPU ? getTierBadge(currentGPU.tier) : null;

  return (
    <div className="space-y-4">
      {/* Tier Tabs - Vertical Stack */}
      <div className="space-y-1">
        {(['datacenter', 'professional', 'consumer', 'custom'] as GPUTier[]).map((tier) => (
          <button
            key={tier}
            onClick={() => handleTierChange(tier)}
            className={`w-full flex items-center space-x-2 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
              activeTier === tier
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {getTierIcon(tier)}
            <span className="capitalize">{tier}</span>
          </button>
        ))}
      </div>

      {/* Multi-GPU Presets for Datacenter (shown only on datacenter tab) */}
      {activeTier === 'datacenter' && (
        <div className="bg-blue-50 rounded-lg p-3">
          <h4 className="text-xs font-semibold text-blue-900 mb-2">Quick Presets:</h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {datacenterPresets.map((preset) => (
              <button
                key={preset.id}
                onClick={() => handlePresetClick(preset)}
                className={`px-3 py-2 text-xs font-medium rounded-md transition-all ${
                  selectedGPU === preset.gpuId && numGPUs === preset.count
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-white text-blue-700 hover:bg-blue-100 border border-blue-200'
                }`}
              >
                <div>{preset.label}</div>
                <div className="text-[10px] opacity-75">{preset.totalVRAM} GB</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* GPU Selection Dropdown */}
      <div>
        <label htmlFor="gpu-select" className="block text-sm font-medium text-gray-700 mb-2">
          Select GPU
        </label>
        <select
          id="gpu-select"
          value={selectedGPU}
          onChange={(e) => setSelectedGPU(e.target.value)}
          className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
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
        <div className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-blue-500" />
              <span className="text-sm font-medium text-gray-700">GPU Specifications</span>
            </div>
            {currentTierBadge && (
              <span className={`text-xs px-2 py-1 rounded-full ${currentTierBadge.color}`}>
                {currentTierBadge.label}
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">VRAM:</span>
              <span className="ml-2 font-medium">
                {isCustom && customVRAM ? customVRAM : currentGPU.vram_gb} GB
              </span>
            </div>
            <div>
              <span className="text-gray-500">Memory:</span>
              <span className="ml-2 font-medium">{currentGPU.memory_type}</span>
            </div>
            <div>
              <span className="text-gray-500">Bandwidth:</span>
              <span className="ml-2 font-medium">{currentGPU.memory_bandwidth_gbps.toLocaleString()} GB/s</span>
            </div>
            <div>
              <span className="text-gray-500">PCIe:</span>
              <span className="ml-2 font-medium">Gen {currentGPU.pcie_gen}</span>
            </div>
            <div>
              <span className="text-gray-500">FP16:</span>
              <span className="ml-2 font-medium">{currentGPU.compute_tflops_fp16.toLocaleString()} TFLOPS</span>
            </div>
            {currentGPU.compute_tflops_fp8 && (
              <div>
                <span className="text-gray-500">FP8:</span>
                <span className="ml-2 font-medium">{currentGPU.compute_tflops_fp8.toLocaleString()} TFLOPS</span>
              </div>
            )}
            <div>
              <span className="text-gray-500">TDP:</span>
              <span className="ml-2 font-medium">{currentGPU.tdp_watts}W</span>
            </div>
            <div>
              <span className="text-gray-500">Year:</span>
              <span className="ml-2 font-medium">{currentGPU.release_year}</span>
            </div>
            {currentGPU.nvlink_bandwidth_gbps && (
              <div className="col-span-2">
                <span className="text-gray-500">Interconnect:</span>
                <span className="ml-2 font-medium">{currentGPU.nvlink_bandwidth_gbps} GB/s</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Custom VRAM Input */}
      {isCustom && (
        <div className="border-t pt-3">
          <label htmlFor="custom-vram" className="block text-sm font-medium text-gray-700 mb-1">
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
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </div>
      )}

      {/* Number of GPUs */}
      <div>
        <label htmlFor="num-gpus" className="block text-sm font-medium text-gray-700 mb-2">
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
            <span className="text-lg font-semibold">{numGPUs}</span>
            <span className="text-sm text-gray-500 block">GPU{numGPUs > 1 ? 's' : ''}</span>
          </div>
        </div>
        {/* Quick GPU count buttons for common configurations */}
        {currentGPU?.multi_gpu_capable && (
          <div className="flex space-x-2 mt-2">
            {[1, 2, 4, 8, 16].map(count => (
              <button
                key={count}
                onClick={() => setNumGPUs(count)}
                className={`px-3 py-1 text-xs rounded ${
                  numGPUs === count
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {count}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Total VRAM Summary */}
      {totalVRAM > 0 && (
        <div className="bg-blue-50 rounded-lg p-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-900">Total Available VRAM</span>
          </div>
          <span className="text-lg font-bold text-blue-900">{totalVRAM.toLocaleString()} GB</span>
        </div>
      )}

      {/* Multi-GPU Info */}
      {numGPUs > 1 && currentGPU?.multi_gpu_capable && (
        <div className="bg-amber-50 rounded-lg p-3 text-xs text-amber-800">
          <strong>Multi-GPU Configuration:</strong> Using tensor parallelism across {numGPUs} GPUs.
          {currentGPU.nvlink_bandwidth_gbps && (
            <span> Interconnect bandwidth: {(currentGPU.nvlink_bandwidth_gbps * (numGPUs - 1)).toLocaleString()} GB/s total.</span>
          )}
        </div>
      )}

      {numGPUs > 1 && !currentGPU?.multi_gpu_capable && (
        <div className="bg-red-50 rounded-lg p-3 text-xs text-red-800">
          <strong>Warning:</strong> This GPU is not optimized for multi-GPU configurations.
        </div>
      )}
    </div>
  );
};

export default GPUSelector;