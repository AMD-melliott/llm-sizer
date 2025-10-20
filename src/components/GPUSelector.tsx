import React from 'react';
import { Cpu, Zap } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import { GPU } from '../types';

interface GPUSelectorProps {
  gpus: GPU[];
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

  const currentGPU = gpus.find(g => g.id === selectedGPU);
  const isCustom = selectedGPU === 'custom';

  const totalVRAM = currentGPU
    ? (isCustom && customVRAM ? customVRAM : currentGPU.vram_gb) * numGPUs
    : 0;

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="gpu-select" className="block text-sm font-medium text-gray-700 mb-2">
          GPU Selection
        </label>
        <select
          id="gpu-select"
          value={selectedGPU}
          onChange={(e) => setSelectedGPU(e.target.value)}
          className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
        >
          <optgroup label="AMD Enterprise">
            {gpus.filter(g => g.vendor === 'AMD' && g.category === 'enterprise').map(gpu => (
              <option key={gpu.id} value={gpu.id}>
                {gpu.name} ({gpu.vram_gb}GB)
              </option>
            ))}
          </optgroup>
          <optgroup label="NVIDIA Enterprise">
            {gpus.filter(g => g.vendor === 'NVIDIA' && g.category === 'enterprise').map(gpu => (
              <option key={gpu.id} value={gpu.id}>
                {gpu.name} ({gpu.vram_gb}GB)
              </option>
            ))}
          </optgroup>
          <optgroup label="NVIDIA Consumer">
            {gpus.filter(g => g.vendor === 'NVIDIA' && g.category === 'consumer').map(gpu => (
              <option key={gpu.id} value={gpu.id}>
                {gpu.name} ({gpu.vram_gb}GB)
              </option>
            ))}
          </optgroup>
          <optgroup label="Custom">
            <option value="custom">Custom GPU</option>
          </optgroup>
        </select>
      </div>

      {currentGPU && (
        <div className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-700">GPU Specifications</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">VRAM:</span>
              <span className="ml-2 font-medium">
                {isCustom && customVRAM ? customVRAM : currentGPU.vram_gb} GB
              </span>
            </div>
            <div>
              <span className="text-gray-500">Memory BW:</span>
              <span className="ml-2 font-medium">{currentGPU.memory_bandwidth_gbps.toLocaleString()} GB/s</span>
            </div>
            <div>
              <span className="text-gray-500">FP16 TFLOPS:</span>
              <span className="ml-2 font-medium">{currentGPU.compute_tflops_fp16.toLocaleString()}</span>
            </div>
            {currentGPU.compute_tflops_fp8 && (
              <div>
                <span className="text-gray-500">FP8 TFLOPS:</span>
                <span className="ml-2 font-medium">{currentGPU.compute_tflops_fp8.toLocaleString()}</span>
              </div>
            )}
          </div>
        </div>
      )}

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
      </div>

      {totalVRAM > 0 && (
        <div className="bg-blue-50 rounded-lg p-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-900">Total Available VRAM</span>
          </div>
          <span className="text-lg font-bold text-blue-900">{totalVRAM.toLocaleString()} GB</span>
        </div>
      )}
    </div>
  );
};

export default GPUSelector;