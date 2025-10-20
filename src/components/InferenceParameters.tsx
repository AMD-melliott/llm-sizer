import React from 'react';
import { Users, FileText, Layers, HardDrive } from 'lucide-react';
import useAppStore from '../store/useAppStore';

const InferenceParameters: React.FC = () => {
  const {
    batchSize,
    setBatchSize,
    sequenceLength,
    setSequenceLength,
    concurrentUsers,
    setConcurrentUsers,
    enableOffloading,
    setEnableOffloading,
  } = useAppStore();

  // Log scale values for sliders
  const batchSizeSteps = [1, 2, 4, 8, 16, 32, 64, 128];
  const sequenceLengthSteps = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
  const concurrentUserSteps = [1, 2, 4, 8, 16, 32, 64, 128];

  const getBatchSizeIndex = (value: number) => {
    const index = batchSizeSteps.findIndex(v => v >= value);
    return index === -1 ? batchSizeSteps.length - 1 : index;
  };

  const getSequenceLengthIndex = (value: number) => {
    const index = sequenceLengthSteps.findIndex(v => v >= value);
    return index === -1 ? sequenceLengthSteps.length - 1 : index;
  };

  const getConcurrentUsersIndex = (value: number) => {
    const index = concurrentUserSteps.findIndex(v => v >= value);
    return index === -1 ? concurrentUserSteps.length - 1 : index;
  };

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-gray-600" />
            <label className="text-sm font-medium text-gray-700">
              Batch Size
            </label>
          </div>
          <span className="text-lg font-semibold text-blue-600">{batchSize}</span>
        </div>
        <input
          type="range"
          min="0"
          max={batchSizeSteps.length - 1}
          value={getBatchSizeIndex(batchSize)}
          onChange={(e) => setBatchSize(batchSizeSteps[parseInt(e.target.value)])}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>8</span>
          <span>32</span>
          <span>128</span>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <FileText className="w-4 h-4 text-gray-600" />
            <label className="text-sm font-medium text-gray-700">
              Sequence Length (tokens)
            </label>
          </div>
          <span className="text-lg font-semibold text-blue-600">{sequenceLength.toLocaleString()}</span>
        </div>
        <input
          type="range"
          min="0"
          max={sequenceLengthSteps.length - 1}
          value={getSequenceLengthIndex(sequenceLength)}
          onChange={(e) => setSequenceLength(sequenceLengthSteps[parseInt(e.target.value)])}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>512</span>
          <span>4K</span>
          <span>32K</span>
          <span>128K</span>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4 text-gray-600" />
            <label className="text-sm font-medium text-gray-700">
              Concurrent Users
            </label>
          </div>
          <span className="text-lg font-semibold text-blue-600">{concurrentUsers}</span>
        </div>
        <input
          type="range"
          min="0"
          max={concurrentUserSteps.length - 1}
          value={getConcurrentUsersIndex(concurrentUsers)}
          onChange={(e) => setConcurrentUsers(concurrentUserSteps[parseInt(e.target.value)])}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>8</span>
          <span>32</span>
          <span>128</span>
        </div>
      </div>

      <div className="border-t pt-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <HardDrive className="w-4 h-4 text-gray-600" />
            <div>
              <label htmlFor="offloading" className="text-sm font-medium text-gray-700">
                Enable Offloading
              </label>
              <p className="text-xs text-gray-500">
                Offload to CPU/RAM/NVMe when VRAM is exceeded
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              id="offloading"
              type="checkbox"
              checked={enableOffloading}
              onChange={(e) => setEnableOffloading(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
          </label>
        </div>
        {enableOffloading && (
          <div className="mt-2 p-2 bg-amber-50 rounded text-xs text-amber-700">
            <strong>Note:</strong> Offloading will significantly reduce inference speed but allows running models larger than VRAM capacity.
          </div>
        )}
      </div>

      <div className="bg-gray-50 rounded-lg p-3">
        <h4 className="text-xs font-medium text-gray-700 mb-2">Parameter Impact:</h4>
        <ul className="text-xs text-gray-600 space-y-1">
          <li>• <strong>Batch Size:</strong> Processes multiple requests simultaneously</li>
          <li>• <strong>Sequence Length:</strong> Maximum context window for generation</li>
          <li>• <strong>Concurrent Users:</strong> Number of parallel user sessions</li>
        </ul>
      </div>
    </div>
  );
};

export default InferenceParameters;