import React from 'react';
import { HelpCircle } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import { InferenceQuantization, KVCacheQuantization } from '../types';
import { getQuantizationInfo } from '../utils/memoryCalculator';

const QuantizationOptions: React.FC = () => {
  const {
    inferenceQuantization,
    setInferenceQuantization,
    kvCacheQuantization,
    setKVCacheQuantization,
  } = useAppStore();

  const inferenceOptions: { value: InferenceQuantization; label: string; description: string }[] = [
    { value: 'fp16', label: 'FP16', description: 'Full 16-bit precision' },
    { value: 'fp8', label: 'FP8', description: '8-bit floating point' },
    { value: 'int8', label: 'INT8', description: '8-bit integer' },
    { value: 'int4', label: 'INT4', description: '4-bit integer' },
  ];

  const kvCacheOptions: { value: KVCacheQuantization; label: string; description: string }[] = [
    { value: 'fp16_bf16', label: 'FP16/BF16', description: '16-bit KV cache' },
    { value: 'fp8_bf16', label: 'FP8/BF16', description: '8-bit KV cache' },
    { value: 'int8', label: 'INT8', description: '8-bit integer KV cache' },
  ];

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium text-gray-700">
            Inference Quantization
          </label>
          <div className="group relative">
            <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" />
            <div className="absolute right-0 w-64 p-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
              Quantization reduces model size and memory usage by using fewer bits to represent weights. Lower precision typically means faster inference but may reduce quality.
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {inferenceOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setInferenceQuantization(option.value)}
              className={`p-3 rounded-lg border-2 transition-all ${
                inferenceQuantization === option.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 bg-white'
              }`}
            >
              <div className="text-sm font-medium">
                {option.label}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {option.description}
              </div>
            </button>
          ))}
        </div>

        {inferenceQuantization && (
          <div className="mt-2 p-2 bg-gray-50 rounded text-xs text-gray-600">
            {getQuantizationInfo(inferenceQuantization)}
          </div>
        )}
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium text-gray-700">
            KV Cache Quantization
          </label>
          <div className="group relative">
            <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" />
            <div className="absolute right-0 w-64 p-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
              KV cache stores attention keys and values. Quantizing the KV cache can significantly reduce memory usage during long context generation.
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {kvCacheOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setKVCacheQuantization(option.value)}
              className={`p-3 rounded-lg border-2 transition-all ${
                kvCacheQuantization === option.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 bg-white'
              }`}
            >
              <div className="text-sm font-medium">
                {option.label}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {option.description}
              </div>
            </button>
          ))}
        </div>

        {kvCacheQuantization && (
          <div className="mt-2 p-2 bg-gray-50 rounded text-xs text-gray-600">
            {getQuantizationInfo(kvCacheQuantization)}
          </div>
        )}
      </div>

      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <div className="flex space-x-2">
          <HelpCircle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-amber-800">
            <strong>Trade-offs:</strong> Lower quantization reduces memory usage and can improve throughput,
            but may impact model quality. INT4 can save 75% memory vs FP16 but with noticeable quality degradation.
            FP8 offers a good balance for most use cases.
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuantizationOptions;