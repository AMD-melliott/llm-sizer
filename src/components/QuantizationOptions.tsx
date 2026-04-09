import React, { useState } from 'react';
import { HelpCircle, X } from 'lucide-react';
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

  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);

  const inferenceOptions: { value: InferenceQuantization; label: string; description: string }[] = [
    { value: 'fp16', label: 'FP16', description: 'Full 16-bit precision' },
    { value: 'fp8', label: 'FP8', description: '8-bit floating point' },
    { value: 'int8', label: 'INT8', description: '8-bit integer' },
    { value: 'int4', label: 'INT4', description: '4-bit integer' },
  ];

  const kvCacheOptions: { value: KVCacheQuantization; label: string; description: string }[] = [
    { value: 'fp16_bf16', label: 'FP16/BF16', description: '16-bit KV cache' },
    { value: 'fp8_bf16', label: 'FP8/BF16', description: '8-bit KV cache' },
  ];

  const toggleTooltip = (id: string) => {
    setActiveTooltip(activeTooltip === id ? null : id);
  };

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Inference Quantization
          </label>
          <button
            onClick={() => toggleTooltip('inference')}
            className="relative"
          >
            <HelpCircle className="w-4 h-4 cursor-help" style={{ color: 'var(--text-muted)' }} />
            {activeTooltip === 'inference' && (
              <div
                className="absolute right-0 w-64 p-3 rounded-lg text-xs z-10 shadow-lg"
                style={{ backgroundColor: 'var(--surface-raised)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>Quantization</span>
                  <button onClick={() => setActiveTooltip(null)}>
                    <X className="w-3 h-3" style={{ color: 'var(--text-muted)' }} />
                  </button>
                </div>
                Quantization reduces model size and memory usage by using fewer bits to represent weights. Lower precision typically means faster inference but may reduce quality.
              </div>
            )}
          </button>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {inferenceOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setInferenceQuantization(option.value)}
              className="p-3 rounded-lg transition-all"
              style={{
                backgroundColor: inferenceQuantization === option.value ? 'var(--accent-light)' : 'var(--surface)',
                border: `2px solid ${inferenceQuantization === option.value ? 'var(--accent)' : 'var(--border)'}`,
                color: 'var(--text-primary)',
              }}
            >
              <div className="text-sm font-medium font-mono">
                {option.label}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {option.description}
              </div>
            </button>
          ))}
        </div>

        {inferenceQuantization && (
          <div className="mt-2 p-2 rounded text-xs" style={{ backgroundColor: 'var(--surface-inset)', color: 'var(--text-secondary)' }}>
            {getQuantizationInfo(inferenceQuantization)}
          </div>
        )}
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            KV Cache Quantization
          </label>
          <button
            onClick={() => toggleTooltip('kvcache')}
            className="relative"
          >
            <HelpCircle className="w-4 h-4 cursor-help" style={{ color: 'var(--text-muted)' }} />
            {activeTooltip === 'kvcache' && (
              <div
                className="absolute right-0 w-64 p-3 rounded-lg text-xs z-10 shadow-lg"
                style={{ backgroundColor: 'var(--surface-raised)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>KV Cache</span>
                  <button onClick={() => setActiveTooltip(null)}>
                    <X className="w-3 h-3" style={{ color: 'var(--text-muted)' }} />
                  </button>
                </div>
                KV cache stores attention keys and values. Quantizing the KV cache can significantly reduce memory usage during long context generation.
              </div>
            )}
          </button>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {kvCacheOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setKVCacheQuantization(option.value)}
              className="p-3 rounded-lg transition-all"
              style={{
                backgroundColor: kvCacheQuantization === option.value ? 'var(--accent-light)' : 'var(--surface)',
                border: `2px solid ${kvCacheQuantization === option.value ? 'var(--accent)' : 'var(--border)'}`,
                color: 'var(--text-primary)',
              }}
            >
              <div className="text-sm font-medium font-mono">
                {option.label}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {option.description}
              </div>
            </button>
          ))}
        </div>

        {kvCacheQuantization && (
          <div className="mt-2 p-2 rounded text-xs" style={{ backgroundColor: 'var(--surface-inset)', color: 'var(--text-secondary)' }}>
            {getQuantizationInfo(kvCacheQuantization)}
          </div>
        )}
      </div>

      <div className="rounded-lg p-3" style={{ backgroundColor: 'var(--warning-light)', border: '1px solid var(--warning)' }}>
        <div className="flex space-x-2">
          <HelpCircle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'var(--warning)' }} />
          <div className="text-xs" style={{ color: 'var(--warning-text)' }}>
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
