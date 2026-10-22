import React, { useMemo } from 'react';
import { Users, FileText, Layers, Image } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import modelsData from '../data/models.json';
import { Model } from '../types';

const InferenceParameters: React.FC = () => {
  const {
    selectedModel,
    batchSize,
    setBatchSize,
    sequenceLength,
    setSequenceLength,
    concurrentUsers,
    setConcurrentUsers,
    numImages,
    setNumImages,
    imageResolution,
    setImageResolution,
  } = useAppStore();

  const currentModel = useMemo(() => {
    const models = modelsData.models as Model[];
    return models.find(m => m.id === selectedModel);
  }, [selectedModel]);

  const isMultimodal = currentModel?.modality === 'multimodal';

  const batchSizeSteps = [1, 2, 4, 8, 16, 32, 64, 128, 256];
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
            <Layers className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
            <label className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Batch Size
            </label>
          </div>
          <span className="text-lg font-semibold font-mono tabular-nums" style={{ color: 'var(--accent)' }}>{batchSize}</span>
        </div>
        <input
          type="range"
          min="0"
          max={batchSizeSteps.length - 1}
          value={getBatchSizeIndex(batchSize)}
          onChange={(e) => setBatchSize(batchSizeSteps[parseInt(e.target.value)])}
          className="w-full"
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>1</span>
          <span>8</span>
          <span>32</span>
          <span>256</span>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <FileText className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
            <label className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Sequence Length (tokens)
            </label>
          </div>
          <span className="text-lg font-semibold font-mono tabular-nums" style={{ color: 'var(--accent)' }}>{sequenceLength.toLocaleString()}</span>
        </div>
        <input
          type="range"
          min="0"
          max={sequenceLengthSteps.length - 1}
          value={getSequenceLengthIndex(sequenceLength)}
          onChange={(e) => setSequenceLength(sequenceLengthSteps[parseInt(e.target.value)])}
          className="w-full"
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>512</span>
          <span>4K</span>
          <span>32K</span>
          <span>128K</span>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
            <label className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Concurrent Users
            </label>
          </div>
          <span className="text-lg font-semibold font-mono tabular-nums" style={{ color: 'var(--accent)' }}>{concurrentUsers}</span>
        </div>
        <input
          type="range"
          min="0"
          max={concurrentUserSteps.length - 1}
          value={getConcurrentUsersIndex(concurrentUsers)}
          onChange={(e) => setConcurrentUsers(concurrentUserSteps[parseInt(e.target.value)])}
          className="w-full"
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>1</span>
          <span>8</span>
          <span>32</span>
          <span>128</span>
        </div>
      </div>

      {/* Multimodal Parameters */}
      {isMultimodal && (
        <>
          <div style={{ borderTop: '1px solid var(--border)' }} className="pt-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Image className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                <label className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Images per Prompt
                </label>
              </div>
              <span className="text-lg font-semibold font-mono tabular-nums" style={{ color: 'var(--accent)' }}>{numImages}</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              value={numImages}
              onChange={(e) => setNumImages(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              <span>1</span>
              <span>5</span>
              <span>10</span>
            </div>
          </div>

          <div>
            <div className="flex items-center space-x-2 mb-3">
              <Image className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
              <label className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                Image Resolution
              </label>
            </div>
            <select
              value={imageResolution}
              onChange={(e) => setImageResolution(parseInt(e.target.value))}
              className="w-full rounded-lg shadow-sm focus:outline-none focus:ring-2"
              style={{ borderColor: 'var(--border-strong)', color: 'var(--text-primary)', backgroundColor: 'var(--surface)' }}
            >
              <option value={224}>224 x 224</option>
              <option value={336}>336 x 336</option>
              <option value={448}>448 x 448</option>
              <option value={512}>512 x 512</option>
              <option value={1024}>1024 x 1024</option>
            </select>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              Higher resolutions increase memory usage but improve visual understanding
            </p>
          </div>
        </>
      )}

      <div className="rounded-lg p-3" style={{ backgroundColor: 'var(--surface-inset)' }}>
        <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Parameter Impact:</h4>
        <ul className="text-xs space-y-1" style={{ color: 'var(--text-muted)' }}>
          <li><strong style={{ color: 'var(--text-secondary)' }}>Batch Size:</strong> Processes multiple requests simultaneously</li>
          <li><strong style={{ color: 'var(--text-secondary)' }}>Sequence Length:</strong> Maximum context window for generation</li>
          <li><strong style={{ color: 'var(--text-secondary)' }}>Concurrent Users:</strong> Number of parallel user sessions</li>
          {isMultimodal && (
            <>
              <li><strong style={{ color: 'var(--text-secondary)' }}>Images per Prompt:</strong> Number of images processed in each request</li>
              <li><strong style={{ color: 'var(--text-secondary)' }}>Image Resolution:</strong> Higher resolutions = more image tokens & memory</li>
            </>
          )}
        </ul>
      </div>
    </div>
  );
};

export default InferenceParameters;
