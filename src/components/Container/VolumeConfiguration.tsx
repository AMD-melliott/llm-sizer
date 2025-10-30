import React, { useState } from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import type { VolumeMount } from '../../types';

export const VolumeConfiguration: React.FC = () => {
  const {
    modelPath,
    setModelPath,
    mountModelPath,
    setMountModelPath,
    mountHFCache,
    setMountHFCache,
    useContainerToolkit,
    setUseContainerToolkit,
    customVolumes,
    addCustomVolume,
    removeCustomVolume,
  } = useContainerStore();
  
  const [showAddVolume, setShowAddVolume] = useState(false);
  const [newVolume, setNewVolume] = useState<VolumeMount>({
    hostPath: '',
    containerPath: '',
    readOnly: false,
  });
  
  const handleAddVolume = () => {
    if (newVolume.hostPath && newVolume.containerPath) {
      addCustomVolume(newVolume);
      setNewVolume({ hostPath: '', containerPath: '', readOnly: false });
      setShowAddVolume(false);
    }
  };
  
  return (
    <div className="space-y-4">
      {/* Container Toolkit Toggle */}
      <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
        <div className="flex-1">
          <label htmlFor="container-toolkit" className="text-sm font-medium text-gray-700 dark:text-gray-300">
            AMD Container Toolkit
          </label>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Simplified GPU access management (recommended)
          </p>
        </div>
        <input
          id="container-toolkit"
          type="checkbox"
          checked={useContainerToolkit}
          onChange={(e) => setUseContainerToolkit(e.target.checked)}
          className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        />
      </div>
      
      {/* Model Path */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label htmlFor="model-path" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Model Storage Path
          </label>
          <div className="flex items-center">
            <input
              id="mount-model-path"
              type="checkbox"
              checked={mountModelPath}
              onChange={(e) => setMountModelPath(e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="mount-model-path" className="ml-2 text-xs text-gray-600 dark:text-gray-400">
              Mount
            </label>
          </div>
        </div>
        <input
          id="model-path"
          type="text"
          value={modelPath}
          onChange={(e) => setModelPath(e.target.value)}
          disabled={!mountModelPath}
          placeholder="./models"
          className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-white disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
        />
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Host directory for model files (mapped to /models in container)
        </p>
      </div>
      
      {/* HuggingFace Cache */}
      <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
        <div className="flex-1">
          <label htmlFor="mount-hf-cache" className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Mount HuggingFace Cache
          </label>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Share HF cache between host and container
          </p>
        </div>
        <input
          id="mount-hf-cache"
          type="checkbox"
          checked={mountHFCache}
          onChange={(e) => setMountHFCache(e.target.checked)}
          className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        />
      </div>
      
      {/* Custom Volumes */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Additional Volumes
          </label>
          <button
            onClick={() => setShowAddVolume(!showAddVolume)}
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            {showAddVolume ? 'Cancel' : '+ Add Volume'}
          </button>
        </div>
        
        {/* Add Volume Form */}
        {showAddVolume && (
          <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md space-y-2">
            <div>
              <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                Host Path
              </label>
              <input
                type="text"
                value={newVolume.hostPath}
                onChange={(e) => setNewVolume({ ...newVolume, hostPath: e.target.value })}
                placeholder="/path/on/host"
                className="w-full px-2 py-1 text-sm bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-white"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                Container Path
              </label>
              <input
                type="text"
                value={newVolume.containerPath}
                onChange={(e) => setNewVolume({ ...newVolume, containerPath: e.target.value })}
                placeholder="/path/in/container"
                className="w-full px-2 py-1 text-sm bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-white"
              />
            </div>
            <div className="flex items-center">
              <input
                id="read-only"
                type="checkbox"
                checked={newVolume.readOnly}
                onChange={(e) => setNewVolume({ ...newVolume, readOnly: e.target.checked })}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="read-only" className="ml-2 text-xs text-gray-600 dark:text-gray-400">
                Read-only
              </label>
            </div>
            <button
              onClick={handleAddVolume}
              disabled={!newVolume.hostPath || !newVolume.containerPath}
              className="w-full px-3 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              Add Volume
            </button>
          </div>
        )}
        
        {/* Custom Volumes List */}
        {customVolumes.length > 0 && (
          <div className="space-y-2">
            {customVolumes.map((volume, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-700 dark:text-gray-300 truncate">
                    {volume.hostPath} → {volume.containerPath}
                  </p>
                  {volume.readOnly && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      (read-only)
                    </span>
                  )}
                </div>
                <button
                  onClick={() => removeCustomVolume(idx)}
                  className="ml-2 text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
                  title="Remove volume"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
