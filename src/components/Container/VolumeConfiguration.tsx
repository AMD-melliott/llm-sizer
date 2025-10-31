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
    useHostNetwork,
    setUseHostNetwork,
    autoRemoveContainer,
    setAutoRemoveContainer,
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
      
      {/* Host Network Mode */}
      <div className="space-y-2">
        <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
          <div className="flex-1">
            <label htmlFor="use-host-network" className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Use Host Network
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Share the host's network stack (bypasses Docker networking)
            </p>
          </div>
          <input
            id="use-host-network"
            type="checkbox"
            checked={useHostNetwork}
            onChange={(e) => setUseHostNetwork(e.target.checked)}
            className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
        </div>
        
        {/* Security Warning */}
        {useHostNetwork && (
          <div className="mt-2 flex items-start space-x-2 p-2 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded">
            <svg className="w-4 h-4 text-orange-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div className="text-xs text-orange-700 dark:text-orange-300">
              <p className="font-medium">Security Notice:</p>
              <p>Host network mode bypasses Docker's network isolation. The container shares the host's network stack directly. Only use this if you understand the security implications.</p>
            </div>
          </div>
        )}
      </div>
      
      {/* Auto-Remove Container */}
      <div className="space-y-2">
        <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
          <div className="flex-1">
            <label htmlFor="auto-remove" className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Auto-Remove Container (--rm)
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Automatically remove container when it exits (ephemeral mode)
            </p>
          </div>
          <input
            id="auto-remove"
            type="checkbox"
            checked={autoRemoveContainer}
            onChange={(e) => setAutoRemoveContainer(e.target.checked)}
            className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
        </div>
        
        {/* Info notice when auto-remove is enabled */}
        {autoRemoveContainer && (
          <div className="mt-2 flex items-start space-x-2 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
            <svg className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div className="text-xs text-blue-700 dark:text-blue-300">
              <p className="font-medium">Ephemeral Container Mode:</p>
              <p>Container will be removed when stopped. No restart policy will be applied. Mounted volumes will persist, but container state will be lost. Use for testing or temporary deployments.</p>
            </div>
          </div>
        )}
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
