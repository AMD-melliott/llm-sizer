import React from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import engineParametersData from '../../data/engine-parameters.json';

export const EngineSelector: React.FC = () => {
  const { selectedEngineId, setSelectedEngineId } = useContainerStore();
  
  const engines = engineParametersData.engines;
  const selectedEngine = engines.find(e => e.id === selectedEngineId);
  
  return (
    <div className="space-y-2">
      <label htmlFor="engine-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Inference Engine
      </label>
      
      <select
        id="engine-select"
        value={selectedEngineId}
        onChange={(e) => setSelectedEngineId(e.target.value)}
        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-white"
      >
        {engines.map((engine) => (
          <option key={engine.id} value={engine.id}>
            {engine.name} v{engine.version}
          </option>
        ))}
      </select>
      
      {selectedEngine && (
        <div className="mt-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md">
          <p className="text-sm text-gray-700 dark:text-gray-300">
            {selectedEngine.description}
          </p>
          <a
            href={selectedEngine.documentation}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline mt-1 inline-block"
          >
            View Documentation →
          </a>
        </div>
      )}
    </div>
  );
};
