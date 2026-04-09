import React from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import engineParametersData from '../../data/engine-parameters.json';

export const EngineSelector: React.FC = () => {
  const { selectedEngineId, setSelectedEngineId } = useContainerStore();

  const engines = engineParametersData.engines;
  const selectedEngine = engines.find(e => e.id === selectedEngineId);

  return (
    <div className="space-y-2">
      <label htmlFor="engine-select" className="block text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
        Inference Engine
      </label>

      <select
        id="engine-select"
        value={selectedEngineId}
        onChange={(e) => setSelectedEngineId(e.target.value)}
        className="w-full px-3 py-2 rounded-md focus:outline-none focus:ring-2"
        style={{
          backgroundColor: 'var(--surface)',
          border: '1px solid var(--border-strong)',
          color: 'var(--text-primary)',
          boxShadow: '0 1px 3px var(--shadow-color)'
        }}
        onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
        onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-strong)'}
      >
        {engines.map((engine) => (
          <option key={engine.id} value={engine.id}>
            {engine.name}
          </option>
        ))}
      </select>

      {selectedEngine && (
        <div className="mt-2">
          <a
            href={selectedEngine.documentation}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm hover:underline inline-block"
            style={{ color: 'var(--accent)' }}
          >
            View Documentation →
          </a>
        </div>
      )}
    </div>
  );
};
