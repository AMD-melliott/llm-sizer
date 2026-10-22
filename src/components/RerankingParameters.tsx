import React from 'react';
import useAppStore from '../store/useAppStore';

const RerankingParameters: React.FC = () => {
  const {
    rerankingBatchSize,
    setRerankingBatchSize,
    numQueries,
    setNumQueries,
    docsPerQuery,
    setDocsPerQuery,
    maxQueryLength,
    setMaxQueryLength,
    maxDocLength,
    setMaxDocLength,
  } = useAppStore();

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="reranking-batch-size" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Reranking Batch Size: <span className="font-mono tabular-nums">{rerankingBatchSize}</span>
        </label>
        <input
          id="reranking-batch-size"
          type="range"
          min="1"
          max="512"
          value={rerankingBatchSize}
          onChange={(e) => setRerankingBatchSize(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>1</span>
          <span>256</span>
          <span>512</span>
        </div>
      </div>

      <div>
        <label htmlFor="num-queries" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Number of Queries: <span className="font-mono tabular-nums">{numQueries}</span>
        </label>
        <input
          id="num-queries"
          type="range"
          min="1"
          max="50"
          value={numQueries}
          onChange={(e) => setNumQueries(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>1</span>
          <span>25</span>
          <span>50</span>
        </div>
      </div>

      <div>
        <label htmlFor="docs-per-query" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Documents per Query: <span className="font-mono tabular-nums">{docsPerQuery}</span>
        </label>
        <input
          id="docs-per-query"
          type="range"
          min="10"
          max="500"
          step="10"
          value={docsPerQuery}
          onChange={(e) => setDocsPerQuery(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>10</span>
          <span>250</span>
          <span>500</span>
        </div>
      </div>

      <div>
        <label htmlFor="max-query-length" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Max Query Length (tokens): <span className="font-mono tabular-nums">{maxQueryLength}</span>
        </label>
        <input
          id="max-query-length"
          type="range"
          min="64"
          max="1024"
          step="64"
          value={maxQueryLength}
          onChange={(e) => setMaxQueryLength(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>64</span>
          <span>512</span>
          <span>1024</span>
        </div>
      </div>

      <div>
        <label htmlFor="max-doc-length" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Max Document Length (tokens): <span className="font-mono tabular-nums">{maxDocLength}</span>
        </label>
        <input
          id="max-doc-length"
          type="range"
          min="128"
          max="4096"
          step="128"
          value={maxDocLength}
          onChange={(e) => setMaxDocLength(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>128</span>
          <span>2048</span>
          <span>4096</span>
        </div>
      </div>

      <div className="rounded-xl p-3 mt-4" style={{ backgroundColor: 'var(--accent-light)' }}>
        <p className="text-sm" style={{ color: 'var(--accent-text)' }}>
          <strong>Reranking:</strong> Processing <span className="font-mono tabular-nums">{numQueries}</span> queries, each with <span className="font-mono tabular-nums">{docsPerQuery}</span> documents.
          Total query-document pairs: <span className="font-mono tabular-nums">{numQueries * docsPerQuery}</span>.
        </p>
      </div>
    </div>
  );
};

export default RerankingParameters;
