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
        <label htmlFor="reranking-batch-size" className="block text-sm font-medium text-gray-700 mb-2">
          Reranking Batch Size: {rerankingBatchSize}
        </label>
        <input
          id="reranking-batch-size"
          type="range"
          min="1"
          max="512"
          value={rerankingBatchSize}
          onChange={(e) => setRerankingBatchSize(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>256</span>
          <span>512</span>
        </div>
      </div>

      <div>
        <label htmlFor="num-queries" className="block text-sm font-medium text-gray-700 mb-2">
          Number of Queries: {numQueries}
        </label>
        <input
          id="num-queries"
          type="range"
          min="1"
          max="50"
          value={numQueries}
          onChange={(e) => setNumQueries(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>25</span>
          <span>50</span>
        </div>
      </div>

      <div>
        <label htmlFor="docs-per-query" className="block text-sm font-medium text-gray-700 mb-2">
          Documents per Query: {docsPerQuery}
        </label>
        <input
          id="docs-per-query"
          type="range"
          min="10"
          max="500"
          step="10"
          value={docsPerQuery}
          onChange={(e) => setDocsPerQuery(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>10</span>
          <span>250</span>
          <span>500</span>
        </div>
      </div>

      <div>
        <label htmlFor="max-query-length" className="block text-sm font-medium text-gray-700 mb-2">
          Max Query Length (tokens): {maxQueryLength}
        </label>
        <input
          id="max-query-length"
          type="range"
          min="64"
          max="1024"
          step="64"
          value={maxQueryLength}
          onChange={(e) => setMaxQueryLength(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>64</span>
          <span>512</span>
          <span>1024</span>
        </div>
      </div>

      <div>
        <label htmlFor="max-doc-length" className="block text-sm font-medium text-gray-700 mb-2">
          Max Document Length (tokens): {maxDocLength}
        </label>
        <input
          id="max-doc-length"
          type="range"
          min="128"
          max="4096"
          step="128"
          value={maxDocLength}
          onChange={(e) => setMaxDocLength(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>128</span>
          <span>2048</span>
          <span>4096</span>
        </div>
      </div>

      <div className="bg-blue-50 rounded-lg p-3 mt-4">
        <p className="text-sm text-blue-800">
          <strong>Reranking:</strong> Processing {numQueries} queries, each with {docsPerQuery} documents.
          Total query-document pairs: {numQueries * docsPerQuery}.
        </p>
      </div>
    </div>
  );
};

export default RerankingParameters;
