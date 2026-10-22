import React from 'react';
import useAppStore from '../store/useAppStore';

const EmbeddingParameters: React.FC = () => {
  const {
    embeddingBatchSize,
    setEmbeddingBatchSize,
    documentsPerBatch,
    setDocumentsPerBatch,
    avgDocumentSize,
    setAvgDocumentSize,
    chunkSize,
    setChunkSize,
    chunkOverlap,
    setChunkOverlap,
  } = useAppStore();

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="embedding-batch-size" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Batch Size: <span className="font-mono tabular-nums">{embeddingBatchSize}</span>
        </label>
        <input
          id="embedding-batch-size"
          type="range"
          min="1"
          max="1024"
          value={embeddingBatchSize}
          onChange={(e) => setEmbeddingBatchSize(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>1</span>
          <span>512</span>
          <span>1024</span>
        </div>
      </div>

      <div>
        <label htmlFor="documents-per-batch" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Documents per Batch: <span className="font-mono tabular-nums">{documentsPerBatch}</span>
        </label>
        <input
          id="documents-per-batch"
          type="range"
          min="1"
          max="256"
          value={documentsPerBatch}
          onChange={(e) => setDocumentsPerBatch(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>1</span>
          <span>128</span>
          <span>256</span>
        </div>
      </div>

      <div>
        <label htmlFor="avg-document-size" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Avg Document Size (tokens): <span className="font-mono tabular-nums">{avgDocumentSize}</span>
        </label>
        <input
          id="avg-document-size"
          type="range"
          min="64"
          max="8192"
          step="64"
          value={avgDocumentSize}
          onChange={(e) => setAvgDocumentSize(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>64</span>
          <span>4096</span>
          <span>8192</span>
        </div>
      </div>

      <div>
        <label htmlFor="chunk-size" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Chunk Size (tokens): <span className="font-mono tabular-nums">{chunkSize}</span>
        </label>
        <input
          id="chunk-size"
          type="range"
          min="64"
          max="8192"
          step="64"
          value={chunkSize}
          onChange={(e) => setChunkSize(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>64</span>
          <span>4096</span>
          <span>8192</span>
        </div>
      </div>

      <div>
        <label htmlFor="chunk-overlap" className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Chunk Overlap (%): <span className="font-mono tabular-nums">{chunkOverlap}</span>
        </label>
        <input
          id="chunk-overlap"
          type="range"
          min="0"
          max="50"
          value={chunkOverlap}
          onChange={(e) => setChunkOverlap(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{ backgroundColor: 'var(--surface-inset)' }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
        </div>
      </div>

      <div className="rounded-xl p-3 mt-4" style={{ backgroundColor: 'var(--accent-light)' }}>
        <p className="text-sm" style={{ color: 'var(--accent-text)' }}>
          <strong>Batch Processing:</strong> Processing <span className="font-mono tabular-nums">{documentsPerBatch}</span> documents
          of ~<span className="font-mono tabular-nums">{avgDocumentSize}</span> tokens each in batches of <span className="font-mono tabular-nums">{embeddingBatchSize}</span>.
        </p>
      </div>
    </div>
  );
};

export default EmbeddingParameters;
