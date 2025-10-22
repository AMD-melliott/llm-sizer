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
        <label htmlFor="embedding-batch-size" className="block text-sm font-medium text-gray-700 mb-2">
          Batch Size: {embeddingBatchSize}
        </label>
        <input
          id="embedding-batch-size"
          type="range"
          min="1"
          max="1024"
          value={embeddingBatchSize}
          onChange={(e) => setEmbeddingBatchSize(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>512</span>
          <span>1024</span>
        </div>
      </div>

      <div>
        <label htmlFor="documents-per-batch" className="block text-sm font-medium text-gray-700 mb-2">
          Documents per Batch: {documentsPerBatch}
        </label>
        <input
          id="documents-per-batch"
          type="range"
          min="1"
          max="256"
          value={documentsPerBatch}
          onChange={(e) => setDocumentsPerBatch(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>128</span>
          <span>256</span>
        </div>
      </div>

      <div>
        <label htmlFor="avg-document-size" className="block text-sm font-medium text-gray-700 mb-2">
          Avg Document Size (tokens): {avgDocumentSize}
        </label>
        <input
          id="avg-document-size"
          type="range"
          min="64"
          max="8192"
          step="64"
          value={avgDocumentSize}
          onChange={(e) => setAvgDocumentSize(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>64</span>
          <span>4096</span>
          <span>8192</span>
        </div>
      </div>

      <div>
        <label htmlFor="chunk-size" className="block text-sm font-medium text-gray-700 mb-2">
          Chunk Size (tokens): {chunkSize}
        </label>
        <input
          id="chunk-size"
          type="range"
          min="64"
          max="8192"
          step="64"
          value={chunkSize}
          onChange={(e) => setChunkSize(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>64</span>
          <span>4096</span>
          <span>8192</span>
        </div>
      </div>

      <div>
        <label htmlFor="chunk-overlap" className="block text-sm font-medium text-gray-700 mb-2">
          Chunk Overlap (%): {chunkOverlap}
        </label>
        <input
          id="chunk-overlap"
          type="range"
          min="0"
          max="50"
          value={chunkOverlap}
          onChange={(e) => setChunkOverlap(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
        </div>
      </div>

      <div className="bg-blue-50 rounded-lg p-3 mt-4">
        <p className="text-sm text-blue-800">
          <strong>Batch Processing:</strong> Processing {documentsPerBatch} documents 
          of ~{avgDocumentSize} tokens each in batches of {embeddingBatchSize}.
        </p>
      </div>
    </div>
  );
};

export default EmbeddingParameters;
