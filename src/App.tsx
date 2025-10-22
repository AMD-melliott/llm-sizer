import { Calculator, Github, Brain } from 'lucide-react';
import useAppStore from './store/useAppStore';
import { useMemoryCalculation } from './hooks/useMemoryCalculation';
import ModelTypeSelector from './components/ModelTypeSelector';
import ModelSelector from './components/ModelSelector';
import GPUSelector from './components/GPUSelector';
import QuantizationOptions from './components/QuantizationOptions';
import InferenceParameters from './components/InferenceParameters';
import EmbeddingParameters from './components/EmbeddingParameters';
import RerankingParameters from './components/RerankingParameters';
import ResultsDisplay from './components/ResultsDisplay';
import MemoryVisualization from './components/MemoryVisualization';

function App() {
  const state = useAppStore();

  const {
    results,
    isCalculating,
    models,
    embeddingModels,
    rerankingModels,
    gpus,
  } = useMemoryCalculation({
    modelType: state.modelType,
    selectedModelId: state.selectedModel,
    selectedGPUId: state.selectedGPU,
    inferenceQuantization: state.inferenceQuantization,
    kvCacheQuantization: state.kvCacheQuantization,
    batchSize: state.batchSize,
    sequenceLength: state.sequenceLength,
    concurrentUsers: state.concurrentUsers,
    numGPUs: state.numGPUs,
    enableOffloading: state.enableOffloading,
    customModelParams: state.customModelParams,
    customHiddenSize: state.customHiddenSize,
    customNumLayers: state.customNumLayers,
    customNumHeads: state.customNumHeads,
    customVRAM: state.customVRAM,
    numImages: state.numImages,
    imageResolution: state.imageResolution,
    embeddingBatchSize: state.embeddingBatchSize,
    documentsPerBatch: state.documentsPerBatch,
    avgDocumentSize: state.avgDocumentSize,
    chunkSize: state.chunkSize,
    chunkOverlap: state.chunkOverlap,
    rerankingBatchSize: state.rerankingBatchSize,
    numQueries: state.numQueries,
    docsPerQuery: state.docsPerQuery,
    maxQueryLength: state.maxQueryLength,
    maxDocLength: state.maxDocLength,
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Brain className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">LLM Inference Calculator</h1>
                <p className="text-sm text-gray-600">Estimate memory requirements and performance for LLM inference</p>
              </div>
            </div>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-500 hover:text-gray-700 transition-colors"
            >
              <Github className="w-6 h-6" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Configuration */}
          <div className="lg:col-span-1 space-y-6">
            {/* Model Type Selection Card */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <ModelTypeSelector />
            </div>

            {/* Model Selection Card */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Calculator className="w-5 h-5 text-blue-600" />
                <h2 className="text-lg font-semibold">Model Configuration</h2>
              </div>
              <ModelSelector 
                models={models} 
                embeddingModels={embeddingModels}
                rerankingModels={rerankingModels}
              />
            </div>

            {/* GPU Selection Card */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Calculator className="w-5 h-5 text-green-600" />
                <h2 className="text-lg font-semibold">Hardware Configuration</h2>
              </div>
              <GPUSelector gpus={gpus} />
            </div>

            {/* Quantization Card */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Calculator className="w-5 h-5 text-purple-600" />
                <h2 className="text-lg font-semibold">Quantization Settings</h2>
              </div>
              <QuantizationOptions />
            </div>

            {/* Parameters Card - Conditional based on model type */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Calculator className="w-5 h-5 text-orange-600" />
                <h2 className="text-lg font-semibold">
                  {state.modelType === 'generation' && 'Inference Parameters'}
                  {state.modelType === 'embedding' && 'Embedding Parameters'}
                  {state.modelType === 'reranking' && 'Reranking Parameters'}
                </h2>
              </div>
              {state.modelType === 'generation' && <InferenceParameters />}
              {state.modelType === 'embedding' && <EmbeddingParameters />}
              {state.modelType === 'reranking' && <RerankingParameters />}
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2 space-y-6">
            <ResultsDisplay results={results} isCalculating={isCalculating} />

            {results && (
              <MemoryVisualization
                memoryBreakdown={results.memoryBreakdown}
                totalVRAM={results.totalVRAM}
              />
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>
              LLM Inference Calculator - Estimate memory requirements for Large Language Models
            </p>
            <p className="mt-2">
              Built with React, TypeScript, and Tailwind CSS
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;