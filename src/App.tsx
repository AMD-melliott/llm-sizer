import { Github, Brain, Cpu, Box, Zap, Layers } from 'lucide-react';
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
import CollapsibleSection from './components/CollapsibleSection';

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
            {/* 1. Hardware Configuration (GPU) */}
            <CollapsibleSection 
              title="Hardware Configuration" 
              icon={Cpu} 
              iconColor="text-green-600"
              defaultOpen={true}
            >
              <GPUSelector gpus={gpus} />
            </CollapsibleSection>

            {/* 2. Model Type Selection */}
            <CollapsibleSection 
              title="Model Type" 
              icon={Box} 
              iconColor="text-indigo-600"
              defaultOpen={true}
            >
              <ModelTypeSelector />
            </CollapsibleSection>

            {/* 3. Model Configuration */}
            <CollapsibleSection 
              title="Model Configuration" 
              icon={Brain} 
              iconColor="text-blue-600"
              defaultOpen={true}
            >
              <ModelSelector 
                models={models} 
                embeddingModels={embeddingModels}
                rerankingModels={rerankingModels}
              />
            </CollapsibleSection>

            {/* 4. Inference Parameters - Conditional based on model type */}
            <CollapsibleSection 
              title={
                state.modelType === 'generation' ? 'Inference Parameters' :
                state.modelType === 'embedding' ? 'Embedding Parameters' :
                'Reranking Parameters'
              }
              icon={Zap} 
              iconColor="text-orange-600"
              defaultOpen={false}
            >
              {state.modelType === 'generation' && <InferenceParameters />}
              {state.modelType === 'embedding' && <EmbeddingParameters />}
              {state.modelType === 'reranking' && <RerankingParameters />}
            </CollapsibleSection>

            {/* 5. Quantization Settings */}
            <CollapsibleSection 
              title="Quantization Settings" 
              icon={Layers} 
              iconColor="text-purple-600"
              defaultOpen={false}
            >
              <QuantizationOptions />
            </CollapsibleSection>
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