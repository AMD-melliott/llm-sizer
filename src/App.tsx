import { useEffect } from 'react';
import { Github, Brain, Cpu, Box, Zap, Layers, BookOpen, Container } from 'lucide-react';
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
import { TabContainer } from './components/Tabs';
import { MultiPageDocLayout } from './components/Documentation';
import { documentationPages } from './content/documentation-pages';
import PartitioningTab from './components/Tabs/PartitioningTab';
import { ContainerConfigTab } from './components/Container';
import ThemeToggle from './components/ThemeToggle';

function App() {
  const state = useAppStore();
  const setResults = useAppStore((state) => state.setResults);

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

  // Store calculation results in global state so other components can access them
  useEffect(() => {
    setResults(results);
  }, [results, setResults]);

  const calculatorContent = (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Configuration */}
          <div className="lg:col-span-1 space-y-4">
            {/* 1. Hardware Configuration (GPU) */}
            <CollapsibleSection
              title="Hardware Configuration"
              icon={Cpu}
              iconColor="text-emerald-500 dark:text-emerald-400"
              defaultOpen={true}
            >
              <GPUSelector gpus={gpus} />
            </CollapsibleSection>

            {/* 2. Model Type Selection */}
            <CollapsibleSection
              title="Model Type"
              icon={Box}
              iconColor="text-indigo-500 dark:text-indigo-400"
              defaultOpen={true}
            >
              <ModelTypeSelector />
            </CollapsibleSection>

            {/* 3. Model Configuration */}
            <CollapsibleSection
              title="Model Configuration"
              icon={Brain}
              iconColor="text-blue-500 dark:text-blue-400"
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
              iconColor="text-amber-500 dark:text-amber-400"
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
              iconColor="text-violet-500 dark:text-violet-400"
              defaultOpen={false}
            >
              <QuantizationOptions />
            </CollapsibleSection>
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2 space-y-4">
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
  );

  return (
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: 'var(--surface-alt)' }}>
      {/* Header */}
      <header style={{ backgroundColor: 'var(--surface)', borderBottom: '1px solid var(--border)', boxShadow: '0 1px 3px var(--shadow-color)' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-xl" style={{ backgroundColor: 'var(--accent-light)' }}>
                <Brain className="w-7 h-7" style={{ color: 'var(--accent)' }} />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
                  LLM Inference Calculator
                </h1>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  GPU memory estimation for inference workloads
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <ThemeToggle />
              <a
                href="https://github.com/amd/llm-sizer"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg transition-colors"
                style={{ color: 'var(--text-muted)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
              >
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation and Content */}
      <TabContainer
        tabs={[
          { id: 'calculator', label: 'Calculator', icon: <Cpu className="w-4 h-4" /> },
          { id: 'partitioning', label: 'Partitioning', icon: <Layers className="w-4 h-4" /> },
          { id: 'container', label: 'Container Config', icon: <Container className="w-4 h-4" /> },
          { id: 'documentation', label: 'Documentation', icon: <BookOpen className="w-4 h-4" /> },
        ]}
        defaultTab="calculator"
        className="flex-1"
      >
        {(activeTab) => (
          <div className="flex-1 tab-content-enter" key={activeTab}>
            {activeTab === 'calculator' && calculatorContent}
            {activeTab === 'partitioning' && <PartitioningTab />}
            {activeTab === 'container' && <ContainerConfigTab />}
            {activeTab === 'documentation' && (
              <MultiPageDocLayout pages={documentationPages} />
            )}
          </div>
        )}
      </TabContainer>

      {/* Footer */}
      <footer style={{ backgroundColor: 'var(--surface)', borderTop: '1px solid var(--border)' }} className="mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              LLM Inference Calculator &mdash; GPU memory estimation for inference workloads
            </p>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/amd/llm-sizer"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs transition-colors"
                style={{ color: 'var(--text-muted)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--accent)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
              >
                GitHub
              </a>
              <a
                href="https://github.com/amd/llm-sizer/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs transition-colors"
                style={{ color: 'var(--text-muted)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--accent)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
              >
                Report Issue
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
