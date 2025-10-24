import { DocSection } from '../components/Documentation';

// Re-export the old structure for backward compatibility
export { documentationSections } from './documentation';
export { glossaryTerms } from './documentation';

// New multi-page structure
export interface PagedDocSection {
  id: string;
  title: string;
  sections: DocSection[];
}

// Import documentation sections
import { documentationSections } from './documentation';

// Helper function to extract sections by IDs
function getSectionsByIds(ids: string[]): DocSection[] {
  return documentationSections.filter((section) => ids.includes(section.id));
}

// Page 1: Overview
const overviewPage: PagedDocSection = {
  id: 'overview-page',
  title: 'Overview',
  sections: getSectionsByIds(['overview']),
};

// Page 2: Calculation Methodology
const methodologyPage: PagedDocSection = {
  id: 'methodology-page',
  title: 'Calculation Methodology',
  sections: getSectionsByIds(['methodology']),
};

// Page 3: Embedding Models
const embeddingPage: PagedDocSection = {
  id: 'embedding-page',
  title: 'Embedding Models',
  sections: getSectionsByIds(['embedding-models']),
};

// Page 4: Reranking Models
const rerangkingPage: PagedDocSection = {
  id: 'reranking-page',
  title: 'Reranking Models',
  sections: getSectionsByIds(['reranking-models']),
};

// Page 5: Example Calculations
const examplesPage: PagedDocSection = {
  id: 'examples-page',
  title: 'Example Calculations',
  sections: getSectionsByIds(['examples']),
};

// Page 6: Best Practices & Performance
const bestPracticesPage: PagedDocSection = {
  id: 'best-practices-page',
  title: 'Best Practices & Performance',
  sections: getSectionsByIds(['best-practices', 'performance']),
};

// Page 7: Glossary
const glossaryPage: PagedDocSection = {
  id: 'glossary-page',
  title: 'Glossary',
  sections: getSectionsByIds(['glossary']),
};

// Export all pages
export const documentationPages: PagedDocSection[] = [
  overviewPage,
  methodologyPage,
  embeddingPage,
  rerangkingPage,
  examplesPage,
  bestPracticesPage,
  glossaryPage,
];
