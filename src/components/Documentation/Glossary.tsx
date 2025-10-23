import { useState } from 'react';
import { Search } from 'lucide-react';

export interface GlossaryTerm {
  term: string;
  definition: string;
  category?: 'model' | 'hardware' | 'quantization' | 'performance' | 'general';
}

interface GlossaryProps {
  terms: GlossaryTerm[];
}

export function Glossary({ terms }: GlossaryProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const categories = [
    { id: 'all', label: 'All Terms' },
    { id: 'model', label: 'Model' },
    { id: 'hardware', label: 'Hardware' },
    { id: 'quantization', label: 'Quantization' },
    { id: 'performance', label: 'Performance' },
    { id: 'general', label: 'General' },
  ];

  const filteredTerms = terms.filter((term) => {
    const matchesSearch = term.term.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         term.definition.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || term.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const getCategoryColor = (category?: string) => {
    switch (category) {
      case 'model': return 'bg-blue-100 text-blue-800';
      case 'hardware': return 'bg-green-100 text-green-800';
      case 'quantization': return 'bg-purple-100 text-purple-800';
      case 'performance': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="my-6">
      {/* Search and Filter */}
      <div className="mb-6 space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search terms..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="flex flex-wrap gap-2">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`
                px-3 py-1.5 text-sm font-medium rounded-md transition-colors
                ${
                  selectedCategory === category.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }
              `}
            >
              {category.label}
            </button>
          ))}
        </div>
      </div>

      {/* Glossary Terms */}
      <div className="space-y-6">
        {filteredTerms.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No terms found matching your search.
          </div>
        ) : (
          filteredTerms.map((term, index) => (
            <div key={index} className="border-l-4 border-blue-500 pl-4">
              <div className="flex items-start justify-between gap-4">
                <dt className="text-lg font-semibold text-gray-900">
                  {term.term}
                </dt>
                {term.category && (
                  <span className={`
                    px-2 py-1 text-xs font-medium rounded-full whitespace-nowrap
                    ${getCategoryColor(term.category)}
                  `}>
                    {term.category}
                  </span>
                )}
              </div>
              <dd className="mt-2 text-gray-700">{term.definition}</dd>
            </div>
          ))
        )}
      </div>

      {/* Statistics */}
      <div className="mt-6 pt-6 border-t border-gray-200 text-sm text-gray-600">
        Showing {filteredTerms.length} of {terms.length} terms
      </div>
    </div>
  );
}
