import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface FormulaBlockProps {
  title?: string;
  formula: string;
  explanation?: string;
  variables?: { symbol: string; description: string }[];
}

export function FormulaBlock({ title, formula, explanation, variables }: FormulaBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(formula);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-6 border border-gray-200 rounded-lg overflow-hidden bg-gray-50">
      {title && (
        <div className="bg-gray-100 px-4 py-2 border-b border-gray-200">
          <h4 className="text-sm font-semibold text-gray-900">{title}</h4>
        </div>
      )}
      
      <div className="p-4">
        <div className="relative bg-white rounded-md border border-gray-300 p-4">
          <button
            onClick={handleCopy}
            className="absolute top-2 right-2 p-2 text-gray-400 hover:text-gray-600 transition-colors"
            title="Copy formula"
          >
            {copied ? <Check className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4" />}
          </button>
          <code className="text-sm font-mono text-gray-900 block pr-10">
            {formula}
          </code>
        </div>
        
        {explanation && (
          <p className="mt-3 text-sm text-gray-700">{explanation}</p>
        )}
        
        {variables && variables.length > 0 && (
          <div className="mt-4">
            <h5 className="text-xs font-semibold text-gray-700 uppercase mb-2">Variables:</h5>
            <dl className="space-y-2">
              {variables.map((variable, index) => (
                <div key={index} className="flex gap-3 text-sm">
                  <dt className="font-mono font-semibold text-gray-900 min-w-[80px]">
                    {variable.symbol}
                  </dt>
                  <dd className="text-gray-700">{variable.description}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </div>
    </div>
  );
}
