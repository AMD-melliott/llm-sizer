
import { Calculator } from 'lucide-react';

interface CalculationStep {
  label: string;
  calculation: string;
  result: string;
}

interface ExampleCalculationProps {
  title: string;
  description: string;
  parameters: { label: string; value: string }[];
  steps: CalculationStep[];
  totalMemory: string;
  notes?: string[];
}

export function ExampleCalculation({
  title,
  description,
  parameters,
  steps,
  totalMemory,
  notes
}: ExampleCalculationProps) {
  return (
    <div className="my-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
      <div className="flex items-start gap-3 mb-4">
        <Calculator className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
        <div>
          <h4 className="text-lg font-semibold text-gray-900">{title}</h4>
          <p className="text-sm text-gray-700 mt-1">{description}</p>
        </div>
      </div>

      {/* Parameters */}
      <div className="bg-white rounded-md p-4 mb-4">
        <h5 className="text-sm font-semibold text-gray-900 mb-3">Input Parameters:</h5>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {parameters.map((param, index) => (
            <div key={index} className="flex justify-between items-center">
              <dt className="text-sm text-gray-600">{param.label}:</dt>
              <dd className="text-sm font-mono font-semibold text-gray-900">{param.value}</dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Calculation Steps */}
      <div className="bg-white rounded-md p-4 mb-4">
        <h5 className="text-sm font-semibold text-gray-900 mb-3">Calculation Steps:</h5>
        <ol className="space-y-3">
          {steps.map((step, index) => (
            <li key={index} className="text-sm">
              <div className="font-medium text-gray-900 mb-1">
                {index + 1}. {step.label}
              </div>
              <div className="font-mono text-xs text-gray-700 bg-gray-50 p-2 rounded mb-1">
                {step.calculation}
              </div>
              <div className="font-semibold text-blue-700">
                = {step.result}
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* Total */}
      <div className="bg-blue-600 text-white rounded-md p-4 mb-4">
        <div className="flex justify-between items-center">
          <span className="font-semibold">Total Memory Required:</span>
          <span className="text-2xl font-bold">{totalMemory}</span>
        </div>
      </div>

      {/* Notes */}
      {notes && notes.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <h5 className="text-sm font-semibold text-yellow-900 mb-2">Notes:</h5>
          <ul className="space-y-1">
            {notes.map((note, index) => (
              <li key={index} className="text-sm text-yellow-800">
                • {note}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
