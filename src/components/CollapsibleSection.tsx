import React, { useState, ReactNode } from 'react';
import { ChevronDown, ChevronRight, LucideIcon } from 'lucide-react';

interface CollapsibleSectionProps {
  title: string;
  icon: LucideIcon;
  iconColor?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  icon: Icon,
  iconColor = 'text-blue-600',
  children,
  defaultOpen = false,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header - Clickable to toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-6 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <Icon className={`w-5 h-5 ${iconColor}`} />
          <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        </div>
        <div className="flex-shrink-0 ml-4">
          {isOpen ? (
            <ChevronDown className="w-5 h-5 text-gray-500" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-500" />
          )}
        </div>
      </button>

      {/* Content - Collapsible */}
      {isOpen && (
        <div className="px-6 pb-6 animate-fadeIn">
          {children}
        </div>
      )}
    </div>
  );
};

export default CollapsibleSection;
