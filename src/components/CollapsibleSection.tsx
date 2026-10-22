import React, { useState, ReactNode } from 'react';
import { ChevronDown, LucideIcon } from 'lucide-react';

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
  iconColor = 'text-blue-500',
  children,
  defaultOpen = false,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div
      className="rounded-xl transition-shadow duration-200"
      style={{
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border)',
        boxShadow: '0 1px 3px var(--shadow-color)',
      }}
    >
      {/* Header - Clickable to toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-5 py-4 rounded-xl transition-colors"
        style={{ color: 'var(--text-primary)' }}
      >
        <div className="flex items-center space-x-2.5">
          <Icon className={`w-[18px] h-[18px] ${iconColor}`} />
          <h2 className="text-sm font-semibold">{title}</h2>
        </div>
        <ChevronDown
          className={`w-4 h-4 transition-transform duration-200 ${isOpen ? '' : '-rotate-90'}`}
          style={{ color: 'var(--text-muted)' }}
        />
      </button>

      {/* Content - Animated collapse */}
      <div className={`collapsible-content ${isOpen ? 'open' : ''}`}>
        <div>
          <div className="px-5 pb-5">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CollapsibleSection;
