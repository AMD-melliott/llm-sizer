import React, { useState } from 'react';

export type TabId = 'calculator' | 'partitioning' | 'container' | 'documentation';

interface Tab {
  id: TabId;
  label: string;
  icon?: React.ReactNode;
}

interface TabContainerProps {
  tabs: Tab[];
  defaultTab?: TabId;
  children: (activeTab: TabId) => React.ReactNode;
  className?: string;
}

export function TabContainer({ tabs, defaultTab = 'calculator', children, className = '' }: TabContainerProps) {
  const [activeTab, setActiveTab] = useState<TabId>(defaultTab);

  const handleKeyDown = (e: React.KeyboardEvent, tabId: TabId) => {
    const currentIndex = tabs.findIndex(t => t.id === tabId);
    
    if (e.key === 'ArrowLeft' && currentIndex > 0) {
      e.preventDefault();
      setActiveTab(tabs[currentIndex - 1].id);
    } else if (e.key === 'ArrowRight' && currentIndex < tabs.length - 1) {
      e.preventDefault();
      setActiveTab(tabs[currentIndex + 1].id);
    }
  };

  return (
    <div className={className}>
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 bg-white">
        <nav 
          className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1" 
          role="tablist"
          aria-label="Main navigation tabs"
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`tabpanel-${tab.id}`}
              id={`tab-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={(e) => handleKeyDown(e, tab.id)}
              className={`
                flex items-center gap-2 px-4 sm:px-6 py-3 text-sm font-medium
                border-b-2 transition-colors
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                ${
                  activeTab === tab.id
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-600 hover:text-gray-800 hover:border-gray-300'
                }
              `}
            >
              {tab.icon && <span className="w-5 h-5">{tab.icon}</span>}
              <span className="hidden sm:inline">{tab.label}</span>
              <span className="sm:hidden">{tab.label.split(' ')[0]}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
        {children(activeTab)}
      </div>
    </div>
  );
}
