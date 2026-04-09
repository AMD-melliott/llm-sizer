import React, { useState, useRef, useEffect, useCallback } from 'react';

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
  const tabRefs = useRef<Map<TabId, HTMLButtonElement>>(new Map());
  const indicatorRef = useRef<HTMLDivElement>(null);

  const updateIndicator = useCallback(() => {
    const activeButton = tabRefs.current.get(activeTab);
    const indicator = indicatorRef.current;
    if (activeButton && indicator) {
      const parent = activeButton.parentElement;
      if (parent) {
        const parentRect = parent.getBoundingClientRect();
        const buttonRect = activeButton.getBoundingClientRect();
        indicator.style.left = `${buttonRect.left - parentRect.left}px`;
        indicator.style.width = `${buttonRect.width}px`;
      }
    }
  }, [activeTab]);

  useEffect(() => {
    updateIndicator();
    window.addEventListener('resize', updateIndicator);
    return () => window.removeEventListener('resize', updateIndicator);
  }, [updateIndicator]);

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
      <div style={{ backgroundColor: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
        <nav
          className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 relative"
          role="tablist"
          aria-label="Main navigation tabs"
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              ref={(el) => { if (el) tabRefs.current.set(tab.id, el); }}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`tabpanel-${tab.id}`}
              id={`tab-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={(e) => handleKeyDown(e, tab.id)}
              className="flex items-center gap-2 px-4 sm:px-5 py-3 text-sm font-medium border-b-2 border-transparent transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2"
              style={{
                color: activeTab === tab.id ? 'var(--accent)' : 'var(--text-muted)',
                borderBottomColor: 'transparent',
                ['--tw-ring-color' as string]: 'var(--ring-color)',
              }}
              onMouseEnter={(e) => {
                if (activeTab !== tab.id) e.currentTarget.style.color = 'var(--text-secondary)';
              }}
              onMouseLeave={(e) => {
                if (activeTab !== tab.id) e.currentTarget.style.color = 'var(--text-muted)';
              }}
            >
              {tab.icon && <span className="w-4 h-4">{tab.icon}</span>}
              <span className="hidden sm:inline">{tab.label}</span>
              <span className="sm:hidden">{tab.label.split(' ')[0]}</span>
            </button>
          ))}
          {/* Animated indicator */}
          <div
            ref={indicatorRef}
            className="absolute bottom-0 h-0.5 tab-indicator"
            style={{ backgroundColor: 'var(--accent)' }}
          />
        </nav>
      </div>

      {/* Tab Content */}
      <div role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
        {children(activeTab)}
      </div>
    </div>
  );
}
