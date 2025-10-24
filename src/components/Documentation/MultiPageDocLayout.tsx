import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronLeft } from 'lucide-react';

export interface DocSection {
  id: string;
  title: string;
  content: React.ReactNode;
  subsections?: DocSection[];
}

interface PagedSection {
  id: string;
  title: string;
  sections: DocSection[];
}

interface PageNavigationProps {
  pages: PagedSection[];
  currentPageIndex: number;
  onPageChange: (index: number) => void;
}

function PageNavigation({ pages, currentPageIndex, onPageChange }: PageNavigationProps) {
  return (
    <nav className="flex items-center justify-between border-t border-gray-200 bg-white px-4 sm:px-6 lg:px-8 py-4">
      <button
        onClick={() => onPageChange(currentPageIndex - 1)}
        disabled={currentPageIndex === 0}
        className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <ChevronLeft className="w-4 h-4" />
        Previous
      </button>

      <div className="text-sm text-gray-600">
        Page {currentPageIndex + 1} of {pages.length}
      </div>

      <button
        onClick={() => onPageChange(currentPageIndex + 1)}
        disabled={currentPageIndex === pages.length - 1}
        className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Next
        <ChevronRight className="w-4 h-4" />
      </button>
    </nav>
  );
}

interface TableOfContentsProps {
  sections: DocSection[];
  activeSection: string;
  onSectionClick: (id: string) => void;
}

function TableOfContents({ sections, activeSection, onSectionClick }: TableOfContentsProps) {
  return (
    <nav className="space-y-1" aria-label="Table of contents">
      <h2 className="text-sm font-semibold text-gray-900 mb-4">On this page</h2>
      {sections.map((section) => (
        <div key={section.id}>
          <button
            onClick={() => onSectionClick(section.id)}
            className={`
              w-full text-left text-sm py-1.5 px-3 rounded transition-colors
              ${
                activeSection === section.id
                  ? 'bg-blue-50 text-blue-700 font-medium'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }
            `}
          >
            {section.title}
          </button>
          {section.subsections && (
            <div className="ml-4 space-y-1 mt-1">
              {section.subsections.map((subsection) => (
                <button
                  key={subsection.id}
                  onClick={() => onSectionClick(subsection.id)}
                  className={`
                    w-full text-left text-xs py-1 px-3 rounded transition-colors flex items-center gap-1
                    ${
                      activeSection === subsection.id
                        ? 'bg-blue-50 text-blue-700 font-medium'
                        : 'text-gray-500 hover:text-gray-900 hover:bg-gray-50'
                    }
                  `}
                >
                  <ChevronRight className="w-3 h-3" />
                  {subsection.title}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </nav>
  );
}

interface MultiPageDocLayoutProps {
  pages: PagedSection[];
}

export function MultiPageDocLayout({ pages }: MultiPageDocLayoutProps) {
  // Flatten all pages into a single list of sections for TOC
  const allSections = pages.flatMap(page => page.sections);
  
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [activeSection, setActiveSection] = useState<string>(
    pages[0]?.sections[0]?.id || ''
  );

  useEffect(() => {
    // Update active section when page changes
    setActiveSection(pages[currentPageIndex]?.sections[0]?.id || '');
  }, [currentPageIndex, pages]);

  useEffect(() => {
    // Observe sections to update active section on scroll
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: '-20% 0px -35% 0px', threshold: 0.1 }
    );

    const allSections = document.querySelectorAll('[data-section-id]');
    allSections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, [currentPageIndex]);

  const scrollToSection = (id: string) => {
    // Find which page contains this section
    const pageIndex = pages.findIndex(page => 
      page.sections.some(section => section.id === id || 
        section.subsections?.some(sub => sub.id === id)
      )
    );
    
    if (pageIndex !== -1 && pageIndex !== currentPageIndex) {
      // Switch to the correct page first
      setCurrentPageIndex(pageIndex);
      // Wait for the page to render, then scroll
      setTimeout(() => {
        const element = document.getElementById(id);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'start' });
          setActiveSection(id);
        }
      }, 100);
    } else {
      // Same page, just scroll
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setActiveSection(id);
      }
    }
  };

  const renderSection = (section: DocSection, level: number = 0) => {
    const HeadingTag = level === 0 ? 'h2' : 'h3';
    const headingClass = level === 0 
      ? 'text-2xl font-bold text-gray-900 mb-4' 
      : 'text-xl font-semibold text-gray-900 mb-3 mt-6';

    return (
      <section 
        key={section.id} 
        id={section.id} 
        data-section-id={section.id}
        className={level === 0 ? 'mb-12 scroll-mt-6' : 'mb-8 scroll-mt-6'}
      >
        <HeadingTag className={headingClass}>
          {section.title}
        </HeadingTag>
        <div className="prose prose-blue max-w-none">
          {section.content}
        </div>
        {section.subsections && (
          <div className="mt-6 space-y-6">
            {section.subsections.map((subsection) => renderSection(subsection, level + 1))}
          </div>
        )}
      </section>
    );
  };

  const currentPage = pages[currentPageIndex];
  const currentSections = currentPage?.sections || [];

  return (
    <div className="w-full h-full flex flex-col bg-gray-50">
      <div className="flex-1 flex overflow-hidden">
        {/* Table of Contents - Desktop - Shows ALL sections */}
        <aside className="hidden lg:block lg:w-64 flex-shrink-0 bg-white border-r border-gray-200 overflow-y-auto">
          <div className="sticky top-0 p-4">
            <TableOfContents
              sections={allSections}
              activeSection={activeSection}
              onSectionClick={scrollToSection}
            />
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto flex flex-col">
          <div className="flex-1 px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto w-full">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 sm:p-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-8">{currentPage?.title || 'Documentation'}</h1>
              {currentSections.map((section) => renderSection(section))}
            </div>
          </div>

          {/* Page Navigation */}
          <PageNavigation 
            pages={pages} 
            currentPageIndex={currentPageIndex}
            onPageChange={setCurrentPageIndex}
          />
        </main>

        {/* Mobile TOC - Collapsible - Shows ALL sections */}
        <div className="lg:hidden fixed bottom-4 right-4 z-40">
          <details className="bg-white rounded-lg border border-gray-200 shadow-lg p-4 max-w-sm">
            <summary className="cursor-pointer font-semibold text-gray-900 hover:text-blue-600">
              📋 Table of Contents
            </summary>
            <div className="mt-4 max-h-96 overflow-y-auto">
              <TableOfContents
                sections={allSections}
                activeSection={activeSection}
                onSectionClick={scrollToSection}
              />
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
