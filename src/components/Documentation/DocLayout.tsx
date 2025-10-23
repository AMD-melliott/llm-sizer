import React, { useState, useEffect } from 'react';
import { ChevronRight } from 'lucide-react';

export interface DocSection {
  id: string;
  title: string;
  content: React.ReactNode;
  subsections?: DocSection[];
}

interface TableOfContentsProps {
  sections: DocSection[];
  activeSection: string;
  onSectionClick: (id: string) => void;
}

function TableOfContents({ sections, activeSection, onSectionClick }: TableOfContentsProps) {
  return (
    <nav className="sticky top-4 space-y-1" aria-label="Table of contents">
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

interface DocLayoutProps {
  sections: DocSection[];
}

export function DocLayout({ sections }: DocLayoutProps) {
  const [activeSection, setActiveSection] = useState<string>(sections[0]?.id || '');

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

    // Observe all section elements
    const allSections = document.querySelectorAll('[data-section-id]');
    allSections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setActiveSection(id);
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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="lg:grid lg:grid-cols-12 lg:gap-8">
        {/* Table of Contents - Desktop */}
        <aside className="hidden lg:block lg:col-span-3">
          <div className="pr-4">
            <TableOfContents
              sections={sections}
              activeSection={activeSection}
              onSectionClick={scrollToSection}
            />
          </div>
        </aside>

        {/* Main Content */}
        <main className="lg:col-span-9">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 sm:p-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-8">Documentation</h1>
            {sections.map((section) => renderSection(section))}
          </div>
        </main>

        {/* Mobile TOC - Collapsible */}
        <div className="lg:hidden mt-6 border-t pt-4">
          <details className="bg-white rounded-lg border border-gray-200 p-4">
            <summary className="cursor-pointer font-semibold text-gray-900">
              Table of Contents
            </summary>
            <div className="mt-4">
              <TableOfContents
                sections={sections}
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
