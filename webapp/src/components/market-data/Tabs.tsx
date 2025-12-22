import React, { useState, useRef, type KeyboardEvent } from 'react';

export interface Tab {
  label: string;
  content: React.ReactNode;
}

export interface TabsProps {
  tabs: Tab[];
}

const Tabs: React.FC<TabsProps> = ({ tabs }) => {
  const [activeTab, setActiveTab] = useState(0);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex = null;
    if (e.key === 'ArrowRight') {
      nextIndex = (index + 1) % tabs.length;
    } else if (e.key === 'ArrowLeft') {
      nextIndex = (index - 1 + tabs.length) % tabs.length;
    } else if (e.key === 'Home') {
      nextIndex = 0;
    } else if (e.key === 'End') {
      nextIndex = tabs.length - 1;
    }

    if (nextIndex !== null) {
      e.preventDefault();
      setActiveTab(nextIndex);
      tabRefs.current[nextIndex]?.focus();
    }
  };

  return (
    <div className="w-full">
      <div
        role="tablist"
        className="flex border-b border-cyber-cyan/20 mb-4 overflow-x-auto scrollbar-hide"
        aria-label="Data Sections"
      >
        {tabs.map((tab, index) => {
          const isActive = activeTab === index;
          return (
            <button
              key={index}
              ref={(el) => { tabRefs.current[index] = el; }}
              role="tab"
              aria-selected={isActive}
              aria-controls={`tabpanel-${index}`}
              id={`tab-${index}`}
              tabIndex={isActive ? 0 : -1}
              onClick={() => setActiveTab(index)}
              onKeyDown={(e) => handleKeyDown(e, index)}
              className={`
                px-4 py-2 text-sm font-mono transition-all outline-none focus-visible:ring-2 focus-visible:ring-cyber-cyan focus-visible:ring-offset-1 focus-visible:ring-offset-cyber-black whitespace-nowrap
                ${isActive
                  ? 'text-cyber-cyan border-b-2 border-cyber-cyan bg-cyber-cyan/5 shadow-[0_4px_10px_-4px_rgba(6,182,212,0.3)] font-bold'
                  : 'text-cyber-text/60 hover:text-cyber-text hover:bg-cyber-slate/30 border-b-2 border-transparent'}
              `}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div
        role="tabpanel"
        id={`tabpanel-${activeTab}`}
        aria-labelledby={`tab-${activeTab}`}
        className="animate-fade-in focus:outline-none"
        tabIndex={0}
      >
        {tabs[activeTab].content}
      </div>
    </div>
  );
};

export default Tabs;
