import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Wifi, Server, Database, FileText, User, X } from 'lucide-react';
import Fuse from 'fuse.js';
import { dataManager } from '../utils/DataManager';

interface GlobalNavProps {
  isOffline: boolean;
}

const GlobalNav: React.FC<GlobalNavProps> = ({ isOffline }) => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(-1);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [manifest, setManifest] = useState<any>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    dataManager.getManifest().then(setManifest);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
      if (e.key === 'Escape' && document.activeElement === searchInputRef.current) {
        setSearchTerm('');
        searchInputRef.current?.blur();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const searchResults = useMemo(() => {
    if (!searchTerm || !manifest) return [];

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const agents = (manifest.agents || []).map((a: any) => ({ ...a, type: 'agent' }));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const reports = (manifest.reports || []).map((r: any) => ({ ...r, type: 'report' }));

    const fuse = new Fuse([...agents, ...reports], {
      keys: ['name', 'title', 'docstring', 'content'],
      threshold: 0.3
    });

    return fuse.search(searchTerm).slice(0, 5).map(r => r.item);
  }, [searchTerm, manifest]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleSelection = (item: any) => {
    setSearchTerm('');
    setSelectedIndex(-1);

    if (item.type === 'agent') {
      navigate('/agents');
    } else if (item.type === 'report') {
      navigate('/reports');
    }
  };

  const handleInputKeyDown = (e: React.KeyboardEvent) => {
    if (searchResults.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => (prev < searchResults.length - 1 ? prev + 1 : prev));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => (prev > -1 ? prev - 1 : -1));
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      handleSelection(searchResults[selectedIndex]);
    }
  };

  return (
    <header className="h-16 bg-cyber-black/90 border-b border-cyber-cyan/20 flex items-center justify-between px-6 backdrop-blur sticky top-0 z-50">

      {/* Left: Brand */}
      <div className="flex items-center gap-4">
        <div className="flex flex-col">
          <h1 className="text-xl font-bold tracking-widest text-cyber-cyan glitch-text cursor-default">
            ADAM v23.5
          </h1>
          <span className="text-[10px] text-cyber-text/60 font-mono tracking-widest">
            ADAPTIVE FINANCIAL INTELLIGENCE
          </span>
        </div>
      </div>

      {/* Center: Search */}
      <div className="flex-1 max-w-2xl mx-8 relative z-50">
        <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-cyber-cyan/50 group-hover:text-cyber-cyan transition-colors" />
          </div>
          <input
            ref={searchInputRef}
            type="text"
            className="block w-full pl-10 pr-10 py-2 border border-cyber-slate/50 rounded-sm leading-5 bg-cyber-slate/30 text-cyber-text placeholder-cyber-text/30 focus:outline-none focus:bg-cyber-slate/50 focus:border-cyber-cyan/50 focus:ring-1 focus:ring-cyber-cyan/30 sm:text-sm font-mono transition-all"
            placeholder="SEARCH SYSTEM KNOWLEDGE [CTRL+K]..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setSelectedIndex(-1);
            }}
            onKeyDown={handleInputKeyDown}
            aria-label="Search system knowledge"
            aria-activedescendant={selectedIndex >= 0 ? `result-${selectedIndex}` : undefined}
            aria-controls="search-results-listbox"
            aria-expanded={searchResults.length > 0}
            role="combobox"
            aria-autocomplete="list"
          />
          {searchTerm && (
            <button
              onClick={() => {
                setSearchTerm('');
                searchInputRef.current?.focus();
              }}
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-cyber-cyan/50 hover:text-cyber-cyan transition-colors"
              aria-label="Clear search"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Search Results Dropdown */}
        {searchResults.length > 0 && (
            <div
              className="absolute top-full left-0 right-0 mt-2 bg-cyber-slate border border-cyber-cyan/30 rounded shadow-xl max-h-96 overflow-y-auto"
              role="listbox"
              id="search-results-listbox"
            >
                {searchResults.map((item, idx) => (
                    <div
                      key={idx}
                      id={`result-${idx}`}
                      role="option"
                      aria-selected={idx === selectedIndex}
                      onClick={() => handleSelection(item)}
                      className={`p-3 cursor-pointer border-b border-white/5 last:border-0 flex items-start gap-3 transition-colors ${
                        idx === selectedIndex ? 'bg-cyber-cyan/20' : 'hover:bg-cyber-cyan/10'
                      }`}
                    >
                        {item.type === 'agent' ? <User className="h-4 w-4 text-purple-400 mt-1" /> : <FileText className="h-4 w-4 text-cyber-cyan mt-1" />}
                        <div>
                            <div className="text-sm font-bold text-white">{item.name || item.title}</div>
                            <div className="text-xs text-cyber-text/60 line-clamp-1">{item.docstring || item.content}</div>
                        </div>
                    </div>
                ))}
            </div>
        )}
      </div>

      {/* Right: Status Modules */}
      <div className="flex items-center gap-6 font-mono text-xs">

        {/* Network Status */}
        <div className="flex items-center gap-2 group cursor-help" title={isOffline ? "Network: OFFLINE" : "Network: CONNECTED"}>
          <Wifi className={`h-4 w-4 ${isOffline ? 'text-cyber-danger' : 'text-cyber-success'} animate-pulse`} />
          <span className={isOffline ? 'text-cyber-danger' : 'text-cyber-success'}>
            {isOffline ? 'OFFLINE' : 'ONLINE'}
          </span>
        </div>

        {/* System Mode */}
        <div className="flex items-center gap-2">
          <Server className="h-4 w-4 text-cyber-warning" />
          <span className="text-cyber-warning">MODE: {isOffline ? 'SIMULATION' : 'LIVE'}</span>
        </div>

        {/* Database */}
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-cyber-cyan" />
          <span className="text-cyber-cyan">KG: READY</span>
        </div>

        {/* Archive Link */}
         <a href="/showcase/index.html" className="px-3 py-1 border border-cyber-cyan/30 rounded hover:bg-cyber-cyan/10 text-cyber-cyan transition-colors">
            ARCHIVES
         </a>

      </div>
    </header>
  );
};

export default GlobalNav;
