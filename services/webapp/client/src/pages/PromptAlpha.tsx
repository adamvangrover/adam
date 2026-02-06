import React, { useState, useEffect, useMemo } from 'react';
import { usePromptStore } from '../stores/promptStore';
import { usePromptFeed } from '../hooks/usePromptFeed';
import { useSwarmSimulation } from '../hooks/useSwarmSimulation';
import { useUserStore } from '../stores/userStore';

// Components
import { TickerTape } from '../components/promptAlpha/TickerTape';
import { BriefingModal } from '../components/promptAlpha/BriefingModal';
import { AlphaChart } from '../components/promptAlpha/AlphaChart';
import { CommandBar } from '../components/promptAlpha/CommandBar';
import { SystemMonitor } from '../components/promptAlpha/SystemMonitor';
import { NewsWire } from '../components/promptAlpha/NewsWire';
import { MarketIndicators } from '../components/promptAlpha/MarketIndicators';
import { SwarmActivity } from '../components/promptAlpha/SwarmActivity';
import { AlphaAlert } from '../components/promptAlpha/AlphaAlert';

// Icons
import { 
  Search, RefreshCw, Cpu, Wifi, Activity, Database, Zap, Terminal, 
  Layout, Network, Hexagon, BarChart2, ChevronLeft, ChevronRight, 
  Filter, ArrowUp, ArrowDown, Grid, List, Play, Eye, Share2, 
  Settings, Radio, Layers, Lock, ShieldCheck
} from 'lucide-react';

const PromptAlpha: React.FC = () => {
  // 1. Initialize Simulation Engines
  usePromptFeed();
  useSwarmSimulation();

  // 2. Stores & State
  const prompts = usePromptStore((state) => state.prompts);
  const selectPrompt = usePromptStore((state) => state.selectPrompt);
  const preferences = usePromptStore((state) => state.preferences);
  const updatePreferences = usePromptStore((state) => state.updatePreferences);
  const feeds = usePromptStore((state) => state.feeds);
  const { alphaPoints, rank } = useUserStore();

  // 3. UI State
  const [filter, setFilter] = useState('');
  const [viewMode, setViewMode] = useState<'STREAM' | 'VAULT' | 'NEWS' | 'SYS' | 'MARKET' | 'SWARM' | 'ANALYTICS' | 'NETWORK'>('STREAM');
  const [layoutMode, setLayoutMode] = useState<'LIST' | 'GRID'>('LIST');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
  
  // 4. Sorting State
  const [sortBy, setSortBy] = useState<'ALPHA' | 'TIME' | 'SOURCE'>('TIME');
  const [sortDir, setSortDir] = useState<'ASC' | 'DESC'>('DESC');

  // 5. Hotkeys Listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey) {
        switch(e.key.toLowerCase()) {
          case 's': setViewMode('STREAM'); break;
          case 'v': setViewMode('VAULT'); break;
          case 'n': setViewMode('NEWS'); break;
          case 'm': setViewMode('MARKET'); break;
          case 'y': setViewMode('SYS'); break;
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // 6. Advanced Filter & Sort Logic
  const processedPrompts = useMemo(() => {
    // A. Filter View
    let data = prompts
      .filter(p => viewMode === 'VAULT' ? p.isFavorite : true)
      .filter(p =>
        p.title.toLowerCase().includes(filter.toLowerCase()) ||
        p.content.toLowerCase().includes(filter.toLowerCase()) ||
        p.tags.some(t => t.toLowerCase().includes(filter.toLowerCase()))
      );

    // B. Sort
    return data.sort((a, b) => {
      let res = 0;
      if (sortBy === 'ALPHA') res = a.alphaScore - b.alphaScore;
      else if (sortBy === 'TIME') res = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      else if (sortBy === 'SOURCE') res = a.source.localeCompare(b.source);
      return sortDir === 'ASC' ? res : -res;
    });
  }, [prompts, viewMode, filter, sortBy, sortDir]);

  // Handle Command Bar Inputs
  const handleCommand = (cmd: string) => {
      switch(cmd.toUpperCase()) {
          case 'NEWS': setViewMode('NEWS'); break;
          case 'SYS': setViewMode('SYS'); break;
          case 'PRMT': setViewMode('STREAM'); break;
          case 'VAULT': setViewMode('VAULT'); break;
          case 'MARKET': setViewMode('MARKET'); break;
          case 'SWARM': setViewMode('SWARM'); break;
          case 'ANALYTICS': setViewMode('ANALYTICS'); break;
          case 'NETWORK': setViewMode('NETWORK'); break;
          case 'TOGGLE_SIM': updatePreferences({ useSimulation: !preferences.useSimulation }); break;
          default: console.log("Unknown command");
      }
  };

  const toggleSort = (field: 'ALPHA' | 'TIME' | 'SOURCE') => {
    if (sortBy === field) {
      setSortDir(prev => prev === 'ASC' ? 'DESC' : 'ASC');
    } else {
      setSortBy(field);
      setSortDir('DESC');
    }
  };

  return (
    <div className="h-screen bg-black text-gray-300 font-mono flex flex-col overflow-hidden selection:bg-cyan-500/30">
      <AlphaAlert />

      {/* --- TOP LEVEL: TICKER --- */}
      <TickerTape />

      {/* --- LEVEL 2: COMMAND & NAVIGATION --- */}
      <CommandBar onCommand={handleCommand} currentView={viewMode} />

      {/* --- LEVEL 3: MAIN WORKSPACE --- */}
      <div className="flex-1 flex overflow-hidden">

        {/* === LEFT SIDEBAR === */}
        <div className={`${sidebarCollapsed ? 'w-12' : 'w-64'} bg-slate-950 border-r border-cyan-900/30 flex flex-col transition-all duration-300 shrink-0 relative z-20`}>
          
          {/* Header / Brand */}
          <div className="p-4 border-b border-cyan-900/30 flex items-center justify-between overflow-hidden whitespace-nowrap">
             {!sidebarCollapsed && (
                <div className="flex flex-col">
                  <h1 className="text-xl font-bold text-cyan-400 tracking-tighter flex items-center gap-2">
                    <Cpu size={20}/> P<span className="text-white">ALPHA</span>
                  </h1>
                </div>
             )}
             <button onClick={() => setSidebarCollapsed(!sidebarCollapsed)} className="text-slate-500 hover:text-white transition-colors">
               {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
             </button>
          </div>

          {/* User Profile (Hidden if collapsed) */}
          {!sidebarCollapsed && (
            <div className="p-4 bg-slate-900/50 border-b border-cyan-900/30 animate-in fade-in slide-in-from-left-4 duration-300">
              <div className="flex justify-between items-baseline mb-1">
                  <span className="text-[10px] text-slate-400 font-bold">RANK</span>
                  <span className="text-xs text-amber-400 font-bold">{rank}</span>
              </div>
              <div className="flex justify-between items-baseline">
                  <span className="text-[10px] text-slate-400 font-bold">CAPTURED</span>
                  <span className="text-sm text-cyan-400 font-mono">{alphaPoints.toLocaleString()}</span>
              </div>
            </div>
          )}

          {/* Navigation Items */}
          <div className="flex-1 overflow-y-auto overflow-x-hidden p-2 flex flex-col gap-1">
              <div className={`text-[10px] text-gray-600 font-bold mb-1 mt-2 px-2 ${sidebarCollapsed ? 'text-center' : ''}`}>
               {sidebarCollapsed ? 'FN' : 'FUNCTIONS'}
              </div>
              
              <NavButton collapsed={sidebarCollapsed} label="STREAM" icon={Zap} active={viewMode === 'STREAM'} onClick={() => setViewMode('STREAM')} />
              <NavButton collapsed={sidebarCollapsed} label="VAULT" icon={Database} active={viewMode === 'VAULT'} onClick={() => setViewMode('VAULT')} />
              <NavButton collapsed={sidebarCollapsed} label="SWARM" icon={Network} active={viewMode === 'SWARM'} onClick={() => setViewMode('SWARM')} />
              <NavButton collapsed={sidebarCollapsed} label="ANALYTICS" icon={BarChart2} active={viewMode === 'ANALYTICS'} onClick={() => setViewMode('ANALYTICS')} />
              <NavButton collapsed={sidebarCollapsed} label="NETWORK" icon={Hexagon} active={viewMode === 'NETWORK'} onClick={() => setViewMode('NETWORK')} />
              <NavButton collapsed={sidebarCollapsed} label="SYSTEM" icon={Terminal} active={viewMode === 'SYS'} onClick={() => setViewMode('SYS')} />
              <NavButton collapsed={sidebarCollapsed} label="NEWS" icon={Layout} active={viewMode === 'NEWS'} onClick={() => setViewMode('NEWS')} />
          </div>

          {/* Simulation Toggle & Mini Charts */}
          <div className="mt-auto border-t border-slate-800 p-2">
            {!sidebarCollapsed && <AlphaChart />}
            <div className={`pt-2 flex ${sidebarCollapsed ? 'justify-center' : 'justify-between'} items-center`}>
               {!sidebarCollapsed && <span className="text-[10px] text-slate-500 font-bold">SIMULATION</span>}
               <div 
                 className={`w-3 h-3 rounded-full cursor-pointer transition-colors ${preferences.useSimulation ? 'bg-cyan-400 shadow-[0_0_8px_cyan]' : 'bg-slate-700'}`}
                 onClick={() => updatePreferences({ useSimulation: !preferences.useSimulation })}
                 title="Toggle Simulation"
               />
            </div>
          </div>
        </div>

        {/* === CENTER MAIN CONTENT === */}
        <div className="flex-1 flex flex-col bg-black relative min-w-0">

          {/* Top Indicators */}
          <MarketIndicators />

          {/* Main View Area */}
          <div className="flex-1 overflow-hidden relative flex flex-col">

            {/* --- VIEW: STREAM & VAULT --- */}
            {(viewMode === 'STREAM' || viewMode === 'VAULT') && (
                <>
                    {/* Toolbar */}
                    <div className="h-12 border-b border-cyan-900/30 flex items-center px-4 justify-between bg-slate-950/30 backdrop-blur shrink-0 gap-4">
                        {/* Search */}
                        <div className="flex items-center gap-2 flex-1 max-w-md bg-slate-900/50 border border-slate-800 rounded px-2 py-1.5 focus-within:border-cyan-500/50 transition-colors">
                            <Search size={14} className="text-gray-500" />
                            <input
                                type="text"
                                placeholder={viewMode === 'STREAM' ? "SEARCH LIVE FEED [ALT+S]" : "SEARCH VAULT [ALT+V]"}
                                className="bg-transparent border-none outline-none text-xs w-full text-white placeholder-gray-600 font-mono uppercase"
                                value={filter}
                                onChange={(e) => setFilter(e.target.value)}
                            />
                        </div>

                        {/* Layout & Sort Controls */}
                        <div className="flex items-center gap-2">
                            <div className="h-4 w-[1px] bg-slate-800 mx-2" />
                            
                            {/* Sort Buttons */}
                            <button onClick={() => toggleSort('ALPHA')} className={`px-2 py-1 rounded text-[10px] font-bold flex items-center gap-1 ${sortBy === 'ALPHA' ? 'text-cyan-400 bg-cyan-950' : 'text-slate-500 hover:text-slate-300'}`}>
                              ROI {sortBy === 'ALPHA' && (sortDir === 'ASC' ? <ArrowUp size={10}/> : <ArrowDown size={10}/>)}
                            </button>
                            <button onClick={() => toggleSort('TIME')} className={`px-2 py-1 rounded text-[10px] font-bold flex items-center gap-1 ${sortBy === 'TIME' ? 'text-cyan-400 bg-cyan-950' : 'text-slate-500 hover:text-slate-300'}`}>
                              TIME {sortBy === 'TIME' && (sortDir === 'ASC' ? <ArrowUp size={10}/> : <ArrowDown size={10}/>)}
                            </button>

                            <div className="h-4 w-[1px] bg-slate-800 mx-2" />
                            
                            {/* View Toggle */}
                            <button onClick={() => setLayoutMode('LIST')} className={`p-1.5 rounded ${layoutMode === 'LIST' ? 'bg-cyan-900/50 text-cyan-400' : 'text-slate-600 hover:text-slate-300'}`}>
                                <List size={14} />
                            </button>
                            <button onClick={() => setLayoutMode('GRID')} className={`p-1.5 rounded ${layoutMode === 'GRID' ? 'bg-cyan-900/50 text-cyan-400' : 'text-slate-600 hover:text-slate-300'}`}>
                                <Grid size={14} />
                            </button>
                        </div>
                    </div>

                    {/* Content Container */}
                    <div className="flex-1 overflow-y-auto bg-black/50 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent p-0">
                        
                        {/* LIST VIEW */}
                        {layoutMode === 'LIST' && (
                          <table className="w-full text-left text-sm border-collapse table-fixed">
                              <thead className="sticky top-0 bg-slate-950/95 backdrop-blur z-10 border-b border-slate-800 shadow-lg">
                                  <tr>
                                      <th className="p-3 font-bold text-cyan-500/80 text-[10px] uppercase w-24 tracking-wider cursor-pointer hover:bg-slate-900" onClick={() => toggleSort('ALPHA')}>Alpha Score</th>
                                      <th className="p-3 font-bold text-slate-500 text-[10px] uppercase w-auto">Signal Payload</th>
                                      <th className="p-3 font-bold text-slate-500 text-[10px] uppercase w-48 hidden md:table-cell">Vectors</th>
                                      <th className="p-3 font-bold text-slate-500 text-[10px] uppercase w-32 hidden lg:table-cell cursor-pointer" onClick={() => toggleSort('SOURCE')}>Origin</th>
                                      <th className="p-3 font-bold text-slate-500 text-[10px] uppercase w-32 text-right cursor-pointer" onClick={() => toggleSort('TIME')}>Timestamp</th>
                                      <th className="p-3 w-16 text-right"></th>
                                  </tr>
                              </thead>
                              <tbody>
                                  {processedPrompts.map(prompt => (
                                      <tr 
                                          key={prompt.id} 
                                          onClick={() => selectPrompt(prompt.id)}
                                          className="border-b border-slate-900/50 hover:bg-cyan-900/5 cursor-pointer transition-colors group relative"
                                      >
                                          <td className="p-3">
                                              <div className="flex items-center gap-2">
                                                <div className={`w-1 h-8 rounded-sm ${prompt.alphaScore >= 80 ? 'bg-green-500' : prompt.alphaScore >= 50 ? 'bg-yellow-500' : 'bg-slate-700'}`} />
                                                <span className={`font-mono font-bold text-lg ${prompt.alphaScore >= 80 ? 'text-green-400' : prompt.alphaScore >= 50 ? 'text-yellow-500' : 'text-slate-500'}`}>
                                                    {prompt.alphaScore}
                                                </span>
                                              </div>
                                          </td>
                                          <td className="p-3 overflow-hidden">
                                              <div className="text-gray-200 font-medium group-hover:text-cyan-300 transition-colors truncate">
                                                  {prompt.title}
                                              </div>
                                              <div className="text-[11px] text-slate-500 mt-1 line-clamp-1 font-mono opacity-80">
                                                  {prompt.content}
                                              </div>
                                          </td>
                                          <td className="p-3 hidden md:table-cell">
                                              <div className="flex gap-1 flex-wrap">
                                                  {prompt.tags.slice(0, 3).map(tag => (
                                                      <span key={tag} className="text-[9px] text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800 whitespace-nowrap">
                                                          {tag}
                                                      </span>
                                                  ))}
                                              </div>
                                          </td>
                                          <td className="p-3 hidden lg:table-cell">
                                            <div className="flex items-center gap-2">
                                              <Activity size={10} className="text-slate-600" />
                                              <span className="text-xs text-slate-400 font-mono uppercase">{prompt.source}</span>
                                            </div>
                                          </td>
                                          <td className="p-3 text-xs text-slate-600 font-mono text-right">
                                              {new Date(prompt.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}
                                          </td>
                                          <td className="p-3 text-right">
                                            {/* Hover Actions */}
                                            <div className="hidden group-hover:flex items-center justify-end gap-2">
                                              <button className="p-1 hover:bg-cyan-500/20 rounded text-cyan-400" title="Simulate"><Play size={12} /></button>
                                              <button className="p-1 hover:bg-cyan-500/20 rounded text-cyan-400" title="Inspect"><Eye size={12} /></button>
                                            </div>
                                          </td>
                                      </tr>
                                  ))}
                              </tbody>
                          </table>
                        )}

                        {/* GRID VIEW */}
                        {layoutMode === 'GRID' && (
                           <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                              {processedPrompts.map(prompt => (
                                <div key={prompt.id} onClick={() => selectPrompt(prompt.id)} className="bg-slate-900/40 border border-slate-800 hover:border-cyan-500/50 p-4 rounded flex flex-col gap-3 cursor-pointer group hover:bg-slate-900/80 transition-all">
                                   <div className="flex justify-between items-start">
                                      <span className={`text-xl font-bold font-mono ${prompt.alphaScore >= 80 ? 'text-green-400' : 'text-slate-500'}`}>{prompt.alphaScore}</span>
                                      <span className="text-[10px] text-slate-600 border border-slate-800 px-1 rounded">{prompt.source}</span>
                                   </div>
                                   <div className="flex-1">
                                      <h3 className="text-sm font-bold text-gray-200 group-hover:text-cyan-400 line-clamp-2">{prompt.title}</h3>
                                      <p className="text-[10px] text-slate-500 mt-2 line-clamp-3 font-mono">{prompt.content}</p>
                                   </div>
                                   <div className="flex justify-between items-end border-t border-slate-800/50 pt-2">
                                      <div className="flex gap-1">
                                        {prompt.tags.slice(0, 2).map(t => <div key={t} className="w-1.5 h-1.5 rounded-full bg-slate-700" title={t} />)}
                                      </div>
                                      <span className="text-[10px] text-slate-600">{new Date(prompt.timestamp).toLocaleTimeString()}</span>
                                   </div>
                                </div>
                              ))}
                           </div>
                        )}

                        {/* Empty State */}
                        {processedPrompts.length === 0 && (
                            <div className="h-full flex flex-col items-center justify-center text-slate-600">
                                <Activity size={48} className="mb-4 opacity-20 animate-pulse" />
                                <p className="font-mono text-sm">NO SIGNAL DETECTED</p>
                                <p className="text-xs mt-2 opacity-50">Adjust filters or check feed connection</p>
                            </div>
                        )}
                    </div>
                </>
            )}

            {/* --- OTHER VIEW MODES --- */}
            {viewMode === 'SYS' && <div className="p-4 h-full overflow-hidden animate-in fade-in"><SystemMonitor /></div>}
            {viewMode === 'NEWS' && <div className="p-4 h-full overflow-hidden animate-in fade-in"><NewsWire /></div>}
            {viewMode === 'SWARM' && <div className="p-4 h-full overflow-hidden animate-in fade-in"><SwarmActivity /></div>}
            
            {/* Analytics Placeholder */}
            {viewMode === 'ANALYTICS' && (
               <div className="p-8 h-full flex flex-col items-center justify-center text-slate-600 bg-slate-950/30">
                  <BarChart2 size={64} className="mb-4 text-cyan-900" />
                  <h2 className="text-xl font-bold text-slate-400">ANALYTICS MODULE</h2>
                  <p className="text-sm font-mono mt-2">Data visualization engine requires Level 2 clearance.</p>
               </div>
            )}
            
            {/* Network Placeholder */}
            {viewMode === 'NETWORK' && (
               <div className="p-8 h-full flex flex-col items-center justify-center text-slate-600 bg-slate-950/30">
                  <Hexagon size={64} className="mb-4 text-cyan-900 animate-pulse" />
                  <h2 className="text-xl font-bold text-slate-400">NEURAL NETWORK</h2>
                  <p className="text-sm font-mono mt-2">Mapping node topology...</p>
               </div>
            )}

          </div>

          {/* --- SYSTEM STATUS FOOTER --- */}
          <div className="h-6 bg-slate-950 border-t border-cyan-900/30 flex items-center px-4 justify-between text-[9px] font-mono text-slate-500 shrink-0 select-none">
             <div className="flex gap-4">
                <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-green-500"></span> ONLINE</span>
                <span className="hidden md:inline">LATENCY: 12ms</span>
                <span className="hidden md:inline">UPTIME: 99.9%</span>
             </div>
             <div className="flex gap-4 uppercase">
                <span>MEM: 43%</span>
                <span>CPU: 12%</span>
                <span>VER: 2.4.0-ALPHA</span>
             </div>
          </div>

        </div>

        {/* === RIGHT SIDEBAR (Collapsible) === */}
        {viewMode !== 'NEWS' && (
           <div className={`${rightPanelCollapsed ? 'w-0' : 'w-80'} hidden xl:flex border-l border-cyan-900/30 bg-black/40 flex-col transition-all duration-300 relative`}>
             {/* Toggle Handle */}
             <button 
               onClick={() => setRightPanelCollapsed(!rightPanelCollapsed)}
               className="absolute -left-3 top-1/2 w-3 h-12 bg-slate-900 border border-slate-700 border-r-0 rounded-l flex items-center justify-center text-slate-500 hover:text-cyan-400 z-50"
             >
               {rightPanelCollapsed ? <ChevronLeft size={10} /> : <ChevronRight size={10} />}
             </button>
             
             {!rightPanelCollapsed && (
                 <>
                   <div className="p-3 border-b border-cyan-900/20 text-[10px] font-bold text-cyan-500 uppercase flex justify-between">
                     <span>Global Wire</span>
                     <Radio size={12} className="animate-pulse" />
                   </div>
                   <div className="flex-1 overflow-hidden">
                      <NewsWire />
                   </div>
                 </>
             )}
           </div>
        )}

      </div>

      {/* Modal Overlay */}
      <BriefingModal />
    </div>
  );
};

// --- Helper Components ---

interface NavButtonProps {
    label: string;
    icon: any;
    active: boolean;
    onClick: () => void;
    collapsed: boolean;
}

const NavButton: React.FC<NavButtonProps> = ({ label, icon: Icon, active, onClick, collapsed }) => (
    <button
        onClick={onClick}
        title={label}
        className={`w-full flex items-center gap-3 px-3 py-2 rounded text-xs font-bold transition-all relative overflow-hidden group
        ${active ? 'bg-cyan-900/20 text-cyan-400 border border-cyan-900/50' : 'text-slate-500 hover:text-white hover:bg-slate-900'}
        ${collapsed ? 'justify-center px-0' : ''}`}
    >
        <Icon size={14} className={active ? 'text-cyan-400' : 'group-hover:text-cyan-200'} />
        {!collapsed && <span>{label}</span>}
        {active && !collapsed && <div className="absolute right-2 w-1.5 h-1.5 bg-cyan-400 rounded-full shadow-[0_0_5px_cyan]" />}
    </button>
);

export default PromptAlpha;