import React, { useState } from 'react';
import { usePromptStore } from '../stores/promptStore';
import { usePromptFeed } from '../hooks/usePromptFeed';
import { useSwarmSimulation } from '../hooks/useSwarmSimulation';
import { useUserStore } from '../stores/userStore';

import { TickerTape } from '../components/promptAlpha/TickerTape';
import { BriefingModal } from '../components/promptAlpha/BriefingModal';
import { AlphaChart } from '../components/promptAlpha/AlphaChart';
import { CommandBar } from '../components/promptAlpha/CommandBar';
import { SystemMonitor } from '../components/promptAlpha/SystemMonitor';
import { NewsWire } from '../components/promptAlpha/NewsWire';
import { MarketIndicators } from '../components/promptAlpha/MarketIndicators';
import { SwarmActivity } from '../components/promptAlpha/SwarmActivity';
import { AlphaAlert } from '../components/promptAlpha/AlphaAlert';

import { Search, RefreshCw, Cpu, Wifi, Activity, Database, Zap, Terminal, Layout, Network, Hexagon } from 'lucide-react';

const PromptAlpha: React.FC = () => {
  // Initialize Simulation Engines
  usePromptFeed();
  useSwarmSimulation();

  // Stores
  const prompts = usePromptStore((state) => state.prompts);
  const selectPrompt = usePromptStore((state) => state.selectPrompt);
  const preferences = usePromptStore((state) => state.preferences);
  const updatePreferences = usePromptStore((state) => state.updatePreferences);
  const feeds = usePromptStore((state) => state.feeds);
  const { alphaPoints, rank } = useUserStore();

  const [filter, setFilter] = useState('');
  const [viewMode, setViewMode] = useState<'STREAM' | 'VAULT' | 'NEWS' | 'SYS' | 'MARKET' | 'SWARM'>('STREAM');

  // Filter logic
  const filteredPrompts = prompts
    .filter(p => viewMode === 'VAULT' ? p.isFavorite : true)
    .filter(p =>
      p.title.toLowerCase().includes(filter.toLowerCase()) ||
      p.content.toLowerCase().includes(filter.toLowerCase()) ||
      p.tags.some(t => t.toLowerCase().includes(filter.toLowerCase()))
    );

  const handleCommand = (cmd: string) => {
      switch(cmd) {
          case 'NEWS': setViewMode('NEWS'); break;
          case 'SYS': setViewMode('SYS'); break;
          case 'PRMT': setViewMode('STREAM'); break;
          case 'VAULT': setViewMode('VAULT'); break;
          case 'MARKET': setViewMode('MARKET'); break;
          case 'SWARM': setViewMode('SWARM'); break;
          default: console.log("Unknown command");
      }
  };

  return (
    <div className="h-screen bg-black text-gray-300 font-mono flex flex-col overflow-hidden selection:bg-cyan-500/30">
      <AlphaAlert />

      {/* 1. Ticker Tape */}
      <TickerTape />

      {/* 2. Command Bar */}
      <CommandBar onCommand={handleCommand} currentView={viewMode} />

      {/* 3. Main Workspace */}
      <div className="flex-1 flex overflow-hidden">

        {/* Left Sidebar (Navigation/Status) */}
        <div className="w-64 bg-slate-950 border-r border-cyan-900/30 flex flex-col hidden md:flex shrink-0">
          <div className="p-4 border-b border-cyan-900/30">
            <h1 className="text-xl font-bold text-cyan-400 tracking-tighter flex items-center gap-2">
              <Cpu /> PROMPT<span className="text-white">ALPHA</span>
            </h1>
            <div className="text-[10px] text-gray-500 mt-1 uppercase">Autonomous Intelligence Stream</div>
          </div>

          {/* User Profile */}
          <div className="p-4 bg-slate-900/50 border-b border-cyan-900/30">
            <div className="flex justify-between items-baseline mb-1">
                <span className="text-[10px] text-slate-400 font-bold">OPERATOR RANK</span>
                <span className="text-xs text-amber-400 font-bold">{rank}</span>
            </div>
            <div className="flex justify-between items-baseline">
                <span className="text-[10px] text-slate-400 font-bold">ALPHA CAPTURED</span>
                <span className="text-sm text-cyan-400 font-mono">{alphaPoints.toLocaleString()}</span>
            </div>
          </div>

          <div className="p-4 overflow-y-auto flex-1 flex flex-col gap-6">

            {/* View Switcher (Clickable) */}
            <div className="flex flex-col gap-1">
                <div className="text-[10px] text-gray-600 font-bold mb-1">FUNCTIONS</div>
                <NavButton label="STREAM" icon={Zap} active={viewMode === 'STREAM'} onClick={() => setViewMode('STREAM')} />
                <NavButton label="VAULT" icon={Database} active={viewMode === 'VAULT'} onClick={() => setViewMode('VAULT')} />
                <NavButton label="SWARM" icon={Network} active={viewMode === 'SWARM'} onClick={() => setViewMode('SWARM')} />
                <NavButton label="SYSTEM" icon={Terminal} active={viewMode === 'SYS'} onClick={() => setViewMode('SYS')} />
                <NavButton label="NEWS" icon={Layout} active={viewMode === 'NEWS'} onClick={() => setViewMode('NEWS')} />
            </div>

            {/* Feeds Status */}
            <div>
                <div className="text-xs text-gray-500 mb-3 uppercase tracking-widest font-bold">Active Feeds</div>
                <div className="space-y-2">
                {feeds.map(feed => (
                    <div key={feed.id} className="p-2 bg-slate-900 border border-slate-800 rounded flex items-center justify-between">
                    <div className="flex flex-col">
                        <span className="text-[10px] font-bold text-gray-400">{feed.name}</span>
                    </div>
                    {feed.status === 'loading' ? <RefreshCw size={10} className="animate-spin text-cyan-500" /> : <div className="w-2 h-2 bg-cyan-500 rounded-full shadow-[0_0_5px_cyan]" />}
                    </div>
                ))}
                </div>
            </div>

            {/* Mini Charts */}
            <AlphaChart />

            {/* Simulation Toggle */}
            <div className="mt-auto pt-4 border-t border-slate-800">
              <label className="flex items-center justify-between cursor-pointer group">
                  <span className="text-xs text-gray-500 font-bold group-hover:text-cyan-400 transition-colors">SIMULATION MODE</span>
                  <div className={`w-8 h-4 rounded-full relative transition-colors ${preferences.useSimulation ? 'bg-cyan-900' : 'bg-slate-800'}`} onClick={() => updatePreferences({ useSimulation: !preferences.useSimulation })}>
                      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${preferences.useSimulation ? 'left-4.5 bg-cyan-400' : 'left-0.5 bg-gray-500'}`} style={{left: preferences.useSimulation ? '18px' : '2px'}} />
                  </div>
              </label>
            </div>

          </div>
        </div>

        {/* Center/Main Content Area */}
        <div className="flex-1 flex flex-col bg-black relative">

          {/* Market Indicators */}
          <MarketIndicators />

          {/* Dynamic Content Views */}
          <div className="flex-1 overflow-hidden relative p-0">

            {/* View: STREAM & VAULT */}
            {(viewMode === 'STREAM' || viewMode === 'VAULT') && (
                <div className="h-full flex flex-col">
                    {/* Filter Toolbar */}
                    <div className="h-10 border-b border-cyan-900/30 flex items-center px-4 justify-between bg-slate-950/30 backdrop-blur shrink-0">
                        <div className="flex items-center gap-2 flex-1 max-w-md bg-slate-900/50 border border-slate-800 rounded px-2 py-1 focus-within:border-cyan-500/50 transition-colors">
                        <Search size={12} className="text-gray-500" />
                        <input
                            type="text"
                            placeholder={viewMode === 'STREAM' ? "SEARCH LIVE FEED..." : "SEARCH SECURE VAULT..."}
                            className="bg-transparent border-none outline-none text-[10px] w-full text-white placeholder-gray-600 font-mono uppercase"
                            value={filter}
                            onChange={(e) => setFilter(e.target.value)}
                        />
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {filteredPrompts.map(prompt => (
                            <div
                            key={prompt.id}
                            onClick={() => selectPrompt(prompt.id)}
                            className="bg-slate-950 border border-slate-800 hover:border-cyan-500/50 p-4 rounded cursor-pointer group transition-all duration-200 hover:shadow-[0_0_15px_rgba(34,211,238,0.1)] flex flex-col h-48 relative overflow-hidden"
                            >
                            {/* Subtle Scanline */}
                            <div className="absolute top-0 left-0 w-full h-1 bg-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />

                            <div className="flex justify-between items-start mb-2">
                                <span className={`text-[10px] font-bold border px-1 rounded ${prompt.alphaScore > 80 ? 'text-cyan-400 border-cyan-500/50 bg-cyan-900/20' : 'text-slate-500 border-slate-700'}`}>
                                ALPHA: {prompt.alphaScore}
                                </span>
                                <span className="text-[10px] text-slate-600">{new Date(prompt.timestamp).toLocaleTimeString()}</span>
                            </div>

                            <h3 className="text-sm font-bold text-gray-200 mb-2 line-clamp-2 group-hover:text-cyan-400 transition-colors">
                                {prompt.title}
                            </h3>

                            <p className="text-xs text-slate-500 line-clamp-3 mb-auto font-mono opacity-80">
                                {prompt.content}
                            </p>

                            <div className="mt-3 flex items-center justify-between border-t border-slate-900 pt-2">
                                <div className="flex gap-1">
                                {prompt.tags.slice(0, 2).map(tag => (
                                    <span key={tag} className="text-[10px] text-slate-600 bg-slate-900 px-1 rounded">#{tag}</span>
                                ))}
                                </div>
                                <span className="text-[10px] text-slate-600 uppercase">{prompt.source}</span>
                            </div>
                            </div>
                        ))}

                        {filteredPrompts.length === 0 && (
                            <div className="col-span-full h-64 flex flex-col items-center justify-center text-slate-600">
                                <Activity size={48} className="mb-4 opacity-20 animate-pulse" />
                                <p className="font-mono text-sm">NO DATA FOUND</p>
                            </div>
                        )}
                        </div>
                    </div>
                </div>
            )}

            {/* View: SYSTEM */}
            {viewMode === 'SYS' && <div className="p-4 h-full overflow-hidden"><SystemMonitor /></div>}

            {/* View: NEWS */}
            {viewMode === 'NEWS' && <div className="p-4 h-full overflow-hidden"><NewsWire /></div>}

            {/* View: SWARM */}
            {viewMode === 'SWARM' && <div className="p-4 h-full overflow-hidden"><SwarmActivity /></div>}

          </div>
        </div>

        {/* Right Sidebar (News Wire - always visible on large screens unless mode is NEWS) */}
        {viewMode !== 'NEWS' && (
            <div className="hidden xl:block w-80 border-l border-cyan-900/30">
                <NewsWire />
            </div>
        )}

      </div>

      {/* Modal */}
      <BriefingModal />
    </div>
  );
};

const NavButton: React.FC<{ label: string, icon: any, active: boolean, onClick: () => void }> = ({ label, icon: Icon, active, onClick }) => (
    <button
        onClick={onClick}
        className={`w-full flex items-center gap-3 px-3 py-2 rounded text-xs font-bold transition-all ${active ? 'bg-cyan-900/20 text-cyan-400 border border-cyan-900/50' : 'text-slate-500 hover:text-white hover:bg-slate-900'}`}
    >
        <Icon size={14} />
        {label}
    </button>
);

export default PromptAlpha;
