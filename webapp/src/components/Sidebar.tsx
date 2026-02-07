import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  BarChart2,
  BrainCircuit,
  Network,
  Terminal,
  FileText,
  Users,
  Settings
} from 'lucide-react';

const Sidebar: React.FC = () => {
  const navItems = [
    { path: '/', label: 'MISSION CONTROL', icon: LayoutDashboard },
    { path: '/terminal', label: 'UFOs TERMINAL', icon: Terminal },
    { path: '/market-data', label: 'MARKET DATA', icon: BarChart2 },
    { path: '/agents', label: 'AGENT REGISTRY', icon: Users },
    { path: '/knowledge-graph', label: 'KNOWLEDGE GRAPH', icon: Network },
    { path: '/deep-dive', label: 'DEEP DIVE', icon: BrainCircuit },
    { path: '/reports', label: 'REPORTS VAULT', icon: FileText },
    { path: '/settings', label: 'SYSTEM CONFIG', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-cyber-black/95 border-r border-cyber-cyan/10 flex flex-col h-[calc(100vh-64px)] overflow-y-auto font-mono">
      <div className="py-6 px-4">
        <div className="text-xs text-cyber-text/40 mb-2 uppercase tracking-widest pl-3">Navigation</div>
        <nav className="space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 text-sm transition-all border-l-2 ${
                  isActive
                    ? 'border-cyber-cyan bg-cyber-cyan/10 text-cyber-cyan shadow-[0_0_10px_rgba(6,182,212,0.1)]'
                    : 'border-transparent text-cyber-text/70 hover:text-cyber-cyan hover:bg-cyber-slate/50'
                }`
              }
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="mt-auto p-6 border-t border-cyber-cyan/10">
        <div className="bg-cyber-slate/30 rounded p-3 border border-cyber-cyan/10">
          <div className="text-[10px] text-cyber-text/50 mb-1" id="resource-usage-label">SYSTEM RESOURCE</div>
          <div
            className="h-1 w-full bg-cyber-black rounded-full overflow-hidden mb-1"
            role="progressbar"
            aria-valuenow={65}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-labelledby="resource-usage-label"
          >
            <div className="h-full bg-cyber-cyan w-[65%] shadow-[0_0_5px_rgba(6,182,212,0.5)]"></div>
          </div>
          <div className="flex justify-between text-[10px] text-cyber-cyan">
            <span>CPU: 65%</span>
            <span>MEM: 12GB</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
