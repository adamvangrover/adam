import React from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/', label: 'Mission Control', icon: '🚀' },
  { path: '/prompt-alpha', label: 'Prompt Alpha', icon: '🧠' },
  { path: '/synthesizer', label: 'Synthesizer', icon: '🎛️' },
  { path: '/terminal', label: 'UFOs Terminal', icon: '💻' },
  { path: '/market-data', label: 'Market Intelligence', icon: '📈' },
  { path: '/knowledge-graph', label: 'Knowledge Graph', icon: '🕸️' },
  { path: '/agents', label: 'Agent Status', icon: '🤖' },
  { path: '/vault', label: 'Archives & Reports', icon: '🗄️' },
  { path: '/simulation-tools', label: 'Simulations', icon: '🎲' },
];

const Sidebar: React.FC = () => {
  return (
    <aside className="cyber-panel flex flex-col w-[260px] h-[calc(100vh-60px)] border-r border-[var(--primary-color)]">
      <nav aria-label="Main Navigation" className="flex-grow py-5">
        <ul className="list-none p-0 m-0">
          {navItems.map((item) => (
            <li key={item.path} className="mb-1">
              <NavLink
                to={item.path}
                className={({ isActive }) => `
                  flex items-center px-6 py-3 transition-all duration-200
                  group relative outline-none
                  focus-visible:ring-2 focus-visible:ring-[var(--primary-color)] focus-visible:ring-inset
                  ${isActive
                    ? 'bg-[var(--primary-color)] text-[var(--bg-color)] border-l-4 border-[var(--accent-color)]'
                    : 'text-[#aaa] border-l-4 border-transparent hover:bg-[rgba(0,243,255,0.1)] hover:text-[var(--primary-color)] hover:border-[var(--primary-color)]/50'
                  }
                `}
              >
                <span className="mr-2.5 w-5 text-center">{item.icon}</span>
                <span className="font-semibold text-sm font-mono">{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <div className="p-5 border-t border-[#333] text-xs text-[#666] font-mono">
        <div>Build: v23.5.0-RC1</div>
        <div>Env: {process.env.NODE_ENV?.toUpperCase()}</div>
        <div className="mt-1 text-[var(--primary-color)]">System: ONLINE</div>
      </div>
    </aside>
  );
};

export default Sidebar;
