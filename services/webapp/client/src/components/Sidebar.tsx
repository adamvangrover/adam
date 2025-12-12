import React from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/', label: 'Mission Control', icon: '🚀' },
  { path: '/terminal', label: 'UFOs Terminal', icon: '💻' },
  { path: '/market-data', label: 'Market Intelligence', icon: '📈' },
  { path: '/knowledge-graph', label: 'Knowledge Graph', icon: '🕸️' },
  { path: '/agents', label: 'Agent Status', icon: '🤖' },
  { path: '/vault', label: 'Archives & Reports', icon: '🗄️' },
  { path: '/simulation-tools', label: 'Simulations', icon: '🎲' },
];

const Sidebar: React.FC = () => {
  return (
    <aside className="glass-panel" style={{ width: '260px', height: 'calc(100vh - 60px)', display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--primary-color)' }}>
      <nav style={{ flexGrow: 1, padding: '20px 0' }}>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {navItems.map((item) => (
            <li key={item.path} style={{ marginBottom: '5px' }}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  isActive ? "text-cyan-glow" : ""
                }
                style={({ isActive }) => ({
                  display: 'flex', alignItems: 'center', padding: '12px 25px',
                  color: isActive ? 'var(--bg-color)' : '#aaa',
                  backgroundColor: isActive ? 'var(--primary-color)' : 'transparent',
                  textDecoration: 'none',
                  borderLeft: isActive ? '4px solid var(--accent-color)' : '4px solid transparent',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.85rem',
                  transition: 'all 0.2s'
                })}
              >
                <span style={{ marginRight: '10px' }}>{item.icon}</span>
                <span style={{ fontWeight: 600 }}>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <div style={{ padding: '20px', borderTop: '1px solid #333', fontSize: '0.75rem', color: '#666', fontFamily: 'var(--font-mono)' }}>
        <div>Build: v23.5.0-STABLE</div>
        <div>Env: {process.env.NODE_ENV?.toUpperCase() || 'PRODUCTION'}</div>
      </div>
    </aside>
  );
};

export default Sidebar;
