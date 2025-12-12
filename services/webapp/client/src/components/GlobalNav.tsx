import React, { useState, useEffect } from 'react';
import { dataManager } from '../utils/DataManager';
import { useNavigate } from 'react-router-dom';

const GlobalNav: React.FC = () => {
  const [status, setStatus] = useState('CONNECTING...');
  const [mode, setMode] = useState<'LIVE' | 'ARCHIVE'>('LIVE');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [manifest, setManifest] = useState<any>(null);
  const navigate = useNavigate();

  useEffect(() => {
    dataManager.checkConnection().then(s => setStatus(s.status));
    dataManager.getManifest().then(m => setManifest(m));
  }, []);

  const toggleMode = () => {
      const newMode = mode === 'LIVE' ? 'ARCHIVE' : 'LIVE';
      setMode(newMode);
      dataManager.setMode(newMode);
  };

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
      const q = e.target.value;
      setQuery(q);

      if (q.length > 1 && manifest) {
          const hits: any[] = [];
          // Search Agents
          manifest.agents?.forEach((a: any) => {
              if (a.name.toLowerCase().includes(q.toLowerCase()) || a.specialization.toLowerCase().includes(q.toLowerCase())) {
                  hits.push({ type: 'AGENT', title: a.name, path: '/agents' });
              }
          });
          // Search Reports
          manifest.reports?.forEach((r: any) => {
              if (r.title.toLowerCase().includes(q.toLowerCase())) {
                  hits.push({ type: 'REPORT', title: r.title, path: '/vault' }); // Could deep link
              }
          });

          setResults(hits.slice(0, 5));
      } else {
          setResults([]);
      }
  };

  const handleResultClick = (res: any) => {
      setQuery('');
      setResults([]);
      navigate(res.path);
  };

  return (
    <header className="glass-panel" style={{
      height: '60px', display: 'flex', alignItems: 'center',
      padding: '0 20px', borderBottom: '1px solid var(--primary-color)',
      justifyContent: 'space-between', zIndex: 1000, position: 'relative'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <h1 className="text-cyan-glow" style={{ margin: 0, fontSize: '1.5rem', letterSpacing: '2px', fontFamily: 'var(--font-mono)' }}>ADAM v23.5</h1>
        <div className="mono-font" style={{ fontSize: '0.8rem', color: '#666', fontFamily: 'var(--font-mono)' }}>UNIFIED FINANCIAL OS</div>
      </div>

      {/* Global Search */}
      <div style={{ flexGrow: 1, maxWidth: '600px', margin: '0 40px', position: 'relative' }}>
        <input
          type="text"
          className="cyber-input"
          placeholder="GLOBAL SEARCH (Ctrl+K)..."
          style={{ borderRadius: '4px' }}
          value={query}
          onChange={handleSearch}
        />
        {results.length > 0 && (
            <div className="glass-panel" style={{ position: 'absolute', top: '100%', left: 0, width: '100%', padding: '10px' }}>
                {results.map((res, i) => (
                    <div
                        key={i}
                        style={{ padding: '8px', cursor: 'pointer', borderBottom: '1px solid #333' }}
                        onClick={() => handleResultClick(res)}
                        className="hover:bg-slate-800"
                    >
                        <span className="cyber-badge badge-cyan" style={{marginRight: '10px'}}>{res.type}</span>
                        {res.title}
                    </div>
                ))}
            </div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>

        {/* Mode Switcher */}
        <div
            onClick={toggleMode}
            style={{
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                background: 'rgba(0,0,0,0.3)',
                padding: '5px 10px',
                borderRadius: '4px',
                border: '1px solid #444'
            }}
        >
            <div style={{
                width: '10px', height: '10px', borderRadius: '50%',
                background: mode === 'LIVE' ? 'var(--primary-color)' : '#666',
                marginRight: '8px',
                boxShadow: mode === 'LIVE' ? '0 0 10px var(--primary-color)' : 'none'
            }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: mode === 'LIVE' ? '#fff' : '#aaa' }}>
                {mode}
            </span>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '0.7rem', color: '#888', fontFamily: 'var(--font-mono)' }}>SYSTEM STATUS</div>
          <div style={{
            color: status === 'ONLINE' ? 'var(--success-color)' : 'var(--warning-color)',
            fontWeight: 'bold', fontSize: '0.9rem', fontFamily: 'var(--font-mono)'
          }}>{status}</div>
        </div>
      </div>
    </header>
  );
};

export default GlobalNav;
