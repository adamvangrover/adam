import React, { useState, useEffect, useRef } from 'react';
import { dataManager } from '../utils/DataManager';
import { useNavigate } from 'react-router-dom';

const GlobalNav: React.FC = () => {
  const [status, setStatus] = useState('CONNECTING...');
  const [mode, setMode] = useState<'LIVE' | 'ARCHIVE'>('LIVE');
  const navigate = useNavigate();
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const checkStatus = async () => {
      const s = await dataManager.checkConnection();
      setStatus(s.status);
      setMode(s.status === 'ONLINE' ? 'LIVE' : 'ARCHIVE');
    };

    checkStatus();
    const interval = setInterval(checkStatus, 10000); // Check every 10s
    return () => clearInterval(interval);
  }, []);

  // Palette: Keyboard shortcut for Global Search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const [isMockMode, setIsMockMode] = useState(false);
  useEffect(() => {
    // Check if system is in Mock Mode via API
    fetch('/api/system/status')
      .then(res => res.json())
      .then(data => setIsMockMode(data.mode === 'MOCK'))
      .catch(() => setIsMockMode(false));
  }, []);

  const toggleMode = () => {
    if (mode === 'LIVE') {
      setMode('ARCHIVE');
      dataManager.toggleSimulationMode(true);
      navigate('/vault');
    } else {
      setMode('LIVE');
      dataManager.checkConnection().then(s => {
        setStatus(s.status);
        if (s.status === 'ONLINE') {
            dataManager.toggleSimulationMode(false);
            navigate('/');
        } else {
            alert("Cannot switch to LIVE: Backend unreachable.");
        }
      });
    }
  };

  return (
    <header className="cyber-panel" style={{
      height: '60px', display: 'flex', alignItems: 'center',
      padding: '0 20px', borderBottom: '1px solid var(--primary-color)',
      justifyContent: 'space-between',
      background: 'rgba(5, 11, 20, 0.95)',
      zIndex: 100
    }}>
      {/* Brand */}
      <button
        style={{ display: 'flex', alignItems: 'center', gap: '20px', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}
        onClick={() => navigate('/')}
        aria-label="Go to Dashboard"
      >
        <div style={{ position: 'relative' }}>
            <h1 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--primary-color)', letterSpacing: '2px', textShadow: '0 0 10px var(--primary-color)' }}>ADAM v23.5</h1>
            <div className="scan-line" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, opacity: 0.5, pointerEvents: 'none' }}></div>
        </div>
        <div className="mono-font" style={{ fontSize: '0.8rem', color: '#666', borderLeft: '1px solid #333', paddingLeft: '10px' }}>UNIFIED FINANCIAL OS</div>
      </button>

      {/* Global Search */}
      <div style={{ flexGrow: 1, maxWidth: '600px', margin: '0 40px', position: 'relative' }}>
        <span aria-hidden="true" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#444' }}>🔍</span>
        <input
          ref={searchInputRef}
          type="text"
          placeholder="GLOBAL SEARCH (Ctrl+K)..."
          aria-label="Global Search"
          className="cyber-input"
          style={{
            width: '100%', background: 'rgba(0,0,0,0.3)', border: '1px solid #333',
            padding: '8px 15px 8px 35px', color: '#fff', borderRadius: '2px',
            fontFamily: 'monospace'
          }}
        />
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>

        {/* Mode Switcher */}
        <button
            onClick={toggleMode}
            aria-label={`Switch to ${mode === 'LIVE' ? 'Archive' : 'Live'} Mode`}
            className="hover:opacity-80 focus:ring-2 focus:ring-cyan-500 outline-none"
            style={{
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '10px',
                padding: '5px 10px', border: '1px solid #333', borderRadius: '4px',
                background: mode === 'LIVE' ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 165, 0, 0.1)',
                color: 'inherit', font: 'inherit'
            }}
        >
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: mode === 'LIVE' ? '#0f0' : '#ffa500' }}></div>
            <span style={{ fontSize: '0.8rem', fontWeight: 'bold', color: mode === 'LIVE' ? '#0f0' : '#ffa500' }}>{mode} MODE</span>
        </button>

        {/* Mock Mode Indicator */}
        {isMockMode && (
          <div style={{
             padding: '5px 10px', border: '1px solid #ff00ff', borderRadius: '4px',
             background: 'rgba(255, 0, 255, 0.1)', color: '#ff00ff', fontSize: '0.8rem', fontWeight: 'bold'
          }}>
            MOCK DATA
          </div>
        )}

        {/* System Status */}
        <div style={{ textAlign: 'right', borderLeft: '1px solid #333', paddingLeft: '20px' }}>
          <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '2px' }}>SYSTEM STATUS</div>
          <div style={{
            color: status === 'ONLINE' ? '#0f0' : (status === 'SIMULATED' ? '#ffa500' : '#f00'),
            fontWeight: 'bold', fontSize: '0.85rem', letterSpacing: '1px'
          }}>{status}</div>
        </div>

        {/* Swarm Link */}
        <a
           href="/swarm"
           onClick={(e) => { e.preventDefault(); navigate('/swarm'); }}
           style={{ color: '#0ff', fontSize: '1.2rem', textDecoration: 'none' }}
           title="Swarm Telemetry"
           aria-label="Swarm Telemetry"
         >
           🧠
         </a>

        {/* Archive Link */}
         <a
            href="/showcase/index.html"
            style={{ color: '#666', fontSize: '1.2rem', textDecoration: 'none' }}
            title="Static Showcase"
            aria-label="Open Static Showcase"
          >
            📂
          </a>

      </div>
    </header>
  );
};

export default GlobalNav;
