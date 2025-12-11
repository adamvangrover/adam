import React, { useState, useEffect } from 'react';
import { dataManager } from '../utils/DataManager';

const GlobalNav: React.FC = () => {
  const [status, setStatus] = useState('CONNECTING...');

  useEffect(() => {
    dataManager.checkConnection().then(s => setStatus(s.status));
  }, []);

  return (
    <header className="cyber-panel" style={{
      height: '60px', display: 'flex', alignItems: 'center',
      padding: '0 20px', borderBottom: '1px solid var(--primary-color)',
      justifyContent: 'space-between'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <h1 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--primary-color)', letterSpacing: '2px' }}>ADAM v23.5</h1>
        <div className="mono-font" style={{ fontSize: '0.8rem', color: '#666' }}>UNIFIED FINANCIAL OS</div>
      </div>

      {/* Global Search - Phase 5 Placeholder */}
      <div style={{ flexGrow: 1, maxWidth: '600px', margin: '0 40px' }}>
        <input
          type="text"
          placeholder="GLOBAL SEARCH (Ctrl+K)..."
          style={{
            width: '100%', background: 'rgba(0,0,0,0.5)', border: '1px solid #444',
            padding: '8px 15px', color: '#fff', borderRadius: '4px'
          }}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '0.7rem', color: '#888' }}>SYSTEM STATUS</div>
          <div style={{
            color: status === 'ONLINE' ? '#0f0' : '#f7d51d',
            fontWeight: 'bold', fontSize: '0.9rem'
          }}>{status}</div>
        </div>
      </div>
    </header>
  );
};

export default GlobalNav;
