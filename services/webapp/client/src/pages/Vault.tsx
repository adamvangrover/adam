import React, { useEffect, useState } from 'react';
import { dataManager } from '../utils/DataManager';

const Vault: React.FC = () => {
  const [items, setItems] = useState<any[]>([]);

  useEffect(() => {
    dataManager.getManifest().then(data => setItems(data.reports));
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h2 className="text-cyan mono-font">/// SECURE DATA VAULT</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px', marginTop: '20px' }}>
        {items.map(item => (
          <div key={item.id} className="cyber-panel" style={{ padding: '20px', cursor: 'pointer', transition: 'transform 0.2s' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--accent-color)', marginBottom: '5px' }}>{item.type.toUpperCase()}</div>
            <h3 style={{ margin: '0 0 10px 0', fontSize: '1.1rem' }}>{item.title}</h3>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
              <span>{item.date}</span>
              <span>ACCESS &rarr;</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Vault;
