import React, { useEffect, useState } from 'react';
import { dataManager } from '../utils/DataManager';
import { useNavigate } from 'react-router-dom';

const Vault: React.FC = () => {
  const [items, setItems] = useState<any[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    dataManager.getManifest().then(data => setItems(data.reports));
  }, []);

  const openReport = (id: string, type: string) => {
      // If it's a Deep Dive or JSON, open in DeepDive viewer
      if (type === 'JSON' || type === 'SNC' || id.includes('Deep_Dive')) {
          navigate(`/deep-dive/${id}`);
      } else {
          // For now, just show alert or fallback
          alert(`Opening artifact: ${id} (${type}) - Content Viewer implementation pending for this type.`);
      }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2 className="text-cyan mono-font" style={{ borderBottom: '1px solid var(--primary-color)', paddingBottom: '10px' }}>{'///'} SECURE DATA VAULT</h2>

      {items.length === 0 && <div className="text-muted" style={{ marginTop: '20px' }}>Loading archives from Manifest...</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px', marginTop: '20px' }}>
        {items.map(item => (
          <div
            key={item.id}
            className="cyber-panel"
            style={{ padding: '20px', cursor: 'pointer', transition: 'all 0.2s', border: '1px solid rgba(0, 243, 255, 0.2)' }}
            onClick={() => openReport(item.id, item.type)}
            onMouseOver={(e) => e.currentTarget.style.borderColor = 'var(--primary-color)'}
            onMouseOut={(e) => e.currentTarget.style.borderColor = 'rgba(0, 243, 255, 0.2)'}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--accent-color)', border: '1px solid var(--accent-color)', padding: '2px 4px', borderRadius: '2px' }}>
                    {item.type.toUpperCase()}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#666' }}>ID: {item.id}</div>
            </div>

            <h3 style={{ margin: '0 0 10px 0', fontSize: '1.1rem', color: '#e0e0e0' }}>{item.title}</h3>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888', marginTop: '15px' }}>
              <span>{item.date}</span>
              <span style={{ color: 'var(--primary-color)' }}>ACCESS &rarr;</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Vault;
