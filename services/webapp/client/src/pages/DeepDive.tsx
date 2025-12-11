import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { dataManager } from '../utils/DataManager';

const DeepDive: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('financials');

  useEffect(() => {
    const fetchData = async () => {
        setLoading(true);
        // Try to find the report in manifest to get path
        try {
            const manifest = await dataManager.getManifest();
            const reportMeta = manifest.reports.find(r => r.id === id);

            if (reportMeta) {
                // In real app, we would fetch reportMeta.path
                // For now, since we don't have an API to serve files outside public,
                // we mock the content based on ID
                setData({
                    title: reportMeta.title,
                    meta: reportMeta,
                    financials: { revenue: '24.5B', growth: '+12%', ebitda: '10.2B', debt_ratio: '2.4x' },
                    sentiment: { score: 0.85, label: 'BULLISH', sources: 124, key_drivers: ['AI Demand', 'Data Center Growth'] },
                    risks: { market: 'High Volatility', operational: 'Supply Chain Constraints', regulatory: 'Antitrust' },
                    strategy: { recommendation: 'BUY', conviction: 'HIGH', target_price: '$1250' }
                });
            } else {
                 // Fallback Mock
                 setData({
                    title: `Deep Dive: ${id || 'UNKNOWN'}`,
                    financials: { revenue: 'N/A', growth: 'N/A' },
                    sentiment: { score: 0.5, label: 'NEUTRAL' },
                    risks: { market: 'Unknown' },
                    strategy: { recommendation: 'HOLD' }
                });
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };
    fetchData();
  }, [id]);

  if (loading) return <div className="text-cyan mono-font" style={{ padding: '20px' }}>ACCESSING SECURE MAINFRAME...</div>;
  if (!data) return <div className="text-danger mono-font" style={{ padding: '20px' }}>ERROR: ARTIFACT NOT FOUND.</div>;

  const renderContent = () => {
      switch(activeTab) {
          case 'financials':
              return <pre className="code-block">{JSON.stringify(data.financials, null, 2)}</pre>;
          case 'sentiment':
              return <pre className="code-block">{JSON.stringify(data.sentiment, null, 2)}</pre>;
          case 'risks':
              return <pre className="code-block">{JSON.stringify(data.risks, null, 2)}</pre>;
          case 'strategy':
              return <pre className="code-block">{JSON.stringify(data.strategy, null, 2)}</pre>;
          default:
              return <div>Select a module.</div>;
      }
  };

  return (
    <div style={{ padding: '20px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <header style={{ marginBottom: '20px', borderBottom: '1px solid #333', paddingBottom: '10px' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--accent-color)' }}>DEEP DIVE PROTOCOL v23.5</div>
          <h1 style={{ margin: 0, color: '#fff' }}>{data.title}</h1>
          <div style={{ fontSize: '0.8rem', color: '#666' }}>ID: {id} | TYPE: {data.meta?.type || 'UNKNOWN'}</div>
      </header>

      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          {['financials', 'sentiment', 'risks', 'strategy'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                    padding: '10px 20px',
                    background: activeTab === tab ? 'var(--primary-color)' : 'rgba(0,0,0,0.3)',
                    color: activeTab === tab ? '#000' : 'var(--primary-color)',
                    border: '1px solid var(--primary-color)',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    fontFamily: 'monospace'
                }}
              >
                  {tab}
              </button>
          ))}
      </div>

      <div className="cyber-panel" style={{ flexGrow: 1, padding: '20px', overflow: 'auto', backgroundColor: 'rgba(0,0,0,0.2)' }}>
          {renderContent()}
      </div>
    </div>
  );
};

export default DeepDive;
