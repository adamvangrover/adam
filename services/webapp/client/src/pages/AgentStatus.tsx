import React, { useState, useEffect } from 'react';
import { dataManager } from '../utils/DataManager';

const AgentStatus: React.FC = () => {
  const [agents, setAgents] = useState<any[]>([]);

  useEffect(() => {
    const fetchAgents = async () => {
        const manifest = await dataManager.getManifest();
        setAgents(manifest.agents || []);
    };
    fetchAgents();
  }, []);

  return (
    <div style={{ padding: '20px' }}>
        <h2 className="text-cyan-glow" style={{ marginBottom: '20px' }}>ACTIVE AGENT ROSTER</h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
            {agents.map(agent => (
                <div key={agent.id} className="cyber-card" style={{ padding: '20px', position: 'relative', overflow: 'hidden' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                        <h3 style={{ margin: 0, fontSize: '1.1rem' }}>{agent.name}</h3>
                        <div className={`cyber-badge ${agent.status === 'active' ? 'badge-green' : 'badge-amber'}`}>
                            {agent.status.toUpperCase()}
                        </div>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '15px', height: '40px', overflow: 'hidden' }}>
                        {agent.description}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: '#666', borderTop: '1px solid #333', paddingTop: '10px' }}>
                        <span>{agent.specialization}</span>
                        <span>Last Active: {agent.last_active}</span>
                    </div>
                </div>
            ))}
        </div>
    </div>
  );
};

export default AgentStatus;
