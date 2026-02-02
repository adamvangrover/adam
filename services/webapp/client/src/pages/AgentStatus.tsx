import React, { useEffect, useState } from 'react';
import { dataManager, DataManifest } from '../utils/DataManager';
import { Search } from 'lucide-react';

const AgentStatus: React.FC = () => {
  const [agents, setAgents] = useState<DataManifest['agents']>([]);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    dataManager.getManifest().then(data => setAgents(data.agents));
  }, []);

  const filteredAgents = agents.filter(a =>
      a.name.toLowerCase().includes(filter.toLowerCase()) ||
      a.specialization.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <h2 className="text-cyan mono-font">{'///'} AGENT REGISTRY</h2>
        <div style={{ position: 'relative', width: '300px' }}>
            <Search
                size={16}
                color="#666"
                style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }}
                aria-hidden="true"
            />
            <input
                type="text"
                aria-label="Filter agents"
                placeholder="FILTER AGENTS..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="cyber-input"
                style={{
                    width: '100%',
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid #333',
                    color: '#fff',
                    padding: '8px 10px 8px 35px', // Added left padding for icon
                    borderRadius: '2px'
                }}
            />
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--primary-color)', color: 'var(--primary-color)', textAlign: 'left' }}>
              <th style={{ padding: '10px' }}>ID</th>
              <th style={{ padding: '10px' }}>AGENT DESIGNATION</th>
              <th style={{ padding: '10px' }}>STATUS</th>
              <th style={{ padding: '10px' }}>SPECIALIZATION</th>
            </tr>
          </thead>
          <tbody>
            {filteredAgents.map(agent => (
              <tr key={agent.id} style={{ borderBottom: '1px solid #333', background: 'rgba(0,0,0,0.1)' }}>
                <td style={{ padding: '10px', fontFamily: 'monospace', color: '#888' }}>{agent.id}</td>
                <td style={{ padding: '10px', fontWeight: 'bold' }}>{agent.name}</td>
                <td style={{ padding: '10px' }}>
                    <span style={{
                        color: agent.status === 'Active' ? '#0f0' : (agent.status === 'Processing' ? '#0ff' : '#888'),
                        border: `1px solid ${agent.status === 'Active' ? '#0f0' : (agent.status === 'Processing' ? '#0ff' : '#888')}`,
                        padding: '2px 6px', fontSize: '0.7rem', borderRadius: '2px'
                    }}>
                        {agent.status.toUpperCase()}
                    </span>
                </td>
                <td style={{ padding: '10px', color: '#aaa' }}>{agent.specialization}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: '20px', fontSize: '0.8rem', color: '#666' }}>
          TOTAL AGENTS ONLINE: {filteredAgents.length}
      </div>
    </div>
  );
};

export default AgentStatus;
