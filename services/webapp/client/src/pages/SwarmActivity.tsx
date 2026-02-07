import React, { useState, useEffect } from 'react';

interface LogEntry {
  timestamp: string;
  event_type: string;
  agent_id: string;
  details: any;
}

const SwarmActivity: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState('ALL');

  useEffect(() => {
    // Mock polling of telemetry log
    // In production, this would hit /api/telemetry
    const mockLogs: LogEntry[] = [
      { timestamp: new Date().toISOString(), event_type: "TASK_START", agent_id: "MetaOrchestrator", details: { query: "Analyze AAPL" } },
      { timestamp: new Date(Date.now() + 1000).toISOString(), event_type: "THOUGHT_TRACE", agent_id: "MetaOrchestrator", details: { content: "Routing to Deep Dive..." } },
      { timestamp: new Date(Date.now() + 2000).toISOString(), event_type: "TOOL_EXECUTION", agent_id: "DeepDiveAgent", details: { tool: "azure_ai_search", parameters: { query: "AAPL debt" } } },
      { timestamp: new Date(Date.now() + 3000).toISOString(), event_type: "CRITIQUE", agent_id: "Reflector", details: { score: 0.85, comment: "Good data coverage." } }
    ];
    setLogs(mockLogs);
  }, []);

  const filteredLogs = filter === 'ALL' ? logs : logs.filter(l => l.event_type === filter);

  return (
    <div style={{ padding: '20px', color: '#fff' }}>
      <h2 className="cyber-text">Swarm Neural Telemetry</h2>

      {/* Controls */}
      <div style={{ marginBottom: '20px' }}>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="cyber-input"
          style={{ padding: '10px', background: '#111', color: '#0f0', border: '1px solid #333' }}
        >
          <option value="ALL">ALL EVENTS</option>
          <option value="TASK_START">TASKS</option>
          <option value="THOUGHT_TRACE">THOUGHTS</option>
          <option value="TOOL_EXECUTION">TOOLS</option>
        </select>
      </div>

      {/* Log Feed */}
      <div className="cyber-panel" style={{ height: '500px', overflowY: 'auto', fontFamily: 'monospace' }}>
        {filteredLogs.map((log, idx) => (
          <div key={idx} style={{
            padding: '10px',
            borderBottom: '1px solid #333',
            background: log.event_type === 'ERROR' ? 'rgba(255,0,0,0.1)' : 'transparent'
          }}>
            <span style={{ color: '#666', marginRight: '15px' }}>[{log.timestamp.split('T')[1].split('.')[0]}]</span>
            <span style={{ color: '#0ff', fontWeight: 'bold', marginRight: '15px' }}>{log.agent_id}</span>
            <span style={{ color: '#aaa', marginRight: '10px' }}>{log.event_type}</span>
            <span style={{ color: '#fff' }}>{JSON.stringify(log.details)}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SwarmActivity;
