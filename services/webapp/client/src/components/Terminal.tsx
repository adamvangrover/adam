import React, { useState, useEffect, useRef } from 'react';
import { dataManager } from '../utils/DataManager';

const Terminal: React.FC = () => {
  const [history, setHistory] = useState<string[]>(['> ADAM v23.5 KERNEL INITIALIZED', '> TYPE "help" FOR COMMANDS']);
  const [input, setInput] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  const handleCommand = async (cmd: string) => {
    const newHistory = [...history, `> ${cmd}`];

    switch (cmd.toLowerCase()) {
      case 'help':
        newHistory.push('AVAILABLE COMMANDS: status, query [ticker], scan agents, clear');
        break;
      case 'status':
        const status = await dataManager.checkConnection();
        newHistory.push(`SYSTEM STATUS: ${status.status} | LATENCY: ${status.latency}ms`);
        break;
      case 'scan agents':
        const manifest = await dataManager.getManifest();
        newHistory.push('--- AGENT ROSTER ---');
        manifest.agents.forEach(a => newHistory.push(`[${a.status.toUpperCase()}] ${a.name} :: ${a.specialization}`));
        break;
      case 'clear':
        setHistory([]);
        setInput('');
        return;
      default:
        if (cmd.startsWith('query ')) {
            newHistory.push(`INITIATING DEEP DIVE SIMULATION FOR: ${cmd.split(' ')[1].toUpperCase()}...`);
            newHistory.push(`[INFO] Fetching 10-K... DONE`);
            newHistory.push(`[INFO] Running Sentiment Analysis... DONE`);
            newHistory.push(`[RESULT] See "Analysis Tools" for full report.`);
        } else {
            newHistory.push(`ERROR: UNKNOWN COMMAND "${cmd}"`);
        }
    }
    setHistory(newHistory);
    setInput('');
  };

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  return (
    <div className="cyber-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '15px', fontFamily: 'monospace' }}>
      <div style={{ flexGrow: 1, overflowY: 'auto', marginBottom: '10px', color: '#00f3ff' }}>
        {history.map((line, i) => <div key={i}>{line}</div>)}
        <div ref={endRef} />
      </div>
      <div style={{ display: 'flex', borderTop: '1px solid #333', paddingTop: '10px' }}>
        <span style={{ color: '#f7d51d', marginRight: '10px' }}>$</span>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleCommand(input)}
          style={{
            background: 'transparent', border: 'none', color: '#fff',
            flexGrow: 1, outline: 'none', fontFamily: 'monospace'
          }}
          autoFocus
        />
      </div>
    </div>
  );
};

export default Terminal;
