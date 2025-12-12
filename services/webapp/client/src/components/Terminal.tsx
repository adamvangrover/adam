import React, { useState, useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { dataManager } from '../utils/DataManager';

const Terminal: React.FC = () => {
  const [history, setHistory] = useState<string[]>(['> ADAM v23.5 KERNEL INITIALIZED', '> TYPE "help" FOR COMMANDS']);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'LIVE' | 'SIMULATED'>('SIMULATED');
  const endRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Attempt connection
    const socket = io('http://localhost:5000', {
        transports: ['websocket'],
        autoConnect: false
    });

    socket.on('connect', () => {
        setMode('LIVE');
        setHistory(h => [...h, '> [SYSTEM] CONNECTED TO CORE KERNEL UPLINK.']);
    });

    socket.on('disconnect', () => {
        setMode('SIMULATED');
        setHistory(h => [...h, '> [SYSTEM] CONNECTION LOST. REVERTING TO LOCAL SIMULATION.']);
    });

    socket.on('log', (data) => {
        setHistory(h => [...h, `> [CORE] ${data.message}`]);
    });

    // Try connecting if we think we are live, but for now default to simulated if backend is not explicitly up
    dataManager.checkConnection().then(res => {
        if (res.status === 'ONLINE') {
            socket.connect();
            socketRef.current = socket;
        }
    });

    return () => {
        if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  const handleCommand = async (cmd: string) => {
    const newHistory = [...history, `> ${cmd}`];
    setHistory(newHistory); // optimistically update
    setInput('');

    if (mode === 'LIVE' && socketRef.current?.connected) {
        socketRef.current.emit('command', { command: cmd });
        // Response will come via log or specific event, but for now we might want a simple ack
        return;
    }

    // SIMULATION MODE LOGIC
    await processLocalCommand(cmd, newHistory);
  };

  const processLocalCommand = async (cmd: string, currentHistory: string[]) => {
      const args = cmd.split(' ');
      const command = args[0].toLowerCase();

      let output: string[] = [];

      await new Promise(r => setTimeout(r, 300)); // Simulate processing delay

      switch (command) {
        case 'help':
            output = [
                'AVAILABLE COMMANDS:',
                '  status           - Check system status',
                '  query [ticker]   - Run deep dive analysis',
                '  agents           - List active agents',
                '  graph            - Visualize knowledge graph',
                '  clear            - Clear terminal',
                '  mode             - Show current operating mode'
            ];
            break;
        case 'status':
            const status = await dataManager.checkConnection();
            output = [`SYSTEM STATUS: ${status.status}`, `LATENCY: ${status.latency}ms`, `MODE: ${mode}`];
            break;
        case 'agents':
            const manifest = await dataManager.getManifest();
            output = ['--- AGENT ROSTER ---'];
            manifest.agents.slice(0, 10).forEach((a: any) =>
                output.push(`[${a.status.toUpperCase()}] ${a.name.padEnd(30)} :: ${a.specialization}`)
            );
            if (manifest.agents.length > 10) output.push(`... and ${manifest.agents.length - 10} more.`);
            break;
        case 'query':
            if (args[1]) {
                const ticker = args[1].toUpperCase();
                output = [`INITIATING DEEP DIVE SIMULATION FOR: ${ticker}...`];
                setHistory([...currentHistory, ...output]);

                // Simulate steps
                setTimeout(() => setHistory(h => [...h, `> [INFO] Fetching 10-K for ${ticker}... DONE`]), 800);
                setTimeout(() => setHistory(h => [...h, `> [INFO] Running Sentiment Analysis... DONE`]), 1600);
                setTimeout(() => setHistory(h => [...h, `> [INFO] Calculating Intrinsic Value... DONE`]), 2400);
                setTimeout(() => setHistory(h => [...h, `> [RESULT] Report generated. Check "Archives".`]), 3200);
                return; // Special async handling
            } else {
                output = ['Usage: query [ticker]'];
            }
            break;
        case 'clear':
            setHistory([]);
            return;
        case 'mode':
            output = [`CURRENT OPERATING MODE: ${mode}`];
            break;
        default:
            output = [`ERROR: UNKNOWN COMMAND "${cmd}"`];
      }

      setHistory(h => [...h, ...output.map(l => `> ${l}`)]);
  };

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  return (
    <div className="glass-panel" style={{
        height: '100%', display: 'flex', flexDirection: 'column', padding: '20px',
        fontFamily: 'var(--font-mono)', backgroundColor: '#000', border: '1px solid #333'
    }}>
      <div className="scanline" style={{position: 'absolute', top:0, left:0, right:0, bottom:0, pointerEvents: 'none'}}></div>

      <div style={{ flexGrow: 1, overflowY: 'auto', marginBottom: '10px', color: mode === 'LIVE' ? '#0f0' : '#f7d51d' }}>
        {history.map((line, i) => <div key={i} style={{marginBottom: '4px', textShadow: mode === 'LIVE' ? '0 0 5px #0f0' : 'none'}}>{line}</div>)}
        <div ref={endRef} />
      </div>

      <div style={{ display: 'flex', borderTop: '1px solid #333', paddingTop: '10px', alignItems: 'center' }}>
        <span style={{ color: mode === 'LIVE' ? '#0f0' : '#f7d51d', marginRight: '10px' }}>user@adam-{mode.toLowerCase()}:~$</span>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleCommand(input)}
          style={{
            background: 'transparent', border: 'none', color: '#fff',
            flexGrow: 1, outline: 'none', fontFamily: 'var(--font-mono)', fontSize: '1rem'
          }}
          autoFocus
          placeholder="Enter command..."
        />
      </div>
    </div>
  );
};

export default Terminal;
