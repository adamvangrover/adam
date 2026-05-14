import React, { useState, useEffect, useRef, memo } from 'react';
import { dataManager } from '../utils/DataManager';

interface HistoryLine {
    id: string;
    text: string;
}

// Bolt: Memoized component for history rendering.
// This prevents the entire history list (potentially hundreds of DOM nodes)
// from re-rendering on every keystroke in the input field.
const TerminalHistory = memo(({ history }: { history: HistoryLine[] }) => {
    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [history]);

    return (
        <>
            {history.map((line) => (
                <div key={line.id} style={{
                    marginBottom: '4px',
                    color: line.text.startsWith('ERROR') ? '#f00' : (line.text.includes('[INFO]') ? '#0ff' : '#0f0')
                }}>
                    {line.text}
                </div>
            ))}
            <div ref={endRef} />
        </>
    );
});

const Terminal: React.FC = () => {
  const [history, setHistory] = useState<HistoryLine[]>([
      { id: 'init-1', text: '> ADAM v23.5 KERNEL INITIALIZED' },
      { id: 'init-2', text: '> CONNECTED TO NEURO-SYMBOLIC CORE...' },
      { id: 'init-3', text: '> TYPE "help" FOR COMMANDS' }
  ]);
  const [input, setInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyPointer, setHistoryPointer] = useState<number>(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  const createLine = (text: string): HistoryLine => ({
      id: Math.random().toString(36).substr(2, 9),
      text
  });

  const handleCommand = async (cmd: string) => {
    const timestamp = new Date().toISOString().split('T')[1].slice(0,8);
    const newHistory = [...history, createLine(`[${timestamp}] $ ${cmd}`)];

    const cleanCmd = cmd.trim().toLowerCase();

    // Add to command history if not empty
    if (cmd.trim()) {
        setCommandHistory(prev => [...prev, cmd]);
        setHistoryPointer(-1);
    }

    switch (cleanCmd) {
      case 'help':
        newHistory.push(
            createLine('AVAILABLE COMMANDS:'),
            createLine('  status       - Check system connectivity and health'),
            createLine('  scan agents  - List all active agents and their tasks'),
            createLine('  query [sym]  - Initiate Deep Dive analysis for a ticker'),
            createLine('  mode [type]  - Switch mode (live/archive)'),
            createLine('  clear        - Clear terminal screen')
        );
        break;
      case 'status':
        const status = await dataManager.checkConnection();
        newHistory.push(
            createLine(`SYSTEM STATUS: ${status.status}`),
            createLine(`LATENCY: ${status.latency}ms`),
            createLine(`VERSION: ${status.version}`),
            createLine(`MODE: ${dataManager.isOfflineMode() ? 'OFFLINE / SIMULATED' : 'ONLINE / CONNECTED'}`)
        );
        break;
      case 'scan agents':
        newHistory.push(createLine('SCANNING AGENT NETWORK...'));
        const manifest = await dataManager.getManifest();
        if (manifest.agents) {
            manifest.agents.forEach(a => {
                newHistory.push(createLine(`[${a.status.toUpperCase()}] ${a.name.padEnd(20)} :: ${a.specialization}`));
            });
        } else {
            newHistory.push(createLine('NO AGENTS DETECTED.'));
        }
        break;
      case 'clear':
        setHistory([]);
        setInput('');
        return; // Early return to avoid adding more history
      default:
        if (cleanCmd.startsWith('query ')) {
            const ticker = cleanCmd.split(' ')[1]?.toUpperCase();
            if (ticker) {
                newHistory.push(createLine(`INITIATING DEEP DIVE SIMULATION FOR: ${ticker}...`));
                setTimeout(() => setHistory(h => [...h, createLine(`[INFO] Fetching 10-K for ${ticker}... DONE`)]), 500);
                setTimeout(() => setHistory(h => [...h, createLine(`[INFO] Running Sentiment Analysis (BERT)... DONE`)]), 1200);
                setTimeout(() => setHistory(h => [...h, createLine(`[INFO] Calculating Monte Carlo Risk... DONE`)]), 2000);
                setTimeout(() => setHistory(h => [...h, createLine(`[RESULT] Analysis Complete. Report generated in Vault.`)]), 2500);
            } else {
                newHistory.push(createLine('ERROR: Ticker required. Usage: query [ticker]'));
            }
        } else if (cleanCmd.startsWith('mode ')) {
             const m = cleanCmd.split(' ')[1];
             if (m === 'live') {
                 dataManager.toggleSimulationMode(false);
                 newHistory.push(createLine('SWITCHING TO LIVE MODE...'));
             } else if (m === 'archive') {
                 dataManager.toggleSimulationMode(true);
                 newHistory.push(createLine('SWITCHING TO ARCHIVE MODE...'));
             } else {
                 newHistory.push(createLine('ERROR: Unknown mode. Use "live" or "archive".'));
             }
        } else {
            newHistory.push(createLine(`ERROR: UNKNOWN COMMAND "${cmd}"`));
        }
    }
    setHistory(newHistory);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
        handleCommand(input);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (commandHistory.length === 0) return;
        const newPointer = historyPointer === -1 ? commandHistory.length - 1 : Math.max(0, historyPointer - 1);
        setHistoryPointer(newPointer);
        setInput(commandHistory[newPointer]);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (historyPointer === -1) return;
        if (historyPointer < commandHistory.length - 1) {
            const newPointer = historyPointer + 1;
            setHistoryPointer(newPointer);
            setInput(commandHistory[newPointer]);
        } else {
            setHistoryPointer(-1);
            setInput('');
        }
    } else if (e.key === 'Tab') {
        e.preventDefault();
        const commands = ['help', 'status', 'scan agents', 'query', 'mode', 'clear'];
        const match = commands.find(c => c.startsWith(input.toLowerCase()));
        if (match) setInput(match);
    }
  };

  const focusInput = () => {
      inputRef.current?.focus();
  }

  return (
    <div className="cyber-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '15px', fontFamily: 'monospace', backgroundColor: '#000', color: '#0f0', border: '1px solid #333', boxShadow: '0 0 10px rgba(0,255,0,0.1)' }} onClick={focusInput}>

      {/* Scanline Effect Overlay */}
      <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
          background: 'linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))',
          backgroundSize: '100% 2px, 3px 100%',
          pointerEvents: 'none',
          zIndex: 5
      }}></div>

      <div
        className="flex-grow overflow-y-auto mb-2.5 relative z-10 focus:outline-none focus:ring-1 focus:ring-[var(--primary-color)]"
        tabIndex={0}
        role="log"
        aria-live="polite"
        aria-label="Terminal Output"
      >
        <TerminalHistory history={history} />
      </div>
      <div style={{ display: 'flex', borderTop: '1px solid #333', paddingTop: '10px', alignItems: 'center', position: 'relative', zIndex: 1 }}>
        <span style={{ color: '#f7d51d', marginRight: '10px' }}>$</span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          aria-label="Terminal Command Input"
          placeholder="Type 'help' for commands..."
          spellCheck={false}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          style={{
            background: 'transparent', border: 'none', color: '#fff',
            flexGrow: 1, outline: 'none', fontFamily: 'monospace',
            fontSize: '1rem'
          }}
          autoFocus
          autoComplete="off"
        />
      </div>
    </div>
  );
};

export default Terminal;
