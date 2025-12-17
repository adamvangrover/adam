import React, { useState, useEffect, useRef } from 'react';

const Terminal: React.FC = () => {
  const [output, setOutput] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const [isLive] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const simulateBoot = async () => {
    const bootSequence = [
      "INITIALIZING ADAM v23.5 KERNEL...",
      "LOADING NEURO-SYMBOLIC PLANNER...",
      "CONNECTING TO QUANTUM RISK ENGINE... [MOCKED]",
      "ESTABLISHING SECURE UPLINK TO OMNI-GRAPH...",
      "SYSTEM READY."
    ];
    for (const line of bootSequence) {
        setOutput(prev => [...prev, `> ${line}`]);
        await new Promise(r => setTimeout(r, 600));
    }
  };

  useEffect(() => {
    simulateBoot();
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [output]);

  const handleCommand = (cmd: string) => {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    setOutput(prev => [...prev, `[${timestamp}] $ ${cmd}`]);

    if (cmd === 'help') {
        setOutput(prev => [...prev, "AVAILABLE COMMANDS:", "  help - Show this menu", "  status - System Health", "  query [ticker] - Deep Dive", "  clear - Clear Screen"]);
    } else if (cmd === 'status') {
        setOutput(prev => [...prev, "SYSTEM STATUS: ONLINE (SIMULATION MODE)", "CPU: 65%", "MEMORY: 12GB/32GB", "AGENTS: 97 ACTIVE"]);
    } else if (cmd.startsWith('query')) {
        const ticker = cmd.split(' ')[1] || 'UNKNOWN';
        setOutput(prev => [...prev, `INITIATING DEEP DIVE FOR ${ticker.toUpperCase()}...`, "FETCHING SEC FILINGS... DONE", "RUNNING SENTIMENT ANALYSIS... DONE", "CALCULATING INTRINSIC VALUE..."]);
        setTimeout(() => {
            setOutput(prev => [...prev, `REPORT GENERATED FOR ${ticker.toUpperCase()}. SEE 'REPORTS' MODULE.`]);
        }, 2000);
    } else if (cmd === 'clear') {
        setOutput([]);
    } else {
        setOutput(prev => [...prev, `Unknown command: ${cmd}`]);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
        handleCommand(input);
        setInput('');
    }
  };

  return (
    <div className="h-[600px] bg-black border border-cyber-cyan/30 rounded-lg p-4 font-mono text-sm relative overflow-hidden shadow-[0_0_20px_rgba(6,182,212,0.1)]">
        <div className="scan-line opacity-20"></div>
        <div className="absolute top-2 right-4 text-xs text-cyber-cyan/50">
            CONNECTION: {isLive ? 'WEBSOCKET' : 'LOCAL_SIM'}
        </div>

        <div className="h-full flex flex-col">
            <div
                className="flex-1 overflow-y-auto space-y-1 text-green-400 p-2 focus:outline-none focus:ring-1 focus:ring-cyber-cyan/30"
                ref={scrollRef}
                role="log"
                aria-live="polite"
                aria-atomic="false"
                aria-label="Terminal Output"
                tabIndex={0}
            >
                {output.map((line, i) => (
                    <div key={i} className="break-all">{line}</div>
                ))}
            </div>
            <div className="mt-2 flex items-center border-t border-cyber-cyan/20 pt-2">
                <span className="text-cyber-cyan mr-2">{'>'}</span>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onKeyDown}
                    className="flex-1 bg-transparent border-none outline-none text-cyber-cyan focus:ring-0 placeholder-cyber-cyan/50 w-full"
                    placeholder="ENTER COMMAND..."
                    aria-label="Terminal Command Input"
                    autoFocus
                />
            </div>
        </div>
    </div>
  );
};

export default Terminal;
