import React, { useState, useEffect } from 'react';
import { Terminal, X, Minimize2, Maximize2 } from 'lucide-react';

interface AgentThought {
  id: number;
  agent: string;
  message: string;
  time: string;
}

const mockThoughts = [
  { agent: "Credit Risk Agent", message: "Analyzing ticker AAPL debt-to-equity ratios..." },
  { agent: "Macro Sentinel", message: "Detected yield curve inversion early signals." },
  { agent: "Fundamental Analyst", message: "Evaluating NVDA Q4 earnings forecast." },
  { agent: "Market Sentiment", message: "Bullish divergence observed on social media indices." },
  { agent: "Risk Officer", message: "Holding position: Portfolio volatility exceeding thresholds." }
];

const AgentIntercom: React.FC = () => {
  const [thoughts, setThoughts] = useState<AgentThought[]>([]);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    let idCounter = 0;
    const interval = setInterval(() => {
      const newThought = {
        id: idCounter++,
        agent: mockThoughts[Math.floor(Math.random() * mockThoughts.length)].agent,
        message: mockThoughts[Math.floor(Math.random() * mockThoughts.length)].message,
        time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
      };

      setThoughts(prev => {
        const next = [newThought, ...prev];
        return next.slice(0, 5); // Keep only the 5 most recent
      });
    }, 4000); // New thought every 4 seconds

    return () => clearInterval(interval);
  }, []);

  if (!isVisible) return null;

  return (
    <div className={`fixed bottom-6 right-6 w-80 bg-cyber-black/90 border border-cyber-cyan/30 rounded-lg shadow-[0_0_20px_rgba(6,182,212,0.2)] font-mono flex flex-col transition-all duration-300 z-50 ${isMinimized ? 'h-12' : 'h-64'}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-2 border-b border-cyber-cyan/30 bg-cyber-slate/50 rounded-t-lg">
        <div className="flex items-center gap-2 text-cyber-cyan text-sm tracking-wider">
          <Terminal size={14} />
          <span>AGENT INTERCOM</span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setIsMinimized(!isMinimized)} className="text-cyber-text/70 hover:text-cyber-cyan transition-colors">
            {isMinimized ? <Maximize2 size={14} /> : <Minimize2 size={14} />}
          </button>
          <button onClick={() => setIsVisible(false)} className="text-cyber-text/70 hover:text-cyber-cyan transition-colors">
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Feed */}
      {!isMinimized && (
        <div className="p-3 overflow-y-auto flex-grow flex flex-col-reverse gap-2 scrollbar-thin scrollbar-thumb-cyber-cyan/30 scrollbar-track-transparent">
          {thoughts.map((t) => (
            <div key={t.id} className="text-xs leading-relaxed opacity-0 animate-fade-in" style={{ animation: 'fadeIn 0.5s ease-out forwards' }}>
              <div className="text-cyber-text/40 mb-1">[{t.time}] <span className="text-cyber-cyan">{t.agent}</span>:</div>
              <div className="text-cyber-text/90 pl-2 border-l border-cyber-cyan/20">{t.message}</div>
            </div>
          ))}
          {thoughts.length === 0 && (
            <div className="text-xs text-cyber-text/50 text-center mt-8 animate-pulse">
              Awaiting agent signals...
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(5px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default AgentIntercom;
