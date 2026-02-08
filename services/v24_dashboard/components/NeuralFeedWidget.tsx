'use client';
import React, { useEffect, useState, useRef } from 'react';

// Step 3: Define TypeScript Interface matching Pydantic
interface Thought {
  timestamp: string;
  agent_name: string;
  content: string;
  conviction_score: number;
}

const NeuralFeedWidget: React.FC = () => {
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<string>('Connecting...');

  useEffect(() => {
    // WebSocket Connection
    const ws = new WebSocket('ws://localhost:8000/ws/stream');

    ws.onopen = () => {
      console.log('Neural Link: Connected');
      setStatus('Connected');
    };

    ws.onmessage = (event) => {
      try {
        const thought: Thought = JSON.parse(event.data);
        setThoughts((prev) => [...prev.slice(-49), thought]); // Keep last 50

        // Auto-scroll to bottom
        setTimeout(() => {
             if (scrollRef.current) {
                scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
             }
        }, 50);

      } catch (err) {
        console.error('Neural Link: Error parsing data', err);
      }
    };

    ws.onclose = () => {
      console.log('Neural Link: Disconnected');
      setStatus('Disconnected');
      // Simple reconnection logic could go here
    };

    return () => {
      ws.close();
    };
  }, []);

  const getConvictionColor = (score: number) => {
    if (score > 0.8) return 'text-green-400';
    if (score > 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="flex flex-col h-full w-full bg-black/90 font-mono text-xs rounded border border-gray-700 p-2 overflow-hidden shadow-lg shadow-green-900/10">
      <div className="flex justify-between items-center border-b border-gray-700 pb-2 mb-2">
        <span className="text-cyan-400 font-bold tracking-wider">NEURAL LINK v30.1</span>
        <span className={`text-[10px] ${status === 'Connected' ? 'text-green-500' : 'text-red-500'}`}>
          ● {status}
        </span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-1 pr-1 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
        {thoughts.map((thought, idx) => (
          <div key={idx} className="flex gap-2 border-b border-gray-800/50 pb-1 hover:bg-white/5 transition-colors">
            <span className="text-gray-500 w-16 shrink-0">[{new Date(thought.timestamp).toLocaleTimeString([], {hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit"})}]</span>
            <span className="text-blue-400 font-bold w-24 shrink-0 truncate" title={thought.agent_name}>{thought.agent_name}:</span>
            <span className="text-gray-300 flex-1 break-words">{thought.content}</span>
            <span className={`${getConvictionColor(thought.conviction_score)} font-bold w-10 text-right shrink-0`}>
              {Math.round(thought.conviction_score * 100)}%
            </span>
          </div>
        ))}
        {thoughts.length === 0 && (
          <div className="text-gray-600 italic text-center mt-10">Waiting for intelligence stream...</div>
        )}
      </div>
    </div>
  );
};

export default NeuralFeedWidget;
