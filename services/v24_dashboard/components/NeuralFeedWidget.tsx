'use client';
import React, { useEffect, useState, useRef, memo } from 'react';

// Define TypeScript Interface matching Pydantic backend model
interface ThoughtPayload {
  timestamp: string;
  agent_name: string;
  content: string;
  conviction_score: number;
}

interface Thought extends ThoughtPayload {
  id: string; // Added for frontend optimization
}

// Determine color based on conviction score with visual flair
// Moved outside component to avoid recreation on every render
const getConvictionColor = (score: number) => {
  // Backend sends 0-100 or 0.0-1.0? Assuming 0-100 based on usage below,
  // but code handles normalization if needed.
  const normalizedScore = score <= 1 ? score * 100 : score;

  if (normalizedScore >= 80) return 'text-green-400 shadow-[0_0_8px_rgba(74,222,128,0.4)]';
  if (normalizedScore >= 50) return 'text-cyan-400';
  return 'text-red-400';
};

// Memoized item component to prevent re-renders of existing items
const ThoughtItem = memo(({ thought }: { thought: Thought }) => (
  <div className="flex gap-2 border-b border-gray-800/50 pb-1 hover:bg-white/5 transition-colors animate-in fade-in slide-in-from-bottom-1 duration-300">
    <span className="text-gray-500 w-16 shrink-0">
      [{new Date(thought.timestamp).toLocaleTimeString([], {hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit"})}]
    </span>
    <span className="text-blue-400 font-bold w-28 shrink-0 truncate" title={thought.agent_name}>
      {thought.agent_name}
    </span>
    <span className="text-gray-300 flex-1 break-words">
      {thought.content}
    </span>
    <span className={`${getConvictionColor(thought.conviction_score)} font-bold w-10 text-right shrink-0`}>
      {Math.round(thought.conviction_score <= 1 ? thought.conviction_score * 100 : thought.conviction_score)}%
    </span>
  </div>
));

ThoughtItem.displayName = 'ThoughtItem';

const NeuralFeedWidget: React.FC = () => {
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<string>('Connecting...');

  // Auto-scroll to bottom whenever new thoughts arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thoughts]);

  useEffect(() => {
    // WebSocket Connection
    const ws = new WebSocket('ws://localhost:8000/ws/stream');

    ws.onopen = () => {
      console.log('Neural Link: Connected');
      setStatus('Connected');
    };

    ws.onmessage = (event) => {
      try {
        const thoughtPayload: ThoughtPayload = JSON.parse(event.data);
        // Add unique ID for React list key optimization
        const thought: Thought = {
          ...thoughtPayload,
          id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substr(2, 9)
        };
        setThoughts((prev) => [...prev.slice(-49), thought]); // Keep last 50 items
      } catch (err) {
        console.error('Neural Link: Error parsing data', err);
      }
    };

    ws.onclose = () => {
      console.log('Neural Link: Disconnected');
      setStatus('Disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="flex flex-col h-full w-full bg-black/90 font-mono text-xs rounded border border-gray-700 p-2 overflow-hidden shadow-lg shadow-green-900/10">
      {/* Header Bar */}
      <div className="flex justify-between items-center border-b border-gray-700 pb-2 mb-2 bg-gray-900/30 px-2 -mx-2 -mt-2 pt-2">
        <span className="text-cyan-400 font-bold tracking-wider">NEURAL LINK v30.1</span>
        <div className="flex items-center space-x-2">
          <span className={`text-[10px] uppercase ${status === 'Connected' ? 'text-green-500' : 'text-red-500'}`}>
            {status}
          </span>
          <div className={`w-2 h-2 rounded-full ${status === 'Connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
        </div>
      </div>

      {/* Feed Stream */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-1 pr-1 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
        {thoughts.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-600 italic space-y-2">
            <span className="animate-pulse">Awaiting neural signals...</span>
          </div>
        )}
        
        {thoughts.map((thought) => (
          <ThoughtItem key={thought.id} thought={thought} />
        ))}
      </div>
    </div>
  );
};

export default NeuralFeedWidget;
