"use client";
import React, { useEffect, useState, useRef } from 'react';

interface Thought {
  timestamp: string;
  agent_name: string;
  content: string;
  conviction_score: number;
}

const NeuralFeedWidget: React.FC = () => {
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Create WebSocket connection
    const ws = new WebSocket('ws://localhost:8000/ws/stream');

    ws.onopen = () => {
      console.log('Connected to Neural Link');
    };

    ws.onmessage = (event) => {
      try {
        const data: Thought = JSON.parse(event.data);
        setThoughts((prev) => [...prev, data].slice(-50)); // Keep last 50 thoughts
      } catch (err) {
        console.error('Failed to parse thought:', err);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from Neural Link');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thoughts]);

  return (
    <div className="flex flex-col h-64 w-full bg-black border border-green-900 rounded-lg overflow-hidden font-mono text-xs shadow-[0_0_10px_rgba(0,255,0,0.1)]">
      <div className="flex items-center justify-between px-3 py-1 bg-green-900/20 border-b border-green-900/50">
        <span className="text-green-500 font-bold tracking-wider">NEURAL_LINK_V30</span>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${thoughts.length > 0 ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-green-700">LIVE</span>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin scrollbar-thumb-green-900 scrollbar-track-black"
      >
        {thoughts.length === 0 && (
          <div className="text-green-900 italic text-center mt-20">Waiting for neural signals...</div>
        )}

        {thoughts.map((thought, idx) => {
          const isHighConviction = thought.conviction_score >= 80;
          const isLowConviction = thought.conviction_score < 50;

          let colorClass = "text-green-400"; // Default
          if (isLowConviction) colorClass = "text-red-500";
          else if (isHighConviction) colorClass = "text-emerald-300 shadow-[0_0_5px_rgba(16,185,129,0.5)]";

          return (
            <div key={idx} className="flex space-x-2 animate-in fade-in duration-300">
              <span className="text-green-800 shrink-0">[{new Date(thought.timestamp).toLocaleTimeString()}]</span>
              <span className="text-cyan-600 font-semibold shrink-0">@{thought.agent_name}:</span>
              <span className={`${colorClass} break-words`}>
                {thought.content}
                <span className="text-xs opacity-50 ml-2">({thought.conviction_score}%)</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default NeuralFeedWidget;
