"use client";
import React, { useEffect, useState, useRef } from 'react';
import { Activity, ShieldCheck, Newspaper, Code, Cpu } from 'lucide-react';

interface Thought {
  timestamp: string;
  agent_name: string;
  content: string;
  conviction_score: number;
}

const AGENT_ICONS: { [key: string]: any } = {
  "NewsBot-V30": Newspaper,
  "CodeWeaver-V30": Code,
  "RiskManager-V30": ShieldCheck,
  "SwarmOrchestrator-V30": Cpu,
  "Watchtower-V30": Activity
};

const HiveMindGrid: React.FC = () => {
  const [agentStates, setAgentStates] = useState<{ [key: string]: Thought }>({});
  const [lastUpdate, setLastUpdate] = useState<{ [key: string]: number }>({});

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream');

    ws.onmessage = (event) => {
      try {
        const data: Thought = JSON.parse(event.data);
        setAgentStates(prev => ({
          ...prev,
          [data.agent_name]: data
        }));
        setLastUpdate(prev => ({
          ...prev,
          [data.agent_name]: Date.now()
        }));
      } catch (err) {
        console.error('Failed to parse thought:', err);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  // Calculate grid columns based on number of active agents
  const agents = Object.keys(agentStates);

  return (
    <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      {agents.length === 0 && (
        <div className="col-span-full text-center text-gray-500 py-10 border border-dashed border-gray-700 rounded-lg">
          Initialize Hive Mind to see active agents...
        </div>
      )}

      {agents.map(agentName => {
        const thought = agentStates[agentName];
        const lastTime = lastUpdate[agentName] || 0;
        const timeSince = (Date.now() - lastTime) / 1000;
        const isActive = timeSince < 5; // consider active if updated in last 5s

        const Icon = AGENT_ICONS[agentName] || Activity;
        const convictionColor = thought.conviction_score > 80 ? 'text-green-400' :
                               thought.conviction_score > 50 ? 'text-yellow-400' : 'text-red-400';

        return (
          <div key={agentName} className={`
            relative p-4 rounded-xl border transition-all duration-500 overflow-hidden
            ${isActive ? 'border-green-500/50 bg-green-900/10 shadow-[0_0_15px_rgba(0,255,0,0.1)]' : 'border-gray-800 bg-gray-900/50'}
          `}>
            {isActive && (
              <div className="absolute top-0 right-0 w-2 h-2 m-2 bg-green-500 rounded-full animate-ping" />
            )}

            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 bg-gray-800 rounded-lg text-cyan-400">
                <Icon size={20} />
              </div>
              <h3 className="font-bold text-gray-200 text-sm">{agentName}</h3>
            </div>

            <div className="min-h-[3rem]">
              <p className="text-xs text-gray-400 font-mono leading-relaxed line-clamp-3">
                {thought.content}
              </p>
            </div>

            <div className="mt-4 flex items-center justify-between text-xs border-t border-gray-800 pt-2">
              <span className="text-gray-600 font-mono">
                {new Date(thought.timestamp).toLocaleTimeString()}
              </span>
              <div className="flex items-center space-x-1">
                <span className="text-gray-600">Conf:</span>
                <span className={`font-bold ${convictionColor}`}>
                  {thought.conviction_score}%
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default HiveMindGrid;
