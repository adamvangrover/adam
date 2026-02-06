import React from 'react';
import { Cpu, MemoryStick, Activity, Server, Shield, Brain, Globe, Database } from 'lucide-react';

const AGENTS = [
  { name: 'RiskAnalyst', type: 'Specialized', status: 'Active', load: 88, icon: Shield },
  { name: 'MarketSentiment', type: 'Specialized', status: 'Active', load: 45, icon: Globe },
  { name: 'FundamentalAnalysis', type: 'Specialized', status: 'Idle', load: 12, icon: Database },
  { name: 'TechnicalAnalysis', type: 'Specialized', status: 'Active', load: 67, icon: Activity },
  { name: 'NarrativeWeaver', type: 'Utility', status: 'Active', load: 34, icon: Brain },
  { name: 'ConsensusEngine', type: 'Core', status: 'Active', load: 92, icon: Server },
  { name: 'BlindspotScanner', type: 'Specialized', status: 'Sleep', load: 5, icon: Activity },
  { name: 'Adjudicator', type: 'Core', status: 'Active', load: 78, icon: Cpu },
];

export const SystemMonitor: React.FC = () => {
  return (
    <div className="h-full flex flex-col gap-4 p-4 overflow-y-auto">
      <h2 className="text-xl font-bold text-cyan-400 font-mono border-b border-cyan-900 pb-2 mb-2 flex items-center gap-2">
        <Server /> SYSTEM KERNEL
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {AGENTS.map((agent) => (
          <div key={agent.name} className="bg-[#002b36] border border-cyan-900/50 p-4 rounded hover:border-cyan-500/50 transition-colors group">
            <div className="flex justify-between items-start mb-3">
              <div className="flex items-center gap-2">
                <div className={`p-2 rounded bg-cyan-950 ${agent.status === 'Active' ? 'text-cyan-400' : 'text-gray-600'}`}>
                  <agent.icon size={18} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-cyan-200 font-mono">{agent.name}</h3>
                  <div className="text-[10px] text-cyan-600 uppercase">{agent.type}</div>
                </div>
              </div>
              <div className={`w-2 h-2 rounded-full ${agent.status === 'Active' ? 'bg-green-500 animate-pulse' : 'bg-gray-700'}`} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-[10px] font-mono text-cyan-600">
                <span>CPU LOAD</span>
                <span>{agent.load}%</span>
              </div>
              <div className="w-full bg-cyan-950 h-1.5 rounded-full overflow-hidden">
                <div
                  className={`h-full ${agent.load > 80 ? 'bg-red-500' : agent.load > 50 ? 'bg-amber-500' : 'bg-cyan-500'} transition-all duration-1000`}
                  style={{ width: `${agent.load}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8">
        <h3 className="text-sm font-bold text-cyan-500 font-mono mb-4 uppercase tracking-widest border-b border-cyan-900/30 pb-1 w-max">Memory Allocation</h3>
        <div className="flex gap-1 h-8 w-full bg-cyan-950/30 rounded overflow-hidden">
            {Array.from({length: 50}).map((_, i) => (
                <div
                    key={i}
                    className={`flex-1 ${Math.random() > 0.7 ? 'bg-cyan-700/50' : Math.random() > 0.9 ? 'bg-amber-600/50' : 'bg-transparent'} border-r border-black/20`}
                />
            ))}
        </div>
        <div className="flex justify-between text-[10px] font-mono text-cyan-800 mt-1">
            <span>0x00000000</span>
            <span>HEAP: 45%</span>
            <span>STACK: 12%</span>
            <span>0xFFFFFFFF</span>
        </div>
      </div>
    </div>
  );
};
