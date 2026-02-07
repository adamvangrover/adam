import React from 'react';
import { useSwarmStore, Agent } from '../../stores/swarmStore';
import { Cpu, Zap, Activity } from 'lucide-react';

const AgentCell: React.FC<{ agent: Agent }> = ({ agent }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'WORKING': return 'border-cyan-400 bg-cyan-900/40 text-cyan-200';
      case 'COMPUTING': return 'border-amber-500 bg-amber-900/40 text-amber-200 animate-pulse';
      case 'IDLE': return 'border-slate-700 bg-slate-900/40 text-slate-500';
      default: return 'border-red-500 bg-red-900/40 text-red-200';
    }
  };

  const Icon = agent.status === 'COMPUTING' ? Zap : (agent.status === 'WORKING' ? Activity : Cpu);

  return (
    <div className={`
      relative group border p-2 flex flex-col gap-1 rounded-sm transition-all duration-300
      ${getStatusColor(agent.status)}
    `}>
      <div className="flex justify-between items-center text-[10px] font-mono">
        <span className="opacity-70">{agent.id}</span>
        <Icon size={10} />
      </div>
      <div className="text-xs font-bold truncate">
        {agent.role.substring(0, 4)}
      </div>

      {/* Tooltip */}
      <div className="absolute opacity-0 group-hover:opacity-100 bottom-full left-0 w-48 bg-black border border-cyan-500 p-2 text-[10px] z-50 pointer-events-none mb-2 shadow-xl">
        <div className="font-bold text-cyan-400">{agent.name}</div>
        <div className="text-slate-300">{agent.role}</div>
        <div className="my-1 border-t border-slate-800"></div>
        <div className="text-white italic">&gt; {agent.currentTask}</div>
        <div className="mt-1 flex justify-between text-slate-500">
           <span>EFF: {agent.efficiency}%</span>
           <span>STATUS: {agent.status}</span>
        </div>
      </div>
    </div>
  );
};

export const SwarmActivity: React.FC = () => {
  const { agents, networkLoad, consensusRate, totalCompute } = useSwarmStore();

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Header Metrics */}
      <div className="grid grid-cols-4 gap-4 mb-2">
        <div className="bg-slate-900/50 p-2 border border-slate-800 rounded">
            <div className="text-[10px] text-slate-500 uppercase">Active Agents</div>
            <div className="text-xl font-mono text-cyan-400">{agents.filter(a => a.status !== 'IDLE').length}/{agents.length}</div>
        </div>
        <div className="bg-slate-900/50 p-2 border border-slate-800 rounded">
            <div className="text-[10px] text-slate-500 uppercase">Network Load</div>
            <div className="text-xl font-mono text-amber-400">{networkLoad.toFixed(1)}%</div>
        </div>
        <div className="bg-slate-900/50 p-2 border border-slate-800 rounded">
            <div className="text-[10px] text-slate-500 uppercase">Consensus</div>
            <div className="text-xl font-mono text-emerald-400">{consensusRate.toFixed(2)}%</div>
        </div>
        <div className="bg-slate-900/50 p-2 border border-slate-800 rounded">
            <div className="text-[10px] text-slate-500 uppercase">Est. Compute</div>
            <div className="text-xl font-mono text-purple-400">{totalCompute.toFixed(1)} TF</div>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 overflow-y-auto pr-2 scrollbar-hide flex-1">
        {agents.map(agent => (
          <AgentCell key={agent.id} agent={agent} />
        ))}
      </div>

      <div className="text-[10px] text-slate-600 font-mono text-center">
        SWARM ORCHESTRATION LAYER v2.1 // CONNECTED
      </div>
    </div>
  );
};
