import React, { useEffect, useState } from 'react';
import { dataManager } from '../utils/DataManager';
import { Users } from 'lucide-react';

const AgentRegistry: React.FC = () => {
  const [agents, setAgents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAgents = async () => {
        const manifest = await dataManager.getManifest();
        setAgents(manifest.agents || []);
        setLoading(false);
    };
    fetchAgents();
  }, []);

  if (loading) return (
    <div className="text-cyber-cyan animate-pulse flex items-center gap-2" role="status" aria-live="polite">
        SCANNING AGENT NETWORK...
    </div>
  );

  return (
    <div>
      <h2 className="text-2xl font-bold text-cyber-cyan mb-6 flex items-center gap-2">
        <Users className="h-6 w-6" />
        AGENT REGISTRY
      </h2>

      {agents.length === 0 ? (
        <div
          className="glass-panel p-12 flex flex-col items-center justify-center text-center space-y-4 rounded border-dashed border-2 border-cyber-cyan/20"
          role="status"
          aria-live="polite"
        >
          <div className="p-4 bg-cyber-cyan/5 rounded-full ring-1 ring-cyber-cyan/20">
            <Users className="h-12 w-12 text-cyber-cyan/50" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-bold text-cyber-cyan tracking-wide">NO AGENTS DETECTED</h3>
            <p className="text-cyber-text/60 max-w-md font-mono text-sm leading-relaxed">
              The agent registry is currently empty. Initialize the simulation to register active agents.
            </p>
          </div>
        </div>
      ) : (
        <div className="overflow-x-auto glass-panel rounded">
          <table className="min-w-full text-left text-sm font-mono">
            <thead className="bg-cyber-cyan/10 text-cyber-cyan uppercase tracking-wider border-b border-cyber-cyan/20">
              <tr>
                <th className="px-6 py-3">Agent ID / Class</th>
                <th className="px-6 py-3">Type</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3">Capabilities</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-cyber-cyan/10">
              {agents.map((agent, idx) => (
                <tr key={idx} className="hover:bg-cyber-cyan/5 transition-colors">
                  <td className="px-6 py-4 font-bold text-white">
                    {agent.name}
                    <div className="text-[10px] text-cyber-text/50 font-normal">{agent.path}</div>
                  </td>
                  <td className="px-6 py-4 text-cyber-text">
                    <span className={`px-2 py-0.5 rounded text-[10px] border ${agent.type === 'Core' ? 'border-purple-500 text-purple-400' : 'border-blue-500 text-blue-400'}`}>
                      {agent.type?.toUpperCase() || 'UNKNOWN'}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-cyber-success animate-pulse"></div>
                      <span className="text-cyber-success text-xs">ACTIVE</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-cyber-text/70 text-xs max-w-md truncate" title={agent.docstring}>
                    {agent.docstring?.split('\n')[0] || 'No capability definition found.'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AgentRegistry;
