import React from 'react';

interface StatusPanelProps {
  metrics: {
    cpu_usage?: number;
    memory_usage?: number;
    active_agents?: number;
    mesh_latency_ms?: number;
  };
}

const StatusPanel: React.FC<StatusPanelProps> = ({ metrics }) => {
  if (!metrics) return null;

  return (
    <div className="flex gap-4 text-xs font-mono text-gray-300">
      <div className="flex flex-col items-center">
        <span className="text-[10px] text-gray-500 uppercase">CPU</span>
        <span className={(metrics.cpu_usage || 0) > 70 ? 'text-red-400' : 'text-green-400'}>
          {metrics.cpu_usage || 0}%
        </span>
      </div>
      <div className="flex flex-col items-center">
        <span className="text-[10px] text-gray-500 uppercase">MEM</span>
        <span className={(metrics.memory_usage || 0) > 80 ? 'text-orange-400' : 'text-blue-400'}>
          {metrics.memory_usage || 0}%
        </span>
      </div>
      <div className="flex flex-col items-center">
        <span className="text-[10px] text-gray-500 uppercase">AGENTS</span>
        <span className="text-purple-400 font-bold">{metrics.active_agents || 0}</span>
      </div>
      <div className="flex flex-col items-center">
        <span className="text-[10px] text-gray-500 uppercase">PING</span>
        <span className={(metrics.mesh_latency_ms || 0) > 100 ? 'text-red-400' : 'text-gray-400'}>
          {metrics.mesh_latency_ms || 0}ms
        </span>
      </div>
    </div>
  );
};

export default StatusPanel;
