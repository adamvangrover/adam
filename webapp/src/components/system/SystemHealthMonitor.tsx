import React, { useState, useEffect } from 'react';
import { Activity, Cpu, Server, AlertCircle, CheckCircle } from 'lucide-react';

// Types for our monitoring data
interface SystemStatus {
  cpuUsage: number;
  memoryUsage: number;
  activeAgents: number;
  queueSize: number;
  status: 'healthy' | 'degraded' | 'critical';
}

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR';
  source: string;
  message: string;
}

const SystemHealthMonitor: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    cpuUsage: 0,
    memoryUsage: 0,
    activeAgents: 0,
    queueSize: 0,
    status: 'healthy',
  });

  const [logs, setLogs] = useState<LogEntry[]>([]);

  // Simulate fetching data - In production, replace with WebSocket or polling API
  useEffect(() => {
    const interval = setInterval(() => {
      // Mock System Status Update
      setSystemStatus({
        cpuUsage: Math.floor(Math.random() * 30) + 10,
        memoryUsage: Math.floor(Math.random() * 40) + 20,
        activeAgents: Math.floor(Math.random() * 5),
        queueSize: Math.floor(Math.random() * 10),
        status: 'healthy',
      });

      // Mock Log Entry
      const newLog: LogEntry = {
        id: Math.random().toString(36).substr(2, 9),
        timestamp: new Date().toLocaleTimeString(),
        level: Math.random() > 0.9 ? 'WARN' : 'INFO',
        source: ['Orchestrator', 'GraphEngine', 'RiskAgent'][Math.floor(Math.random() * 3)],
        message: 'Processing operational cycle...'
      };

      setLogs(prev => [newLog, ...prev].slice(0, 50)); // Keep last 50 logs
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'degraded': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'INFO': return 'text-blue-400';
      case 'WARN': return 'text-yellow-400';
      case 'ERROR': return 'text-red-400';
      default: return 'text-gray-300';
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
      {/* System Metrics Panel */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 shadow-lg lg:col-span-1">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" />
            System Health
          </h3>
          <span className={`flex items-center gap-1 text-sm font-medium ${getStatusColor(systemStatus.status)}`}>
            {systemStatus.status === 'healthy' ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
            {systemStatus.status.toUpperCase()}
          </span>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>CPU Usage</span>
              <span>{systemStatus.cpuUsage}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-indigo-500 h-2 rounded-full transition-all duration-500" 
                style={{ width: `${systemStatus.cpuUsage}%` }}
              ></div>
            </div>
          </div>

          <div>
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Memory Usage</span>
              <span>{systemStatus.memoryUsage}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full transition-all duration-500" 
                style={{ width: `${systemStatus.memoryUsage}%` }}
              ></div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="bg-gray-700/50 p-3 rounded border border-gray-600 text-center">
              <div className="text-2xl font-bold text-white">{systemStatus.activeAgents}</div>
              <div className="text-xs text-gray-400">Active Agents</div>
            </div>
            <div className="bg-gray-700/50 p-3 rounded border border-gray-600 text-center">
              <div className="text-2xl font-bold text-white">{systemStatus.queueSize}</div>
              <div className="text-xs text-gray-400">Tasks Queued</div>
            </div>
          </div>
        </div>
      </div>

      {/* Live Logs Panel */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 shadow-lg lg:col-span-2 flex flex-col h-80">
        <h3 className="text-lg font-semibold text-gray-100 mb-4 flex items-center gap-2">
          <Server className="w-5 h-5 text-indigo-400" />
          Live System Logs
        </h3>
        
        <div className="flex-1 overflow-y-auto bg-gray-900 rounded p-4 font-mono text-sm custom-scrollbar">
          {logs.map((log) => (
            <div key={log.id} className="mb-1 border-b border-gray-800 pb-1 last:border-0">
              <span className="text-gray-500 mr-3">[{log.timestamp}]</span>
              <span className={`font-bold mr-3 ${getLogLevelColor(log.level)}`}>{log.level}</span>
              <span className="text-indigo-300 mr-3">[{log.source}]</span>
              <span className="text-gray-300">{log.message}</span>
            </div>
          ))}
          {logs.length === 0 && (
            <div className="text-gray-500 text-center mt-10">Waiting for system events...</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SystemHealthMonitor;
