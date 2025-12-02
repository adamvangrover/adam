import React, { useEffect, useState, useRef } from 'react';
import GlassCard from '../common/GlassCard';
import { Activity, Cpu, Database, Server, AlertTriangle } from 'lucide-react';
import { dataManager } from '../../utils/DataManager';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const TerminalLog = ({ logs }) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="font-mono text-xs bg-slate-950 p-4 rounded-lg border border-slate-800 h-48 overflow-y-auto text-emerald-500/80 shadow-inner">
            {logs.map((log, i) => (
                <div key={i} className="mb-1">
                    <span className="text-slate-500">[{log.timestamp}]</span> <span className={log.type === 'error' ? 'text-rose-500' : 'text-emerald-400'}>{log.message}</span>
                </div>
            ))}
            <div ref={bottomRef} />
        </div>
    );
};

const MissionControl = () => {
    const [health, setHealth] = useState(null);
    const [agents, setAgents] = useState([]);
    const [logs, setLogs] = useState([]);

    const addLog = (message, type = 'info') => {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prev => [...prev.slice(-50), { timestamp, message, type }]);
    };

    useEffect(() => {
        const fetchData = async () => {
            addLog("Initializing Mission Control...");

            const healthData = await dataManager.getData('/system/health');
            if (healthData) {
                setHealth(healthData);
                addLog(`System Health loaded. CPU: ${healthData.cpu}%`);
            } else {
                addLog("Failed to load System Health", "error");
            }

            const agentsData = await dataManager.getData('/agents/status');
            if (agentsData) {
                setAgents(agentsData);
                addLog(`Loaded ${agentsData.length} active agents.`);
            }

            // Market Data is used in other modules, but we can verify it loads
            const mData = await dataManager.getData('/market/data');
            if (mData) {
                addLog("Market Baseline Data acquired.");
            }
        };

        fetchData();

        // Polling simulation
        const interval = setInterval(() => {
             // Randomly generate logs to simulate activity
             const actions = ["Scanning news feeds...", "Re-calibrating risk models...", "Optimizing knowledge graph...", "Fetching SEC filings..."];
             const randomAction = actions[Math.floor(Math.random() * actions.length)];
             if (Math.random() > 0.7) addLog(randomAction);
        }, 3000);

        return () => clearInterval(interval);
    }, []);

    const chartData = {
        labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
        datasets: [
          {
            label: 'System Load',
            data: [30, 45, 55, 60, 40, 75, 50],
            fill: true,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            tension: 0.4,
          },
        ],
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: { display: false },
        },
        scales: {
            y: { display: false },
            x: { grid: { display: false, drawBorder: false }, ticks: { color: '#64748b' } }
        },
        maintainAspectRatio: false
    };

    return (
        <div className="space-y-6">
            {/* Top Stats Row */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <GlassCard className="flex items-center space-x-4">
                    <div className="p-3 bg-cyan-500/20 rounded-lg text-cyan-400">
                        <Cpu size={24} />
                    </div>
                    <div>
                        <p className="text-slate-400 text-xs uppercase tracking-wider">CPU Load</p>
                        <h3 className="text-2xl font-mono font-bold text-white">{health?.cpu || 45}%</h3>
                    </div>
                </GlassCard>
                <GlassCard className="flex items-center space-x-4">
                    <div className="p-3 bg-emerald-500/20 rounded-lg text-emerald-400">
                        <Activity size={24} />
                    </div>
                    <div>
                        <p className="text-slate-400 text-xs uppercase tracking-wider">Active Agents</p>
                        <h3 className="text-2xl font-mono font-bold text-white">{agents.length || 0}</h3>
                    </div>
                </GlassCard>
                 <GlassCard className="flex items-center space-x-4">
                    <div className="p-3 bg-purple-500/20 rounded-lg text-purple-400">
                        <Database size={24} />
                    </div>
                    <div>
                        <p className="text-slate-400 text-xs uppercase tracking-wider">Knowledge Nodes</p>
                        <h3 className="text-2xl font-mono font-bold text-white">12.5k</h3>
                    </div>
                </GlassCard>
                <GlassCard className="flex items-center space-x-4">
                     <div className="p-3 bg-rose-500/20 rounded-lg text-rose-400">
                        <AlertTriangle size={24} />
                    </div>
                    <div>
                        <p className="text-slate-400 text-xs uppercase tracking-wider">Global Risk</p>
                        <h3 className="text-2xl font-mono font-bold text-white">{health?.risk_level || "MODERATE"}</h3>
                    </div>
                </GlassCard>
            </div>

            {/* Main Dashboard Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Main Visualizer Area */}
                <GlassCard className="lg:col-span-2 min-h-[400px] flex flex-col">
                    <div className="flex justify-between items-center mb-6">
                        <h2 className="text-lg font-semibold text-white flex items-center">
                            <Server className="mr-2 text-cyan-500" size={20} />
                            System Load & Throughput
                        </h2>
                         <div className="flex space-x-2">
                            <span className="w-3 h-3 rounded-full bg-cyan-500 animate-pulse"></span>
                            <span className="text-xs text-cyan-500 font-mono">LIVE</span>
                        </div>
                    </div>
                    <div className="flex-1 w-full h-full">
                         <Line options={chartOptions} data={chartData} />
                    </div>
                </GlassCard>

                {/* Agent Status Panel */}
                <div className="space-y-6">
                    <GlassCard>
                        <h2 className="text-lg font-semibold text-white mb-4">Agent Status</h2>
                        <div className="space-y-3">
                            {agents.map((agent) => (
                                <div key={agent.id} className="flex justify-between items-center p-2 rounded bg-slate-800/50 border border-slate-700/50">
                                    <div className="flex flex-col">
                                        <span className="text-sm font-medium text-slate-200">{agent.name}</span>
                                        <span className="text-xs text-slate-500">{agent.task}</span>
                                    </div>
                                    <div className={`w-2 h-2 rounded-full ${agent.status === 'active' ? 'bg-emerald-500 shadow-lg shadow-emerald-500/50' : 'bg-amber-500'}`}></div>
                                </div>
                            ))}
                            {agents.length === 0 && <div className="text-slate-500 text-sm">Loading agent registry...</div>}
                        </div>
                    </GlassCard>

                     {/* System Terminal */}
                     <div className="relative group">
                         <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg blur opacity-30 group-hover:opacity-75 transition duration-1000 group-hover:duration-200"></div>
                         <div className="relative">
                            <TerminalLog logs={logs} />
                         </div>
                     </div>
                </div>
            </div>
        </div>
    );
};

export default MissionControl;
