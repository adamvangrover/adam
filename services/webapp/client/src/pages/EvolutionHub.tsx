import React, { useState, useEffect, useMemo } from 'react';
import { 
    GitBranch, 
    BarChart3, 
    ShieldAlert, 
    CheckCircle, 
    Activity, 
    Terminal, 
    Cpu, 
    Network, 
    Share2, 
    Zap,
    Box,
    Layers,
    Server,
    Users
} from 'lucide-react';

// --- Types ---

interface AgentDef {
    id: string;
    name: string;
    role: string;
    type: 'fiduciary' | 'analyst' | 'executor' | 'observer';
}

interface VersionMetric {
    label: string;
    value: number;
    unit: string;
    delta?: number; // percent change
}

interface SystemVersion {
    id: string;
    version: string;
    codename: string;
    date: string;
    status: 'DEPRECATED' | 'ARCHIVED' | 'ACTIVE_LTS' | 'ALPHA';
    description: string;
    architectureType: 'monolith' | 'async' | 'graph' | 'swarm';
    highlights: string[];
    metrics: VersionMetric[];
    agents: AgentDef[];
    constraints: string[];
}

// --- Data ---

const AGENT_ROSTER: Record<string, AgentDef> = {
    'market_analyst': { id: 'market_analyst', name: 'Market Analyst', role: 'Data Ingestion', type: 'analyst' },
    'risk_guardian': { id: 'risk_guardian', name: 'Risk Guardian', role: 'Compliance', type: 'fiduciary' },
    'executor': { id: 'executor', name: 'Trade Executor', role: 'Execution', type: 'executor' },
    'sentinel': { id: 'sentinel', name: 'Sentinel', role: 'Monitoring', type: 'observer' },
    'planner': { id: 'planner', name: 'Neuro-Symbolic Planner', role: 'Orchestration', type: 'fiduciary' },
    'swarm_lead': { id: 'swarm_lead', name: 'Swarm Lead', role: 'Coordination', type: 'fiduciary' },
    'coder': { id: 'coder', name: 'Code Weaver', role: 'Development', type: 'executor' },
    'researcher': { id: 'researcher', name: 'Deep Dive Researcher', role: 'Analysis', type: 'analyst' },
};

const VERSIONS: SystemVersion[] = [
    {
        id: 'v21',
        version: "v21.0",
        codename: "Monolith",
        date: "Q3 2023",
        status: "DEPRECATED",
        description: "Single-threaded reasoning loop with limited tool use and static knowledge base.",
        architectureType: 'monolith',
        highlights: [
            "Sequential Chain-of-Thought",
            "Local Vector Store",
            "Basic Tool Use (Search, Calc)"
        ],
        metrics: [
            { label: "Throughput", value: 12, unit: "tps" },
            { label: "Hallucination Rate", value: 15, unit: "%" },
            { label: "Avg Latency", value: 4.5, unit: "s" }
        ],
        agents: [AGENT_ROSTER['market_analyst'], AGENT_ROSTER['executor']],
        constraints: ["Single Thread", "No Memory Persistence"]
    },
    {
        id: 'v22',
        version: "v22.0",
        codename: "Async Flow",
        date: "Q1 2024",
        status: "ARCHIVED",
        description: "Introduction of RabbitMQ for non-blocking operations. First multi-agent prototypes.",
        architectureType: 'async',
        highlights: [
            "Event-Driven Architecture",
            "Shared Postgres Memory",
            "Parallel Execution"
        ],
        metrics: [
            { label: "Throughput", value: 45, unit: "tps", delta: 275 },
            { label: "Hallucination Rate", value: 8, unit: "%", delta: -46 },
            { label: "Avg Latency", value: 2.1, unit: "s", delta: -53 }
        ],
        agents: [AGENT_ROSTER['market_analyst'], AGENT_ROSTER['risk_guardian'], AGENT_ROSTER['executor'], AGENT_ROSTER['sentinel']],
        constraints: ["Async Complexity", "Race Conditions"]
    },
    {
        id: 'v23',
        version: "v23.0",
        codename: "Adaptive Graph",
        date: "Q3 2024",
        status: "ACTIVE_LTS",
        description: "Cyclical reasoning graphs (LangGraph). Neuro-symbolic planning with self-correction.",
        architectureType: 'graph',
        highlights: [
            "Cyclical Reasoning (Loops)",
            "Self-Correction & Reflection",
            "Dynamic Tool Selection"
        ],
        metrics: [
            { label: "Throughput", value: 38, unit: "tps", delta: -15 }, // Slower but smarter
            { label: "Hallucination Rate", value: 1.2, unit: "%", delta: -85 },
            { label: "Reasoning Depth", value: 8, unit: "steps" }
        ],
        agents: [AGENT_ROSTER['planner'], AGENT_ROSTER['market_analyst'], AGENT_ROSTER['risk_guardian'], AGENT_ROSTER['researcher'], AGENT_ROSTER['executor']],
        constraints: ["High Token Cost", "Complex State Management"]
    },
    {
        id: 'v24',
        version: "v2.0 Swarm",
        codename: "Hive Mind",
        date: "Q1 2025",
        status: "ALPHA",
        description: "Distributed swarm intelligence. Emergent behavior. Real-time consensus protocols.",
        architectureType: 'swarm',
        highlights: [
            "Decentralized Decision Making",
            "Swarm Consensus Protocols",
            "Evolutionary Optimization"
        ],
        metrics: [
            { label: "Throughput", value: 120, unit: "tps", delta: 215 },
            { label: "System Entropy", value: 0.4, unit: "H" },
            { label: "Agent Autonomy", value: 95, unit: "%" }
        ],
        agents: Object.values(AGENT_ROSTER),
        constraints: ["Network Overhead", "Fiduciary Verification Hard"]
    }
];

// --- Sub-Components ---

const ArchitectureViz = ({ type }: { type: string }) => {
    // Simple SVG representation of the architecture types
    if (type === 'monolith') {
        return (
            <svg viewBox="0 0 200 120" className="w-full h-48 opacity-80 hover:opacity-100 transition-opacity">
                <rect x="60" y="20" width="80" height="80" rx="4" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
                <text x="100" y="65" textAnchor="middle" fill="#93c5fd" fontSize="12">Monolith</text>
                <circle cx="100" cy="40" r="4" fill="#60a5fa" />
                <circle cx="100" cy="90" r="4" fill="#60a5fa" />
                <path d="M100 44 L100 86" stroke="#60a5fa" strokeWidth="2" strokeDasharray="4 2" />
            </svg>
        );
    }
    if (type === 'async') {
        return (
            <svg viewBox="0 0 200 120" className="w-full h-48 opacity-80 hover:opacity-100 transition-opacity">
                <rect x="20" y="40" width="40" height="40" rx="4" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
                <rect x="140" y="40" width="40" height="40" rx="4" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
                <rect x="80" y="50" width="40" height="20" rx="2" fill="#a855f7" fillOpacity="0.2" stroke="#a855f7" strokeWidth="2" />
                <path d="M60 60 L80 60" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrow)" />
                <path d="M120 60 L140 60" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrow)" />
                <text x="100" y="30" textAnchor="middle" fill="#e2e8f0" fontSize="10">Event Bus</text>
            </svg>
        );
    }
    if (type === 'graph') {
        return (
            <svg viewBox="0 0 200 120" className="w-full h-48 opacity-80 hover:opacity-100 transition-opacity">
                <circle cx="100" cy="60" r="25" fill="#10b981" fillOpacity="0.1" stroke="#10b981" strokeWidth="2" />
                <circle cx="40" cy="60" r="15" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
                <circle cx="160" cy="60" r="15" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
                <circle cx="100" cy="20" r="10" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b" strokeWidth="2" />
                
                {/* Edges */}
                <path d="M55 60 L75 60" stroke="#94a3b8" strokeWidth="2" />
                <path d="M125 60 L145 60" stroke="#94a3b8" strokeWidth="2" />
                <path d="M100 30 L100 35" stroke="#94a3b8" strokeWidth="2" />
                <path d="M100 85 Q 40 85 40 75" fill="none" stroke="#a855f7" strokeWidth="1" strokeDasharray="3 3" />
            </svg>
        );
    }
    return (
        <svg viewBox="0 0 200 120" className="w-full h-48 opacity-80 hover:opacity-100 transition-opacity">
            {[...Array(8)].map((_, i) => (
                <circle 
                    key={i}
                    cx={100 + 40 * Math.cos(i * Math.PI / 4)} 
                    cy={60 + 30 * Math.sin(i * Math.PI / 4)} 
                    r="6" 
                    fill={i % 2 === 0 ? "#a855f7" : "#3b82f6"} 
                    className="animate-pulse"
                    style={{ animationDelay: `${i * 0.1}s` }}
                />
            ))}
            <circle cx="100" cy="60" r="12" fill="#e2e8f0" fillOpacity="0.1" stroke="#e2e8f0" strokeWidth="1" />
            {[...Array(8)].map((_, i) => (
                <line 
                    key={`line-${i}`}
                    x1="100" y1="60"
                    x2={100 + 40 * Math.cos(i * Math.PI / 4)}
                    y2={60 + 30 * Math.sin(i * Math.PI / 4)}
                    stroke="#475569" strokeWidth="0.5"
                />
            ))}
        </svg>
    );
}

// --- Main Component ---

const EvolutionHub = () => {
    const [selectedVersionId, setSelectedVersionId] = useState<string>('v23'); // Default to current LTS
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    const selectedVersion = useMemo(() => 
        VERSIONS.find(v => v.id === selectedVersionId) || VERSIONS[2], 
    [selectedVersionId]);

    const isCurrent = selectedVersion.status === 'ACTIVE_LTS' || selectedVersion.status === 'ALPHA';

    return (
        <div className="min-h-screen bg-[#050b14] text-slate-200 font-mono relative overflow-hidden flex flex-col">
            {/* Custom CSS */}
            <style>{`
                @keyframes scan {
                    0% { top: 0%; }
                    100% { top: 100%; }
                }
                .scan-line {
                    width: 100%;
                    height: 2px;
                    background: rgba(168, 85, 247, 0.3);
                    opacity: 0.5;
                    animation: scan 4s linear infinite;
                    position: fixed;
                    top: 0;
                    pointer-events: none;
                    z-index: 50;
                }
                .glass-panel {
                    background: rgba(13, 20, 31, 0.7);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(59, 130, 246, 0.15);
                    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                }
                .glass-panel-active {
                    border: 1px solid rgba(168, 85, 247, 0.4);
                    box-shadow: 0 0 20px rgba(168, 85, 247, 0.15);
                }
                .text-glow {
                    text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
                }
                /* Scrollbar */
                ::-webkit-scrollbar { width: 8px; }
                ::-webkit-scrollbar-track { background: #0f172a; }
                ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
                ::-webkit-scrollbar-thumb:hover { background: #475569; }
            `}</style>

            <div className="scan-line"></div>

            {/* Top Navigation Bar (Mock) */}
            <nav className="border-b border-slate-800 bg-[#020617] px-6 py-3 flex justify-between items-center z-10 sticky top-0">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
                        A
                    </div>
                    <span className="font-bold text-lg tracking-tight">ADAM Platform</span>
                    <span className="bg-slate-800 text-xs px-2 py-0.5 rounded text-slate-400">v23.5.0</span>
                </div>
                <div className="flex gap-4 text-sm text-slate-400">
                    <div className="flex items-center gap-2 hover:text-white cursor-pointer transition-colors">
                        <Terminal size={14} /> Documentation
                    </div>
                    <div className="flex items-center gap-2 hover:text-white cursor-pointer transition-colors">
                        <Activity size={14} /> System Status
                    </div>
                </div>
            </nav>

            <div className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-0">
                
                {/* Left Sidebar: Timeline Selection */}
                <div className="lg:col-span-3 space-y-6">
                    <div className="glass-panel p-6 rounded-xl">
                        <h2 className="text-sm font-bold text-slate-400 mb-6 flex items-center gap-2 uppercase tracking-wider">
                            <GitBranch className="w-4 h-4 text-blue-400" />
                            Version History
                        </h2>
                        
                        <div className="relative border-l border-slate-700 ml-2 space-y-0">
                            {VERSIONS.map((v, idx) => {
                                const isActive = v.id === selectedVersionId;
                                return (
                                    <div 
                                        key={v.id} 
                                        className={`pl-6 pb-8 relative cursor-pointer group transition-all duration-300 ${idx === VERSIONS.length - 1 ? 'pb-0' : ''}`}
                                        onClick={() => setSelectedVersionId(v.id)}
                                    >
                                        {/* Timeline Dot */}
                                        <div className={`absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full border-2 transition-all duration-300 z-10 ${
                                            isActive 
                                                ? 'bg-purple-500 border-purple-500 scale-125 shadow-[0_0_10px_rgba(168,85,247,0.8)]' 
                                                : 'bg-[#050b14] border-slate-600 group-hover:border-purple-400'
                                        }`}></div>

                                        <div className={`flex flex-col transition-all ${isActive ? 'translate-x-1' : ''}`}>
                                            <span className={`text-lg font-bold transition-colors ${
                                                isActive ? 'text-purple-400 text-glow' : 'text-slate-400 group-hover:text-slate-200'
                                            }`}>
                                                {v.version}
                                            </span>
                                            <span className="text-xs text-slate-500">{v.date}</span>
                                            
                                            {/* Status Badge */}
                                            <div className="mt-2">
                                                <span className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${
                                                    v.status === 'ACTIVE_LTS' ? 'bg-emerald-500/20 text-emerald-300' :
                                                    v.status === 'ALPHA' ? 'bg-purple-500/20 text-purple-300' :
                                                    'bg-slate-800 text-slate-500'
                                                }`}>
                                                    {v.status}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                    
                    {/* Mini Constraints Panel */}
                    <div className="glass-panel p-5 rounded-xl border-red-900/20">
                        <h3 className="text-xs font-bold text-red-400 mb-3 flex items-center gap-2 uppercase tracking-wider">
                            <ShieldAlert className="w-3 h-3" />
                            Active Constraints
                        </h3>
                        <div className="space-y-2">
                            {selectedVersion.constraints.map((c, i) => (
                                <div key={i} className="flex items-center gap-2 text-xs text-slate-300 bg-slate-900/50 p-2 rounded border border-slate-800/50">
                                    <div className="w-1.5 h-1.5 rounded-full bg-red-500"></div>
                                    {c}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="lg:col-span-9 space-y-6">
                    
                    {/* Hero Header for Version */}
                    <div className={`glass-panel p-8 rounded-xl transition-all duration-500 ${isCurrent ? 'glass-panel-active' : ''}`}>
                        <div className="flex flex-col md:flex-row justify-between md:items-center gap-4 mb-6">
                            <div>
                                <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                                    {selectedVersion.version}
                                    <span className="text-slate-500 font-normal">//</span>
                                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
                                        {selectedVersion.codename}
                                    </span>
                                </h1>
                                <p className="text-slate-400 max-w-2xl">{selectedVersion.description}</p>
                            </div>
                            
                            {/* Architecture Icon */}
                            <div className="hidden md:flex items-center justify-center w-16 h-16 rounded-2xl bg-slate-900/50 border border-slate-700/50">
                                {selectedVersion.architectureType === 'monolith' && <Box className="text-blue-500" />}
                                {selectedVersion.architectureType === 'async' && <Layers className="text-blue-500" />}
                                {selectedVersion.architectureType === 'graph' && <Network className="text-emerald-500" />}
                                {selectedVersion.architectureType === 'swarm' && <Share2 className="text-purple-500" />}
                            </div>
                        </div>

                        {/* Highlights Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                            {selectedVersion.highlights.map((h, i) => (
                                <div key={i} className="bg-slate-900/40 p-3 rounded-lg border border-slate-800 flex items-center gap-3">
                                    <CheckCircle className="w-4 h-4 text-emerald-500/80" />
                                    <span className="text-sm text-slate-200">{h}</span>
                                </div>
                            ))}
                        </div>

                        {/* Visualizer Area */}
                        <div className="bg-[#020617] rounded-lg border border-slate-800 p-6 relative overflow-hidden group">
                            <div className="absolute top-3 left-4 text-xs font-mono text-slate-500 uppercase tracking-widest z-10">Architecture Blueprint</div>
                            <ArchitectureViz type={selectedVersion.architectureType} />
                        </div>
                    </div>

                    {/* Lower Section: Metrics & Agents */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        
                        {/* Metrics */}
                        <div className="glass-panel p-6 rounded-xl">
                            <h3 className="text-sm font-bold text-slate-400 mb-5 flex items-center gap-2 uppercase tracking-wider">
                                <BarChart3 className="w-4 h-4 text-blue-400" />
                                Performance Telemetry
                            </h3>
                            <div className="space-y-5">
                                {selectedVersion.metrics.map((m, i) => (
                                    <div key={i} className="group">
                                        <div className="flex justify-between items-end mb-1">
                                            <span className="text-sm text-slate-400 group-hover:text-slate-200 transition-colors">{m.label}</span>
                                            <div className="text-right">
                                                <span className="text-lg font-bold text-white font-mono">{m.value}{m.unit}</span>
                                                {m.delta && (
                                                    <span className={`ml-2 text-xs ${m.delta > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                        {m.delta > 0 ? '+' : ''}{m.delta}%
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                        {/* Mock Progress Bar */}
                                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                            <div 
                                                className="h-full bg-blue-500 rounded-full transition-all duration-1000 ease-out"
                                                style={{ width: mounted ? `${Math.min(m.value * (m.unit === '%' ? 1 : 2), 100)}%` : '0%' }}
                                            ></div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Active Agents */}
                        <div className="glass-panel p-6 rounded-xl">
                            <h3 className="text-sm font-bold text-slate-400 mb-5 flex items-center gap-2 uppercase tracking-wider">
                                <Users className="w-4 h-4 text-purple-400" />
                                Active Agent Roster
                            </h3>
                            <div className="grid grid-cols-1 gap-3">
                                {selectedVersion.agents.map((agent, i) => (
                                    <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-slate-800/30 border border-slate-700/50 hover:bg-slate-800/50 hover:border-purple-500/30 transition-all">
                                        <div className="flex items-center gap-3">
                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                                                agent.type === 'fiduciary' ? 'bg-red-500/20 text-red-400' :
                                                agent.type === 'executor' ? 'bg-emerald-500/20 text-emerald-400' :
                                                'bg-blue-500/20 text-blue-400'
                                            }`}>
                                                {agent.type === 'fiduciary' ? <ShieldAlert size={14} /> : 
                                                 agent.type === 'executor' ? <Zap size={14} /> : 
                                                 <Server size={14} />}
                                            </div>
                                            <div>
                                                <div className="text-sm font-bold text-white">{agent.name}</div>
                                                <div className="text-xs text-slate-500">{agent.role}</div>
                                            </div>
                                        </div>
                                        <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse"></div>
                                    </div>
                                ))}
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    );
};

export default EvolutionHub;
