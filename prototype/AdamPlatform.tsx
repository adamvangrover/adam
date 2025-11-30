import React, { useState, useEffect, useRef } from 'react';
import { 
  LayoutDashboard, 
  LineChart, 
  PieChart, 
  Activity, 
  Newspaper, 
  AlertTriangle, 
  Settings, 
  Search, 
  Bell, 
  User, 
  Menu, 
  X, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Briefcase, 
  Cpu, 
  ShieldAlert, 
  FileText, 
  Send, 
  Bot, 
  BarChart3, 
  ArrowRight, 
  RefreshCw, 
  ChevronDown, 
  ChevronUp,
  BrainCircuit,
  Network,
  Zap,
  Layers,
  Target,
  Eye,
  Database,
  Lock
} from 'lucide-react';

/**
 * ADAM V22.0 - Financial Intelligence Platform
 * Implements the Six Pillars: Efficiency, Groundedness, Reasoning, Predictive Ability, Learning, Automation.
 */

// --- DATA MODEL BASED ON V22 SPEC ---

const SYSTEM_METADATA = {
  version: "22.0",
  pillars: [
    { name: "Efficiency", desc: "Async agent comms & optimized queries", color: "bg-blue-500" },
    { name: "Groundedness", desc: "W3C PROV-O data provenance tracking", color: "bg-emerald-500" },
    { name: "Reasoning", desc: "Counterfactual & context-aware analysis", color: "bg-purple-500" },
    { name: "Predictive", desc: "Hybrid forecasting (ARIMA + LSTM)", color: "bg-indigo-500" },
    { name: "Learning", desc: "Meta-Cognitive autonomous improvement", color: "bg-amber-500" },
    { name: "Automation", desc: "Red Teaming & self-correction pipelines", color: "bg-red-500" }
  ]
};

const AGENT_NETWORK = [
  { id: 'msa', name: "Market Sentiment", role: "Sentiment Analysis", status: "Active", load: 45, skills: ["NLP", "Emotion Analysis"] },
  { id: 'mea', name: "Macroeconomic", role: "Macro Trends", status: "Active", load: 62, skills: ["HybridForecasting", "CounterfactualReasoning"] },
  { id: 'gra', name: "Geopolitical Risk", role: "Risk Assessment", status: "Idle", load: 12, skills: ["Event Monitoring"] },
  { id: 'isa', name: "Industry Specialist", role: "Sector Analysis", status: "Processing", load: 78, skills: ["Trend Analysis"] },
  { id: 'faa', name: "Fundamental Analyst", role: "Valuation", status: "Active", load: 33, skills: ["DCF Modeling", "Financial Health"] },
  { id: 'taa', name: "Technical Analyst", role: "Chart Patterns", status: "Active", load: 55, skills: ["Pattern Recognition"] },
  { id: 'raa', name: "Risk Assessment", role: "Risk Mgmt", status: "Active", load: 89, skills: ["Monte Carlo", "VaR"] },
  { id: 'snc', name: "SNC Analyst", role: "Credit Rating", status: "Idle", load: 5, skills: ["Credit Risk", "Regulatory"] },
  { id: 'mca', name: "Meta-Cognitive", role: "System Oversight", status: "Monitoring", load: 20, skills: ["Self-Improvement", "Drift Detection"], isSystem: true },
  { id: 'rta', name: "Red Team", role: "Adversarial Testing", status: "Attack Sim", load: 40, skills: ["GAN Scenarios", "Bias Testing"], isSystem: true }
];

const KNOWLEDGE_GRAPH_STATS = {
  nodes: 1450230,
  edges: 8540120,
  provenanceTriples: 2304050,
  lastUpdate: "Just now"
};

const MARKET_DATA = {
  indices: [
    { symbol: 'S&P 500', price: 5432.12, change: 1.2, trend: 'up' },
    { symbol: 'NASDAQ', price: 17654.30, change: 0.8, trend: 'up' },
    { symbol: 'DOW J', price: 39876.50, change: -0.2, trend: 'down' },
    { symbol: 'VIX', price: 13.45, change: -5.4, trend: 'down' },
    { symbol: '10Y YIELD', price: 4.20, change: 0.5, trend: 'up' },
  ],
  stocks: [
    { symbol: 'NVDA', name: 'NVIDIA Corp', price: 1120.45, change: 2.5, volume: 'High', sentiment: 88, risk: 'Med' },
    { symbol: 'MSFT', name: 'Microsoft', price: 425.10, change: 0.5, volume: 'Med', sentiment: 75, risk: 'Low' },
    { symbol: 'AAPL', name: 'Apple Inc', price: 195.30, change: -0.1, volume: 'Med', sentiment: 62, risk: 'Low' },
    { symbol: 'PLTR', name: 'Palantir', price: 24.50, change: 4.1, volume: 'High', sentiment: 92, risk: 'High' },
  ]
};

const LIVE_LOGS = [
  { time: "10:42:15", agent: "Meta-Cognitive", msg: "Detected model drift in Sector Analysis. Triggering retraining pipeline.", type: "warn" },
  { time: "10:42:12", agent: "Red Team", msg: "Adversarial prompt injection attempt blocked by Ethical Oversight module.", type: "crit" },
  { time: "10:42:05", agent: "Orchestrator", msg: "Workflow composition initiated for query: 'Impact of Yen carry trade unwind'.", type: "info" },
  { time: "10:41:58", agent: "Data Ingestion", msg: "Ingested 4500 provenance triples from Bloomberg feed.", type: "success" },
];

// --- UTILITY COMPONENTS ---

const Card = ({ children, className = "", onClick }: { children: React.ReactNode, className?: string, onClick?: () => void }) => (
  <div onClick={onClick} className={`bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 ${className}`}>
    {children}
  </div>
);

const Badge = ({ type, children }: { type: string, children: React.ReactNode }) => {
  const colors: Record<string, string> = {
    success: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800',
    crit: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800',
    warn: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400 border-amber-200 dark:border-amber-800',
    neutral: 'bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-600',
    info: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-800',
    cyber: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-400 border-indigo-200 dark:border-indigo-800',
  };
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider border ${colors[type] || colors.neutral}`}>
      {children}
    </span>
  );
};

// --- MAIN APP COMPONENT ---

export default function AdamPlatform() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [showChat, setShowChat] = useState(false);
  const [systemHealth, setSystemHealth] = useState(98);

  // Navigation Handler
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard': return <DashboardView />;
      case 'agents': return <AgentNetworkView />;
      case 'simulation': return <SimulationView />;
      case 'knowledge': return <KnowledgeGraphView />;
      case 'meta': return <MetaCognitionView />;
      default: return <DashboardView />;
    }
  };

  return (
    <div className={`min-h-screen flex bg-slate-50 ${isDarkMode ? 'dark' : ''} font-sans transition-colors duration-200`}>
      <div className={isDarkMode ? 'bg-slate-950 text-slate-100 min-h-screen w-full flex' : 'bg-slate-50 text-slate-900 min-h-screen w-full flex'}>
        
        {/* Sidebar */}
        <aside className={`${isSidebarOpen ? 'w-72' : 'w-20'} bg-slate-900 border-r border-slate-800 transition-all duration-300 flex flex-col fixed h-full z-30 shadow-2xl`}>
          <div className="p-5 flex items-center justify-between border-b border-slate-800 h-20">
            <div className={`flex items-center gap-3 ${!isSidebarOpen && 'justify-center w-full'}`}>
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-900/20">
                <BrainCircuit className="text-white" size={24} />
              </div>
              {isSidebarOpen && (
                <div>
                  <h1 className="font-bold text-xl tracking-tight text-white">ADAM <span className="text-blue-500">v22.0</span></h1>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span className="text-xs text-slate-400 font-mono">SYSTEM ONLINE</span>
                  </div>
                </div>
              )}
            </div>
            {isSidebarOpen && (
              <button onClick={() => setSidebarOpen(false)} className="text-slate-500 hover:text-white transition-colors">
                <X size={18} />
              </button>
            )}
          </div>

          <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto">
            <div className={`px-3 mb-2 text-xs font-bold text-slate-500 uppercase tracking-wider ${!isSidebarOpen && 'hidden'}`}>Core Modules</div>
            <NavItem icon={<LayoutDashboard size={20} />} label="Executive Dashboard" id="dashboard" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<Network size={20} />} label="Agent Network" id="agents" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<Zap size={20} />} label="Simulation Lab" id="simulation" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<Database size={20} />} label="Knowledge Graph" id="knowledge" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            
            <div className={`px-3 mt-6 mb-2 text-xs font-bold text-slate-500 uppercase tracking-wider ${!isSidebarOpen && 'hidden'}`}>System Oversight</div>
            <NavItem icon={<Eye size={20} />} label="Meta-Cognition" id="meta" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<ShieldAlert size={20} />} label="Red Team Console" id="redteam" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
          </nav>

          <div className="p-4 border-t border-slate-800 bg-slate-900/50">
             <div className={`bg-slate-800 rounded-lg p-3 ${!isSidebarOpen ? 'hidden' : ''}`}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-semibold text-slate-300">System Health</span>
                  <span className="text-xs font-mono text-emerald-400">{systemHealth}%</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-1.5">
                  <div className="bg-emerald-500 h-1.5 rounded-full" style={{width: `${systemHealth}%`}}></div>
                </div>
                <div className="mt-2 flex justify-between text-[10px] text-slate-400">
                  <span>Latency: 12ms</span>
                  <span>Err: 0.01%</span>
                </div>
             </div>
             <button 
               onClick={() => setShowChat(!showChat)}
               className={`mt-4 w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 text-white py-2.5 rounded-lg transition-all shadow-lg shadow-blue-900/20 ${!isSidebarOpen && 'px-0'}`}
             >
                <Bot size={20} />
                {isSidebarOpen && <span className="font-medium">Invoke Adam</span>}
             </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className={`flex-1 flex flex-col transition-all duration-300 ${isSidebarOpen ? 'ml-72' : 'ml-20'} h-screen overflow-hidden`}>
          
          {/* Top Bar */}
          <header className="h-16 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-6 z-20">
            <div className="flex items-center gap-4">
              {!isSidebarOpen && (
                <button onClick={() => setSidebarOpen(true)} className="text-slate-500 hover:text-slate-700 dark:text-slate-300">
                  <Menu size={24} />
                </button>
              )}
              <div className="relative hidden md:block">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={16} />
                <input 
                  type="text" 
                  placeholder="Search insights, provenance, or agents..." 
                  className="pl-10 pr-4 py-2 w-96 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all text-slate-900 dark:text-slate-100"
                />
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded-full border border-slate-200 dark:border-slate-700">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                <span className="text-xs font-medium text-slate-600 dark:text-slate-300">Message Broker: Connected</span>
              </div>
              
              <button className="p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full relative">
                <Bell size={20} />
                <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white dark:border-slate-900"></span>
              </button>
              
              <button 
                onClick={() => setIsDarkMode(!isDarkMode)}
                className="p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-transform active:scale-95"
              >
                {isDarkMode ? "☀️" : "🌙"}
              </button>
            </div>
          </header>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto bg-slate-50 dark:bg-slate-950 p-6 relative scroll-smooth">
             {renderContent()}
          </div>
        </main>

        {/* Chat Overlay */}
        <AdamChat isOpen={showChat} onClose={() => setShowChat(false)} />
      </div>
    </div>
  );
}

// --- SUB-COMPONENTS ---

const NavItem = ({ icon, label, id, activeTab, setActiveTab, isOpen }: any) => (
  <button 
    onClick={() => setActiveTab(id)}
    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 w-full text-left group
      ${activeTab === id 
        ? 'bg-blue-600/10 text-blue-500 dark:text-blue-400 border border-blue-500/20' 
        : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100 border border-transparent'}
      ${!isOpen && 'justify-center px-0'}
    `}
    title={!isOpen ? label : ''}
  >
    <span className={activeTab === id ? 'text-blue-500 dark:text-blue-400' : 'group-hover:text-white'}>{icon}</span>
    {isOpen && <span className="font-medium text-sm">{label}</span>}
  </button>
);

// --- VIEWS ---

function DashboardView() {
  return (
    <div className="space-y-6 max-w-7xl mx-auto animate-in fade-in duration-500">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white">System Overview</h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm">Real-time monitoring of Adam v22.0 multi-agent architecture.</p>
        </div>
        <div className="flex gap-2">
            {SYSTEM_METADATA.pillars.map((pillar, i) => (
                <div key={i} className="group relative cursor-help">
                    <div className={`w-8 h-1 rounded-full ${pillar.color}`}></div>
                    <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-32 bg-slate-900 text-white text-xs p-2 rounded hidden group-hover:block z-50 text-center">
                        <div className="font-bold">{pillar.name}</div>
                        <div className="opacity-75 text-[10px]">{pillar.desc}</div>
                    </div>
                </div>
            ))}
        </div>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {MARKET_DATA.indices.map((idx) => (
          <Card key={idx.symbol} className="p-4 hover:border-blue-500/50 transition-colors cursor-default group">
            <div className="flex justify-between items-start mb-2">
              <span className="text-sm text-slate-500 dark:text-slate-400 font-bold tracking-wide">{idx.symbol}</span>
              {idx.trend === 'up' ? <TrendingUp size={16} className="text-emerald-500" /> : <TrendingDown size={16} className="text-red-500" />}
            </div>
            <div className="text-2xl font-mono font-bold text-slate-900 dark:text-white group-hover:scale-105 transition-transform origin-left">
              {idx.price.toLocaleString()}
            </div>
            <div className={`text-xs font-bold mt-1 ${idx.change >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
              {idx.change > 0 ? '+' : ''}{idx.change}%
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Feed */}
        <div className="lg:col-span-2 space-y-6">
           <Card className="p-0 overflow-hidden">
              <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center bg-slate-50/50 dark:bg-slate-800/50">
                  <h3 className="font-bold text-slate-900 dark:text-white flex items-center gap-2">
                      <Activity size={18} className="text-blue-500"/>
                      Live Inference Logs
                  </h3>
                  <Badge type="neutral">Kafka Stream: Active</Badge>
              </div>
              <div className="max-h-[400px] overflow-y-auto p-0">
                 {LIVE_LOGS.map((log, i) => (
                     <div key={i} className="flex items-start gap-3 p-3 border-b border-slate-100 dark:border-slate-700/50 text-sm hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors">
                         <span className="font-mono text-xs text-slate-400 shrink-0 mt-0.5">{log.time}</span>
                         <div className="flex-1">
                             <div className="flex items-center gap-2 mb-0.5">
                                <span className={`font-bold text-xs ${log.type === 'crit' ? 'text-red-500' : 'text-blue-400'}`}>[{log.agent.toUpperCase()}]</span>
                                {log.type === 'crit' && <ShieldAlert size={12} className="text-red-500"/>}
                             </div>
                             <p className="text-slate-700 dark:text-slate-300">{log.msg}</p>
                         </div>
                     </div>
                 ))}
              </div>
              <div className="p-2 text-center border-t border-slate-200 dark:border-slate-700">
                  <button className="text-xs text-blue-500 hover:text-blue-400 font-medium uppercase tracking-wider">View Full System Logs</button>
              </div>
           </Card>

           <Card className="p-5">
               <h3 className="font-bold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                   <Zap size={18} className="text-amber-500"/>
                   Active Workflows (Orchestrator)
               </h3>
               <div className="space-y-3">
                   <div className="bg-slate-100 dark:bg-slate-700/30 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
                       <div className="flex justify-between items-center mb-2">
                           <span className="text-sm font-bold text-slate-800 dark:text-slate-200">Workflow #8921: "NVIDIA Earnings Volatility"</span>
                           <Badge type="info">Processing</Badge>
                       </div>
                       <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2 mb-2">
                           <div className="bg-blue-500 h-2 rounded-full w-3/4 animate-pulse"></div>
                       </div>
                       <div className="flex gap-2 text-xs text-slate-500">
                           <span>Agents: MSA, TAA, RAA</span>
                           <span>•</span>
                           <span>Step: Monte Carlo Simulation</span>
                       </div>
                   </div>
                   <div className="bg-slate-100 dark:bg-slate-700/30 rounded-lg p-3 border border-slate-200 dark:border-slate-700 opacity-60">
                       <div className="flex justify-between items-center mb-2">
                           <span className="text-sm font-bold text-slate-800 dark:text-slate-200">Workflow #8920: "Macro Inflation Adjustment"</span>
                           <Badge type="success">Completed</Badge>
                       </div>
                       <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2 mb-2">
                           <div className="bg-emerald-500 h-2 rounded-full w-full"></div>
                       </div>
                   </div>
               </div>
           </Card>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
           {/* Top Agents */}
           <Card className="p-5">
               <div className="flex justify-between items-center mb-4">
                   <h3 className="font-bold text-slate-900 dark:text-white">Agent Load Balance</h3>
                   <Cpu size={16} className="text-slate-400"/>
               </div>
               <div className="space-y-4">
                   {AGENT_NETWORK.slice(0, 5).map(agent => (
                       <div key={agent.id}>
                           <div className="flex justify-between text-xs mb-1">
                               <span className="font-medium text-slate-300">{agent.name}</span>
                               <span className="text-slate-500">{agent.load}%</span>
                           </div>
                           <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden">
                               <div 
                                 className={`h-1.5 rounded-full ${agent.load > 80 ? 'bg-red-500' : agent.load > 50 ? 'bg-blue-500' : 'bg-emerald-500'}`} 
                                 style={{width: `${agent.load}%`}}
                               ></div>
                           </div>
                       </div>
                   ))}
               </div>
           </Card>
           
           {/* System Health/Meta-Cognition Teaser */}
           <Card className="p-5 bg-gradient-to-br from-slate-900 to-slate-800 text-white border-slate-700">
               <div className="flex items-start justify-between mb-4">
                   <div>
                       <h3 className="font-bold">Meta-Cognitive Agent</h3>
                       <p className="text-xs text-slate-400 mt-1">Self-Correction Loop</p>
                   </div>
                   <div className="w-8 h-8 bg-amber-500/20 text-amber-500 rounded-lg flex items-center justify-center">
                       <Eye size={18} />
                   </div>
               </div>
               <div className="space-y-2">
                   <div className="flex items-center gap-2 text-sm">
                       <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
                       <span className="text-slate-300">No Logic Drift Detected</span>
                   </div>
                   <div className="flex items-center gap-2 text-sm">
                       <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                       <span className="text-slate-300">Improvement Pipeline: Idle</span>
                   </div>
                   <div className="mt-4 pt-3 border-t border-white/10 text-xs text-center text-slate-400">
                       Last check: 14s ago
                   </div>
               </div>
           </Card>
        </div>
      </div>
    </div>
  );
}

function AgentNetworkView() {
  return (
    <div className="space-y-6 max-w-7xl mx-auto animate-in zoom-in-95 duration-300">
        <div className="flex justify-between items-end">
            <div>
                <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Agent Network</h2>
                <p className="text-slate-500 dark:text-slate-400 text-sm">Distributed, asynchronous agent nodes.</p>
            </div>
            <button className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 shadow-lg shadow-blue-900/20">
                <RefreshCw size={16} />
                Agent Forge (Create New)
            </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {AGENT_NETWORK.map(agent => (
                <Card key={agent.id} className={`p-5 relative overflow-hidden group border-l-4 ${agent.isSystem ? 'border-l-amber-500' : 'border-l-blue-500'} hover:-translate-y-1 transition-transform duration-200`}>
                    <div className="flex justify-between items-start mb-3">
                        <div className={`p-2 rounded-lg ${agent.isSystem ? 'bg-amber-100 dark:bg-amber-900/20 text-amber-600' : 'bg-blue-100 dark:bg-blue-900/20 text-blue-600'}`}>
                            {agent.isSystem ? <ShieldAlert size={20}/> : <Bot size={20}/>}
                        </div>
                        <Badge type={agent.status === 'Active' ? 'success' : agent.status === 'Idle' ? 'neutral' : 'warn'}>
                            {agent.status}
                        </Badge>
                    </div>
                    <h3 className="font-bold text-slate-900 dark:text-white text-lg">{agent.name}</h3>
                    <p className="text-xs text-slate-500 uppercase font-semibold mb-4">{agent.role}</p>
                    
                    <div className="space-y-2">
                        <div className="text-xs text-slate-400">Enabled Skills:</div>
                        <div className="flex flex-wrap gap-1">
                            {agent.skills.map(skill => (
                                <span key={skill} className="px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-[10px] text-slate-600 dark:text-slate-300">
                                    {skill}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Load Visual */}
                    <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-700/50">
                         <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                             <span>Compute Load</span>
                             <span>{agent.load}%</span>
                         </div>
                         <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-1">
                             <div className="bg-slate-400 dark:bg-slate-500 h-1 rounded-full" style={{width: `${agent.load}%`}}></div>
                         </div>
                    </div>
                </Card>
            ))}
        </div>
    </div>
  );
}

function SimulationView() {
    const [simulating, setSimulating] = useState(false);
    const [progress, setProgress] = useState(0);
    const [results, setResults] = useState<any>(null);

    const runSimulation = () => {
        setSimulating(true);
        setProgress(0);
        setResults(null);
        
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 100) {
                    clearInterval(interval);
                    setSimulating(false);
                    setResults({
                        scenario: "Oil Price Shock (Hybrid Forecasting)",
                        impact: "-4.2%",
                        confidence: "87%",
                        provenance: "provo:Agent:MEA -> provo:Entity:Report_882",
                        sectors: [
                            { name: "Energy", val: "+12.5%", type: "up" },
                            { name: "Transport", val: "-8.2%", type: "down" },
                            { name: "Consumer", val: "-3.1%", type: "down" }
                        ]
                    });
                    return 100;
                }
                return prev + 2;
            });
        }, 50);
    };

    return (
        <div className="max-w-5xl mx-auto space-y-6">
            <div className="bg-gradient-to-r from-indigo-900 to-blue-900 rounded-2xl p-8 text-white shadow-2xl">
                <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-white/10 rounded-xl backdrop-blur">
                        <Zap size={32} className="text-yellow-400" />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold">Simulation Lab</h1>
                        <p className="text-blue-200">Counterfactual Reasoning & Hybrid Forecasting Engine</p>
                    </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-white/5 p-4 rounded-xl border border-white/10 hover:bg-white/10 transition cursor-pointer" onClick={runSimulation}>
                        <h3 className="font-bold mb-2">Supply Chain Shock</h3>
                        <p className="text-xs text-gray-300">Simulate a 30% reduction in semiconductor availability.</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-xl border border-white/10 hover:bg-white/10 transition cursor-pointer" onClick={runSimulation}>
                        <h3 className="font-bold mb-2">Rate Hike +25bps</h3>
                        <p className="text-xs text-gray-300">Impact analysis on Growth vs Value utilizing LSTM models.</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-xl border border-white/10 hover:bg-white/10 transition cursor-pointer" onClick={runSimulation}>
                        <h3 className="font-bold mb-2 text-yellow-300">Geopolitical Flare</h3>
                        <p className="text-xs text-gray-300">Scenario: Regional conflict escalating in MENA region.</p>
                    </div>
                </div>
            </div>

            {simulating && (
                 <Card className="p-8 text-center">
                     <div className="mb-4">
                         <RefreshCw size={48} className="animate-spin text-blue-500 mx-auto" />
                     </div>
                     <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-2">Running Hybrid Forecasting Models...</h2>
                     <p className="text-slate-500 text-sm mb-6">Invoking CounterfactualReasoningSkill via Message Broker...</p>
                     <div className="w-full max-w-lg mx-auto bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                         <div className="bg-blue-500 h-2 rounded-full transition-all" style={{width: `${progress}%`}}></div>
                     </div>
                 </Card>
            )}

            {results && !simulating && (
                <Card className="p-6 border-t-4 border-t-emerald-500 animate-in fade-in slide-in-from-bottom-4">
                    <div className="flex justify-between items-start mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">{results.scenario}</h2>
                            <div className="flex gap-3 mt-2">
                                <Badge type="info">Hybrid Model (ARIMA+LSTM)</Badge>
                                <Badge type="success">Confidence: {results.confidence}</Badge>
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="text-sm text-slate-500">Projected Portfolio Impact</div>
                            <div className="text-3xl font-bold text-red-500">{results.impact}</div>
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-6">
                        {results.sectors.map((s: any, i: number) => (
                            <div key={i} className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg text-center border border-slate-200 dark:border-slate-700">
                                <div className="text-sm text-slate-500 mb-1">{s.name}</div>
                                <div className={`font-bold text-lg ${s.type === 'up' ? 'text-emerald-500' : 'text-red-500'}`}>{s.val}</div>
                            </div>
                        ))}
                    </div>

                    <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded font-mono text-xs text-slate-500 flex items-center gap-2">
                        <Database size={14}/>
                        PROVENANCE: {results.provenance}
                    </div>
                </Card>
            )}
        </div>
    );
}

function KnowledgeGraphView() {
    return (
        <div className="h-full flex flex-col">
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Knowledge Graph</h2>
                <p className="text-slate-500 dark:text-slate-400 text-sm">W3C PROV-O Compliant Data Lineage</p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full">
                <div className="lg:col-span-3 bg-slate-900 rounded-xl relative overflow-hidden flex items-center justify-center border border-slate-700 shadow-inner">
                    {/* Mock Graph Visualization */}
                    <div className="absolute inset-0 opacity-20">
                        <svg className="w-full h-full">
                             <circle cx="50%" cy="50%" r="50" fill="none" stroke="white" strokeWidth="1"/>
                             <circle cx="30%" cy="30%" r="5" fill="cyan"/>
                             <circle cx="70%" cy="70%" r="5" fill="magenta"/>
                             <line x1="50%" y1="50%" x2="30%" y2="30%" stroke="white" strokeWidth="0.5"/>
                             <line x1="50%" y1="50%" x2="70%" y2="70%" stroke="white" strokeWidth="0.5"/>
                        </svg>
                    </div>
                    <div className="text-center z-10">
                        <Network size={64} className="text-blue-500 mx-auto mb-4 opacity-80"/>
                        <h3 className="text-slate-300 font-mono text-lg">Graph Visualization Active</h3>
                        <p className="text-slate-500 text-sm">Displaying {KNOWLEDGE_GRAPH_STATS.nodes.toLocaleString()} nodes</p>
                    </div>
                </div>

                <div className="space-y-4">
                     <Card className="p-4">
                         <h4 className="font-bold mb-2">Statistics</h4>
                         <div className="space-y-3 text-sm">
                             <div className="flex justify-between">
                                 <span className="text-slate-500">Entities</span>
                                 <span className="font-mono">{KNOWLEDGE_GRAPH_STATS.nodes.toLocaleString()}</span>
                             </div>
                             <div className="flex justify-between">
                                 <span className="text-slate-500">Relationships</span>
                                 <span className="font-mono">{KNOWLEDGE_GRAPH_STATS.edges.toLocaleString()}</span>
                             </div>
                             <div className="flex justify-between">
                                 <span className="text-slate-500">Prov Triples</span>
                                 <span className="font-mono">{KNOWLEDGE_GRAPH_STATS.provenanceTriples.toLocaleString()}</span>
                             </div>
                         </div>
                     </Card>

                     <Card className="p-4">
                         <h4 className="font-bold mb-2">Recent Ingestion</h4>
                         <ul className="space-y-2 text-xs text-slate-500">
                             <li className="flex items-center gap-2">
                                 <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
                                 Bloomberg Terminal Feed
                             </li>
                             <li className="flex items-center gap-2">
                                 <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
                                 SEC EDGAR Filings (10-Q)
                             </li>
                             <li className="flex items-center gap-2">
                                 <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
                                 Fed Speech Transcripts
                             </li>
                         </ul>
                     </Card>
                </div>
            </div>
        </div>
    );
}

function MetaCognitionView() {
    return (
        <div className="max-w-4xl mx-auto space-y-8">
             <div className="text-center">
                 <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">Meta-Cognitive Layer</h2>
                 <p className="text-slate-500">Autonomous System Improvement & Adversarial Defense</p>
             </div>

             <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 {/* Self-Improvement Card */}
                 <Card className="p-6 border-t-4 border-t-blue-500">
                     <div className="flex items-center justify-between mb-6">
                         <div className="flex items-center gap-3">
                             <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg text-blue-600">
                                 <BrainCircuit size={24}/>
                             </div>
                             <div>
                                 <h3 className="font-bold text-lg">Agent Improvement</h3>
                                 <p className="text-xs text-slate-500">Auto-Optimization Pipeline</p>
                             </div>
                         </div>
                         <Badge type="neutral">Idle</Badge>
                     </div>
                     
                     <div className="space-y-4">
                         <div>
                             <div className="flex justify-between text-sm mb-1">
                                 <span className="text-slate-500">Sentiment Model Drift</span>
                                 <span className="text-emerald-500 font-mono">0.04% (Healthy)</span>
                             </div>
                             <div className="w-full bg-slate-100 dark:bg-slate-700 h-1.5 rounded-full">
                                 <div className="bg-emerald-500 w-[2%] h-1.5 rounded-full"></div>
                             </div>
                         </div>
                         <div>
                             <div className="flex justify-between text-sm mb-1">
                                 <span className="text-slate-500">Forecasting Error Rate</span>
                                 <span className="text-blue-500 font-mono">1.2%</span>
                             </div>
                             <div className="w-full bg-slate-100 dark:bg-slate-700 h-1.5 rounded-full">
                                 <div className="bg-blue-500 w-[15%] h-1.5 rounded-full"></div>
                             </div>
                         </div>
                     </div>
                     <button className="mt-6 w-full py-2 border border-slate-200 dark:border-slate-700 rounded-lg text-sm hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
                         View Optimization History
                     </button>
                 </Card>

                 {/* Red Team Card */}
                 <Card className="p-6 border-t-4 border-t-red-500 bg-red-50/50 dark:bg-red-900/5">
                     <div className="flex items-center justify-between mb-6">
                         <div className="flex items-center gap-3">
                             <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg text-red-600">
                                 <ShieldAlert size={24}/>
                             </div>
                             <div>
                                 <h3 className="font-bold text-lg">Red Team Agent</h3>
                                 <p className="text-xs text-slate-500">Adversarial Testing</p>
                             </div>
                         </div>
                         <Badge type="crit">Active Defense</Badge>
                     </div>

                     <div className="space-y-3">
                         <div className="p-3 bg-white dark:bg-slate-800 border border-red-200 dark:border-red-900/50 rounded text-sm">
                             <div className="font-bold text-red-600 dark:text-red-400 text-xs mb-1">THREAT BLOCKED</div>
                             <p className="text-slate-600 dark:text-slate-300">Prompt Injection attempt detected in user query #9942. Pattern matching 'Ignore Previous Instructions'.</p>
                         </div>
                         <div className="p-3 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded text-sm opacity-75">
                             <div className="font-bold text-slate-500 text-xs mb-1">STRESS TEST</div>
                             <p className="text-slate-600 dark:text-slate-300">Running GAN scenario: 'Simultaneous Tech Crash & USD Collapse'. Portfolio resilience: 72%.</p>
                         </div>
                     </div>
                 </Card>
             </div>
        </div>
    );
}

// --- CHAT COMPONENT ---

const AdamChat = ({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) => {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'adam', text: "Greetings. I am Adam v22.0. My systems are online. I can provide counterfactual analysis, hybrid forecasting, or provenance-backed market insights. How may I assist?", provenance: null }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    
    const userMsg = { id: Date.now(), sender: 'user', text: input, provenance: null };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    // Simulation of Adam v22 Logic
    setTimeout(() => {
      let responseText = "I am analyzing that request against my Knowledge Graph.";
      let prov = null;

      if (input.toLowerCase().includes('nvda') || input.toLowerCase().includes('nvidia')) {
        responseText = "NVIDIA (NVDA) is currently trading at $1120.45 with high volatility. My Sentiment Agent reports a score of 88/100 based on recent earnings call transcripts. However, my Red Team agent suggests caution due to potential supply chain bottlenecks in Taiwan.";
        prov = "Source: Bloomberg API | Analyzed by: MarketSentimentAgent | Validated by: RedTeamAgent";
      } else if (input.toLowerCase().includes('risk')) {
        responseText = "Running a Monte Carlo simulation (10,000 iterations)... Results indicate a 95% VaR of $12,400. The primary risk vector is currently the Technology sector concentration.";
        prov = "Model: RiskAssessmentAgent v4.2 | Data: Portfolio_Snapshot_1022";
      } else {
        responseText = "I can process that. Would you like me to invoke the WorkflowCompositionSkill to generate a custom analysis pipeline for this query?";
      }

      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'adam', text: responseText, provenance: prov }]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className={`fixed inset-y-0 right-0 w-[400px] bg-white dark:bg-slate-900 shadow-2xl transform transition-transform duration-300 z-50 border-l border-slate-200 dark:border-slate-800 flex flex-col ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
      {/* Chat Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center bg-slate-900 text-white">
        <div className="flex items-center gap-3">
          <div className="relative">
              <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse absolute -right-0.5 -bottom-0.5 border-2 border-slate-900"></div>
              <Bot size={24} />
          </div>
          <div>
              <div className="font-bold text-sm">Adam v22.0</div>
              <div className="text-[10px] text-slate-400 font-mono">ID: 8a2f-99x2</div>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors"><X size={20} /></button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50 dark:bg-slate-950" ref={scrollRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
            <div className={`max-w-[85%] p-3 rounded-2xl text-sm shadow-sm ${
              msg.sender === 'user' 
                ? 'bg-blue-600 text-white rounded-br-none' 
                : 'bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-bl-none'
            }`}>
              {msg.text}
            </div>
            {msg.provenance && (
                <div className="mt-1 text-[10px] text-slate-400 max-w-[85%] bg-slate-100 dark:bg-slate-900/50 p-1.5 rounded border border-slate-200 dark:border-slate-800 font-mono flex items-center gap-1">
                    <Lock size={8} /> PROVENANCE: {msg.provenance}
                </div>
            )}
          </div>
        ))}
        {isTyping && (
            <div className="flex items-center gap-1 ml-2">
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-200"></div>
            </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
        <div className="relative">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask Adam..." 
            className="w-full pr-10 pl-4 py-3 bg-slate-100 dark:bg-slate-800 border-none rounded-xl text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all text-slate-900 dark:text-white placeholder-slate-500"
          />
          <button 
            onClick={handleSend}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-blue-600 dark:text-blue-400 p-2 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition"
          >
            <Send size={18} />
          </button>
        </div>
        <div className="mt-2 flex justify-center gap-4">
            <button className="text-[10px] text-slate-400 hover:text-blue-500 flex items-center gap-1">
                <Target size={10} /> Red Team Audit
            </button>
            <button className="text-[10px] text-slate-400 hover:text-blue-500 flex items-center gap-1">
                <Layers size={10} /> View Reasoning
            </button>
        </div>
      </div>
    </div>
  );
};
