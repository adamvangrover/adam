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
  ChevronUp
} from 'lucide-react';

/**
 * ADAM PLATFORM - Single File React Implementation
 * Based on v23 Architecture and Webapp mockups
 */

// --- MOCK DATA STORES ---

const MARKET_DATA = {
  indices: [
    { symbol: 'S&P 500', price: 5432.12, change: 1.2, trend: 'up' },
    { symbol: 'NASDAQ', price: 17654.30, change: 0.8, trend: 'up' },
    { symbol: 'DOW J', price: 39876.50, change: -0.2, trend: 'down' },
    { symbol: 'VIX', price: 13.45, change: -5.4, trend: 'down' },
  ],
  stocks: [
    { symbol: 'NVDA', name: 'NVIDIA Corp', price: 1120.45, change: 2.5, volume: '45M', mktCap: '2.8T', sector: 'Technology' },
    { symbol: 'MSFT', name: 'Microsoft', price: 425.10, change: 0.5, volume: '22M', mktCap: '3.1T', sector: 'Technology' },
    { symbol: 'AAPL', name: 'Apple Inc', price: 195.30, change: -0.1, volume: '30M', mktCap: '2.9T', sector: 'Technology' },
    { symbol: 'AMZN', name: 'Amazon.com', price: 182.50, change: 1.1, volume: '28M', mktCap: '1.9T', sector: 'Cons. Disc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc', price: 175.20, change: 0.9, volume: '18M', mktCap: '2.1T', sector: 'Comm. Svcs' },
    { symbol: 'JPM', name: 'JPMorgan Chase', price: 198.40, change: -0.5, volume: '9M', mktCap: '580B', sector: 'Financials' },
  ],
  crypto: [
    { symbol: 'BTC-USD', name: 'Bitcoin', price: 68500.00, change: 3.2, volume: '25B', mktCap: '1.3T' },
    { symbol: 'ETH-USD', name: 'Ethereum', price: 3800.50, change: 2.1, volume: '12B', mktCap: '450B' },
    { symbol: 'SOL-USD', name: 'Solana', price: 145.20, change: 5.4, volume: '3B', mktCap: '65B' },
  ]
};

const PORTFOLIO_DATA = {
  totalValue: 1250450.00,
  dayChange: 12500.50,
  dayChangePercent: 1.01,
  cashBalance: 150000.00,
  allocation: [
    { name: 'Technology', value: 45 },
    { name: 'Financials', value: 20 },
    { name: 'Healthcare', value: 15 },
    { name: 'Crypto', value: 10 },
    { name: 'Cash', value: 10 },
  ],
  holdings: [
    { symbol: 'NVDA', quantity: 200, avgCost: 850.00, currentPrice: 1120.45, return: 31.8 },
    { symbol: 'MSFT', quantity: 500, avgCost: 380.00, currentPrice: 425.10, return: 11.8 },
    { symbol: 'BTC', quantity: 2.5, avgCost: 55000.00, currentPrice: 68500.00, return: 24.5 },
  ]
};

const ALERTS = [
  { id: 1, type: 'critical', title: 'Margin Utilization High', message: 'Portfolio margin utilization has exceeded 75%.', time: '10 mins ago' },
  { id: 2, type: 'warning', title: 'NVDA Volatility', message: 'Unusual options volume detected for NVIDIA Corp.', time: '1 hr ago' },
  { id: 3, type: 'info', title: 'Fed Minutes Released', message: 'FOMC meeting minutes have been published.', time: '2 hrs ago' },
];

const NEWS_FEED = [
  { id: 1, source: 'Bloomberg', title: 'Tech Stocks Rally on New AI Chip Announcements', time: '30m ago', sentiment: 'positive' },
  { id: 2, source: 'Reuters', title: 'Oil Prices Stabilize Amidst Geopolitical Tensions', time: '1h ago', sentiment: 'neutral' },
  { id: 3, source: 'WSJ', title: 'Fed Chair Signals Potential Rate Cut in Q4', time: '2h ago', sentiment: 'positive' },
  { id: 4, source: 'CNBC', title: 'Retail Sales Data Misses Expectations', time: '3h ago', sentiment: 'negative' },
];

// --- UTILITY COMPONENTS ---

const Card = ({ children, className = "" }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}>
    {children}
  </div>
);

const Badge = ({ type, children }) => {
  const colors = {
    success: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    danger: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    warning: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    neutral: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
    info: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  };
  return (
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[type] || colors.neutral}`}>
      {children}
    </span>
  );
};

const TrendIndicator = ({ value }) => {
  const isPositive = value >= 0;
  return (
    <div className={`flex items-center ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
      {isPositive ? <TrendingUp size={16} className="mr-1" /> : <TrendingDown size={16} className="mr-1" />}
      <span className="font-semibold">{Math.abs(value)}%</span>
    </div>
  );
};

// --- MAIN APP COMPONENT ---

export default function AdamPlatform() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showChat, setShowChat] = useState(false);

  // Navigation Handler
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard': return <DashboardView />;
      case 'market': return <MarketDataView />;
      case 'analysis': return <AnalysisView />;
      case 'portfolio': return <PortfolioView />;
      case 'simulation': return <SimulationView />;
      case 'news': return <NewsView />;
      case 'settings': return <SettingsView />;
      default: return <DashboardView />;
    }
  };

  return (
    <div className={`min-h-screen flex bg-gray-50 ${isDarkMode ? 'dark' : ''}`}>
      <div className={isDarkMode ? 'bg-gray-900 text-white min-h-screen w-full flex' : 'bg-gray-50 text-gray-900 min-h-screen w-full flex'}>
        
        {/* Sidebar */}
        <aside className={`${isSidebarOpen ? 'w-64' : 'w-20'} bg-slate-900 text-white transition-all duration-300 ease-in-out flex flex-col fixed h-full z-20`}>
          <div className="p-4 flex items-center justify-between border-b border-slate-800">
            <div className={`flex items-center gap-3 ${!isSidebarOpen && 'justify-center w-full'}`}>
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-xl">A</div>
              {isSidebarOpen && <span className="font-bold text-lg tracking-tight">ADAM v23</span>}
            </div>
            {isSidebarOpen && (
              <button onClick={() => setSidebarOpen(false)} className="text-slate-400 hover:text-white">
                <X size={20} />
              </button>
            )}
          </div>

          <nav className="flex-1 py-6 flex flex-col gap-1 px-3">
            <NavItem icon={<LayoutDashboard size={20} />} label="Dashboard" id="dashboard" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<BarChart3 size={20} />} label="Market Data" id="market" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<Cpu size={20} />} label="Analysis Agents" id="analysis" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<Briefcase size={20} />} label="Portfolio" id="portfolio" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<Activity size={20} />} label="Simulations" id="simulation" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
            <NavItem icon={<Newspaper size={20} />} label="News & Insights" id="news" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
          </nav>

          <div className="p-3 border-t border-slate-800">
             <NavItem icon={<Settings size={20} />} label="Settings" id="settings" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
             <div className="mt-4 flex items-center gap-3 p-2 rounded-lg bg-slate-800 hover:bg-slate-700 cursor-pointer transition">
                <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-sm">JD</div>
                {isSidebarOpen && (
                  <div className="flex-1 overflow-hidden">
                    <p className="text-sm font-medium truncate">John Doe</p>
                    <p className="text-xs text-slate-400 truncate">Portfolio Manager</p>
                  </div>
                )}
             </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className={`flex-1 flex flex-col transition-all duration-300 ${isSidebarOpen ? 'ml-64' : 'ml-20'}`}>
          {/* Header */}
          <header className="h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6 sticky top-0 z-10">
            <div className="flex items-center gap-4">
              {!isSidebarOpen && (
                <button onClick={() => setSidebarOpen(true)} className="text-gray-500 hover:text-gray-700 dark:text-gray-300">
                  <Menu size={24} />
                </button>
              )}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
                <input 
                  type="text" 
                  placeholder="Search assets, reports, or agents..." 
                  className="pl-10 pr-4 py-2 w-64 bg-gray-100 dark:bg-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                />
              </div>
            </div>

            <div className="flex items-center gap-4">
              <button className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full relative">
                <Bell size={20} />
                <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white dark:border-gray-800"></span>
              </button>
              <button 
                onClick={() => setIsDarkMode(!isDarkMode)}
                className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full"
              >
                {isDarkMode ? "☀️" : "🌙"}
              </button>
              <button 
                onClick={() => setShowChat(!showChat)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all ${showChat ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200'}`}
              >
                <Bot size={18} />
                <span className="font-medium">Ask Adam</span>
              </button>
            </div>
          </header>

          {/* Page Content */}
          <div className="p-6 flex-1 overflow-y-auto">
            {renderContent()}
          </div>
        </main>

        {/* Chat Drawer */}
        <AdamChat isOpen={showChat} onClose={() => setShowChat(false)} />
      </div>
    </div>
  );
}

// --- SUB-COMPONENTS ---

const NavItem = ({ icon, label, id, activeTab, setActiveTab, isOpen }) => (
  <button 
    onClick={() => setActiveTab(id)}
    className={`flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-200 w-full text-left
      ${activeTab === id ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'text-slate-400 hover:bg-slate-800 hover:text-white'}
      ${!isOpen && 'justify-center'}
    `}
    title={!isOpen ? label : ''}
  >
    {icon}
    {isOpen && <span className="font-medium">{label}</span>}
  </button>
);

// --- PAGES ---

function DashboardView() {
  return (
    <div className="space-y-6">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {MARKET_DATA.indices.map((idx) => (
          <Card key={idx.symbol} className="p-4 hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start mb-2">
              <span className="text-sm text-gray-500 font-medium">{idx.symbol}</span>
              <span className={`text-xs px-2 py-1 rounded ${idx.trend === 'up' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                {idx.trend === 'up' ? 'BULLISH' : 'BEARISH'}
              </span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {idx.price.toLocaleString()}
            </div>
            <TrendIndicator value={idx.change} />
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart Area */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="p-6 min-h-[400px]">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-bold text-gray-900 dark:text-white">Portfolio Performance</h2>
              <select className="bg-gray-50 border border-gray-300 text-sm rounded-lg p-2 dark:bg-gray-700 dark:border-gray-600 dark:text-white">
                <option>1 Day</option>
                <option>1 Week</option>
                <option>1 Month</option>
                <option>YTD</option>
              </select>
            </div>
            
            {/* Mock Chart Graphic */}
            <div className="w-full h-64 flex items-end gap-2 border-b border-l border-gray-200 dark:border-gray-700 p-4 relative">
               {/* Simple CSS Bar Chart Mockup */}
               {[40, 65, 50, 80, 70, 90, 85, 95, 100, 88, 92, 98].map((h, i) => (
                 <div key={i} className="flex-1 bg-blue-500/20 hover:bg-blue-500/50 rounded-t transition-all relative group" style={{height: `${h}%`}}>
                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                      +{(h/10).toFixed(1)}%
                    </div>
                 </div>
               ))}
            </div>
            <div className="mt-4 flex justify-between text-gray-500 text-sm">
              <span>9:30 AM</span>
              <span>12:00 PM</span>
              <span>4:00 PM</span>
            </div>
          </Card>

          {/* Investment Ideas Table */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-bold text-gray-900 dark:text-white">Adam's Investment Ideas</h2>
              <button className="text-blue-600 text-sm font-medium hover:underline">View All</button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-100 dark:border-gray-700">
                    <th className="pb-3 font-medium">Ticker</th>
                    <th className="pb-3 font-medium">Strategy</th>
                    <th className="pb-3 font-medium">Confidence</th>
                    <th className="pb-3 font-medium">Horizon</th>
                    <th className="pb-3 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                  {[
                    { t: 'PLTR', s: 'AI Expansion Momentum', c: 'High (87%)', h: '3-6m', a: 'Buy' },
                    { t: 'AMD', s: 'Data Center Growth', c: 'Med (65%)', h: '12m', a: 'Hold' },
                    { t: 'TSLA', s: 'Robotaxi Volatility', c: 'High (91%)', h: '1m', a: 'Sell Call' },
                  ].map((item, i) => (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                      <td className="py-3 font-semibold text-blue-600">{item.t}</td>
                      <td className="py-3">{item.s}</td>
                      <td className="py-3">
                        <Badge type={item.c.includes('High') ? 'success' : 'warning'}>{item.c}</Badge>
                      </td>
                      <td className="py-3 text-gray-500">{item.h}</td>
                      <td className="py-3 text-right">
                        <button className="text-xs bg-gray-900 text-white px-3 py-1 rounded hover:bg-gray-700">Analyze</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Right Column - Alerts & Allocation */}
        <div className="space-y-6">
          <Card className="p-6">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
              <ShieldAlert className="text-red-500" size={20} />
              System Alerts
            </h2>
            <div className="space-y-4">
              {ALERTS.map(alert => (
                <div key={alert.id} className="p-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg border-l-4 border-red-500">
                  <div className="flex justify-between items-start">
                    <h3 className="font-semibold text-sm text-gray-900 dark:text-white">{alert.title}</h3>
                    <span className="text-xs text-gray-400">{alert.time}</span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{alert.message}</p>
                </div>
              ))}
            </div>
            <button className="w-full mt-4 py-2 text-sm text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition">View All Alerts</button>
          </Card>

          <Card className="p-6">
            <h2 className="text-lg font-bold mb-4">Asset Allocation</h2>
            <div className="flex items-center justify-center h-48 relative">
               {/* SVG Donut Chart Mock */}
               <svg width="160" height="160" viewBox="0 0 100 100" className="transform -rotate-90">
                 <circle cx="50" cy="50" r="40" fill="transparent" stroke="#e2e8f0" strokeWidth="20" />
                 <circle cx="50" cy="50" r="40" fill="transparent" stroke="#3b82f6" strokeWidth="20" strokeDasharray="120 251" /> 
                 <circle cx="50" cy="50" r="40" fill="transparent" stroke="#10b981" strokeWidth="20" strokeDasharray="60 251" strokeDashoffset="-120" />
               </svg>
               <div className="absolute text-center">
                 <span className="block text-2xl font-bold">1.2M</span>
                 <span className="text-xs text-gray-500">USD</span>
               </div>
            </div>
            <div className="space-y-2 mt-4">
              {PORTFOLIO_DATA.allocation.map(a => (
                <div key={a.name} className="flex justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                    {a.name}
                  </span>
                  <span className="font-medium">{a.value}%</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function MarketDataView() {
  const [subTab, setSubTab] = useState('stocks');
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Market Data</h1>
        <div className="flex bg-gray-200 dark:bg-gray-700 rounded-lg p-1">
          {['Stocks', 'Crypto', 'Forex', 'Bonds'].map(tab => (
            <button 
              key={tab}
              onClick={() => setSubTab(tab.toLowerCase())}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${subTab === tab.toLowerCase() ? 'bg-white dark:bg-gray-600 shadow-sm text-gray-900 dark:text-white' : 'text-gray-500 dark:text-gray-400'}`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <Card className="overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead className="bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 text-xs uppercase">
            <tr>
              <th className="px-6 py-4 font-semibold">Symbol</th>
              <th className="px-6 py-4 font-semibold">Name</th>
              <th className="px-6 py-4 font-semibold">Sector</th>
              <th className="px-6 py-4 font-semibold text-right">Price</th>
              <th className="px-6 py-4 font-semibold text-right">Change</th>
              <th className="px-6 py-4 font-semibold text-right">Volume</th>
              <th className="px-6 py-4 font-semibold text-right">Mkt Cap</th>
              <th className="px-6 py-4 font-semibold text-center">Analysis</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {(subTab === 'stocks' ? MARKET_DATA.stocks : MARKET_DATA.crypto).map((asset, i) => (
              <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors group">
                <td className="px-6 py-4 font-bold text-blue-600 cursor-pointer">{asset.symbol}</td>
                <td className="px-6 py-4 font-medium text-gray-900 dark:text-white">{asset.name}</td>
                <td className="px-6 py-4 text-gray-500">{asset.sector || 'N/A'}</td>
                <td className="px-6 py-4 text-right font-mono">${asset.price.toLocaleString()}</td>
                <td className={`px-6 py-4 text-right font-medium ${asset.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {asset.change > 0 ? '+' : ''}{asset.change}%
                </td>
                <td className="px-6 py-4 text-right text-gray-500">{asset.volume}</td>
                <td className="px-6 py-4 text-right text-gray-500">{asset.mktCap}</td>
                <td className="px-6 py-4 text-center">
                  <button className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 hover:bg-blue-50 text-blue-600 rounded">
                    <Activity size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function AnalysisView() {
  const agents = [
    { name: "Fundamental Analyst", desc: "Deep dive into company financials, 10-K/10-Q reports, and earnings transcripts.", icon: <FileText size={24} className="text-blue-500" />, status: "Active" },
    { name: "Technical Analyst", desc: "Chart pattern recognition, indicator analysis (RSI, MACD), and price target forecasting.", icon: <LineChart size={24} className="text-purple-500" />, status: "Active" },
    { name: "Risk Sentinel", desc: "Real-time assessment of market, credit, and operational risks using macro data.", icon: <ShieldAlert size={24} className="text-red-500" />, status: "Active" },
    { name: "Macro Economist", desc: "Global economic trend analysis focusing on interest rates, GDP, and geopolitical shifts.", icon: <TrendingUp size={24} className="text-green-500" />, status: "Idle" },
    { name: "Sentiment Engine", desc: "Social media and news sentiment analysis to gauge market psychology.", icon: <Activity size={24} className="text-orange-500" />, status: "Processing" },
    { name: "SNC Reviewer", desc: "Shared National Credit program analysis and regulatory compliance checking.", icon: <Briefcase size={24} className="text-gray-500" />, status: "Idle" },
  ];

  return (
    <div className="space-y-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analysis Center</h1>
        <p className="text-gray-500 mt-1">Deploy specialized autonomous agents to analyze specific market segments or companies.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent, i) => (
          <Card key={i} className="p-6 hover:border-blue-400 transition-colors cursor-pointer flex flex-col h-full">
            <div className="flex justify-between items-start mb-4">
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                {agent.icon}
              </div>
              <span className={`text-xs font-medium px-2 py-1 rounded-full flex items-center gap-1 ${
                agent.status === 'Active' ? 'bg-green-100 text-green-700' : 
                agent.status === 'Processing' ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-600'
              }`}>
                {agent.status === 'Processing' && <RefreshCw size={10} className="animate-spin" />}
                {agent.status}
              </span>
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{agent.name}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 flex-1">{agent.desc}</p>
            <button className="mt-6 w-full py-2 bg-gray-900 dark:bg-white dark:text-gray-900 text-white rounded-lg font-medium text-sm hover:opacity-90 transition">
              Deploy Agent
            </button>
          </Card>
        ))}
      </div>

      <div className="mt-8">
        <h2 className="text-xl font-bold mb-4">Recent Agent Reports</h2>
        <div className="grid gap-4">
          {[
            { title: "NVIDIA Corp (NVDA) - Q2 Earnings Preview", agent: "Fundamental Analyst", date: "2 hours ago", tags: ["Tech", "Semis"] },
            { title: "Macro Outlook: Impact of Oil Shock on CPI", agent: "Macro Economist", date: "5 hours ago", tags: ["Macro", "Energy"] },
            { title: "Technical Breakout Detection: SOL-USD", agent: "Technical Analyst", date: "1 day ago", tags: ["Crypto", "Technical"] }
          ].map((report, i) => (
            <Card key={i} className="p-4 flex items-center justify-between hover:bg-gray-50 transition">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-blue-100 rounded flex items-center justify-center text-blue-600">
                  <FileText size={20} />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">{report.title}</h4>
                  <div className="flex items-center gap-2 text-xs text-gray-500 mt-1">
                    <span>{report.agent}</span>
                    <span>•</span>
                    <span>{report.date}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {report.tags.map(tag => (
                  <span key={tag} className="text-xs bg-gray-100 px-2 py-1 rounded text-gray-600">{tag}</span>
                ))}
                <button className="p-2 hover:bg-gray-200 rounded-full"><ArrowRight size={16} /></button>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

function PortfolioView() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Portfolio Management</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="p-5 lg:col-span-3">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold">Holdings Breakdown</h3>
            <button className="text-sm bg-blue-600 text-white px-3 py-1.5 rounded hover:bg-blue-700">Rebalance</button>
          </div>
          <table className="w-full text-left border-collapse">
            <thead className="text-xs uppercase text-gray-500 bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3">Asset</th>
                <th className="px-4 py-3 text-right">Qty</th>
                <th className="px-4 py-3 text-right">Avg Cost</th>
                <th className="px-4 py-3 text-right">Current</th>
                <th className="px-4 py-3 text-right">Total Value</th>
                <th className="px-4 py-3 text-right">Return</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {PORTFOLIO_DATA.holdings.map((pos, i) => (
                <tr key={i} className="border-b border-gray-100">
                  <td className="px-4 py-3 font-medium text-blue-600">{pos.symbol}</td>
                  <td className="px-4 py-3 text-right">{pos.quantity}</td>
                  <td className="px-4 py-3 text-right">${pos.avgCost.toLocaleString()}</td>
                  <td className="px-4 py-3 text-right">${pos.currentPrice.toLocaleString()}</td>
                  <td className="px-4 py-3 text-right font-bold">${(pos.quantity * pos.currentPrice).toLocaleString()}</td>
                  <td className="px-4 py-3 text-right text-green-600">+{pos.return}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card className="p-5">
          <h3 className="font-bold mb-4">Risk Metrics</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-500">Beta</span>
                <span className="font-medium">1.24</span>
              </div>
              <div className="w-full bg-gray-200 h-2 rounded-full"><div className="bg-yellow-500 h-2 rounded-full w-3/4"></div></div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-500">Sharpe Ratio</span>
                <span className="font-medium">1.8</span>
              </div>
              <div className="w-full bg-gray-200 h-2 rounded-full"><div className="bg-green-500 h-2 rounded-full w-2/3"></div></div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-500">VaR (95%)</span>
                <span className="font-medium">$12,400</span>
              </div>
              <div className="w-full bg-gray-200 h-2 rounded-full"><div className="bg-red-500 h-2 rounded-full w-1/4"></div></div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function SimulationView() {
  const [isSimulating, setIsSimulating] = useState(false);
  const [progress, setProgress] = useState(0);

  const runSimulation = () => {
    setIsSimulating(true);
    setProgress(0);
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsSimulating(false);
          return 100;
        }
        return prev + 5;
      });
    }, 150);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Scenario Simulation Engine</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="p-6 border-2 border-transparent hover:border-blue-500 cursor-pointer transition-all">
          <h3 className="font-bold text-lg mb-2">Monte Carlo</h3>
          <p className="text-sm text-gray-500 mb-4">Run 10,000 iterations to project future portfolio value based on historical volatility.</p>
          <button onClick={runSimulation} className="w-full py-2 bg-blue-50 text-blue-600 rounded hover:bg-blue-100 font-medium">Run Simulation</button>
        </Card>
        
        <Card className="p-6 border-2 border-transparent hover:border-red-500 cursor-pointer transition-all">
          <h3 className="font-bold text-lg mb-2">Black Swan Event</h3>
          <p className="text-sm text-gray-500 mb-4">Stress test portfolio against extreme market events (e.g., 2008 crash, Pandemic).</p>
          <button onClick={runSimulation} className="w-full py-2 bg-red-50 text-red-600 rounded hover:bg-red-100 font-medium">Stress Test</button>
        </Card>

        <Card className="p-6 border-2 border-transparent hover:border-purple-500 cursor-pointer transition-all">
          <h3 className="font-bold text-lg mb-2">Inflation Shock</h3>
          <p className="text-sm text-gray-500 mb-4">Simulate impact of sustained 5% inflation and 50bps rate hikes.</p>
          <button onClick={runSimulation} className="w-full py-2 bg-purple-50 text-purple-600 rounded hover:bg-purple-100 font-medium">Analyze Impact</button>
        </Card>
      </div>

      {/* Simulation Output Area */}
      <Card className="p-8 min-h-[300px] flex flex-col items-center justify-center">
        {isSimulating ? (
          <div className="w-full max-w-md text-center">
            <RefreshCw size={40} className="animate-spin text-blue-600 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Running Complex Models...</h3>
            <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
              <div className="bg-blue-600 h-2.5 rounded-full transition-all duration-200" style={{ width: `${progress}%` }}></div>
            </div>
            <p className="text-sm text-gray-500 mt-2">Calculating risk vectors and liquidity correlations...</p>
          </div>
        ) : progress === 100 ? (
          <div className="w-full">
            <div className="flex items-center gap-2 mb-6 text-green-600">
              <div className="w-2 h-2 rounded-full bg-green-600 animate-pulse"></div>
              <span className="font-bold uppercase text-sm">Simulation Complete</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                    <div className="text-xs text-gray-500 uppercase">Est. Loss</div>
                    <div className="text-xl font-bold text-red-600">-12.4%</div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                    <div className="text-xs text-gray-500 uppercase">Recovery Time</div>
                    <div className="text-xl font-bold text-gray-900">14 Months</div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                    <div className="text-xs text-gray-500 uppercase">Vulnerability</div>
                    <div className="text-xl font-bold text-orange-600">Tech Sector</div>
                </div>
            </div>
            <div className="h-64 bg-gray-50 rounded-lg border border-dashed border-gray-300 flex items-center justify-center text-gray-400">
                [Interactive Projection Chart Visualization Would Render Here]
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-400">
            <Activity size={48} className="mx-auto mb-3 opacity-50" />
            <p>Select a scenario above to generate risk projections.</p>
          </div>
        )}
      </Card>
    </div>
  );
}

function NewsView() {
  return (
    <div className="space-y-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Market Intelligence</h1>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
                {NEWS_FEED.map(news => (
                    <Card key={news.id} className="p-4 flex gap-4 hover:bg-gray-50 cursor-pointer">
                        <div className="w-24 h-24 bg-gray-200 rounded-lg flex-shrink-0 overflow-hidden">
                            <img src={`/api/placeholder/100/100`} alt="news thumbnail" className="w-full h-full object-cover opacity-50" />
                        </div>
                        <div className="flex-1">
                            <div className="flex justify-between mb-1">
                                <span className="text-xs font-bold text-blue-600 uppercase">{news.source}</span>
                                <span className="text-xs text-gray-400">{news.time}</span>
                            </div>
                            <h3 className="font-bold text-gray-900 text-lg leading-tight mb-2">{news.title}</h3>
                            <p className="text-sm text-gray-500">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua...</p>
                            <div className="mt-3 flex gap-2">
                                <span className={`text-xs px-2 py-0.5 rounded ${news.sentiment === 'positive' ? 'bg-green-100 text-green-700' : news.sentiment === 'negative' ? 'bg-red-100 text-red-700' : 'bg-gray-100'}`}>
                                    {news.sentiment} sentiment
                                </span>
                            </div>
                        </div>
                    </Card>
                ))}
            </div>
            <div className="space-y-6">
                <Card className="p-5">
                    <h3 className="font-bold mb-3">Trending Topics</h3>
                    <div className="flex flex-wrap gap-2">
                        {['#AI', '#Inflation', '#Fed', '#Nvidia', '#Bitcoin', '#Oil', '#Recession', '#Earnings'].map(tag => (
                            <span key={tag} className="text-sm bg-gray-100 px-3 py-1 rounded-full text-gray-600 hover:bg-gray-200 cursor-pointer">{tag}</span>
                        ))}
                    </div>
                </Card>
                <Card className="p-5 bg-blue-600 text-white">
                    <h3 className="font-bold mb-2 flex items-center gap-2"><Bot size={18}/> Insight Summary</h3>
                    <p className="text-sm opacity-90 leading-relaxed">
                        Market sentiment is currently shifting towards "Risk-On" driven by stronger than expected tech earnings. However, bond yields suggest caution regarding long-term inflation targets.
                    </p>
                </Card>
            </div>
        </div>
    </div>
  );
}

function SettingsView() {
    return (
        <div className="space-y-6 max-w-4xl">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">System Configuration</h1>
            <Card className="divide-y divide-gray-200">
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900">API Keys</h3>
                        <p className="text-sm text-gray-500">Manage connections to Bloomberg, Alpaca, and OpenAI</p>
                    </div>
                    <button className="text-blue-600 font-medium text-sm">Manage</button>
                </div>
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900">Notification Preferences</h3>
                        <p className="text-sm text-gray-500">Email and push notification settings for high-risk alerts</p>
                    </div>
                    <button className="text-blue-600 font-medium text-sm">Edit</button>
                </div>
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900">Risk Parameters</h3>
                        <p className="text-sm text-gray-500">Set thresholds for VaR and Drawdown alerts</p>
                    </div>
                    <button className="text-blue-600 font-medium text-sm">Configure</button>
                </div>
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900">Data Sources</h3>
                        <p className="text-sm text-gray-500">Connect external databases and knowledge graphs</p>
                    </div>
                    <button className="text-blue-600 font-medium text-sm">Connect</button>
                </div>
            </Card>
        </div>
    )
}

// --- CHAT COMPONENT ---

const AdamChat = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'adam', text: "Hello, I'm Adam. I'm monitoring your portfolio and the broader markets. How can I assist with your analysis today?" }
  ]);
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    
    const userMsg = { id: Date.now(), sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");

    // Mock Response Logic
    setTimeout(() => {
      let responseText = "I'm processing that request. I can run a Monte Carlo simulation or analyze specific tickers.";
      if (input.toLowerCase().includes('nvda')) {
        responseText = "NVIDIA (NVDA) is currently trading at $1120.45. My technical agents detect a bullish flag pattern forming on the 4H chart. Volume is 20% above average.";
      } else if (input.toLowerCase().includes('risk')) {
        responseText = "Your current portfolio VaR (95%) is $12,400. The largest contributor to risk is your position in the Technology sector (45% allocation).";
      } else if (input.toLowerCase().includes('buy') || input.toLowerCase().includes('sell')) {
        responseText = "I cannot provide specific financial advice, but I can generate a counterfactual scenario report to see how that trade might impact your portfolio's Sharpe ratio.";
      }

      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'adam', text: responseText }]);
    }, 1000);
  };

  return (
    <div className={`fixed inset-y-0 right-0 w-96 bg-white dark:bg-gray-900 shadow-2xl transform transition-transform duration-300 z-50 border-l border-gray-200 dark:border-gray-700 flex flex-col ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
      {/* Chat Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex justify-between items-center bg-slate-900 text-white">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
          <span className="font-bold">Adam Assistant</span>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={18} /></button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50 dark:bg-gray-900" ref={scrollRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-lg text-sm shadow-sm ${
              msg.sender === 'user' 
                ? 'bg-blue-600 text-white rounded-br-none' 
                : 'bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-700 rounded-bl-none'
            }`}>
              {msg.text}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="relative">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Type a command or question..." 
            className="w-full pr-10 pl-4 py-3 bg-gray-100 dark:bg-gray-800 border-none rounded-xl text-sm focus:ring-2 focus:ring-blue-500 outline-none"
          />
          <button 
            onClick={handleSend}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-blue-600 p-1 hover:bg-blue-50 rounded transition"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};
