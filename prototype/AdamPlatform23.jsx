import React, { useState, useEffect, useRef, useMemo } from 'react';
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
  Globe,
  Zap,
  Lock,
  Eye,
  Download,
  Filter
} from 'lucide-react';

/**
 * ADAM PLATFORM v23.1 - Enhanced Single File Implementation
 * Features: Real-time data simulation, SVG Charting, Agent Orchestration, Interactive UI
 */

// --- UTILITIES & HOOKS ---

const formatCurrency = (value) => {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
};

const formatCompactNumber = (number) => {
  return new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(number);
};

// Hook to simulate live data updates
const useLiveMarketData = (initialData) => {
  const [data, setData] = useState(initialData);

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prevData => {
        const updatePrice = (asset) => {
          const volatility = 0.002; // 0.2% volatility
          const change = 1 + (Math.random() * volatility * 2 - volatility);
          const newPrice = asset.price * change;
          return {
            ...asset,
            price: newPrice,
            change: asset.change + (change - 1) * 100 // Update percentage change roughly
          };
        };

        return {
          ...prevData,
          indices: prevData.indices.map(updatePrice),
          stocks: prevData.stocks.map(updatePrice),
          crypto: prevData.crypto.map(updatePrice)
        };
      });
    }, 3000); // Update every 3 seconds

    return () => clearInterval(interval);
  }, []);

  return data;
};

// --- MOCK DATA STORES ---

const INITIAL_MARKET_DATA = {
  indices: [
    { symbol: 'S&P 500', price: 5432.12, change: 1.2, trend: 'up', history: [5350, 5380, 5360, 5400, 5410, 5432] },
    { symbol: 'NASDAQ', price: 17654.30, change: 0.8, trend: 'up', history: [17200, 17300, 17250, 17400, 17550, 17654] },
    { symbol: 'DOW J', price: 39876.50, change: -0.2, trend: 'down', history: [40000, 39950, 39900, 39920, 39850, 39876] },
    { symbol: 'VIX', price: 13.45, change: -5.4, trend: 'down', history: [14.2, 14.0, 13.8, 13.9, 13.6, 13.45] },
  ],
  stocks: [
    { symbol: 'NVDA', name: 'NVIDIA Corp', price: 1120.45, change: 2.5, volume: '45M', mktCap: '2.8T', sector: 'Technology', pe: 72.5, div: '0.04%' },
    { symbol: 'MSFT', name: 'Microsoft', price: 425.10, change: 0.5, volume: '22M', mktCap: '3.1T', sector: 'Technology', pe: 35.2, div: '0.71%' },
    { symbol: 'AAPL', name: 'Apple Inc', price: 195.30, change: -0.1, volume: '30M', mktCap: '2.9T', sector: 'Technology', pe: 28.4, div: '0.52%' },
    { symbol: 'AMZN', name: 'Amazon.com', price: 182.50, change: 1.1, volume: '28M', mktCap: '1.9T', sector: 'Cons. Disc.', pe: 51.2, div: '-' },
    { symbol: 'GOOGL', name: 'Alphabet Inc', price: 175.20, change: 0.9, volume: '18M', mktCap: '2.1T', sector: 'Comm. Svcs', pe: 24.8, div: '0.46%' },
    { symbol: 'JPM', name: 'JPMorgan Chase', price: 198.40, change: -0.5, volume: '9M', mktCap: '580B', sector: 'Financials', pe: 11.5, div: '2.32%' },
    { symbol: 'TSLA', name: 'Tesla Inc', price: 178.90, change: -1.2, volume: '35M', mktCap: '570B', sector: 'Cons. Disc.', pe: 45.1, div: '-' },
    { symbol: 'META', name: 'Meta Platforms', price: 475.60, change: 1.8, volume: '15M', mktCap: '1.2T', sector: 'Comm. Svcs', pe: 29.3, div: '0.42%' },
  ],
  crypto: [
    { symbol: 'BTC-USD', name: 'Bitcoin', price: 68500.00, change: 3.2, volume: '25B', mktCap: '1.3T' },
    { symbol: 'ETH-USD', name: 'Ethereum', price: 3800.50, change: 2.1, volume: '12B', mktCap: '450B' },
    { symbol: 'SOL-USD', name: 'Solana', price: 145.20, change: 5.4, volume: '3B', mktCap: '65B' },
    { symbol: 'XRP-USD', name: 'XRP', price: 0.52, change: -0.5, volume: '800M', mktCap: '29B' },
  ]
};

const PORTFOLIO_DATA = {
  totalValue: 1250450.00,
  dayChange: 12500.50,
  dayChangePercent: 1.01,
  cashBalance: 150000.00,
  marginUsed: 25000.00,
  buyingPower: 275000.00,
  allocation: [
    { name: 'Technology', value: 45, color: '#3b82f6' },
    { name: 'Financials', value: 20, color: '#10b981' },
    { name: 'Healthcare', value: 15, color: '#8b5cf6' },
    { name: 'Crypto', value: 10, color: '#f59e0b' },
    { name: 'Cash', value: 10, color: '#64748b' },
  ],
  holdings: [
    { symbol: 'NVDA', quantity: 200, avgCost: 850.00, currentPrice: 1120.45, return: 31.8, type: 'Stock' },
    { symbol: 'MSFT', quantity: 500, avgCost: 380.00, currentPrice: 425.10, return: 11.8, type: 'Stock' },
    { symbol: 'BTC', quantity: 2.5, avgCost: 55000.00, currentPrice: 68500.00, return: 24.5, type: 'Crypto' },
    { symbol: 'AAPL', quantity: 300, avgCost: 180.00, currentPrice: 195.30, return: 8.5, type: 'Stock' },
    { symbol: 'JPM', quantity: 150, avgCost: 190.00, currentPrice: 198.40, return: 4.4, type: 'Stock' },
  ],
  transactions: [
    { id: 'TXN-001', date: '2023-10-24', symbol: 'NVDA', type: 'Buy', quantity: 50, price: 1115.00, total: 55750.00 },
    { id: 'TXN-002', date: '2023-10-23', symbol: 'AAPL', type: 'Sell', quantity: 100, price: 194.50, total: 19450.00 },
    { id: 'TXN-003', date: '2023-10-20', symbol: 'BTC-USD', type: 'Buy', quantity: 0.5, price: 67200.00, total: 33600.00 },
  ]
};

const ALERTS = [
  { id: 1, type: 'critical', title: 'Margin Utilization High', message: 'Portfolio margin utilization has exceeded 75%. Immediate action recommended.', time: '10 mins ago' },
  { id: 2, type: 'warning', title: 'NVDA Volatility Alert', message: 'Unusual options volume detected for NVIDIA Corp. Implied volatility +15%.', time: '1 hr ago' },
  { id: 3, type: 'info', title: 'Fed Minutes Released', message: 'FOMC meeting minutes have been published. Market sentiment shifting slightly dovish.', time: '2 hrs ago' },
  { id: 4, type: 'success', title: 'Dividend Received', message: 'Received $450.25 dividend payment from JPMorgan Chase.', time: '1 day ago' },
];

const NEWS_FEED = [
  { id: 1, source: 'Bloomberg', title: 'Tech Stocks Rally on New AI Chip Announcements', time: '30m ago', sentiment: 'positive', summary: 'Major semiconductor companies saw a boost in pre-market trading as new architecture was revealed.' },
  { id: 2, source: 'Reuters', title: 'Oil Prices Stabilize Amidst Geopolitical Tensions', time: '1h ago', sentiment: 'neutral', summary: 'Brent crude holds steady at $82/barrel as supply chain concerns ease slightly.' },
  { id: 3, source: 'WSJ', title: 'Fed Chair Signals Potential Rate Cut in Q4', time: '2h ago', sentiment: 'positive', summary: 'In a surprise speech, the Chair hinted that inflation targets are approaching acceptable levels.' },
  { id: 4, source: 'CNBC', title: 'Retail Sales Data Misses Expectations', time: '3h ago', sentiment: 'negative', summary: 'Consumer spending slowed in September, raising fears of a potential recessionary cool-down.' },
  { id: 5, source: 'TechCrunch', title: 'Quantum Computing Breakthrough', time: '5h ago', sentiment: 'positive', summary: 'Researchers achieve stable qubits at room temperature for 5ms.' },
];

// --- CUSTOM SVG CHART COMPONENTS ---

const SimpleLineChart = ({ data, color = "#3b82f6", height = 60 }) => {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  
  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = 100 - ((val - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible" preserveAspectRatio="none">
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        points={points}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
};

const AreaChart = ({ data, height = 200 }) => {
  if (!data || data.length === 0) return null;
  const min = Math.min(...data) * 0.99;
  const max = Math.max(...data) * 1.01;
  const range = max - min;
  
  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = 100 - ((val - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="w-full" style={{ height: `${height}px` }}>
      <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible" preserveAspectRatio="none">
        <defs>
          <linearGradient id="chartGradient" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.4"/>
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0"/>
          </linearGradient>
        </defs>
        <polygon
          points={`0,100 ${points} 100,100`}
          fill="url(#chartGradient)"
        />
        <polyline
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
          points={points}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </div>
  );
};

const DonutChart = ({ data }) => {
  let cumulativePercent = 0;
  
  const getCoordinatesForPercent = (percent) => {
    const x = Math.cos(2 * Math.PI * percent);
    const y = Math.sin(2 * Math.PI * percent);
    return [x, y];
  };

  return (
    <svg viewBox="-1 -1 2 2" className="transform -rotate-90 w-full h-full">
      {data.map((slice, i) => {
        const startPercent = cumulativePercent;
        const endPercent = cumulativePercent + (slice.value / 100);
        cumulativePercent = endPercent;

        const [startX, startY] = getCoordinatesForPercent(startPercent);
        const [endX, endY] = getCoordinatesForPercent(endPercent);
        const largeArcFlag = slice.value > 50 ? 1 : 0;

        const pathData = [
          `M ${startX} ${startY}`,
          `A 1 1 0 ${largeArcFlag} 1 ${endX} ${endY}`,
          `L 0 0`,
        ].join(' ');

        return (
          <path 
            key={i} 
            d={pathData} 
            fill={slice.color} 
            stroke="white" 
            strokeWidth="0.05"
            className="hover:opacity-80 transition-opacity cursor-pointer"
          />
        );
      })}
      <circle cx="0" cy="0" r="0.6" fill="currentColor" className="text-white dark:text-gray-800" />
    </svg>
  );
};

// --- UTILITY COMPONENTS ---

const Card = ({ children, className = "", title, action }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col ${className}`}>
    {(title || action) && (
      <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center">
        {title && <h3 className="font-semibold text-gray-900 dark:text-white">{title}</h3>}
        {action && <div>{action}</div>}
      </div>
    )}
    <div className="flex-1 relative">
      {children}
    </div>
  </div>
);

const Badge = ({ type, children }) => {
  const colors = {
    success: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    danger: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    warning: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    neutral: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
    info: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
  };
  return (
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[type] || colors.neutral}`}>
      {children}
    </span>
  );
};

const TrendIndicator = ({ value, className="" }) => {
  const isPositive = value >= 0;
  return (
    <div className={`flex items-center ${isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'} ${className}`}>
      {isPositive ? <TrendingUp size={16} className="mr-1" /> : <TrendingDown size={16} className="mr-1" />}
      <span className="font-semibold">{Math.abs(value).toFixed(2)}%</span>
    </div>
  );
};

const Modal = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-2xl overflow-hidden transform transition-all scale-100">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">{title}</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white">
            <X size={24} />
          </button>
        </div>
        <div className="p-6 max-h-[80vh] overflow-y-auto">
          {children}
        </div>
      </div>
    </div>
  );
};

// --- MAIN APP COMPONENT ---

export default function AdamPlatform() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false); // Default to light for better readability initially
  const [showChat, setShowChat] = useState(false);
  const marketData = useLiveMarketData(INITIAL_MARKET_DATA);

  // Effect to toggle body class for Tailwind Dark Mode
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard': return <DashboardView data={marketData} />;
      case 'market': return <MarketDataView data={marketData} />;
      case 'analysis': return <AnalysisView />;
      case 'portfolio': return <PortfolioView data={marketData} />;
      case 'simulation': return <SimulationView />;
      case 'news': return <NewsView />;
      case 'settings': return <SettingsView isDarkMode={isDarkMode} setIsDarkMode={setIsDarkMode} />;
      default: return <DashboardView data={marketData} />;
    }
  };

  return (
    <div className={`min-h-screen flex bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-white font-sans transition-colors duration-200`}>
      
      {/* Sidebar */}
      <aside className={`${isSidebarOpen ? 'w-64' : 'w-20'} bg-slate-900 text-white transition-all duration-300 ease-in-out flex flex-col fixed h-full z-20 shadow-xl`}>
        <div className="p-4 flex items-center justify-between border-b border-slate-800 h-16">
          <div className={`flex items-center gap-3 ${!isSidebarOpen && 'justify-center w-full'}`}>
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-xl shadow-lg shadow-blue-500/30">A</div>
            {isSidebarOpen && <span className="font-bold text-lg tracking-tight">ADAM v23.1</span>}
          </div>
          {isSidebarOpen && (
            <button onClick={() => setSidebarOpen(false)} className="text-slate-400 hover:text-white transition-colors">
              <X size={20} />
            </button>
          )}
        </div>

        <nav className="flex-1 py-6 flex flex-col gap-2 px-3 overflow-y-auto scrollbar-hide">
          <NavItem icon={<LayoutDashboard size={20} />} label="Dashboard" id="dashboard" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
          <NavItem icon={<BarChart3 size={20} />} label="Market Data" id="market" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
          <NavItem icon={<Cpu size={20} />} label="Analysis Agents" id="analysis" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
          <NavItem icon={<Briefcase size={20} />} label="Portfolio" id="portfolio" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
          <NavItem icon={<Activity size={20} />} label="Simulations" id="simulation" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
          <NavItem icon={<Newspaper size={20} />} label="News & Insights" id="news" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
        </nav>

        <div className="p-3 border-t border-slate-800 mt-auto">
           <NavItem icon={<Settings size={20} />} label="Settings" id="settings" activeTab={activeTab} setActiveTab={setActiveTab} isOpen={isSidebarOpen} />
           <div className={`mt-4 flex items-center gap-3 p-2 rounded-lg bg-slate-800/50 hover:bg-slate-800 cursor-pointer transition ${!isSidebarOpen && 'justify-center'}`}>
             <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-sm font-medium shadow-md">JD</div>
             {isSidebarOpen && (
               <div className="flex-1 overflow-hidden">
                 <p className="text-sm font-medium truncate">John Doe</p>
                 <p className="text-xs text-slate-400 truncate">Portfolio Manager</p>
               </div>
             )}
           </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className={`flex-1 flex flex-col transition-all duration-300 ${isSidebarOpen ? 'ml-64' : 'ml-20'}`}>
        {/* Header */}
        <header className="h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6 sticky top-0 z-10 shadow-sm">
          <div className="flex items-center gap-4">
            {!isSidebarOpen && (
              <button onClick={() => setSidebarOpen(true)} className="text-gray-500 hover:text-gray-700 dark:text-gray-300">
                <Menu size={24} />
              </button>
            )}
            <div className="relative hidden md:block">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
              <input 
                type="text" 
                placeholder="Search assets, reports, or agents..." 
                className="pl-10 pr-4 py-2 w-64 lg:w-96 bg-gray-100 dark:bg-gray-700/50 border border-transparent focus:border-blue-500 dark:focus:border-blue-500 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all"
              />
            </div>
          </div>

          <div className="flex items-center gap-3 sm:gap-4">
            <div className="hidden sm:flex items-center gap-2 bg-green-50 dark:bg-green-900/20 px-3 py-1.5 rounded-full border border-green-100 dark:border-green-800/30">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                <span className="text-xs font-medium text-green-700 dark:text-green-400">Market Open</span>
            </div>

            <button className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full relative transition-colors">
              <Bell size={20} />
              <span className="absolute top-1.5 right-1.5 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white dark:border-gray-800"></span>
            </button>
            
            <button 
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
              title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {isDarkMode ? <Zap size={20} className="text-yellow-400 fill-current" /> : <Activity size={20} />}
            </button>

            <button 
              onClick={() => setShowChat(!showChat)}
              className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all shadow-sm hover:shadow-md ${showChat ? 'bg-blue-600 text-white ring-2 ring-blue-300 dark:ring-blue-900' : 'bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 text-gray-700 dark:text-gray-200'}`}
            >
              <Bot size={18} />
              <span className="font-medium hidden sm:inline">Ask Adam</span>
            </button>
          </div>
        </header>

        {/* Page Content */}
        <div className="p-6 flex-1 overflow-y-auto custom-scrollbar">
          {renderContent()}
        </div>
      </main>

      {/* Chat Drawer */}
      <AdamChat isOpen={showChat} onClose={() => setShowChat(false)} currentContext={activeTab} />
    </div>
  );
}

// --- SUB-COMPONENTS & VIEWS ---

const NavItem = ({ icon, label, id, activeTab, setActiveTab, isOpen }) => (
  <button 
    onClick={() => setActiveTab(id)}
    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 w-full text-left group
      ${activeTab === id ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'text-slate-400 hover:bg-slate-800 hover:text-white'}
      ${!isOpen && 'justify-center'}
    `}
    title={!isOpen ? label : ''}
  >
    <span className={`${activeTab === id ? 'text-white' : 'text-slate-400 group-hover:text-white'}`}>{icon}</span>
    {isOpen && <span className="font-medium text-sm">{label}</span>}
    {activeTab === id && isOpen && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-white shadow-sm"></div>}
  </button>
);

// --- VIEW: DASHBOARD ---

function DashboardView({ data }) {
  const historyData = [45000, 46000, 45500, 47000, 48200, 47800, 49000, 50500, 51200, 50800, 52000, 53500];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header Stats */}
      <div>
        <h1 className="text-2xl font-bold mb-1">Dashboard Overview</h1>
        <p className="text-gray-500 dark:text-gray-400 text-sm">Welcome back, John. Here is your daily portfolio digest.</p>
      </div>

      {/* Market Indices Ticker */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {data.indices.map((idx) => (
          <Card key={idx.symbol} className="p-4 hover:shadow-md transition-all group">
            <div className="flex justify-between items-start mb-2">
              <span className="text-sm text-gray-500 font-bold tracking-wide">{idx.symbol}</span>
              <span className={`text-xs px-2 py-0.5 rounded font-medium uppercase ${idx.trend === 'up' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                {idx.trend}
              </span>
            </div>
            <div className="flex items-end justify-between">
              <div>
                <div className="text-2xl font-bold font-mono tracking-tight">{idx.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                <TrendIndicator value={idx.change} className="text-xs mt-1" />
              </div>
              <div className="w-16 h-10 opacity-50 group-hover:opacity-100 transition-opacity">
                <SimpleLineChart data={idx.history || [10, 12, 11, 14]} color={idx.trend === 'up' ? '#16a34a' : '#dc2626'} />
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart Area */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="p-6" title="Portfolio Performance" action={
            <select defaultValue="1 Month" className="bg-gray-50 border border-gray-200 text-xs rounded-lg px-2 py-1 outline-none focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white">
              <option>1 Day</option>
              <option>1 Week</option>
              <option>1 Month</option>
              <option>YTD</option>
            </select>
          }>
            <div className="mb-4 flex gap-8">
              <div>
                <p className="text-xs text-gray-500 uppercase">Total Value</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">{formatCurrency(PORTFOLIO_DATA.totalValue)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase">Day Gain</p>
                <p className="text-3xl font-bold text-green-600">+{formatCurrency(PORTFOLIO_DATA.dayChange)}</p>
              </div>
            </div>
            
            <AreaChart data={historyData} height={280} />
            <div className="mt-2 flex justify-between text-xs text-gray-400">
              <span>Oct 01</span>
              <span>Oct 08</span>
              <span>Oct 15</span>
              <span>Oct 22</span>
              <span>Today</span>
            </div>
          </Card>

          {/* Investment Ideas Table */}
          <Card title="Adam's Investment Ideas" action={<button className="text-blue-600 text-sm font-medium hover:underline">View All</button>}>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50">
                    <th className="pl-6 py-3 font-medium">Ticker</th>
                    <th className="py-3 font-medium">Strategy</th>
                    <th className="py-3 font-medium">Confidence</th>
                    <th className="py-3 font-medium">Horizon</th>
                    <th className="pr-6 py-3 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                  {[
                    { t: 'PLTR', s: 'AI Expansion Momentum', c: 'High (87%)', h: '3-6m', a: 'Buy' },
                    { t: 'AMD', s: 'Data Center Growth', c: 'Med (65%)', h: '12m', a: 'Hold' },
                    { t: 'TSLA', s: 'Robotaxi Volatility', c: 'High (91%)', h: '1m', a: 'Sell Call' },
                  ].map((item, i) => (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                      <td className="pl-6 py-3 font-semibold text-blue-600">{item.t}</td>
                      <td className="py-3">{item.s}</td>
                      <td className="py-3">
                        <Badge type={item.c.includes('High') ? 'success' : 'warning'}>{item.c}</Badge>
                      </td>
                      <td className="py-3 text-gray-500">{item.h}</td>
                      <td className="pr-6 py-3 text-right">
                        <button className="text-xs bg-gray-900 dark:bg-white text-white dark:text-gray-900 px-3 py-1 rounded hover:opacity-90 transition">Analyze</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          <Card title="System Alerts" className="p-0">
            <div className="p-4 space-y-3">
              {ALERTS.map(alert => (
                <div key={alert.id} className={`p-3 rounded-lg border-l-4 shadow-sm ${
                  alert.type === 'critical' ? 'bg-red-50 border-red-500 dark:bg-red-900/10' : 
                  alert.type === 'warning' ? 'bg-yellow-50 border-yellow-500 dark:bg-yellow-900/10' : 
                  alert.type === 'success' ? 'bg-green-50 border-green-500 dark:bg-green-900/10' :
                  'bg-blue-50 border-blue-500 dark:bg-blue-900/10'
                }`}>
                  <div className="flex justify-between items-start">
                    <h3 className="font-semibold text-sm text-gray-900 dark:text-white">{alert.title}</h3>
                    <span className="text-[10px] text-gray-400">{alert.time}</span>
                  </div>
                  <p className="text-xs text-gray-600 dark:text-gray-300 mt-1 leading-snug">{alert.message}</p>
                </div>
              ))}
            </div>
            <div className="p-4 border-t border-gray-100 dark:border-gray-700">
              <button className="w-full py-2 text-sm text-gray-600 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition">View All Alerts</button>
            </div>
          </Card>

          <Card title="Allocation" className="p-6">
            <div className="flex items-center justify-center h-48 relative mb-4">
              <div className="w-40 h-40">
                <DonutChart data={PORTFOLIO_DATA.allocation} />
              </div>
              <div className="absolute text-center pointer-events-none">
                <span className="block text-2xl font-bold text-gray-900 dark:text-white">1.2M</span>
                <span className="text-xs text-gray-500">USD</span>
              </div>
            </div>
            <div className="space-y-2.5">
              {PORTFOLIO_DATA.allocation.map(a => (
                <div key={a.name} className="flex justify-between items-center text-sm">
                  <span className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: a.color }}></span>
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

// --- VIEW: MARKET DATA ---

function MarketDataView({ data }) {
  const [subTab, setSubTab] = useState('stocks');
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Market Intelligence</h1>
          <p className="text-sm text-gray-500">Real-time feeds from global exchanges.</p>
        </div>
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {['Stocks', 'Crypto', 'Forex', 'Bonds'].map(tab => (
            <button 
              key={tab}
              onClick={() => setSubTab(tab.toLowerCase())}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${subTab === tab.toLowerCase() ? 'bg-white dark:bg-gray-600 shadow-sm text-gray-900 dark:text-white' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700'}`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <Card className="overflow-hidden">
        <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex gap-4">
          <button className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 border px-3 py-1.5 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"><Filter size={14} /> Filter</button>
          <button className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 border px-3 py-1.5 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"><Download size={14} /> Export</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead className="bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 text-xs uppercase sticky top-0">
              <tr>
                <th className="px-6 py-4 font-semibold">Symbol</th>
                <th className="px-6 py-4 font-semibold">Name</th>
                {subTab === 'stocks' && <th className="px-6 py-4 font-semibold">Sector</th>}
                <th className="px-6 py-4 font-semibold text-right">Price</th>
                <th className="px-6 py-4 font-semibold text-right">Change</th>
                <th className="px-6 py-4 font-semibold text-right">Volume</th>
                <th className="px-6 py-4 font-semibold text-right">Mkt Cap</th>
                {subTab === 'stocks' && <th className="px-6 py-4 font-semibold text-right">P/E Ratio</th>}
                <th className="px-6 py-4 font-semibold text-center">Analysis</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {(subTab === 'stocks' ? data.stocks : data.crypto).map((asset, i) => (
                <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors group cursor-pointer">
                  <td className="px-6 py-4 font-bold text-blue-600">{asset.symbol}</td>
                  <td className="px-6 py-4 font-medium text-gray-900 dark:text-white">{asset.name}</td>
                  {subTab === 'stocks' && <td className="px-6 py-4 text-gray-500 text-sm">{asset.sector || 'N/A'}</td>}
                  <td className="px-6 py-4 text-right font-mono text-sm">${asset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                  <td className={`px-6 py-4 text-right font-medium text-sm ${asset.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {asset.change > 0 ? '+' : ''}{asset.change.toFixed(2)}%
                  </td>
                  <td className="px-6 py-4 text-right text-gray-500 text-sm">{asset.volume}</td>
                  <td className="px-6 py-4 text-right text-gray-500 text-sm">{asset.mktCap}</td>
                  {subTab === 'stocks' && <td className="px-6 py-4 text-right text-gray-500 text-sm">{asset.pe || '-'}</td>}
                  <td className="px-6 py-4 text-center">
                    <button className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 hover:bg-blue-50 dark:hover:bg-blue-900/30 text-blue-600 rounded">
                      <Activity size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

// --- VIEW: ANALYSIS CENTER ---

function AnalysisView() {
  const [agents, setAgents] = useState([
    { id: 1, name: "Fundamental Analyst", desc: "Deep dive into company financials, 10-K/10-Q reports, and earnings transcripts.", icon: <FileText size={24} className="text-blue-500" />, status: "Active", load: 45 },
    { id: 2, name: "Technical Analyst", desc: "Chart pattern recognition, indicator analysis (RSI, MACD), and price target forecasting.", icon: <LineChart size={24} className="text-purple-500" />, status: "Active", load: 30 },
    { id: 3, name: "Risk Sentinel", desc: "Real-time assessment of market, credit, and operational risks using macro data.", icon: <ShieldAlert size={24} className="text-red-500" />, status: "Active", load: 12 },
    { id: 4, name: "Macro Economist", desc: "Global economic trend analysis focusing on interest rates, GDP, and geopolitical shifts.", icon: <Globe size={24} className="text-green-500" />, status: "Idle", load: 0 },
    { id: 5, name: "Sentiment Engine", desc: "Social media and news sentiment analysis to gauge market psychology.", icon: <Activity size={24} className="text-orange-500" />, status: "Processing", load: 89 },
    { id: 6, name: "SNC Reviewer", desc: "Shared National Credit program analysis and regulatory compliance checking.", icon: <Briefcase size={24} className="text-gray-500" />, status: "Idle", load: 0 },
  ]);

  const toggleAgent = (id) => {
    setAgents(prev => prev.map(a => {
      if (a.id === id) {
        const newStatus = a.status === 'Idle' ? 'Processing' : 'Idle';
        return { ...a, status: newStatus, load: newStatus === 'Idle' ? 0 : Math.floor(Math.random() * 40) + 20 };
      }
      return a;
    }));
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analysis Center</h1>
        <p className="text-gray-500 mt-1">Deploy specialized autonomous agents to analyze specific market segments or companies.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <Card key={agent.id} className="p-6 hover:border-blue-400 transition-all cursor-pointer flex flex-col h-full group">
            <div className="flex justify-between items-start mb-4">
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg group-hover:scale-110 transition-transform">
                {agent.icon}
              </div>
              <span className={`text-xs font-medium px-2 py-1 rounded-full flex items-center gap-1 ${
                agent.status === 'Active' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 
                agent.status === 'Processing' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
              }`}>
                {agent.status === 'Processing' && <RefreshCw size={10} className="animate-spin" />}
                {agent.status}
              </span>
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{agent.name}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 flex-1 mb-4">{agent.desc}</p>
            
            {/* Load Bar */}
            {agent.status !== 'Idle' && (
              <div className="mb-4">
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>System Load</span>
                  <span>{agent.load}%</span>
                </div>
                <div className="w-full bg-gray-100 dark:bg-gray-700 rounded-full h-1.5">
                  <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-500" style={{ width: `${agent.load}%` }}></div>
                </div>
              </div>
            )}

            <button 
              onClick={() => toggleAgent(agent.id)}
              className={`mt-auto w-full py-2 rounded-lg font-medium text-sm transition shadow-sm
                ${agent.status === 'Idle' 
                  ? 'bg-gray-900 dark:bg-white text-white dark:text-gray-900 hover:opacity-90' 
                  : 'bg-red-50 text-red-600 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-400'}
              `}
            >
              {agent.status === 'Idle' ? 'Deploy Agent' : 'Stop Agent'}
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
            <Card key={i} className="p-4 flex flex-col md:flex-row items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/30 transition group cursor-pointer">
              <div className="flex items-center gap-4 w-full md:w-auto mb-3 md:mb-0">
                <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded flex-shrink-0 flex items-center justify-center text-blue-600 dark:text-blue-400">
                  <FileText size={20} />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 transition-colors">{report.title}</h4>
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span>{report.agent}</span>
                    <span>•</span>
                    <span>{report.date}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3 w-full md:w-auto justify-end">
                {report.tags.map(tag => (
                  <span key={tag} className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-gray-600 dark:text-gray-300">{tag}</span>
                ))}
                <button className="p-2 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-full text-gray-400"><ArrowRight size={16} /></button>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- VIEW: PORTFOLIO ---

function PortfolioView({ data }) {
  const [activeView, setActiveView] = useState('holdings');

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Portfolio Management</h1>
          <div className="text-sm text-gray-500 mt-1 flex gap-4">
            <span>Buying Power: <span className="font-mono font-bold text-gray-800 dark:text-gray-300">{formatCurrency(PORTFOLIO_DATA.buyingPower)}</span></span>
            <span>Margin Used: <span className="font-mono font-bold text-gray-800 dark:text-gray-300">{formatCurrency(PORTFOLIO_DATA.marginUsed)}</span></span>
          </div>
        </div>
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          <button onClick={() => setActiveView('holdings')} className={`px-3 py-1 text-sm rounded-md ${activeView === 'holdings' ? 'bg-white dark:bg-gray-600 shadow' : 'text-gray-500'}`}>Holdings</button>
          <button onClick={() => setActiveView('transactions')} className={`px-3 py-1 text-sm rounded-md ${activeView === 'transactions' ? 'bg-white dark:bg-gray-600 shadow' : 'text-gray-500'}`}>Transactions</button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3 space-y-6">
          {activeView === 'holdings' ? (
            <Card className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold">Current Positions</h3>
                <button className="text-sm bg-blue-600 text-white px-3 py-1.5 rounded hover:bg-blue-700 flex items-center gap-2">
                  <RefreshCw size={14} /> Rebalance
                </button>
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
                  {PORTFOLIO_DATA.holdings.map((pos, i) => {
                    // Find live price from mock update if available
                    const liveAsset = data.stocks.find(s => s.symbol === pos.symbol) || data.crypto.find(c => c.symbol === pos.symbol + "-USD");
                    const price = liveAsset ? liveAsset.price : pos.currentPrice;
                    const totalVal = pos.quantity * price;
                    const ret = ((price - pos.avgCost) / pos.avgCost) * 100;

                    return (
                      <tr key={i} className="border-b border-gray-100 dark:border-gray-700">
                        <td className="px-4 py-3">
                          <div className="font-medium text-blue-600">{pos.symbol}</div>
                          <div className="text-xs text-gray-400">{pos.type}</div>
                        </td>
                        <td className="px-4 py-3 text-right">{pos.quantity}</td>
                        <td className="px-4 py-3 text-right text-gray-500">{formatCurrency(pos.avgCost)}</td>
                        <td className="px-4 py-3 text-right font-medium">{formatCurrency(price)}</td>
                        <td className="px-4 py-3 text-right font-bold">{formatCurrency(totalVal)}</td>
                        <td className={`px-4 py-3 text-right font-medium ${ret >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {ret > 0 ? '+' : ''}{ret.toFixed(2)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Card>
          ) : (
            <Card className="p-5">
              <h3 className="font-bold mb-4">Transaction History</h3>
              <table className="w-full text-left border-collapse">
                <thead className="text-xs uppercase text-gray-500 bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <th className="px-4 py-3">Date</th>
                    <th className="px-4 py-3">ID</th>
                    <th className="px-4 py-3">Symbol</th>
                    <th className="px-4 py-3">Type</th>
                    <th className="px-4 py-3 text-right">Price</th>
                    <th className="px-4 py-3 text-right">Total</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  {PORTFOLIO_DATA.transactions.map((txn, i) => (
                    <tr key={i} className="border-b border-gray-100 dark:border-gray-700">
                      <td className="px-4 py-3 text-gray-500">{txn.date}</td>
                      <td className="px-4 py-3 font-mono text-xs text-gray-400">{txn.id}</td>
                      <td className="px-4 py-3 font-medium">{txn.symbol}</td>
                      <td className="px-4 py-3">
                        <Badge type={txn.type === 'Buy' ? 'success' : 'neutral'}>{txn.type}</Badge>
                      </td>
                      <td className="px-4 py-3 text-right">{formatCurrency(txn.price)}</td>
                      <td className="px-4 py-3 text-right">{formatCurrency(txn.total)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Card className="p-5">
            <h3 className="font-bold mb-4">Risk Metrics</h3>
            <div className="space-y-5">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-500">Beta</span>
                  <span className="font-medium">1.24</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full"><div className="bg-yellow-500 h-2 rounded-full w-3/4"></div></div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-500">Sharpe Ratio</span>
                  <span className="font-medium">1.8</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full"><div className="bg-green-500 h-2 rounded-full w-2/3"></div></div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-500">VaR (95%)</span>
                  <span className="font-medium">$12,400</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full"><div className="bg-red-500 h-2 rounded-full w-1/4"></div></div>
              </div>
            </div>
          </Card>
          
          <Card className="bg-gradient-to-br from-blue-600 to-indigo-700 text-white p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-blue-100 text-sm font-medium">Projected Income</p>
                <h3 className="text-2xl font-bold mt-1">$24,500<span className="text-sm font-normal text-blue-200">/yr</span></h3>
              </div>
              <div className="bg-white/20 p-2 rounded-lg">
                <DollarSign size={24} className="text-white" />
              </div>
            </div>
            <div className="mt-4 text-xs text-blue-100 opacity-80">
              Based on current dividend yields and bond coupons.
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

// --- VIEW: SIMULATION ---

function SimulationView() {
  const [activeScenario, setActiveScenario] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [progress, setProgress] = useState(0);
  
  // Data for the result chart
  const [projectionData, setProjectionData] = useState([]);

  const scenarios = [
    { id: 'monte_carlo', name: 'Monte Carlo', desc: '10,000 iterations projecting future value based on volatility.', color: 'blue' },
    { id: 'black_swan', name: 'Black Swan Event', desc: 'Stress test against extreme events like the 2008 Financial Crisis.', color: 'red' },
    { id: 'inflation', name: 'Inflation Shock', desc: 'Impact of sustained 5% inflation and 50bps rate hikes.', color: 'purple' }
  ];

  const runSimulation = (id) => {
    setActiveScenario(id);
    setIsSimulating(true);
    setProgress(0);
    setProjectionData([]);

    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsSimulating(false);
          // Generate mock projection data
          const newData = Array.from({ length: 20 }, (_, i) => {
            let base = 100;
            if (id === 'monte_carlo') base += i * 2 + Math.random() * 10;
            if (id === 'black_swan') base -= i * 5 + Math.random() * 5;
            if (id === 'inflation') base -= i * 1 + Math.random() * 2;
            return base;
          });
          setProjectionData(newData);
          return 100;
        }
        return prev + 5;
      });
    }, 100);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Scenario Simulation Engine</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {scenarios.map(scen => (
          <Card 
            key={scen.id} 
            className={`p-6 border-2 cursor-pointer transition-all ${activeScenario === scen.id ? `border-${scen.color}-500 ring-1 ring-${scen.color}-500` : 'border-transparent hover:border-gray-300 dark:hover:border-gray-600'}`}
            onClick={() => runSimulation(scen.id)}
          >
            <h3 className={`font-bold text-lg mb-2 text-${scen.color}-600 dark:text-${scen.color}-400`}>{scen.name}</h3>
            <p className="text-sm text-gray-500 mb-4 min-h-[40px]">{scen.desc}</p>
            <button className={`w-full py-2 bg-${scen.color}-50 text-${scen.color}-600 dark:bg-${scen.color}-900/20 dark:text-${scen.color}-300 rounded hover:bg-${scen.color}-100 font-medium transition`}>
              Run Simulation
            </button>
          </Card>
        ))}
      </div>

      {/* Simulation Output Area */}
      <Card className="p-8 min-h-[350px] flex flex-col">
        {isSimulating ? (
          <div className="flex-1 flex flex-col items-center justify-center">
            <RefreshCw size={40} className="animate-spin text-blue-600 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Running Complex Models...</h3>
            <div className="w-full max-w-md bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
              <div className="bg-blue-600 h-2.5 rounded-full transition-all duration-200" style={{ width: `${progress}%` }}></div>
            </div>
            <p className="text-sm text-gray-500 mt-2">Calculating risk vectors and liquidity correlations...</p>
          </div>
        ) : activeScenario && progress === 100 ? (
          <div className="w-full animate-fade-in">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-2 text-green-600">
                <div className="w-2 h-2 rounded-full bg-green-600 animate-pulse"></div>
                <span className="font-bold uppercase text-sm">Simulation Complete</span>
              </div>
              <button onClick={() => {setActiveScenario(null); setProgress(0);}} className="text-sm text-gray-500 hover:text-gray-900">Reset</button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 h-64 border border-gray-100 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-800/50">
                <SimpleLineChart data={projectionData} color={activeScenario === 'black_swan' ? '#ef4444' : '#3b82f6'} />
              </div>
              <div className="space-y-4">
                 <div className="p-4 bg-gray-50 dark:bg-gray-700/30 rounded-lg text-center border border-gray-100 dark:border-gray-600">
                    <div className="text-xs text-gray-500 uppercase">Projected Impact</div>
                    <div className={`text-2xl font-bold ${activeScenario === 'monte_carlo' ? 'text-green-600' : 'text-red-600'}`}>
                      {activeScenario === 'monte_carlo' ? '+8.4%' : activeScenario === 'black_swan' ? '-24.2%' : '-12.4%'}
                    </div>
                 </div>
                 <div className="p-4 bg-gray-50 dark:bg-gray-700/30 rounded-lg text-center border border-gray-100 dark:border-gray-600">
                    <div className="text-xs text-gray-500 uppercase">Recovery Time</div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">
                       {activeScenario === 'monte_carlo' ? 'N/A' : '18 Months'}
                    </div>
                 </div>
                 <div className="p-4 bg-gray-50 dark:bg-gray-700/30 rounded-lg text-center border border-gray-100 dark:border-gray-600">
                    <div className="text-xs text-gray-500 uppercase">Most Vulnerable</div>
                    <div className="text-xl font-bold text-orange-600">Tech Sector</div>
                 </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-400">
            <Activity size={48} className="mx-auto mb-3 opacity-20" />
            <p>Select a scenario above to generate risk projections.</p>
          </div>
        )}
      </Card>
    </div>
  );
}

// --- VIEW: NEWS ---

function NewsView() {
  return (
    <div className="space-y-6 animate-fade-in">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Market Intelligence Feed</h1>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
                {NEWS_FEED.map(news => (
                    <Card key={news.id} className="p-4 flex gap-4 hover:bg-gray-50 dark:hover:bg-gray-700/30 cursor-pointer transition">
                        <div className="w-24 h-24 bg-gray-200 dark:bg-gray-700 rounded-lg flex-shrink-0 overflow-hidden flex items-center justify-center">
                            <Globe className="text-gray-400" />
                        </div>
                        <div className="flex-1">
                            <div className="flex justify-between mb-1">
                                <span className="text-xs font-bold text-blue-600 uppercase">{news.source}</span>
                                <span className="text-xs text-gray-400">{news.time}</span>
                            </div>
                            <h3 className="font-bold text-gray-900 dark:text-white text-lg leading-tight mb-2">{news.title}</h3>
                            <p className="text-sm text-gray-500 dark:text-gray-400 line-clamp-2">{news.summary}</p>
                            <div className="mt-3 flex gap-2">
                                <span className={`text-xs px-2 py-0.5 rounded uppercase font-medium ${news.sentiment === 'positive' ? 'bg-green-100 text-green-700' : news.sentiment === 'negative' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>
                                    {news.sentiment}
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
                            <span key={tag} className="text-sm bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 cursor-pointer transition">{tag}</span>
                        ))}
                    </div>
                </Card>
                <Card className="p-5 bg-blue-600 text-white border-none shadow-lg shadow-blue-500/30">
                    <h3 className="font-bold mb-2 flex items-center gap-2"><Bot size={18}/> Insight Summary</h3>
                    <p className="text-sm opacity-90 leading-relaxed">
                        Market sentiment is currently shifting towards "Risk-On" driven by stronger than expected tech earnings. However, bond yields suggest caution regarding long-term inflation targets. Recommendation: Maintain diversification.
                    </p>
                </Card>
            </div>
        </div>
    </div>
  );
}

// --- VIEW: SETTINGS ---

function SettingsView({ isDarkMode, setIsDarkMode }) {
    return (
        <div className="space-y-6 max-w-4xl animate-fade-in">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">System Configuration</h1>
            <Card className="divide-y divide-gray-200 dark:divide-gray-700">
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">Appearance</h3>
                        <p className="text-sm text-gray-500">Toggle between light and dark themes</p>
                    </div>
                    <button 
                      onClick={() => setIsDarkMode(!isDarkMode)}
                      className={`w-12 h-6 rounded-full p-1 transition-colors ${isDarkMode ? 'bg-blue-600' : 'bg-gray-300'}`}
                    >
                      <div className={`w-4 h-4 rounded-full bg-white shadow-sm transform transition-transform ${isDarkMode ? 'translate-x-6' : 'translate-x-0'}`}></div>
                    </button>
                </div>
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">API Keys</h3>
                        <p className="text-sm text-gray-500">Manage connections to Bloomberg, Alpaca, and OpenAI</p>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                        <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded text-xs">Connected</span>
                        <button className="text-blue-600 font-medium hover:underline ml-2">Manage</button>
                    </div>
                </div>
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">Notification Preferences</h3>
                        <p className="text-sm text-gray-500">Email and push notification settings for high-risk alerts</p>
                    </div>
                    <button className="text-blue-600 font-medium text-sm hover:underline">Edit</button>
                </div>
                <div className="p-6 flex justify-between items-center">
                    <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">Risk Parameters</h3>
                        <p className="text-sm text-gray-500">Set thresholds for VaR and Drawdown alerts</p>
                    </div>
                    <button className="text-blue-600 font-medium text-sm hover:underline">Configure</button>
                </div>
            </Card>
        </div>
    )
}

// --- CHAT COMPONENT ---

const AdamChat = ({ isOpen, onClose, currentContext }) => {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'adam', text: "Hello, I'm Adam. I'm monitoring your portfolio and the broader markets. How can I assist with your analysis today?" }
  ]);
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isOpen]);

  const handleSend = () => {
    if (!input.trim()) return;
    
    const userMsg = { id: Date.now(), sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");

    // Mock Context-Aware Response Logic
    setTimeout(() => {
      let responseText = "I'm processing that request. Let me query my internal models.";
      
      if (input.toLowerCase().includes('nvda')) {
        responseText = "NVIDIA (NVDA) is showing strong momentum. My technical agents detect a bullish flag pattern forming on the 4H chart. Volume is 20% above average, suggesting institutional accumulation.";
      } else if (input.toLowerCase().includes('risk')) {
        responseText = "Your current portfolio VaR (95%) is $12,400. The largest contributor to risk is your position in the Technology sector (45% allocation). I recommend hedging with defensive ETFs.";
      } else if (currentContext === 'simulation') {
        responseText = "I see you're in the Simulation Engine. Would you like me to run a custom stress test based on specific interest rate parameters?";
      } else if (currentContext === 'portfolio') {
        responseText = "Your portfolio is currently outperforming the S&P 500 by 1.2% today. Your cash position is 10%, providing ample dry powder for new opportunities.";
      } else {
        responseText = "I've logged that query. My analysis agents are currently reviewing global macro data to provide a comprehensive answer.";
      }

      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'adam', text: responseText }]);
    }, 800);
  };

  return (
    <div className={`fixed inset-y-0 right-0 w-full sm:w-96 bg-white dark:bg-gray-900 shadow-2xl transform transition-transform duration-300 z-50 border-l border-gray-200 dark:border-gray-700 flex flex-col ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
      {/* Chat Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex justify-between items-center bg-slate-900 text-white">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
          <div className="flex flex-col">
            <span className="font-bold text-sm">Adam Assistant</span>
            <span className="text-[10px] text-gray-400 uppercase tracking-wider">v23.1 Online</span>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={18} /></button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50 dark:bg-gray-800/50" ref={scrollRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] p-3 rounded-2xl text-sm shadow-sm ${
              msg.sender === 'user' 
                ? 'bg-blue-600 text-white rounded-br-none' 
                : 'bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-600 rounded-bl-none'
            }`}>
              {msg.text}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
        <div className="relative">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder={`Ask about ${currentContext}...`} 
            className="w-full pr-12 pl-4 py-3 bg-gray-100 dark:bg-gray-800 border-none rounded-xl text-sm focus:ring-2 focus:ring-blue-500 outline-none dark:text-white"
          />
          <button 
            onClick={handleSend}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-blue-600 p-1.5 hover:bg-blue-50 dark:hover:bg-gray-700 rounded-lg transition"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};
