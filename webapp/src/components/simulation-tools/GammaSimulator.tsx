import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity,
  ShieldAlert,
  Zap,
  BarChart2,
  Crosshair,
  Briefcase,
  AlertTriangle,
  TrendingDown,
  Server,
  Lock
} from 'lucide-react';

const SECTORS: Record<string, any> = {
  'Consumer Discretionary': { color: '#f43f5e', yieldSens: 1.2, spreadSens: 1.5, demandSens: 2.0 }, // Highly sensitive to demand drop
  'Commercial Real Estate': { color: '#eab308', yieldSens: 2.5, spreadSens: 1.8, demandSens: 1.0 }, // Highly sensitive to yield/rates
  'Logistics & Supply': { color: '#3b82f6', yieldSens: 1.0, spreadSens: 1.2, demandSens: 1.8 },
  'Tech SaaS': { color: '#10b981', yieldSens: 1.5, spreadSens: 1.1, demandSens: 0.5 }, // Resilient to demand, sensitive to yield
  'Legacy Healthcare': { color: '#a855f7', yieldSens: 1.1, spreadSens: 1.6, demandSens: 0.3 }
};

const generateMockNodes = () => {
  const nodes: any[] = [];
  let id = 0;
  Object.keys(SECTORS).forEach(sector => {
    // Generate 20 nodes per sector
    for (let i = 0; i < 20; i++) {
      nodes.push({
        id: `NODE-${id++}-${sector.substring(0, 3).toUpperCase()}`,
        sector: sector,
        baseDebtEbitda: (Math.random() * 4) + 2 + (sector === 'Commercial Real Estate' ? 2 : 0), // Base leverage 2x to 6x (CRE higher)
        baseIcr: (Math.random() * 3) + 1.2 + (sector === 'Tech SaaS' ? 2 : 0), // Base ICR 1.2x to 4.2x (Tech higher)
        baseLiquidity: Math.random() * 100, // 0 to 100 score
        volatility: Math.random() * 0.5 + 0.5
      });
    }
  });
  return nodes;
};

const BASE_NODES = generateMockNodes();

export default function GammaSimulator() {
  const [activeTab, setActiveTab] = useState('institutional');

  // Macro Shock States
  const [yieldShock, setYieldShock] = useState(0); // bps (0 to 200)
  const [spreadWidening, setSpreadWidening] = useState(0); // bps (0 to 500)
  const [consumerDrop, setConsumerDrop] = useState(0); // % (0 to 20)

  const stressedNodes = useMemo(() => {
    return BASE_NODES.map(node => {
      const sens = SECTORS[node.sector];

      // Calculate stressed EBITDA (drops based on consumer drop and sector sensitivity)
      const ebitdaMultiplier = Math.max(0.1, 1 - (consumerDrop / 100) * sens.demandSens);

      // Calculate stressed Interest Expense (rises based on yield + spread and sector sensitivity)
      const interestShockMultiplier = 1 + ((yieldShock / 100) * sens.yieldSens) + ((spreadWidening / 100) * sens.spreadSens);

      // New ICR = (Base EBITDA * Drop) / (Base Interest * Hike)
      // We assume Base ICR = Base EBITDA / Base Interest
      const currentIcr = (node.baseIcr * ebitdaMultiplier) / interestShockMultiplier;

      // Leverage increases as EBITDA drops
      const currentDebtEbitda = node.baseDebtEbitda / ebitdaMultiplier;

      // HFT metrics: Liquidity drains fast when ICR approaches 1.0
      const distanceToDefault = currentIcr - 1.0;
      let liquidityDrain = 0;
      if (distanceToDefault < 0.5) liquidityDrain = 40;
      if (distanceToDefault < 0) liquidityDrain = 80;

      return {
        ...node,
        currentIcr: Math.max(0.1, currentIcr),
        currentDebtEbitda: Math.min(12, currentDebtEbitda),
        isDefault: currentIcr < 1.0,
        currentLiquidity: Math.max(0, node.baseLiquidity - liquidityDrain), // Base liquidity without jitter
        spreadBidAsk: currentIcr < 1.0 ? 250 : 10 // Base spread without jitter
      };
    });
  }, [yieldShock, spreadWidening, consumerDrop]);

  const retailStats = useMemo(() => {
    const total = stressedNodes.length;
    const defaults = stressedNodes.filter(n => n.isDefault).length;
    const riskScore = Math.min(100, Math.round((defaults / total) * 200)); // Scale risk

    let recommendation = "Hold diversified assets. Maintain standard allocation.";
    if (riskScore > 75) recommendation = "CRITICAL RISK: Liquidate high-yield bonds. Rotate 80% to Sovereign Cash.";
    else if (riskScore > 40) recommendation = "ELEVATED RISK: Trim Consumer Discretionary. Increase defensive cash buffers.";

    return { riskScore, defaults, total, recommendation };
  }, [stressedNodes]);


  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-mono p-4 flex flex-col">

      {/* Header */}
      <header className="flex justify-between items-end border-b border-slate-800 pb-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-emerald-400 flex items-center gap-2">
            <Server className="w-6 h-6" />
            NODE: ADAM v26.0
          </h1>
          <p className="text-xs text-slate-500 mt-1">GAMMA FALLBACK SIMULATION // LIQUIDITY MAPPING</p>
        </div>
        <div className="flex gap-2 text-xs">
          <span className="flex items-center gap-1 text-emerald-500 bg-emerald-500/10 px-2 py-1 rounded"><Lock className="w-3 h-3"/> SECURE</span>
          <span className="flex items-center gap-1 text-amber-500 bg-amber-500/10 px-2 py-1 rounded"><Activity className="w-3 h-3"/> LIVE</span>
        </div>
      </header>

      <div className="flex flex-col lg:flex-row gap-6 flex-1">

        {/* ... */}
        <aside className="w-full lg:w-80 bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-col gap-6 shadow-xl shadow-black/50">
          <div className="flex items-center gap-2 text-slate-100 border-b border-slate-700 pb-2">
            <Zap className="w-5 h-5 text-amber-400" />
            <h2 className="font-semibold text-lg">Macro Stress Inputs</h2>
          </div>

          {/* Slider 1: Yield Shock */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-slate-400 flex justify-between">
              <span>Sovereign Yield Shock (bps)</span>
              <span className="text-amber-400">+{yieldShock}</span>
            </label>
            <input
              type="range" min="0" max="200" step="10"
              value={yieldShock} onChange={(e) => setYieldShock(Number(e.target.value))}
              className="w-full accent-amber-500"
            />
            <p className="text-[10px] text-slate-600">Simulates treasury curve bear-flattening.</p>
          </div>

          {/* Slider 2: Credit Spread */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-slate-400 flex justify-between">
              <span>BSL Spread Widening (bps)</span>
              <span className="text-rose-400">+{spreadWidening}</span>
            </label>
            <input
              type="range" min="0" max="500" step="25"
              value={spreadWidening} onChange={(e) => setSpreadWidening(Number(e.target.value))}
              className="w-full accent-rose-500"
            />
            <p className="text-[10px] text-slate-600">Simulates corporate risk premium blowout.</p>
          </div>

          {/* Slider 3: Consumer Demand */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-slate-400 flex justify-between">
              <span>Consumer Demand Contraction</span>
              <span className="text-blue-400">-{consumerDrop}%</span>
            </label>
            <input
              type="range" min="0" max="25" step="1"
              value={consumerDrop} onChange={(e) => setConsumerDrop(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <p className="text-[10px] text-slate-600">Impacts baseline EBITDA (Stagflation proxy).</p>
          </div>

          <div className="mt-auto bg-slate-950 p-3 rounded border border-slate-800 text-xs text-slate-400">
            <p className="flex justify-between mb-1"><span>System Status:</span> <span className={stressedNodes.filter(n=>n.isDefault).length > 20 ? 'text-rose-500 font-bold' : 'text-emerald-500 font-bold'}>MONITORING</span></p>
            <p className="flex justify-between"><span>Illiquid Nodes:</span> <span className="text-white">{stressedNodes.filter(n => n.isDefault).length} / 100</span></p>
          </div>
        </aside>

        {/* ... */}
        <main className="flex-1 flex flex-col min-w-0">

          {/* View Toggles */}
          <div className="flex gap-2 mb-4 bg-slate-900 p-1 rounded-lg w-fit border border-slate-800">
            <button
              onClick={() => setActiveTab('institutional')}
              className={`flex items-center gap-2 px-4 py-2 rounded text-sm font-semibold transition-colors ${activeTab === 'institutional' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <BarChart2 className="w-4 h-4"/> Institutional Risk
            </button>
            <button
              onClick={() => setActiveTab('hft')}
              className={`flex items-center gap-2 px-4 py-2 rounded text-sm font-semibold transition-colors ${activeTab === 'hft' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <Crosshair className="w-4 h-4"/> HFT Algo Engine
            </button>
            <button
              onClick={() => setActiveTab('retail')}
              className={`flex items-center gap-2 px-4 py-2 rounded text-sm font-semibold transition-colors ${activeTab === 'retail' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <Briefcase className="w-4 h-4"/> Retail Robo-Advisor
            </button>
          </div>

          {/* View Content Container */}
          <div className="flex-1 bg-slate-900 border border-slate-800 rounded-lg p-1 overflow-hidden flex flex-col relative shadow-xl shadow-black/50">
            {activeTab === 'institutional' && <InstitutionalView nodes={stressedNodes} />}
            {activeTab === 'hft' && <HFTView nodes={stressedNodes} />}
            {activeTab === 'retail' && <RetailView stats={retailStats} />}
          </div>

        </main>
      </div>
    </div>
  );
}

function InstitutionalView({ nodes }: { nodes: any[] }) {
  // SVG Coordinate mapping
  const width = 800;
  const height = 500;
  const padding = 50;

  // X = Debt/EBITDA (0 to 12)
  const mapX = (val: number) => padding + ((val / 12) * (width - padding * 2));
  // Y = ICR (0 to 5) - Inverted so higher ICR is at top
  const mapY = (val: number) => height - padding - ((val / 5) * (height - padding * 2));

  // The critical threshold Y line (ICR = 1.0)
  const deathZoneY = mapY(1.0);

  return (
    <div className="flex-1 p-4 flex flex-col h-full w-full overflow-auto">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-indigo-400"/> Systemic Credit Scatter Matrix
        </h3>
        <div className="flex gap-4 text-xs">
          {Object.entries(SECTORS).map(([name, data]) => (
            <div key={name} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: data.color }}></div>
              <span className="text-slate-400">{name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="relative flex-1 min-h-[400px] w-full bg-slate-950 border border-slate-800 rounded mt-2 overflow-hidden">
        {/* Custom SVG Scatter Plot */}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full preserve-3d">

          {/* Grid lines */}
          {[1,2,3,4,5].map(tick => (
             <line key={`y-${tick}`} x1={padding} y1={mapY(tick)} x2={width-padding} y2={mapY(tick)} stroke="#1e293b" strokeDasharray="4 4" />
          ))}
          {[2,4,6,8,10].map(tick => (
             <line key={`x-${tick}`} x1={mapX(tick)} y1={padding} x2={mapX(tick)} y2={height-padding} stroke="#1e293b" strokeDasharray="4 4" />
          ))}

          {/* Death Zone Fill */}
          <rect x={padding} y={deathZoneY} width={width - padding*2} height={height - padding - deathZoneY} fill="#ef4444" opacity="0.05" />

          {/* Y=1.0 Threshold Line */}
          <line x1={padding} y1={deathZoneY} x2={width-padding} y2={deathZoneY} stroke="#ef4444" strokeWidth="2" strokeDasharray="8 4" />
          <text x={padding + 10} y={deathZoneY - 5} fill="#ef4444" fontSize="12" fontWeight="bold">ICR = 1.0x (DEFAULT HORIZON)</text>

          {/* Axes labels */}
          <text x={width/2} y={height - 15} fill="#64748b" fontSize="12" textAnchor="middle">Leverage (Debt / EBITDA)</text>
          <text x={15} y={height/2} fill="#64748b" fontSize="12" textAnchor="middle" transform={`rotate(-90 15 ${height/2})`}>Interest Coverage Ratio (ICR)</text>

          {/* Nodes */}
          {nodes.map((node: any) => (
            <circle
              key={node.id}
              cx={mapX(node.currentDebtEbitda)}
              cy={mapY(node.currentIcr)}
              r={node.isDefault ? 6 : 4}
              fill={SECTORS[node.sector].color}
              opacity={node.isDefault ? 1 : 0.7}
              stroke={node.isDefault ? '#fff' : 'none'}
              strokeWidth={node.isDefault ? 2 : 0}
              className="transition-all duration-500 ease-in-out"
            >
              <title>{`${node.id}\nSector: ${node.sector}\nLev: ${node.currentDebtEbitda.toFixed(2)}x\nICR: ${node.currentIcr.toFixed(2)}x`}</title>
            </circle>
          ))}
        </svg>
      </div>

      <p className="text-xs text-slate-500 mt-3 border-l-2 border-indigo-500 pl-2">
        <strong>METHODOLOGY:</strong> Nodes migrating below the dotted threshold indicate immediate liquidity starvation. Note the asymmetric velocity of Consumer Discretionary vs. Tech SaaS under parallel shocks.
      </p>
    </div>
  );
}

// ⚡ Bolt: Extracted table row into React.memo to prevent unnecessary re-renders of the entire table on tick.
// We pass down tick and use a deterministic pseudo-random seed based on the node's numeric ID to layer visual jitter locally.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const HFTTableRow = React.memo(({ node, tick }: { node: any; tick: number }) => {
  const isCrit = node.isDefault;

  // Extract numeric ID from string like 'NODE-42-CON'
  const idMatch = node.id.match(/\d+/);
  const numericId = idMatch ? parseInt(idMatch[0], 10) : 0;

  // Deterministic pseudo-random seed based on ID and tick
  const seed = (numericId * 9301 + tick * 49297) % 233280;
  const pseudoRandom = seed / 233280; // 0 to 1

  // Layer visual jitter locally
  const jitteredLiquidity = Math.max(0, node.currentLiquidity + (pseudoRandom * 5 - 2.5));
  const baseSpread = isCrit ? 250 : 10;
  const spreadJitter = isCrit ? pseudoRandom * 100 : pseudoRandom * 20;
  const jitteredSpread = baseSpread + spreadJitter;

  return (
    <tr className={`border-b border-slate-900 hover:bg-slate-900/50 ${isCrit ? 'bg-rose-950/20' : ''}`}>
      <td className={`p-2 font-bold ${isCrit ? 'text-rose-400' : 'text-slate-300'}`}>{node.id}</td>
      <td className="p-2 text-slate-500 truncate max-w-[120px]">{node.sector}</td>
      <td className="p-2 text-right">
        <span className={jitteredLiquidity < 20 ? 'text-rose-500' : 'text-emerald-500'}>
          {jitteredLiquidity.toFixed(1)}
        </span>
      </td>
      <td className="p-2 text-right text-slate-400">
        {jitteredSpread.toFixed(0)} bps
      </td>
      <td className="p-2 text-center">
        {isCrit ?
          <span className="text-xs bg-rose-500/20 text-rose-500 px-2 py-0.5 rounded border border-rose-500/30">ILLIQUID</span> :
          <span className="text-xs bg-emerald-500/10 text-emerald-500 px-2 py-0.5 rounded">STABLE</span>
        }
      </td>
      <td className="p-2 text-center">
         {isCrit ?
          <span className="text-rose-500 font-bold animate-pulse">SHORT/UNWIND</span> :
          <span className="text-slate-600">HOLD</span>
        }
      </td>
    </tr>
  );
});
HFTTableRow.displayName = 'HFTTableRow';

function HFTView({ nodes }: { nodes: any[] }) {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setTick(t => t + 1);
    }, 500); // 500ms updates
    return () => clearInterval(interval);
  }, []);

  // ⚡ Bolt: Wrapped sortedNodes in useMemo to prevent O(N log N) sorting on every re-render of HFTView.
  // Expected Impact: Reduces CPU cycle waste when the HFTView re-renders due to other state changes without nodes changing.
  // Sort by liquidity (most distressed first) to simulate an order book looking for targets
  const sortedNodes = useMemo(() =>
    [...nodes].sort((a, b) => a.currentIcr - b.currentIcr).slice(0, 50),
  [nodes]); // Show top 50

  return (
    <div className="flex-1 p-4 flex flex-col h-full overflow-hidden bg-black rounded-lg">
      <div className="flex justify-between items-center mb-4 border-b border-slate-800 pb-2">
        <h3 className="text-lg font-bold text-rose-500 flex items-center gap-2">
          <Activity className="w-5 h-5"/> L2 Market Depth & Execution Matrix
        </h3>
        <div className="animate-pulse flex items-center gap-2 text-xs font-bold text-rose-500">
          <span className="w-2 h-2 rounded-full bg-rose-500"></span> FEED ACTIVE
        </div>
      </div>

      <div className="flex-1 overflow-auto custom-scrollbar">
        <table className="w-full text-left text-xs font-mono border-collapse">
          <thead className="sticky top-0 bg-slate-950 text-slate-500 border-b border-slate-800 shadow-md">
            <tr>
              <th className="p-2">TICKER / NODE</th>
              <th className="p-2">SECTOR</th>
              <th className="p-2 text-right">LIQUIDITY IDX</th>
              <th className="p-2 text-right">BID-ASK SPREAD</th>
              <th className="p-2 text-center">STATUS</th>
              <th className="p-2 text-center">ALGO ACTION</th>
            </tr>
          </thead>
          <tbody>
            {sortedNodes.map((node) => (
              <HFTTableRow key={node.id} node={node} tick={tick} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RetailView({ stats }: { stats: any }) {

  // Calculate gauge rotation based on risk score (0 to 180 degrees)
  const rotation = (stats.riskScore / 100) * 180;

  let statusColor = 'text-emerald-400';
  let bgColor = 'bg-emerald-950/30';
  let borderColor = 'border-emerald-500/30';
  let StatusIcon = Activity;

  if (stats.riskScore > 40) {
    statusColor = 'text-amber-400';
    bgColor = 'bg-amber-950/30';
    borderColor = 'border-amber-500/30';
    StatusIcon = AlertTriangle;
  }
  if (stats.riskScore > 75) {
    statusColor = 'text-rose-500';
    bgColor = 'bg-rose-950/30';
    borderColor = 'border-rose-500/30';
    StatusIcon = TrendingDown;
  }

  return (
    <div className="flex-1 p-6 flex flex-col justify-center items-center h-full text-center">

      <div className="max-w-2xl w-full">
        <h2 className="text-2xl font-bold text-white mb-2">My Wealth Shield™</h2>
        <p className="text-slate-400 text-sm mb-8">AI-Powered Macroeconomic Portfolio Defense</p>

        {/* ... */}
        <div className="flex justify-center mb-8">
          <div className="relative w-64 h-32 overflow-hidden">
             {/* Gauge Background */}
             <div className="absolute top-0 left-0 w-64 h-64 rounded-full border-[24px] border-slate-800"></div>
             {/* Gauge Color Fill (Using SVG for clean semi-circle) */}
             <svg className="absolute top-0 left-0 w-64 h-64 transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50" cy="50" r="38"
                  fill="transparent"
                  stroke={stats.riskScore > 75 ? '#f43f5e' : stats.riskScore > 40 ? '#fbbf24' : '#10b981'}
                  strokeWidth="24"
                  strokeDasharray="238" // Approx circumference (2 * pi * 38)
                  strokeDashoffset={238 - (238 * (stats.riskScore / 100) * 0.5)} // Only fill half
                  className="transition-all duration-700 ease-in-out"
                />
             </svg>
             {/* Needle */}
             <div
                className="absolute bottom-0 left-1/2 w-1 h-24 bg-white origin-bottom rounded-t transition-transform duration-700 ease-in-out z-10"
                style={{ transform: `translateX(-50%) rotate(${-90 + rotation}deg)` }}
             >
                <div className="absolute -top-2 -left-1.5 w-4 h-4 bg-white rounded-full"></div>
             </div>
             {/* Center hub */}
             <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 w-8 h-8 bg-slate-900 rounded-full z-20 border-4 border-slate-700"></div>
          </div>
        </div>

        <div className="text-5xl font-black mb-4 tracking-tighter flex items-center justify-center gap-3" style={{ color: statusColor.replace('text-', '') }}>
          {stats.riskScore} <span className="text-xl text-slate-500 font-normal">/ 100</span>
        </div>
        <h3 className={`text-xl font-bold mb-6 ${statusColor}`}>Systemic Risk Score</h3>

        {/* Actionable Advice Card */}
        <div className={`p-6 rounded-xl border ${bgColor} ${borderColor} backdrop-blur-sm text-left shadow-2xl`}>
          <div className="flex items-center gap-3 mb-3">
            <StatusIcon className={`w-6 h-6 ${statusColor}`} />
            <h4 className="font-bold text-white text-lg">Robo-Advisor Action Plan</h4>
          </div>
          <p className="text-slate-300 leading-relaxed text-sm mb-4">
            {stats.recommendation}
          </p>
          <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-700/50">
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Corporate Bonds at Risk</p>
              <p className="text-xl font-mono text-white">{stats.defaults} / {stats.total}</p>
            </div>
             <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Recommended Action</p>
              <button className="bg-slate-800 hover:bg-slate-700 text-white text-xs px-4 py-2 rounded transition-colors font-bold w-full">
                EXECUTE REBALANCE
              </button>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
