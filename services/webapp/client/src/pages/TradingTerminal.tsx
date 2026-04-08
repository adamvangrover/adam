import React from 'react';
import AlgoStrategies from '../components/AlgoStrategies';
import RoboAdvisor from '../components/RoboAdvisor';
// Import other components if needed (Terminal, MarketSentiment) - Assuming they exist or are mocked for now.

const TradingTerminal = () => {
    return (
        <div className="animate-fade-in pb-10">
            <header className="mb-8 border-b border-cyan-900/30 pb-4">
                <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tight mb-2">Trading Terminal</h1>
                <div className="flex gap-4 text-xs font-mono text-slate-400">
                    <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">STATUS: <span className="text-emerald-400 font-bold">ONLINE</span></span>
                    <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">LATENCY: <span className="text-amber-400">24ms</span></span>
                    <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">UPTIME: 99.99%</span>
                </div>
            </header>

            <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Left Column: Algorithmic Strategies & Execution */}
                <div className="lg:col-span-2 space-y-8">
                    <AlgoStrategies />

                    {/* Placeholder for Interactive Chart or Order Entry if not in AlgoStrategies */}
                    <div className="bg-[#0f172a] p-8 rounded-2xl shadow-inner border border-slate-700/50 h-80 flex flex-col items-center justify-center text-slate-500 relative overflow-hidden group">
                        <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
                        <div className="z-10 bg-slate-900/80 px-6 py-3 rounded-lg border border-slate-800 shadow-xl backdrop-blur-sm group-hover:border-cyan-900/50 transition-colors">
                            <span className="font-mono text-sm tracking-widest uppercase">[Interactive Charting Module Offline]</span>
                        </div>
                    </div>
                </div>

                {/* Right Column: Robo-Advisor & System Health */}
                <div className="space-y-8">
                    <RoboAdvisor />

                    {/* System Notifications / Alerts */}
                    <div className="bg-[#0f172a] p-6 rounded-2xl shadow-inner border border-slate-700/50 relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/5 rounded-full blur-3xl"></div>
                        <h3 className="text-lg font-bold text-slate-200 mb-4 border-b border-slate-800 pb-2">System Alerts</h3>
                        <ul className="space-y-3 text-sm font-medium">
                            <li className="flex items-start space-x-3 text-amber-400 bg-amber-950/20 p-3 rounded-lg border border-amber-900/30">
                                <span className="mt-0.5 animate-pulse">⚠</span>
                                <span>Volatility Spike Detected (VIX &gt; 25)</span>
                            </li>
                            <li className="flex items-start space-x-3 text-cyan-400 bg-cyan-950/20 p-3 rounded-lg border border-cyan-900/30">
                                <span className="mt-0.5">ℹ</span>
                                <span>Rebalancing scheduled for 16:00 EST.</span>
                            </li>
                            <li className="flex items-start space-x-3 text-emerald-400 bg-emerald-950/20 p-3 rounded-lg border border-emerald-900/30">
                                <span className="mt-0.5">✓</span>
                                <span>Data Feeds Synced (AlphaVantage, Polygon).</span>
                            </li>
                        </ul>
                    </div>
                </div>

            </main>
        </div>
    );
};

export default TradingTerminal;
