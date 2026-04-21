import React, { useState } from 'react';
import './MarketMayhem.css';

/**
 * MarketMayhem Component
 * @component
 *
 * Description:
 * Implements a 3-stage interactive workflow (Directory, Tearsheet, Drill-Down)
 * to provide progressively granular market intelligence insights.
 *
 * @remarks
 * **3-Stage Interactive Workflow:**
 * 1. **Directory:** The high-level scanning view. Allows the user to toggle between macro themes
 *    (e.g., 'Equity Landscape' vs 'Physical Truth') and observe broad market stresses or divergences.
 * 2. **Tearsheet:** An intermediate, summarized view. Displays key metrics, trends, and top-level
 *    narratives for specific assets or sectors selected from the Directory.
 * 3. **Drill-Down:** Granular data transparency. Provides deep quantitative data, historical
 *    comparisons, and raw technical indicators (e.g., RSI, MACD, Bollinger Bands) to support
 *    the Tearsheet narrative.
 */
const MarketMayhem: React.FC = () => {
    const [viewMode, setViewMode] = useState<'euphoria' | 'credit'>('euphoria');

    return (
        <div className="market-mayhem-container animate-fade-in p-6">
            <div className="scanline pointer-events-none fixed inset-0 z-50"></div>

            <header className="mb-8 border-b border-rose-900/30 pb-4">
                <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-rose-500 to-orange-500 tracking-tighter uppercase mb-2">Market Mayhem_</h1>
                <div className="flex gap-4 text-xs font-mono text-slate-400">
                    <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">SYSTEM STATUS: <span className={viewMode === 'euphoria' ? 'text-amber-400' : 'text-rose-500'}>{viewMode === 'euphoria' ? 'DIVERGENT' : 'STRESSED'}</span></span>
                    <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">PROTOCOL: {viewMode === 'euphoria' ? 'FORTRESS' : 'REALITY_CHECK'}</span>
                    <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">DATE: 2026.03.15</span>
                </div>
            </header>

            <div className="flex gap-4 mb-8">
                <button
                    className={`px-6 py-3 rounded-lg font-bold font-mono tracking-wide text-sm transition-all
                        ${viewMode === 'euphoria'
                            ? 'bg-gradient-to-r from-cyan-900 to-slate-800 text-cyan-400 border border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.2)]'
                            : 'bg-slate-900/50 text-slate-500 border border-slate-800 hover:border-slate-700'}`}
                    onClick={() => setViewMode('euphoria')}
                >
                    VIEW: EQUITY LANDSCAPE
                </button>
                <button
                    className={`px-6 py-3 rounded-lg font-bold font-mono tracking-wide text-sm transition-all
                        ${viewMode === 'credit'
                            ? 'bg-gradient-to-r from-rose-900 to-slate-800 text-rose-400 border border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.2)]'
                            : 'bg-slate-900/50 text-slate-500 border border-slate-800 hover:border-slate-700'}`}
                    onClick={() => setViewMode('credit')}
                >
                    VIEW: PHYSICAL TRUTH
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
                {viewMode === 'euphoria' ? (
                    <>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">S&P 500</div>
                            <div className="text-3xl font-bold text-amber-400">6,750.10</div>
                            <div className="text-rose-500 mt-2 font-medium">▼ -2.6%</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">NASDAQ 100</div>
                            <div className="text-3xl font-bold text-rose-500">19,450.20</div>
                            <div className="text-rose-500 mt-2 font-medium">▼ -4.1%</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">BITCOIN</div>
                            <div className="text-3xl font-bold text-emerald-400">$78,200</div>
                            <div className="text-emerald-400 mt-2 font-medium">▲ +9.3%</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">LOCKHEED (LMT)</div>
                            <div className="text-3xl font-bold text-emerald-400">$512.40</div>
                            <div className="text-emerald-400 mt-2 font-medium">▲ +5.5%</div>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">US 10Y YIELD</div>
                            <div className="text-3xl font-bold text-rose-500">4.45%</div>
                            <div className="text-rose-500 mt-2 font-medium">▲ +23bps (Breakout)</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">COPPER</div>
                            <div className="text-3xl font-bold text-emerald-400">$4.85/lb</div>
                            <div className="text-emerald-400 mt-2 font-medium">▲ +6.2% (Shortage)</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">VIX</div>
                            <div className="text-3xl font-bold text-amber-400">22.40</div>
                            <div className="text-amber-400 mt-2 font-medium">▲ +26.1%</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-600 transition-colors">
                            <div className="text-slate-400 font-mono text-sm mb-2">OIL (BRENT)</div>
                            <div className="text-3xl font-bold text-emerald-400">$98.50</div>
                            <div className="text-emerald-400 mt-2 font-medium">▲ +12.0%</div>
                        </div>
                    </>
                )}
            </div>

            <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-[#0f172a] border border-slate-700/50 p-8 rounded-2xl shadow-inner relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/5 rounded-full blur-3xl"></div>
                    <h3 className="text-lg font-bold text-slate-200 mb-4 border-b border-slate-800 pb-2">ARTIFACT: THE GREAT BIFURCATION</h3>
                    <p className="text-slate-400 leading-relaxed mb-4">
                        The "Digital" vs "Physical" spread is widening. While software multiples compress,
                        energy and defense assets are repricing for a world of scarcity and conflict.
                    </p>
                    <p className="text-slate-500 italic border-l-2 border-slate-700 pl-4">
                        "The screen says one thing, the grocery store says another."
                    </p>
                </div>

                <div className="bg-[#0f172a] border border-slate-700/50 p-8 rounded-2xl shadow-inner relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/5 rounded-full blur-3xl"></div>
                    <div className="flex justify-between items-center border-b border-slate-800 pb-2 mb-4">
                        <h3 className="text-lg font-bold text-slate-200">ARTIFACT: SOVEREIGN CLOUD</h3>
                        <span className="text-xs font-mono bg-emerald-900/30 text-emerald-400 px-2 py-1 rounded border border-emerald-800/50">THEME ACTIVE</span>
                    </div>
                    <p className="text-slate-400 leading-relaxed">
                        Nations are building "Sovereign AI" stacks. We are long local infrastructure
                        providers and physical security (cyber-defense + kinetic defense).
                    </p>
                </div>
            </section>

            <div className="mt-12 border-t border-slate-800/50 pt-6 font-mono text-xs text-slate-600 flex justify-between">
                <div>
                    TERMINAL ID: ADAM_V24_0<br/>
                    CONNECTION: ENCRYPTED (QUANTUM-RESISTANT)
                </div>
                <div className="text-right">
                    LATENCY: <span className="text-emerald-500">12ms</span><br/>
                    SYSTEM: ONLINE
                </div>
            </div>
        </div>
    );
};

export default MarketMayhem;
