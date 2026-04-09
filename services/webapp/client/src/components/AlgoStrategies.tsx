import React, { useState } from 'react';

const AlgoStrategies = () => {
    const [strategies] = useState([
        { id: 1, name: 'Momentum Alpha', type: 'Trend Following', roi: 12.5, drawdown: 3.2, status: 'Active' },
        { id: 2, name: 'Mean Reversion Gamma', type: 'Mean Reversion', roi: 8.1, drawdown: 1.5, status: 'Active' },
        { id: 3, name: 'Adaptive Volatility', type: 'Reinforcement Learning', roi: 15.3, drawdown: 4.8, status: 'Active' },
        { id: 4, name: 'Arbitrage Delta', type: 'Arbitrage', roi: 5.2, drawdown: 0.8, status: 'Paused' }
    ]);

    const [trades] = useState([
        { id: 101, strategy: 'Momentum Alpha', symbol: 'NVDA', action: 'BUY', price: 135.20, time: '10:30 AM' },
        { id: 102, strategy: 'Adaptive Volatility', symbol: 'SPY', action: 'SELL', price: 585.50, time: '11:15 AM' },
        { id: 103, strategy: 'Mean Reversion Gamma', symbol: 'TSLA', action: 'BUY', price: 242.00, time: '01:45 PM' }
    ]);

    return (
        <div className="bg-[#0f172a] p-8 rounded-2xl shadow-inner border border-slate-700/50 mb-8 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl"></div>
            <h2 className="text-xl font-bold mb-6 border-b border-slate-800 pb-3 tracking-wide text-slate-200">Algorithmic Strategies</h2>

            {/* Strategy Performance Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8 relative z-10">
                {strategies.map(strat => (
                    <div key={strat.id} className={`p-5 rounded-xl border transition-all duration-300 hover:shadow-lg ${strat.status === 'Active' ? 'border-cyan-500/50 bg-slate-900/80 hover:border-cyan-400' : 'border-slate-700 bg-slate-900/40 opacity-70'}`}>
                        <div className="flex justify-between items-center mb-3">
                            <h3 className="font-bold text-slate-200 truncate pr-2">{strat.name}</h3>
                            <span className={`text-xs px-2.5 py-1 rounded-md font-mono font-bold ${strat.status === 'Active' ? 'bg-cyan-900/30 text-cyan-400 border border-cyan-800/50' : 'bg-slate-800 text-slate-400 border border-slate-700'}`}>
                                {strat.status}
                            </span>
                        </div>
                        <p className="text-xs font-mono text-slate-500 mb-4">{strat.type}</p>
                        <div className="flex justify-between text-sm bg-black/20 p-3 rounded-lg border border-slate-800/50">
                            <div>
                                <span className="block text-xs text-slate-500 font-medium mb-1">ROI</span>
                                <span className="text-emerald-400 font-mono font-bold">+{strat.roi}%</span>
                            </div>
                            <div className="text-right">
                                <span className="block text-xs text-slate-500 font-medium mb-1">Max DD</span>
                                <span className="text-rose-400 font-mono font-bold">-{strat.drawdown}%</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Recent Trades Log */}
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800 relative z-10 backdrop-blur-sm">
                <h3 className="text-sm font-bold tracking-wider uppercase mb-4 text-slate-400">Recent Executions</h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm whitespace-nowrap">
                        <thead className="text-slate-500 border-b border-slate-800 text-xs font-mono">
                            <tr>
                                <th className="pb-3 font-medium">TIME</th>
                                <th className="pb-3 font-medium">STRATEGY</th>
                                <th className="pb-3 font-medium">SYMBOL</th>
                                <th className="pb-3 font-medium">ACTION</th>
                                <th className="pb-3 text-right font-medium">PRICE</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                            {trades.map(trade => (
                                <tr key={trade.id} className="hover:bg-slate-800/30 transition-colors group">
                                    <td className="py-3 text-slate-500 font-mono text-xs">{trade.time}</td>
                                    <td className="py-3 text-cyan-400/80 group-hover:text-cyan-400 transition-colors">{trade.strategy}</td>
                                    <td className="py-3 font-bold text-slate-200">{trade.symbol}</td>
                                    <td className="py-3">
                                        <span className={`px-2 py-1 rounded text-xs font-bold border ${trade.action === 'BUY' ? 'text-emerald-400 bg-emerald-950/20 border-emerald-900/30' : 'text-rose-400 bg-rose-950/20 border-rose-900/30'}`}>
                                            {trade.action}
                                        </span>
                                    </td>
                                    <td className="py-3 text-right font-mono font-medium text-slate-300">${trade.price.toFixed(2)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default AlgoStrategies;
