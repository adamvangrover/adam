import React, { useState } from 'react';

const RoboAdvisor = () => {
    const [allocation] = useState([
        { id: 1, name: 'Equities', target: 50, current: 42, color: 'bg-blue-500' },
        { id: 2, name: 'Fixed Income', target: 30, current: 35, color: 'bg-purple-500' },
        { id: 3, name: 'Crypto', target: 10, current: 12, color: 'bg-orange-500' },
        { id: 4, name: 'Cash', target: 10, current: 11, color: 'bg-green-500' }
    ]);

    const [riskProfile] = useState('Moderate'); // Conservative, Moderate, Aggressive

    return (
        <div className="bg-[#0f172a] p-8 rounded-2xl shadow-inner border border-slate-700/50 text-slate-200 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-48 h-48 bg-blue-500/5 rounded-full blur-3xl"></div>
            <h2 className="text-xl font-bold mb-6 border-b border-slate-800 pb-3 flex justify-between items-center relative z-10">
                <span className="tracking-wide">AI Asset Allocation</span>
                <div className="text-xs font-mono bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
                    RISK PROFILE: <span className="text-amber-400 ml-1">{riskProfile.toUpperCase()}</span>
                </div>
            </h2>

            {/* Allocation Bars */}
            <div className="space-y-6 mb-8 relative z-10">
                {allocation.map(asset => (
                    <div key={asset.id} className="group">
                        <div className="flex justify-between text-sm mb-2">
                            <span className="font-bold text-slate-300">{asset.name}</span>
                            <span className="text-slate-500 font-mono text-xs">
                                TARGET: {asset.target}% <span className="mx-1 text-slate-700">|</span> CURRENT: <span className={asset.current !== asset.target ? 'text-amber-400 font-bold ml-1' : 'ml-1'}>{asset.current}%</span>
                            </span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden relative border border-slate-700/50">
                            {/* Current Bar */}
                            <div className={`${asset.color} h-3 rounded-full absolute top-0 left-0 opacity-80 shadow-[0_0_10px_rgba(0,0,0,0.5)] transition-all duration-500`} style={{ width: `${asset.current}%`, backgroundImage: 'linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.2) 100%)' }}></div>
                            {/* Target Marker */}
                            <div className="h-full w-1 bg-white absolute top-0 z-10 shadow-[0_0_5px_white]" style={{ left: `${asset.target}%`, marginLeft: '-2px' }}></div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Rebalancing Recommendations */}
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800 relative z-10 backdrop-blur-sm">
                <h3 className="text-sm font-bold tracking-wider uppercase mb-4 text-slate-400">Rebalancing Engine</h3>
                <div className="grid grid-cols-1 gap-3 text-sm">
                    {allocation.map(asset => {
                        const diff = asset.current - asset.target;
                        if (Math.abs(diff) < 2) return null; // Ignore small drifts
                        return (
                            <div key={asset.id} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700/50 hover:border-slate-600 transition-colors">
                                <span className="font-bold text-slate-300">{asset.name}</span>
                                <span className={`px-3 py-1 rounded text-xs font-bold border ${diff > 0 ? 'text-rose-400 bg-rose-950/20 border-rose-900/30' : 'text-emerald-400 bg-emerald-950/20 border-emerald-900/30'}`}>
                                    {diff > 0 ? `SELL ${diff}%` : `BUY ${Math.abs(diff)}%`}
                                </span>
                            </div>
                        );
                    })}
                </div>
                {allocation.every(a => Math.abs(a.current - a.target) < 2) && (
                    <div className="p-4 border border-dashed border-slate-700 rounded-lg text-slate-500 text-center text-sm font-medium bg-slate-800/20">
                        Portfolio is optimally balanced within tracking error constraints.
                    </div>
                )}
            </div>
        </div>
    );
};

export default RoboAdvisor;
