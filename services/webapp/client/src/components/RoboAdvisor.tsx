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
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg text-white">
            <h2 className="text-2xl font-bold mb-4 border-b border-gray-700 pb-2 flex justify-between items-center">
                <span>Robo-Advisor Allocation</span>
                <div className="text-sm font-normal text-gray-400">
                    Risk Profile: <span className="font-bold text-yellow-400">{riskProfile}</span>
                </div>
            </h2>

            {/* Allocation Bars */}
            <div className="space-y-4 mb-6">
                {allocation.map(asset => (
                    <div key={asset.id}>
                        <div className="flex justify-between text-sm mb-1">
                            <span>{asset.name}</span>
                            <span className="text-gray-400">Target: {asset.target}% | Current: <span className={asset.current !== asset.target ? 'text-yellow-400 font-bold' : ''}>{asset.current}%</span></span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden relative">
                            {/* Current Bar */}
                            <div className={`${asset.color} h-2.5 rounded-full absolute top-0 left-0 opacity-80`} style={{ width: `${asset.current}%` }}></div>
                            {/* Target Marker (Simplified as a slightly different shade or overlay) */}
                            <div className="h-4 w-1 bg-white absolute top-[-3px] opacity-50" style={{ left: `${asset.target}%` }}></div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Rebalancing Recommendations */}
            <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-gray-300">Rebalancing Recommendations</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    {allocation.map(asset => {
                        const diff = asset.current - asset.target;
                        if (Math.abs(diff) < 2) return null; // Ignore small drifts
                        return (
                            <div key={asset.id} className="flex items-center justify-between p-2 bg-gray-800 rounded border border-gray-700">
                                <span className="font-bold">{asset.name}</span>
                                <span className={diff > 0 ? 'text-red-400' : 'text-green-400'}>
                                    {diff > 0 ? `SELL ${diff}%` : `BUY ${Math.abs(diff)}%`}
                                </span>
                            </div>
                        );
                    })}
                </div>
                {allocation.every(a => Math.abs(a.current - a.target) < 2) && (
                    <p className="text-gray-500 italic text-center">Portfolio is balanced within tolerance.</p>
                )}
            </div>
        </div>
    );
};

export default RoboAdvisor;
