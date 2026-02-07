import React, { useState } from 'react';

const AlgoStrategies = () => {
    const [strategies, setStrategies] = useState([
        { id: 1, name: 'Momentum Alpha', type: 'Trend Following', roi: 12.5, drawdown: 3.2, status: 'Active' },
        { id: 2, name: 'Mean Reversion Gamma', type: 'Mean Reversion', roi: 8.1, drawdown: 1.5, status: 'Active' },
        { id: 3, name: 'Adaptive Volatility', type: 'Reinforcement Learning', roi: 15.3, drawdown: 4.8, status: 'Active' },
        { id: 4, name: 'Arbitrage Delta', type: 'Arbitrage', roi: 5.2, drawdown: 0.8, status: 'Paused' }
    ]);

    const [trades, setTrades] = useState([
        { id: 101, strategy: 'Momentum Alpha', symbol: 'NVDA', action: 'BUY', price: 135.20, time: '10:30 AM' },
        { id: 102, strategy: 'Adaptive Volatility', symbol: 'SPY', action: 'SELL', price: 585.50, time: '11:15 AM' },
        { id: 103, strategy: 'Mean Reversion Gamma', symbol: 'TSLA', action: 'BUY', price: 242.00, time: '01:45 PM' }
    ]);

    return (
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-6 text-white">
            <h2 className="text-2xl font-bold mb-4 border-b border-gray-700 pb-2">Algorithmic Strategies</h2>

            {/* Strategy Performance Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                {strategies.map(strat => (
                    <div key={strat.id} className={`p-4 rounded-lg border ${strat.status === 'Active' ? 'border-green-500 bg-gray-700' : 'border-gray-600 bg-gray-800 opacity-75'}`}>
                        <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold text-lg">{strat.name}</h3>
                            <span className={`text-xs px-2 py-1 rounded ${strat.status === 'Active' ? 'bg-green-900 text-green-300' : 'bg-gray-600 text-gray-300'}`}>
                                {strat.status}
                            </span>
                        </div>
                        <p className="text-sm text-gray-400 mb-2">{strat.type}</p>
                        <div className="flex justify-between text-sm">
                            <div>
                                <span className="block text-gray-500">ROI</span>
                                <span className="text-green-400 font-mono">+{strat.roi}%</span>
                            </div>
                            <div>
                                <span className="block text-gray-500">Max DD</span>
                                <span className="text-red-400 font-mono">-{strat.drawdown}%</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Recent Trades Log */}
            <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-gray-300">Recent Trades</h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead className="text-gray-500 border-b border-gray-700">
                            <tr>
                                <th className="pb-2">Time</th>
                                <th className="pb-2">Strategy</th>
                                <th className="pb-2">Symbol</th>
                                <th className="pb-2">Action</th>
                                <th className="pb-2 text-right">Price</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-800">
                            {trades.map(trade => (
                                <tr key={trade.id} className="hover:bg-gray-800 transition-colors">
                                    <td className="py-2 text-gray-400">{trade.time}</td>
                                    <td className="py-2 text-blue-300">{trade.strategy}</td>
                                    <td className="py-2 font-bold">{trade.symbol}</td>
                                    <td className={`py-2 font-bold ${trade.action === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                                        {trade.action}
                                    </td>
                                    <td className="py-2 text-right font-mono">${trade.price.toFixed(2)}</td>
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
