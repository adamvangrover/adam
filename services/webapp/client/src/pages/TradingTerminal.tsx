import React, { useState } from 'react';
import AlgoStrategies from '../components/AlgoStrategies';
import RoboAdvisor from '../components/RoboAdvisor';
// Import other components if needed (Terminal, MarketSentiment) - Assuming they exist or are mocked for now.

const TradingTerminal = () => {
    return (
        <div className="flex flex-col min-h-screen bg-gray-900 text-white font-sans">
            <header className="bg-gray-800 p-4 border-b border-gray-700 flex justify-between items-center shadow-md">
                <h1 className="text-xl font-bold tracking-wider">ADAM v26.0 | Trading Terminal</h1>
                <div className="flex space-x-4 text-sm text-gray-400">
                    <span>Status: <span className="text-green-400 font-bold">Online</span></span>
                    <span>Latency: <span className="text-yellow-400">24ms</span></span>
                </div>
            </header>

            <main className="flex-grow p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Column: Algorithmic Strategies & Execution */}
                <div className="lg:col-span-2 space-y-6">
                    <AlgoStrategies />

                    {/* Placeholder for Interactive Chart or Order Entry if not in AlgoStrategies */}
                    <div className="bg-gray-800 p-6 rounded-lg shadow-lg h-64 flex items-center justify-center text-gray-500 border border-dashed border-gray-600">
                        [Interactive Charting Module Placeholder]
                    </div>
                </div>

                {/* Right Column: Robo-Advisor & System Health */}
                <div className="space-y-6">
                    <RoboAdvisor />

                    {/* System Notifications / Alerts */}
                    <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
                        <h3 className="text-lg font-bold mb-3 border-b border-gray-700 pb-2">System Alerts</h3>
                        <ul className="space-y-2 text-sm">
                            <li className="flex items-start space-x-2 text-yellow-300">
                                <span className="mt-1">⚠</span>
                                <span>Volatility Spike Detected (VIX &gt; 25)</span>
                            </li>
                            <li className="flex items-start space-x-2 text-blue-300">
                                <span className="mt-1">ℹ</span>
                                <span>Rebalancing scheduled for 16:00 EST.</span>
                            </li>
                            <li className="flex items-start space-x-2 text-green-300">
                                <span className="mt-1">✓</span>
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
