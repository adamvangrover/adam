import React, { useState, useEffect, useRef } from 'react';

interface ScenarioSimulatorProps {
    onSimulate: (params: { volatility: number; stress: number }) => void;
}

export const ScenarioSimulator: React.FC<ScenarioSimulatorProps> = ({ onSimulate }) => {
    const [volatility, setVolatility] = useState(0.20);
    const [stress, setStress] = useState(0); // 0-100
    const isMounted = useRef(false);

    // Bolt: Debounce simulation triggers to prevent Monte Carlo engine spam during slider dragging
    useEffect(() => {
        // Skip the initial render effect to match original behavior (no simulation on load)
        if (!isMounted.current) {
            isMounted.current = true;
            return;
        }

        const timer = setTimeout(() => {
            onSimulate({ volatility, stress });
        }, 500);
        return () => clearTimeout(timer);
    }, [volatility, stress, onSimulate]);

    const handleVolChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = parseFloat(e.target.value);
        setVolatility(val);
    };

    const handleStressChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = parseInt(e.target.value, 10);
        setStress(val);
    };

    return (
        <div className="p-4 bg-white rounded-lg shadow-md border border-gray-200">
            <h3 className="text-lg font-bold mb-4 text-gray-800">Scenario Simulator</h3>

            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    EBITDA Volatility: {(volatility * 100).toFixed(0)}%
                </label>
                <input
                    type="range"
                    min="0.05"
                    max="0.50"
                    step="0.01"
                    value={volatility}
                    onChange={handleVolChange}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Low (5%)</span>
                    <span>High (50%)</span>
                </div>
            </div>

            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Geopolitical Stress Index: {stress}
                </label>
                <input
                    type="range"
                    min="0"
                    max="100"
                    step="1"
                    value={stress}
                    onChange={handleStressChange}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Stable (0)</span>
                    <span>Crisis (100)</span>
                </div>
            </div>

            <div className="mt-2 text-xs text-blue-600 italic">
                *Adjusting these parameters will trigger a re-run of the Monte Carlo engine.
            </div>
        </div>
    );
};
