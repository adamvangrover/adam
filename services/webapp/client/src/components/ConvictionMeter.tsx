import React from 'react';

interface ConvictionMeterProps {
    score: number; // 1-10
    reasoning: string[];
}

export const ConvictionMeter: React.FC<ConvictionMeterProps> = ({ score, reasoning }) => {
    // Determine color based on score
    const getColor = (s: number) => {
        if (s >= 8) return '#10B981'; // Green
        if (s >= 5) return '#F59E0B'; // Orange
        return '#EF4444'; // Red
    };

    return (
        <div className="p-4 bg-white rounded-lg shadow-md">
            <h3 className="text-lg font-bold mb-2">Conviction Level</h3>
            <div className="flex items-center mb-4">
                <span className="text-2xl font-bold mr-2">{score}</span>
                <span className="text-gray-500">/ 10</span>
            </div>

            {/* Meter Bar */}
            <div className="w-full bg-gray-200 rounded-full h-4 mb-4 overflow-hidden">
                <div
                    className="h-full transition-all duration-500 ease-out"
                    style={{
                        width: `${Math.min(Math.max(score, 0), 10) * 10}%`,
                        backgroundColor: getColor(score)
                    }}
                ></div>
            </div>

            {/* Reasoning Trace */}
            <div className="mt-4">
                <h4 className="font-semibold text-sm text-gray-700 mb-2">Reasoning Trace:</h4>
                <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                    {reasoning && reasoning.map((r, i) => (
                        <li key={i}>{r}</li>
                    ))}
                </ul>
            </div>
        </div>
    );
};
