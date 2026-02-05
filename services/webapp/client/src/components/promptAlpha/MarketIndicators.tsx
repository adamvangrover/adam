import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// Synthetic Market Data Generator
const generateData = (trend: number) => {
  return Array.from({ length: 50 }, (_, i) => ({
    time: i,
    value: 100 + i * trend + Math.random() * 20 - 10
  }));
};

const DATA_COMPUTE = generateData(0.5);
const DATA_ALPHA = generateData(0.2);
const DATA_ENERGY = generateData(0.8);

export const MarketIndicators: React.FC = () => {
  return (
    <div className="grid grid-cols-3 gap-2 h-24 w-full bg-[#002b36] p-2 border-b border-cyan-900/30">
      <IndicatorCard title="COMPUTE INDEX" value="145.20" change="+2.4%" data={DATA_COMPUTE} color="#22c55e" />
      <IndicatorCard title="ALPHA YIELD" value="8.45%" change="+0.1%" data={DATA_ALPHA} color="#eab308" />
      <IndicatorCard title="ENERGY COST" value="$0.14/kWh" change="+1.2%" data={DATA_ENERGY} color="#ef4444" />
    </div>
  );
};

const IndicatorCard: React.FC<{ title: string, value: string, change: string, data: any[], color: string }> = ({ title, value, change, data, color }) => (
  <div className="flex flex-col justify-between border-r border-cyan-900/30 last:border-0 pr-2">
    <div className="flex justify-between items-baseline">
      <span className="text-[9px] text-cyan-600 font-bold tracking-wider">{title}</span>
      <span className={`text-[9px] font-bold ${change.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>{change}</span>
    </div>
    <div className="text-lg font-mono font-bold text-cyan-100">{value}</div>
    <div className="h-8 w-full opacity-50">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
);
