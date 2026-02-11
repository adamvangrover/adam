'use client';
import React, { useState, useEffect, useRef } from 'react';

// Define the assets to track
const INITIAL_ASSETS = [
  { symbol: 'SPX', name: 'S&P 500', price: 6964.82, change: 0.47 },
  { symbol: 'DJI', name: 'Dow Jones', price: 50135.87, change: 0.65 },
  { symbol: 'NDX', name: 'Nasdaq 100', price: 21543.12, change: 0.92 },
  { symbol: 'VIX', name: 'Volatility', price: 17.76, change: -18.42 },
  { symbol: 'ORCL', name: 'Oracle Corp', price: 178.45, change: 9.60 },
  { symbol: 'BTC', name: 'Bitcoin', price: 70351.00, change: 0.06 },
  { symbol: 'DXY', name: 'US Dollar', price: 96.83, change: -0.80 },
  { symbol: 'TNX', name: '10Y Yield', price: 4.21, change: -0.05 },
];

interface Asset {
  symbol: string;
  name: string;
  price: number;
  change: number;
  history: number[]; // For sparklines
}

const MarketDashboard: React.FC = () => {
  const [assets, setAssets] = useState<Asset[]>(() =>
    INITIAL_ASSETS.map(a => ({ ...a, history: Array(20).fill(a.price) }))
  );
  const isMounted = useRef(true);

  useEffect(() => {
    isMounted.current = true;
    const interval = setInterval(() => {
      if (!isMounted.current) return;

      setAssets(currentAssets =>
        currentAssets.map(asset => {
          // Simulate random walk
          const volatility = asset.symbol === 'VIX' ? 0.05 : asset.symbol === 'BTC' ? 0.002 : 0.0005;
          const change = (Math.random() - 0.5) * asset.price * volatility;
          const newPrice = asset.price + change;
          const newHistory = [...asset.history.slice(1), newPrice];

          return {
            ...asset,
            price: newPrice,
            change: asset.change + (Math.random() - 0.5) * 0.1, // Drift change pct slightly
            history: newHistory
          };
        })
      );
    }, 2000);

    return () => {
      isMounted.current = false;
      clearInterval(interval);
    };
  }, []);

  // Simple SVG Sparkline
  const renderSparkline = (history: number[], color: string) => {
    const min = Math.min(...history);
    const max = Math.max(...history);
    const range = max - min || 1;
    const width = 60;
    const height = 20;

    const points = history.map((price, i) => {
      const x = (i / (history.length - 1)) * width;
      const y = height - ((price - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width={width} height={height} className="overflow-visible">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    );
  };

  return (
    <div className="flex flex-col h-full w-full bg-slate-900/50 rounded-lg overflow-hidden font-mono text-xs">
      <div className="flex justify-between items-center p-2 border-b border-slate-700 bg-slate-800/50">
        <span className="text-cyan-400 font-bold">LIVE MARKET DATA</span>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          <span className="text-slate-400 text-[10px]">REALTIME</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="text-slate-500 border-b border-slate-800">
              <th className="pb-2 font-normal">SYMBOL</th>
              <th className="pb-2 font-normal text-right">PRICE</th>
              <th className="pb-2 font-normal text-right">CHG%</th>
              <th className="pb-2 font-normal text-right pr-2">TREND</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {assets.map((asset) => {
              const isPositive = asset.change >= 0;
              const colorClass = isPositive ? 'text-green-400' : 'text-red-400';
              const strokeColor = isPositive ? '#4ade80' : '#f87171';

              return (
                <tr key={asset.symbol} className="hover:bg-white/5 transition-colors">
                  <td className="py-2 pl-1">
                    <div className="font-bold text-slate-200">{asset.symbol}</div>
                    <div className="text-[10px] text-slate-500 hidden sm:block">{asset.name}</div>
                  </td>
                  <td className={`py-2 text-right font-bold ${colorClass}`}>
                    {asset.symbol === 'BTC' || asset.symbol === 'DJI' || asset.symbol === 'SPX' || asset.symbol === 'NDX'
                      ? asset.price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                      : asset.price.toFixed(2)
                    }
                  </td>
                  <td className={`py-2 text-right ${colorClass}`}>
                    {asset.change > 0 ? '+' : ''}{asset.change.toFixed(2)}%
                  </td>
                  <td className="py-2 text-right pr-1">
                    <div className="flex justify-end">
                      {renderSparkline(asset.history, strokeColor)}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MarketDashboard;
