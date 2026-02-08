import React from 'react';

interface MarketTickerProps {
  data: Record<string, any>;
}

const MarketTicker: React.FC<MarketTickerProps> = ({ data }) => {
  return (
    <div className="flex bg-gray-900 border-b border-gray-800 text-xs font-mono py-1 overflow-x-auto whitespace-nowrap scrollbar-none">
      {Object.entries(data).map(([symbol, info]) => (
        <div key={symbol} className="px-3 flex gap-1 border-r border-gray-800 items-center">
          <span className="font-bold text-gray-300">{symbol}</span>
          <span className={info.change_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
            ${info.price.toFixed(2)} ({info.change_pct > 0 ? '+' : ''}{info.change_pct}%)
          </span>
        </div>
      ))}
      {Object.keys(data).length === 0 && (
        <span className="text-gray-500 italic px-4">Waiting for market feed...</span>
      )}
    </div>
  );
};

export default MarketTicker;
