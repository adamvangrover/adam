import React from 'react';
import { TrendingUp, TrendingDown, Newspaper, Activity } from 'lucide-react';

interface CardProps {
  title: string;
  value: string;
  change: string;
  percent: string;
}

const Card: React.FC<CardProps> = ({ title, value, change, percent }) => {
  const isPositive = change.startsWith('+');

  return (
    <div className="bg-cyber-black/50 border border-cyber-cyan/20 p-4 rounded hover:border-cyber-cyan/50 transition-colors group">
      <h4 className="text-cyber-text/70 text-xs uppercase tracking-wider mb-2 font-mono">{title}</h4>
      <div className="text-2xl font-mono text-white font-bold mb-2">{value}</div>
      <div className={`text-xs flex items-center gap-1 font-mono ${isPositive ? 'text-cyber-success' : 'text-cyber-danger'}`}>
        {isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
        <span>{change} ({percent})</span>
      </div>
    </div>
  );
};

const MarketSummary: React.FC = () => {
  // Mock data - normally would come from an API
  const sentimentValue = 65;

  return (
    <div className="glass-panel p-6 rounded-lg mb-6 cyber-border">
      <h3 className="text-lg font-bold text-cyber-cyan mb-6 flex items-center gap-2 tracking-wide">
        <Activity className="h-5 w-5" />
        MARKET SUMMARY
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card title="S&P 500" value="5,432.10" change="+12.34" percent="0.23%" />
        <Card title="Dow Jones" value="34,567.89" change="-56.78" percent="0.16%" />
        <Card title="Nasdaq" value="17,890.12" change="+98.76" percent="0.55%" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-6 border-t border-cyber-cyan/10">
        {/* News Ticker */}
        <div>
          <h4 className="text-cyber-text/50 text-xs uppercase tracking-wider mb-3 flex items-center gap-2">
            <Newspaper className="h-3 w-3" />
            News Ticker
          </h4>
          <div className="bg-cyber-black/30 p-3 rounded border border-cyber-cyan/10">
            <p className="text-sm text-cyber-text font-mono truncate animate-pulse">
              <span className="text-cyber-warning mr-2">[BREAKING]</span>
              Fed hints at potential rate cuts later this year...
            </p>
          </div>
        </div>

        {/* Market Sentiment */}
        <div>
          <h4 className="text-cyber-text/50 text-xs uppercase tracking-wider mb-3 flex justify-between">
            <span>Market Sentiment</span>
            <span className="text-cyber-cyan">{sentimentValue}% Bullish</span>
          </h4>
          <div
            className="h-6 bg-cyber-black rounded-full overflow-hidden border border-cyber-cyan/20 relative"
            role="progressbar"
            aria-valuenow={sentimentValue}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Market Sentiment: Bullish"
          >
            {/* Grid lines for visual effect */}
            <div className="absolute inset-0 grid grid-cols-10 z-10 opacity-20">
               {[...Array(10)].map((_, i) => (
                 <div key={i} className="border-r border-cyber-slate h-full"></div>
               ))}
            </div>

            <div
              className="h-full bg-gradient-to-r from-cyber-cyan/50 to-cyber-success/80 transition-all duration-1000 ease-out relative z-0"
              style={{ width: `${sentimentValue}%` }}
            >
              <div className="absolute right-0 top-0 bottom-0 w-[1px] bg-white/50 shadow-[0_0_10px_white]"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketSummary;
