import React, { useState } from 'react';
import { Loader2, Plus, Trash2, RefreshCw } from 'lucide-react';

const PortfolioEditor: React.FC = () => {
  const [isRebalancing, setIsRebalancing] = useState(false);

  const handleRebalance = () => {
    setIsRebalancing(true);
    // Simulate API call
    setTimeout(() => setIsRebalancing(false), 2000);
  };

  return (
    <div className="glass-panel p-6 rounded-lg mb-6 cyber-border">
      <h3 className="text-lg font-bold text-cyber-cyan mb-6 flex items-center gap-2 tracking-wide">
        EDIT PORTFOLIO
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Add Holding */}
        <div className="bg-cyber-black/30 p-4 rounded border border-cyber-cyan/10">
          <h4 className="text-cyber-text/70 text-xs uppercase tracking-wider mb-4 font-mono font-bold">Add Holding</h4>

          <div className="mb-3">
            <label htmlFor="add-symbol" className="block text-xs text-cyber-text mb-1">Symbol:</label>
            <input
              id="add-symbol"
              type="text"
              className="bg-cyber-black/50 border border-cyber-cyan/20 text-cyber-text p-2 rounded w-full text-sm focus:outline-none focus:border-cyber-cyan focus:ring-1 focus:ring-cyber-cyan transition-all"
              placeholder="e.g. AAPL"
            />
          </div>

          <div className="mb-4">
            <label htmlFor="add-quantity" className="block text-xs text-cyber-text mb-1">Quantity:</label>
            <input
              id="add-quantity"
              type="number"
              className="bg-cyber-black/50 border border-cyber-cyan/20 text-cyber-text p-2 rounded w-full text-sm focus:outline-none focus:border-cyber-cyan focus:ring-1 focus:ring-cyber-cyan transition-all"
              placeholder="0"
            />
          </div>

          <button className="bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/50 p-2 rounded hover:bg-cyber-cyan/30 focus-visible:ring-2 focus-visible:ring-cyber-cyan focus-visible:ring-offset-2 focus-visible:ring-offset-cyber-black transition-all w-full uppercase text-xs font-bold tracking-wider flex items-center justify-center gap-2">
            <Plus className="h-3 w-3" /> Add Position
          </button>
        </div>

        {/* Remove Holding */}
        <div className="bg-cyber-black/30 p-4 rounded border border-cyber-cyan/10">
          <h4 className="text-cyber-text/70 text-xs uppercase tracking-wider mb-4 font-mono font-bold">Remove Holding</h4>

          <div className="mb-4">
            <label htmlFor="remove-symbol" className="block text-xs text-cyber-text mb-1">Symbol:</label>
            <select
              id="remove-symbol"
              className="bg-cyber-black/50 border border-cyber-cyan/20 text-cyber-text p-2 rounded w-full text-sm focus:outline-none focus:border-cyber-cyan focus:ring-1 focus:ring-cyber-cyan transition-all"
            >
              <option value="TC">TC</option>
              <option value="GEC">GEC</option>
            </select>
          </div>

          <button className="bg-cyber-danger/10 text-cyber-danger border border-cyber-danger/50 p-2 rounded hover:bg-cyber-danger/20 focus-visible:ring-2 focus-visible:ring-cyber-danger focus-visible:ring-offset-2 focus-visible:ring-offset-cyber-black transition-all w-full uppercase text-xs font-bold tracking-wider flex items-center justify-center gap-2 mt-auto">
            <Trash2 className="h-3 w-3" /> Remove
          </button>
        </div>

        {/* Rebalance */}
        <div className="bg-cyber-black/30 p-4 rounded border border-cyber-cyan/10 flex flex-col">
          <h4 className="text-cyber-text/70 text-xs uppercase tracking-wider mb-4 font-mono font-bold">Rebalance</h4>
          <p className="text-xs text-cyber-text/60 mb-4 flex-grow">
            Automatically adjust portfolio weights to match target allocation strategy.
          </p>
          <button
            onClick={handleRebalance}
            disabled={isRebalancing}
            className="bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/50 p-2 rounded hover:bg-cyber-cyan/30 disabled:opacity-50 disabled:cursor-not-allowed focus-visible:ring-2 focus-visible:ring-cyber-cyan focus-visible:ring-offset-2 focus-visible:ring-offset-cyber-black transition-all w-full uppercase text-xs font-bold tracking-wider flex items-center justify-center gap-2"
          >
            {isRebalancing ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <RefreshCw className="h-3 w-3" />
                Rebalance Portfolio
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PortfolioEditor;
