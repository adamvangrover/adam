import React, { useState } from 'react';

const FundamentalAnalysis: React.FC = () => {
  const [loading, setLoading] = useState(false);

  const handleRunAnalysis = () => {
    setLoading(true);
    // Simulate analysis delay
    setTimeout(() => setLoading(false), 2000);
  };

  return (
    <div className="border border-cyber-cyan/30 p-4 my-4 rounded-lg bg-cyber-dark/50 shadow-sm">
      <h3 className="mt-0 text-cyber-cyan text-lg font-mono mb-4 border-b border-cyber-cyan/20 pb-2">
        Fundamental Analysis
      </h3>
      <div className="flex gap-6 flex-col md:flex-row">
        <div className="flex-1">
          <h4 className="text-cyber-text text-xs uppercase tracking-wider mb-4 font-bold">
            Configuration
          </h4>

          <label htmlFor="ticker-input" className="block text-sm text-cyber-text mb-1">
            Company Ticker
          </label>
          <input
            id="ticker-input"
            type="text"
            defaultValue="TC"
            className="w-full p-2 mb-4 bg-cyber-black border border-cyber-slate text-cyber-cyan focus:border-cyber-neon focus:ring-1 focus:ring-cyber-neon outline-none rounded transition-all font-mono"
            placeholder="e.g. AAPL"
          />

          <label htmlFor="model-select" className="block text-sm text-cyber-text mb-1">
            Valuation Model
          </label>
          <div className="relative mb-4">
            <select
              id="model-select"
              className="w-full p-2 bg-cyber-black border border-cyber-slate text-cyber-cyan focus:border-cyber-neon focus:ring-1 focus:ring-cyber-neon outline-none rounded appearance-none cursor-pointer transition-all"
            >
              <option>Discounted Cash Flow (DCF)</option>
              <option>Comparable Company Analysis</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-cyber-cyan">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
              </svg>
            </div>
          </div>

          <button
            onClick={handleRunAnalysis}
            className="w-full p-3 bg-cyber-cyan/10 border border-cyber-cyan text-cyber-cyan hover:bg-cyber-cyan/20 active:bg-cyber-cyan/30 transition-all rounded uppercase font-mono text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            disabled={loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-cyber-cyan" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Analyzing...</span>
              </>
            ) : (
              'Run Analysis'
            )}
          </button>
        </div>

        <div className="flex-[2] border border-cyber-slate p-4 rounded bg-cyber-black/30 flex flex-col">
          <h4 className="text-cyber-text text-xs uppercase tracking-wider mb-4 font-bold flex justify-between items-center">
            <span>Results Output</span>
            {loading && <span className="text-cyber-neon text-[10px] animate-pulse">LIVE UPDATE...</span>}
          </h4>

          <div className="mb-4 p-3 bg-cyber-black/50 rounded border-l-2 border-cyber-success">
            <p className="text-cyber-text text-sm">Intrinsic Value Estimate</p>
            <p className="text-2xl font-mono text-cyber-success font-bold mt-1">
              $165.00 <span className="text-xs text-cyber-text font-normal ml-2">(+12% Upside)</span>
            </p>
          </div>

          <div
            className="flex-1 min-h-[150px] flex flex-col justify-center items-center text-cyber-slate border border-dashed border-cyber-slate/30 rounded bg-cyber-black/20"
            role="img"
            aria-label="Valuation Chart Placeholder"
          >
            <svg className="w-12 h-12 mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path>
            </svg>
            <span className="text-sm font-mono opacity-70">[Valuation Chart Visualization]</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FundamentalAnalysis;
