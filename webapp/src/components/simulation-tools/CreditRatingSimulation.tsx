import React from 'react';
import { Activity, Play, TrendingUp } from 'lucide-react';

const CreditRatingSimulation: React.FC = () => {
  return (
    <div className="glass-panel p-6 rounded-lg mb-6 cyber-border">
      <h3 className="text-lg font-bold text-cyber-cyan mb-6 flex items-center gap-2 tracking-wide">
        <Activity className="h-5 w-5" />
        CREDIT RATING SIMULATION
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Inputs Column */}
        <div className="md:col-span-1 space-y-4">
          <h4 className="text-cyber-text/70 text-xs uppercase tracking-wider mb-2 font-mono border-b border-cyber-cyan/10 pb-1">
            Simulation Inputs
          </h4>

          <div>
            <label htmlFor="ticker-input" className="block text-sm font-medium text-cyber-text mb-1">
              Company Ticker
            </label>
            <input
              id="ticker-input"
              type="text"
              defaultValue="TC"
              className="block w-full px-3 py-2 bg-cyber-black/50 border border-cyber-slate rounded focus:outline-none focus:border-cyber-cyan focus:ring-1 focus:ring-cyber-cyan text-sm text-white font-mono placeholder-cyber-text/30 transition-all"
            />
          </div>

          <div>
            <label htmlFor="scenario-select" className="block text-sm font-medium text-cyber-text mb-1">
              Macroeconomic Scenario
            </label>
            <select
              id="scenario-select"
              className="block w-full px-3 py-2 bg-cyber-black/50 border border-cyber-slate rounded focus:outline-none focus:border-cyber-cyan focus:ring-1 focus:ring-cyber-cyan text-sm text-white font-mono transition-all"
            >
              <option>Baseline</option>
              <option>Recession</option>
              <option>Rapid Growth</option>
            </select>
          </div>

          <button className="w-full px-4 py-2 mt-2 bg-cyber-cyan/10 border border-cyber-cyan/50 text-cyber-cyan font-bold rounded hover:bg-cyber-cyan/20 hover:border-cyber-cyan hover:shadow-[0_0_10px_rgba(6,182,212,0.2)] transition-all uppercase tracking-wider text-xs flex items-center justify-center gap-2 group">
            <Play className="h-3 w-3 group-hover:scale-110 transition-transform" />
            Run Simulation
          </button>
        </div>

        {/* Results Column */}
        <div className="md:col-span-2 bg-cyber-black/30 border border-cyber-cyan/10 rounded p-6 flex flex-col items-center justify-center relative overflow-hidden group">
          {/* Background decoration */}
          <div className="absolute inset-0 bg-gradient-to-br from-cyber-cyan/5 via-transparent to-transparent opacity-50"></div>

          <h4 className="relative z-10 text-cyber-text/70 text-xs uppercase tracking-wider mb-4 font-mono">
            Predicted Credit Rating
          </h4>

          <div className="relative z-10 text-6xl font-bold text-cyber-success mb-2 text-glow font-mono animate-pulse">
            AA
          </div>

          <div className="relative z-10 flex items-center gap-2 text-sm text-cyber-text/70 bg-cyber-black/50 px-3 py-1 rounded-full border border-cyber-cyan/10">
            <TrendingUp className="h-3 w-3 text-cyber-success" />
            <span>Confidence Score: <span className="text-white font-bold">92%</span></span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreditRatingSimulation;
