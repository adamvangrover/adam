import React, { useState, useEffect } from 'react';

const Synthesizer: React.FC = () => {
  const [confidenceScore, setConfidenceScore] = useState<number>(50);

  useEffect(() => {
    // Simulate real-time signal aggregation
    const interval = setInterval(() => {
      setConfidenceScore(prev => {
        const variation = (Math.random() - 0.5) * 10;
        let newScore = prev + variation;
        if (newScore < 0) newScore = 0;
        if (newScore > 100) newScore = 100;
        return Number(newScore.toFixed(1));
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8 text-cyber-cyan font-mono bg-cyber-black min-h-screen">
      <h1 className="text-3xl font-bold mb-6 tracking-widest uppercase border-b border-cyber-cyan/30 pb-4">
        Synthesizer Dashboard
      </h1>

      <div className="bg-cyber-slate/20 p-6 rounded-lg border border-cyber-cyan/30 shadow-[0_0_15px_rgba(6,182,212,0.15)] max-w-2xl mx-auto mt-10">
        <h2 className="text-xl mb-4 uppercase">Aggregated Confidence Score</h2>

        <div className="relative pt-8 pb-8 flex items-center justify-center">
          {/* Circular Gauge Representation */}
          <div className="relative w-64 h-64 rounded-full border-4 border-cyber-slate/50 flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.2)]">
            <div
              className="absolute inset-0 rounded-full border-4 border-t-cyber-cyan border-r-cyber-cyan border-b-transparent border-l-transparent transition-transform duration-1000 ease-in-out"
              style={{ transform: `rotate(${(confidenceScore / 100) * 360 - 45}deg)` }}
            ></div>
            <div className="text-5xl font-bold text-white tracking-widest">
              {confidenceScore.toFixed(0)}<span className="text-2xl text-cyber-cyan">%</span>
            </div>
          </div>
        </div>

        <div className="mt-8 grid grid-cols-3 gap-4 text-center text-sm">
          <div className="p-3 bg-cyber-black/50 rounded border border-cyber-cyan/20">
            <div className="text-cyber-text/50 mb-1">MARKET SENTIMENT</div>
            <div className="text-cyber-cyan">{(confidenceScore * 0.9).toFixed(1)}</div>
          </div>
          <div className="p-3 bg-cyber-black/50 rounded border border-cyber-cyan/20">
            <div className="text-cyber-text/50 mb-1">RISK ASSESSMENT</div>
            <div className="text-cyber-cyan">{(100 - confidenceScore * 0.5).toFixed(1)}</div>
          </div>
          <div className="p-3 bg-cyber-black/50 rounded border border-cyber-cyan/20">
            <div className="text-cyber-text/50 mb-1">FUNDAMENTAL</div>
            <div className="text-cyber-cyan">{(confidenceScore * 1.1 > 100 ? 100 : confidenceScore * 1.1).toFixed(1)}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Synthesizer;
