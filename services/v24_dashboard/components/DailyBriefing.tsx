'use client';
import React, { useState, useEffect, useRef } from 'react';

const DailyBriefing: React.FC = () => {
  const [isStressTesting, setIsStressTesting] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [testComplete, setTestComplete] = useState(false);
  const isMounted = useRef(true);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  const runStressTest = () => {
    if (isStressTesting || testComplete) return;

    setIsStressTesting(true);
    setLogs(['> ACCESSING PRIMARY DEALER NODES...']);

    setTimeout(() => {
      if (isMounted.current) {
        setLogs(prev => [...prev, '> INJECTING SYNTHETIC LIQUIDITY SHOCK...']);
      }
    }, 800);

    setTimeout(() => {
      if (isMounted.current) {
        setLogs(prev => [...prev, '> MONITORING REPO MARKET SPREADS...']);
      }
    }, 1600);

    setTimeout(() => {
      if (isMounted.current) {
        setLogs(prev => [...prev, '> ANALYSIS COMPLETE: COLLATERAL CHAINS STABLE.', '> RISK VECTOR: LOW.', '> CONCLUSION: RUMORS UNFOUNDED.']);
        setIsStressTesting(false);
        setTestComplete(true);
      }
    }, 3000);
  };

  return (
    <div className="w-full bg-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-2xl font-sans text-slate-200">
      <div className="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
        <h2 className="text-2xl font-bold text-green-500 font-mono tracking-tight flex items-center gap-3">
          <span className="inline-block w-3 h-3 bg-green-500 rounded-full animate-pulse shadow-[0_0_10px_#22c55e]"></span>
          DAILY BRIEFING
        </h2>
        <div className="text-xs font-mono text-slate-500 border border-slate-700 rounded px-2 py-1">
          FEB 10, 2026 // SYSTEM: NOMINAL
        </div>
      </div>

      <div className="space-y-6">
        {/* Signal Integrity Section */}
        <section>
          <h3 className="text-lg font-bold text-cyan-400 font-mono mb-2">📡 Signal Integrity: The Super Bowl Hangover</h3>
          <p className="text-slate-300 leading-relaxed text-sm">
            The simulation has rebooted after a weekend of high-velocity betting, and for now, the <span className="text-green-400 font-bold">"Dow 50K"</span> firewall is holding.
            The S&P 500 drifted +0.47% higher to 6,964.82, while the Dow Jones Industrial Average clawed out a new closing record at 50,135.87.
            However, the architecture reveals a distinct lack of structural conviction. While tech-heavy Nasdaq gains (+0.9%) painted a pretty picture,
            the VIX plummeted <span className="text-red-400">-18.42%</span> to 17.76, a massive "volatility crush" typical of a post-event reset.
          </p>
          <div className="mt-3 p-3 bg-slate-800/50 rounded border-l-2 border-yellow-500 text-xs text-slate-400">
            <strong>Credit Dominance Check:</strong> Signal Neutral-to-Constructive. 10-Year Treasury Yields cooled to 4.21%.
            High Yield spreads tightened. No "trap" detected, but volume feels synthetic.
          </div>
        </section>

        {/* Artifacts Grid */}
        <section>
          <h3 className="text-lg font-bold text-cyan-400 font-mono mb-3">🏮 Artifacts</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-800/40 p-3 rounded border border-slate-700 hover:border-cyan-500/50 transition-colors">
              <div className="font-bold text-white mb-1">Oracle (ORCL)</div>
              <div className="text-green-400 font-mono text-sm mb-2">+9.6%</div>
              <p className="text-xs text-slate-400">OpenAI beneficiaries narrative turned legacy monolith into high-frame-rate flyer.</p>
            </div>
            <div className="bg-slate-800/40 p-3 rounded border border-slate-700 hover:border-cyan-500/50 transition-colors">
              <div className="font-bold text-white mb-1">Bitcoin</div>
              <div className="text-green-400 font-mono text-sm mb-2">$70,351 (+0.06%)</div>
              <p className="text-xs text-slate-400">Flatline recovery after $60k flush. Waiting for volatility injection.</p>
            </div>
            <div className="bg-slate-800/40 p-3 rounded border border-slate-700 hover:border-cyan-500/50 transition-colors">
              <div className="font-bold text-white mb-1">DXY (Dollar)</div>
              <div className="text-red-400 font-mono text-sm mb-2">-0.8%</div>
              <p className="text-xs text-slate-400">Liquidity release keeping equity indexes green despite yield pressure.</p>
            </div>
          </div>
        </section>

        {/* The Glitch */}
        <section className="relative overflow-hidden group">
          <div className="absolute inset-0 bg-red-500/5 group-hover:bg-red-500/10 transition-colors rounded"></div>
          <h3 className="text-lg font-bold text-red-400 font-mono mb-2 relative z-10">🌀 The Glitch</h3>
          <blockquote className="text-sm text-slate-300 italic border-l-2 border-red-500 pl-4 py-1 relative z-10">
            "We just witnessed nearly a billion dollars wagered on the flight of a pigskin... The 50,000 Dow is a monument built on the shifting sands of a liquidity pullback... Watch the Chinese Treasury holdings."
          </blockquote>
        </section>

        {/* Interactive Element */}
        <section className="pt-4 border-t border-slate-700 mt-4">
          <div className="flex flex-col items-center text-center">
            <p className="text-sm text-slate-400 mb-4">
              Next Transmission: Tuesday, Feb 10, 18:00 ET. <br/>
              Run 'Liquidity Stress Test' on primary dealers?
            </p>

            <button
              onClick={runStressTest}
              disabled={isStressTesting || testComplete}
              className={`
                px-6 py-2 rounded font-mono font-bold text-sm tracking-widest transition-all duration-300
                ${testComplete
                  ? 'bg-green-500/10 text-green-500 border border-green-500 cursor-default shadow-[0_0_15px_rgba(34,197,94,0.3)]'
                  : 'bg-transparent border border-cyan-500 text-cyan-500 hover:bg-cyan-500/10 hover:shadow-[0_0_15px_rgba(6,182,212,0.3)] active:scale-95'}
                ${isStressTesting ? 'opacity-70 cursor-wait' : ''}
              `}
            >
              {isStressTesting ? 'INITIALIZING...' : testComplete ? 'STRESS TEST COMPLETE' : 'RUN LIQUIDITY STRESS TEST'}
            </button>

            {/* Console Output */}
            {(logs.length > 0) && (
              <div className="mt-4 w-full bg-black/80 rounded p-3 text-left font-mono text-xs text-gray-300 border-l-2 border-cyan-500 animate-in fade-in slide-in-from-top-2">
                {logs.map((log, i) => (
                  <div key={i} className="mb-1 last:mb-0 last:text-green-400 last:font-bold">
                    {log}
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default DailyBriefing;
