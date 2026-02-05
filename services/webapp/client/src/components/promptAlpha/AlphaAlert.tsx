import React, { useEffect, useState } from 'react';
import { usePromptStore } from '../../stores/promptStore';
import { PromptObject } from '../../types/promptAlpha';
import { Zap, X } from 'lucide-react';

export const AlphaAlert: React.FC = () => {
  const latestPrompts = usePromptStore((state) => state.prompts);
  const [alert, setAlert] = useState<PromptObject | null>(null);

  useEffect(() => {
    if (latestPrompts.length === 0) return;

    const latest = latestPrompts[0];
    const isFresh = (Date.now() - latest.timestamp) < 2000; // Under 2 seconds old
    const isHighAlpha = latest.alphaScore >= 90;

    if (isFresh && isHighAlpha) {
      setAlert(latest);
      // Auto dismiss
      const timer = setTimeout(() => setAlert(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [latestPrompts]);

  if (!alert) return null;

  return (
    <div className="fixed top-20 right-4 z-50 animate-in slide-in-from-right fade-in duration-300">
      <div className="bg-cyan-950/90 border border-cyan-400 p-4 rounded shadow-[0_0_30px_rgba(34,211,238,0.3)] backdrop-blur w-80 relative overflow-hidden">

        {/* Shine effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/10 to-transparent translate-x-[-100%] animate-[shimmer_2s_infinite]"></div>

        <button
          onClick={() => setAlert(null)}
          className="absolute top-2 right-2 text-cyan-700 hover:text-cyan-400"
        >
          <X size={14} />
        </button>

        <div className="flex items-start gap-3">
          <div className="bg-cyan-500/20 p-2 rounded-full border border-cyan-400 text-cyan-400 animate-pulse">
            <Zap size={20} />
          </div>
          <div>
            <div className="text-xs font-bold text-cyan-400 tracking-wider mb-1">ALPHA SIGNAL DETECTED</div>
            <div className="text-sm font-bold text-white mb-1 truncate pr-4">{alert.title}</div>
            <div className="text-xs font-mono text-cyan-200">
              SCORE: <span className="text-white text-lg font-bold">{alert.alphaScore.toFixed(1)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
