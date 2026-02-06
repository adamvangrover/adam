import React from 'react';
import { usePromptStore } from '../../stores/promptStore';
import { TrendingUp, Activity } from 'lucide-react';

export const TickerTape: React.FC = () => {
  const prompts = usePromptStore((state) => state.prompts);

  // Filter for high alpha items for the tape
  const topPrompts = prompts
    .filter(p => p.alphaScore > 50)
    .slice(0, 10);

  if (topPrompts.length === 0) return null;

  return (
    <div className="w-full bg-black border-y border-green-900 overflow-hidden h-8 flex items-center relative">
      <div className="absolute left-0 bg-black z-10 px-2 text-green-500 font-bold text-xs flex items-center gap-1">
        <Activity size={12} /> LIVE ALPHA
      </div>
      <div className="animate-marquee whitespace-nowrap flex gap-8 items-center pl-24">
        {topPrompts.map((prompt) => (
          <div key={prompt.id} className="inline-flex items-center gap-2 text-xs font-mono text-green-400">
            <span className="opacity-70">{prompt.source}</span>
            <span className="font-bold text-white">{prompt.title.substring(0, 30)}...</span>
            <span className="text-green-300 flex items-center gap-1">
              <TrendingUp size={10} /> {prompt.alphaScore.toFixed(0)}
            </span>
          </div>
        ))}
        {/* Duplicate for seamless loop if needed, simplified here */}
      </div>
      <style>{`
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-100%); }
        }
        .animate-marquee {
          animation: marquee 30s linear infinite;
        }
      `}</style>
    </div>
  );
};
