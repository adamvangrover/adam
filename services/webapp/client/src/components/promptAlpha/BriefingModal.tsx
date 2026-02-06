import React from 'react';
import { usePromptStore } from '../../stores/promptStore';
import { useUserStore } from '../../stores/userStore';
import { X, Copy, Star, Terminal, Zap, Layers, Hash } from 'lucide-react';

export const BriefingModal: React.FC = () => {
  const selectedPromptId = usePromptStore((state) => state.selectedPromptId);
  const selectPrompt = usePromptStore((state) => state.selectPrompt);
  const prompts = usePromptStore((state) => state.prompts);
  const toggleFavorite = usePromptStore((state) => state.toggleFavorite);
  const { incrementAlpha, recordAction } = useUserStore();

  const prompt = prompts.find(p => p.id === selectedPromptId);

  if (!selectedPromptId || !prompt) return null;

  const handleCopy = () => {
      navigator.clipboard.writeText(prompt.content);
      incrementAlpha(5);
      recordAction();
  };

  const handleFavorite = () => {
      if (!prompt.isFavorite) {
          incrementAlpha(10);
      }
      toggleFavorite(prompt.id);
      recordAction();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="w-full max-w-4xl bg-slate-950 border border-cyan-500/50 shadow-[0_0_20px_rgba(34,211,238,0.2)] rounded-sm flex flex-col max-h-[90vh]">

        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-cyan-900 bg-slate-900">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-cyan-900/20 border border-cyan-500/30 rounded">
               <Terminal className="text-cyan-500" size={20} />
             </div>
             <div>
               <h2 className="text-cyan-400 font-mono text-lg font-bold uppercase tracking-wider">
                 Alpha Briefing // {prompt.id}
               </h2>
               <div className="text-xs text-slate-500 font-mono">
                 DETECTED: {new Date(prompt.timestamp).toLocaleString()} | SOURCE: {prompt.source}
               </div>
             </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleFavorite}
              className={`p-2 hover:bg-slate-800 rounded transition-colors ${prompt.isFavorite ? 'text-yellow-400' : 'text-slate-500'}`}
            >
              <Star size={20} fill={prompt.isFavorite ? "currentColor" : "none"} />
            </button>
            <button
              onClick={() => selectPrompt(null)}
              className="p-2 hover:bg-red-900/20 text-slate-500 hover:text-red-400 rounded transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Content Grid */}
        <div className="flex-1 overflow-hidden flex flex-col md:flex-row">

          {/* Main Content (Left) */}
          <div className="flex-1 p-6 overflow-y-auto border-r border-cyan-900/30 scrollbar-thin scrollbar-thumb-cyan-900 scrollbar-track-transparent">
             <h3 className="text-white font-bold text-xl mb-4 font-mono">{prompt.title}</h3>
             <div className="bg-black border border-slate-800 p-4 rounded font-mono text-sm text-slate-300 whitespace-pre-wrap leading-relaxed selection:bg-cyan-500/30">
               {prompt.content}
             </div>
             <div className="mt-4 flex gap-2">
               <button
                 onClick={handleCopy}
                 className="flex items-center gap-2 px-4 py-2 bg-cyan-900/20 hover:bg-cyan-900/40 text-cyan-400 border border-cyan-900/50 rounded text-sm font-mono transition-colors active:scale-95"
               >
                 <Copy size={14} /> COPY PAYLOAD (+5 ALPHA)
               </button>
             </div>
          </div>

          {/* Metrics Sidebar (Right) */}
          <div className="w-full md:w-80 bg-slate-950 p-6 overflow-y-auto">
             <div className="mb-6">
               <div className="text-xs text-slate-500 font-mono mb-2 uppercase tracking-widest">Alpha Score</div>
               <div className="flex items-end gap-2">
                 <span className="text-4xl font-bold text-cyan-400 font-mono">{prompt.alphaScore}</span>
                 <span className="text-slate-600 mb-1">/ 100</span>
               </div>
               <div className="w-full bg-slate-800 h-1 mt-2 rounded-full overflow-hidden">
                 <div
                   className="h-full bg-cyan-500 shadow-[0_0_10px_rgba(34,211,238,0.5)]"
                   style={{ width: `${prompt.alphaScore}%` }}
                 />
               </div>
             </div>

             <div className="space-y-6">
               <MetricItem
                 icon={<Layers size={16} />}
                 label="Complexity (Length)"
                 value={prompt.metrics.length}
                 max={100}
               />
               <MetricItem
                 icon={<Hash size={16} />}
                 label="Variable Density"
                 value={prompt.metrics.variableDensity}
                 max={30}
               />
               <MetricItem
                 icon={<Zap size={16} />}
                 label="Structural Integrity"
                 value={prompt.metrics.structuralKeywords}
                 max={40}
               />
             </div>

             <div className="mt-8 pt-6 border-t border-slate-800">
               <div className="text-xs text-slate-500 font-mono mb-3 uppercase">Tags</div>
               <div className="flex flex-wrap gap-2">
                 {prompt.tags.map(tag => (
                   <span key={tag} className="px-2 py-1 bg-slate-900 text-xs text-slate-400 rounded font-mono border border-slate-700">
                     #{tag}
                   </span>
                 ))}
               </div>
             </div>
          </div>

        </div>
      </div>
    </div>
  );
};

const MetricItem: React.FC<{ icon: React.ReactNode, label: string, value: number, max: number }> = ({ icon, label, value, max }) => {
  const percent = Math.min(100, (value / max) * 100);
  return (
    <div>
      <div className="flex items-center justify-between text-sm text-slate-400 mb-1 font-mono">
        <div className="flex items-center gap-2">
          {icon} <span>{label}</span>
        </div>
        <span>{value}</span>
      </div>
      <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
        <div className="h-full bg-cyan-700" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
};
