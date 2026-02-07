import React, { useEffect, useState, useRef } from 'react';
import { Newspaper, RefreshCw } from 'lucide-react';

interface NewsItem {
  id: string;
  headline: string;
  source: string;
  timestamp: number;
  priority: 'LOW' | 'MED' | 'HIGH';
}

const SOURCES = ['REUTERS', 'BLOOMBERG', 'FT', 'WSJ', 'ADAM_INTEL', 'REDDIT_FLOW'];
const HEADLINES = [
  "NVIDIA announces new H200 chips causing supply shock",
  "OpenAI releases GPT-5 preview to enterprise partners",
  "Federal Reserve signals rate cuts amid cooling inflation",
  "Meta open sources Llama-4 with 1T context window",
  "Apple integrates local LLMs into all iOS devices",
  "Microsoft acquires nuclear fusion startup for datacenter power",
  "Anthropic publishes 'Constitution for AI Agents'",
  "Regulation: EU AI Act enforces strict transparency on models",
  "Google DeepMind solves protein folding v2",
  "Amazon AWS launches 'Bedrock Serverless' for agents",
  "DeFi protocol hack attributed to rogue AI agent",
  "SoftBank pours $50B into AI hardware manufacturing"
];

export const NewsWire: React.FC = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initial Population
    const initialNews = Array.from({ length: 15 }, () => generateNews());
    setNews(initialNews);

    const interval = setInterval(() => {
        setNews(prev => [generateNews(), ...prev].slice(0, 50));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const generateNews = (): NewsItem => {
    return {
        id: Math.random().toString(36).substr(2, 9),
        headline: HEADLINES[Math.floor(Math.random() * HEADLINES.length)],
        source: SOURCES[Math.floor(Math.random() * SOURCES.length)],
        timestamp: Date.now(),
        priority: Math.random() > 0.8 ? 'HIGH' : Math.random() > 0.5 ? 'MED' : 'LOW'
    };
  };

  return (
    <div className="h-full flex flex-col bg-[#002b36] border-l border-cyan-900/30 w-full md:w-96 overflow-hidden">
        <div className="p-2 border-b border-cyan-900/50 bg-cyan-950/20 flex justify-between items-center">
            <div className="flex items-center gap-2 text-cyan-400 font-mono text-sm font-bold">
                <Newspaper size={14} /> NEWS WIRE
            </div>
            <RefreshCw size={12} className="text-cyan-700 animate-spin-slow" />
        </div>

        <div className="flex-1 overflow-y-auto p-0 scrollbar-hide" ref={scrollRef}>
            {news.map((item, i) => (
                <div key={item.id} className={`p-3 border-b border-cyan-900/20 hover:bg-cyan-900/20 transition-colors cursor-pointer ${i===0 ? 'animate-flash' : ''}`}>
                    <div className="flex justify-between items-start mb-1">
                        <span className="text-[10px] text-cyan-600 font-bold">{new Date(item.timestamp).toLocaleTimeString()}</span>
                        <span className={`text-[9px] px-1 rounded ${item.priority === 'HIGH' ? 'bg-red-900 text-red-200' : 'bg-cyan-900/50 text-cyan-400'}`}>
                            {item.source}
                        </span>
                    </div>
                    <h4 className={`text-xs font-mono leading-tight ${item.priority === 'HIGH' ? 'text-white font-bold' : 'text-cyan-200'}`}>
                        {item.priority === 'HIGH' && <span className="text-red-500 mr-1">FLASH:</span>}
                        {item.headline}
                    </h4>
                </div>
            ))}
        </div>

        <style>{`
            .animate-spin-slow { animation: spin 4s linear infinite; }
            @keyframes spin { 100% { transform: rotate(360deg); } }
            @keyframes flash {
                0% { background-color: rgba(34, 211, 238, 0.2); }
                100% { background-color: transparent; }
            }
            .animate-flash { animation: flash 1s ease-out; }
        `}</style>
    </div>
  );
};
