import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

// Types
interface Prompt {
  id: string;
  name: string;
  path: string;
  category: string;
  type: string;
  content: string;
  metadata: any;
  author: string;
  version: string;
  score: number;
}

const PromptAlpha: React.FC = () => {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPrompt, setSelectedPrompt] = useState<Prompt | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchContext, setSearchContext] = useState<string>('');

  const fetchPrompts = async (context?: string) => {
    setLoading(true);
    setError(null);
    try {
      const url = context
        ? `/api/prompts?context=${encodeURIComponent(context)}`
        : '/api/prompts';

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setPrompts(data);
    } catch (e: any) {
        console.error("Failed to fetch prompts", e);
        setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrompts();
  }, []);

  const handleSearch = (e: React.FormEvent) => {
      e.preventDefault();
      fetchPrompts(searchContext);
  };

  return (
    <div className="h-full flex flex-col p-4 bg-[var(--bg-color)] text-[var(--text-color)] font-mono">
        <header className="mb-4 border-b border-[var(--primary-color)] pb-2 flex justify-between items-end">
            <div>
                <h1 className="text-2xl font-bold text-[var(--primary-color)]">PROMPT ALPHA /// TERMINAL</h1>
                <p className="text-xs text-[#666] mb-2">HIGH-ROI PROMPT AGGREGATION & SCORING SYSTEM</p>

                <form onSubmit={handleSearch} className="flex gap-2">
                    <input
                        type="text"
                        value={searchContext}
                        onChange={(e) => setSearchContext(e.target.value)}
                        placeholder="ENTER CONTEXT / STRATEGY (e.g. 'risk')"
                        className="bg-[#111] border border-[#444] text-[#ccc] px-2 py-1 text-sm w-64 focus:border-[var(--primary-color)] outline-none"
                    />
                    <button
                        type="submit"
                        className="px-3 py-1 bg-[#222] border border-[#444] text-[#aaa] text-sm hover:bg-[var(--primary-color)] hover:text-black transition-colors"
                    >
                        SET_CONTEXT
                    </button>
                </form>
            </div>
            <button
                onClick={() => fetchPrompts(searchContext)}
                className="px-4 py-1 border border-[var(--primary-color)] text-[var(--primary-color)] hover:bg-[var(--primary-color)] hover:text-black transition-colors mb-2"
            >
                REFRESH_FEED
            </button>
        </header>

        {error && (
            <div className="bg-red-900/20 border border-red-500 text-red-500 p-2 mb-4">
                ERROR: {error}
            </div>
        )}

        {loading ? (
             <div className="flex-grow flex items-center justify-center text-[var(--primary-color)] animate-pulse">
                SCANNING REPO... CALCULATING ROI...
             </div>
        ) : (
            <div className="flex-grow overflow-auto border border-[#333] bg-black/50 relative">
                <table className="w-full text-left text-sm border-collapse">
                    <thead className="sticky top-0 bg-black z-10 border-b border-[#333]">
                        <tr>
                            <th className="p-2 font-bold text-[var(--accent-color)] w-24">ROI</th>
                            <th className="p-2 font-bold text-[#aaa]">NAME</th>
                            <th className="p-2 font-bold text-[#aaa]">CATEGORY</th>
                            <th className="p-2 font-bold text-[#aaa]">AUTHOR</th>
                            <th className="p-2 font-bold text-[#aaa]">VERSION</th>
                            <th className="p-2 font-bold text-[#aaa]">TYPE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {prompts.map(prompt => (
                            <tr
                                key={prompt.id}
                                onClick={() => setSelectedPrompt(prompt)}
                                className="border-b border-[#222] hover:bg-[var(--primary-color)]/10 cursor-pointer transition-colors"
                            >
                                <td className="p-2">
                                    <span className={`font-bold ${prompt.score >= 80 ? 'text-green-500' : prompt.score >= 50 ? 'text-yellow-500' : 'text-red-500'}`}>
                                        {prompt.score}
                                    </span>
                                    {prompt.score > 100 && (
                                        <span className="ml-2 text-[10px] bg-[var(--primary-color)] text-black px-1 rounded">HOT</span>
                                    )}
                                </td>
                                <td className="p-2 text-white">
                                    {prompt.name}
                                    {searchContext && prompt.score > 100 && (
                                        <span className="block text-[10px] text-[var(--primary-color)]">MATCHING CONTEXT</span>
                                    )}
                                </td>
                                <td className="p-2 text-[#888]">{prompt.category}</td>
                                <td className="p-2 text-[#888]">{prompt.author}</td>
                                <td className="p-2 text-[#666]">{prompt.version}</td>
                                <td className="p-2 text-xs font-mono text-[#555]">{prompt.type}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        )}

        {selectedPrompt && (
            <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-10 backdrop-blur-sm">
                <div className="bg-[#111] border border-[var(--primary-color)] w-full max-w-4xl h-[85vh] flex flex-col shadow-[0_0_40px_rgba(0,255,255,0.15)]">
                    <header className="flex justify-between items-center p-4 border-b border-[#333] bg-[#1a1a1a]">
                        <div>
                            <h2 className="text-xl font-bold text-white">{selectedPrompt.name}</h2>
                            <span className="text-xs text-[var(--primary-color)]">{selectedPrompt.path}</span>
                        </div>
                        <button
                            onClick={() => setSelectedPrompt(null)}
                            className="text-[#666] hover:text-white font-bold text-xl px-2"
                        >
                            ×
                        </button>
                    </header>
                    <div className="flex-grow overflow-auto p-6 bg-[#0a0a0a]">
                         <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
                             <div className="p-3 border border-[#333] bg-[#111]">
                                 <span className="block text-[#666] text-xs mb-1">ROI SCORE</span>
                                 <span className="text-3xl font-bold text-[var(--primary-color)]">{selectedPrompt.score}</span>
                             </div>
                             <div className="p-3 border border-[#333] bg-[#111]">
                                 <span className="block text-[#666] text-xs mb-1">METADATA</span>
                                 <pre className="text-xs text-[#aaa] whitespace-pre-wrap font-mono">
                                     {JSON.stringify(selectedPrompt.metadata, null, 2)}
                                 </pre>
                             </div>
                         </div>
                         <h3 className="text-[var(--accent-color)] font-bold mb-2 border-b border-[#333] pb-1">CONTENT_PREVIEW</h3>
                         <div className="prose prose-invert prose-sm max-w-none font-mono whitespace-pre-wrap text-[#ccc] p-4 bg-[#111] border border-[#222]">
                            {/* If markdown, render it, else pre tag */}
                            {selectedPrompt.type === 'MARKDOWN' ? (
                                <ReactMarkdown>{selectedPrompt.content}</ReactMarkdown>
                            ) : (
                                <pre>{selectedPrompt.content}</pre>
                            )}
                         </div>
                    </div>
                </div>
            </div>
        )}
    </div>
  );
};

export default PromptAlpha;
