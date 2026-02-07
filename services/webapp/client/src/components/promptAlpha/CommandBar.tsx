import React, { useState, useEffect, useRef } from 'react';
import { Terminal, ArrowRight, Command } from 'lucide-react';

interface CommandBarProps {
  onCommand: (cmd: string) => void;
  currentView: string;
}

const COMMANDS = ['NEWS', 'SYS', 'PRMT', 'VAULT', 'MARKET', 'SWARM', 'HELP'];

export const CommandBar: React.FC<CommandBarProps> = ({ onCommand, currentView }) => {
  const [input, setInput] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Auto-focus input on mount
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    // Simple autocomplete logic
    if (input) {
      const match = COMMANDS.find(c => c.startsWith(input.toUpperCase()));
      if (match) {
        setSuggestion(match);
      } else {
        setSuggestion('');
      }
    } else {
      setSuggestion('');
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    processCommand(input);
  };

  const processCommand = (cmd: string) => {
    const cleanCmd = cmd.toUpperCase().trim();
    if (COMMANDS.includes(cleanCmd)) {
      onCommand(cleanCmd);
      setInput('');
    } else {
        // Handle unknown command visual feedback?
        console.warn("Unknown command");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Tab' && suggestion) {
      e.preventDefault();
      setInput(suggestion);
    }
  };

  return (
    <div className="bg-[#002b36] border-b border-cyan-900/50 p-2 flex items-center gap-4">
      <div className="flex items-center gap-2 px-3 py-1 bg-cyan-950/50 rounded border border-cyan-800 text-cyan-400 text-xs font-mono">
        <Command size={14} />
        <span className="font-bold">{currentView}</span>
      </div>

      <form onSubmit={handleSubmit} className="flex-1 relative flex items-center">
        <Terminal size={16} className="text-cyan-600 absolute left-2" />
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-full bg-[#001f27] border border-cyan-900 text-cyan-300 font-mono text-sm py-1.5 pl-8 pr-20 focus:outline-none focus:border-cyan-500 uppercase placeholder-cyan-900"
          placeholder="ENTER FUNCTION..."
        />
        {suggestion && input && suggestion !== input.toUpperCase() && (
            <div className="absolute left-8 text-cyan-900 pointer-events-none font-mono text-sm">
                <span className="opacity-0">{input}</span>
                <span>{suggestion.substring(input.length)}</span>
            </div>
        )}
        <button
            type="submit"
            className="absolute right-1 px-3 py-0.5 bg-cyan-700 hover:bg-cyan-600 text-white text-[10px] font-bold rounded flex items-center gap-1 font-mono transition-colors"
        >
            GO <ArrowRight size={10} />
        </button>
      </form>

      <div className="hidden md:flex gap-2 text-[10px] font-mono text-cyan-700">
        {COMMANDS.map(cmd => (
            <span key={cmd} className="cursor-pointer hover:text-cyan-400" onClick={() => onCommand(cmd)}>
                {cmd}
            </span>
        ))}
      </div>
    </div>
  );
};
