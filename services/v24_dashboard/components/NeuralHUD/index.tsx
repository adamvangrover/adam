'use client';
import React, { useState } from 'react';
import { useNeuralMesh, NeuralPacket } from '../../hooks/useNeuralMesh';

// Components
import StreamLog from './StreamLog';
import MarketTicker from './MarketTicker';
import StatusPanel from './StatusPanel';

const NeuralHUD: React.FC = () => {
  const { status, feed, marketData, systemMetrics } = useNeuralMesh();
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 bg-cyan-600/80 hover:bg-cyan-500/90 text-white p-2 rounded-full shadow-lg z-50 backdrop-blur"
      >
        <span className="sr-only">Open HUD</span>
        📊
      </button>
    );
  }

  return (
    <div className="fixed top-0 left-0 w-full h-full pointer-events-none z-50 flex flex-col justify-between">
      {/* Top Status Bar */}
      <div className="bg-black/80 backdrop-blur-md border-b border-gray-800 p-2 flex justify-between items-center pointer-events-auto h-12">
        <div className="flex items-center gap-4">
          <span className="text-cyan-400 font-bold tracking-widest text-xs">NEURAL MESH v2.0</span>
          <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded ${status === 'Connected' ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}`}>
            {status}
          </span>
          <StatusPanel metrics={systemMetrics} />
        </div>
        <button onClick={() => setIsOpen(false)} className="text-gray-500 hover:text-white px-2">✕</button>
      </div>

      {/* Middle Content (Pass-through clicks) */}
      <div className="flex-1 pointer-events-none"></div>

      {/* Bottom Log & Ticker */}
      <div className="bg-black/90 backdrop-blur-lg border-t border-gray-800 pointer-events-auto flex flex-col max-h-[300px] transition-all duration-300">
        <MarketTicker data={marketData} />
        <div className="h-48 overflow-hidden">
          <StreamLog packets={feed} />
        </div>
      </div>
    </div>
  );
};

export default NeuralHUD;
