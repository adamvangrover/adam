'use client';
import React from 'react';
import { NeuralPacket } from '../../hooks/useNeuralMesh';

interface StreamLogProps {
  packets: NeuralPacket[];
}

const StreamLog: React.FC<StreamLogProps> = ({ packets }) => {
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    // Auto-scroll logic
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [packets]);

  const getPriorityColor = (priority: number) => {
    switch (priority) {
      case 0: return 'text-red-500 font-bold';
      case 1: return 'text-orange-400';
      case 2: return 'text-gray-300';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className="flex flex-col h-full bg-black/90 text-xs font-mono p-2 border-t border-gray-800">
      <div className="flex justify-between border-b border-gray-800 pb-1 mb-1 text-gray-400 uppercase tracking-widest text-[10px]">
        <span>Event Stream</span>
        <span>Count: {packets.length}</span>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-1 scrollbar-thin scrollbar-thumb-gray-800">
        {packets.map((pkt) => (
          <div key={pkt.id} className="flex gap-2 hover:bg-white/5 py-0.5 transition-colors">
            <span className="text-gray-600 shrink-0 w-16">[{new Date(pkt.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}]</span>
            <span className={`w-24 shrink-0 truncate font-semibold ${pkt.source_agent === 'SystemSupervisor' ? 'text-purple-400' : 'text-blue-400'}`}>
              {pkt.source_agent}
            </span>
            <span className={`flex-1 break-all ${getPriorityColor(pkt.priority)}`}>
              {pkt.packet_type === 'thought' ? pkt.payload.content :
               pkt.packet_type === 'market_data' ? `TICK: ${pkt.payload.symbol} $${pkt.payload.price} (${pkt.payload.change_pct}%)` :
               pkt.packet_type === 'risk_alert' ? `ALERT: ${pkt.payload.alert} (${pkt.payload.severity})` :
               JSON.stringify(pkt.payload)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StreamLog;
