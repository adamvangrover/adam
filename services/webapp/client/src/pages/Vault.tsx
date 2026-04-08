import React, { useEffect, useState } from 'react';
import { dataManager } from '../utils/DataManager';
import { useNavigate } from 'react-router-dom';

const Vault: React.FC = () => {
  const [items, setItems] = useState<any[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    dataManager.getManifest().then(data => setItems(data.reports));
  }, []);

  const openReport = (id: string, type: string) => {
      // If it's a Deep Dive or JSON, open in DeepDive viewer
      if (type === 'JSON' || type === 'SNC' || id.includes('Deep_Dive')) {
          navigate(`/deep-dive/${id}`);
      } else {
          // For now, just show alert or fallback
          alert(`Opening artifact: ${id} (${type}) - Content Viewer implementation pending for this type.`);
      }
  };

  return (
    <div className="animate-fade-in pb-10">
      <header className="mb-8 border-b border-cyan-900/30 pb-4">
        <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tight mb-2">{'/// SECURE DATA VAULT'}</h1>
        <div className="flex gap-4 text-xs font-mono text-slate-400">
            <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">SYSTEM: ARCHIVE</span>
            <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">ENCRYPTION: AES-256</span>
            <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">RECORDS: {items.length}</span>
        </div>
      </header>

      {items.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 text-slate-500">
              <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mb-4"></div>
              <p className="font-mono text-sm tracking-widest uppercase animate-pulse">Decrypting Manifest...</p>
          </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {items.map(item => (
          <div
            key={item.id}
            className="group bg-[#0f172a] p-6 rounded-2xl border border-slate-700/50 hover:border-cyan-500/50 shadow-lg hover:shadow-cyan-900/20 transition-all duration-300 cursor-pointer relative overflow-hidden"
            onClick={() => openReport(item.id, item.type)}
          >
            <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-2xl group-hover:bg-cyan-500/10 transition-colors"></div>

            <div className="flex justify-between items-start mb-4 relative z-10">
                <div className="font-mono text-[10px] font-bold text-amber-400 bg-amber-950/30 border border-amber-900/50 px-2 py-1 rounded">
                    {item.type.toUpperCase()}
                </div>
                <div className="font-mono text-xs text-slate-500 bg-slate-900 px-2 py-1 rounded border border-slate-800">
                    {item.id}
                </div>
            </div>

            <h3 className="text-lg font-bold text-slate-200 mb-6 group-hover:text-cyan-300 transition-colors relative z-10">{item.title}</h3>

            <div className="flex justify-between items-center text-xs text-slate-400 border-t border-slate-800 pt-4 relative z-10">
              <span className="font-mono">{item.date}</span>
              <span className="font-bold text-cyan-500 group-hover:translate-x-1 transition-transform flex items-center gap-1">
                  ACCESS <span className="text-lg leading-none">&rarr;</span>
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Vault;
