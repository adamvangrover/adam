import React, { useEffect, useState } from 'react';
import { dataManager } from '../utils/DataManager';
import { FileText } from 'lucide-react';

const Reports: React.FC = () => {
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReports = async () => {
        const manifest = await dataManager.getManifest();
        setReports(manifest.reports || []);
        setLoading(false);
    };
    fetchReports();
  }, []);

  if (loading) return <div className="text-cyber-cyan animate-pulse">LOADING SECURE VAULT...</div>;

  return (
    <div>
      <h2 className="text-2xl font-bold text-cyber-cyan mb-6 flex items-center gap-2">
        <FileText className="h-6 w-6" />
        INTELLIGENCE ARCHIVES
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {reports.map((report, idx) => (
              <div key={idx} className="glass-panel p-6 rounded border border-cyber-cyan/10 hover:border-cyber-cyan transition-all group cursor-pointer relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-30 transition-opacity">
                    <FileText className="h-24 w-24 text-cyber-cyan" />
                  </div>
                  <h3 className="text-lg font-bold text-cyber-cyan group-hover:text-cyber-neon mb-2 truncate font-mono" title={report.title}>
                      {report.title || 'UNTITLED_ARTIFACT'}
                  </h3>
                  <div className="text-[10px] text-cyber-text/60 font-mono mb-4 bg-cyber-black/50 p-1 rounded inline-block">
                    {report.path}
                  </div>
                  <div className="text-sm text-cyber-text/80 line-clamp-4 font-mono leading-relaxed">
                      {typeof report.content === 'string' ? report.content : "STRUCTURED DATA OBJECT"}
                  </div>
                  <button className="mt-4 text-xs bg-cyber-cyan/10 text-cyber-cyan px-3 py-1 rounded hover:bg-cyber-cyan hover:text-black transition-colors uppercase font-bold tracking-wider">
                    Access Data
                  </button>
              </div>
          ))}
      </div>
    </div>
  );
};

export default Reports;
