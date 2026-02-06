import React, { useState } from 'react';
import { Shield, AlertTriangle, FileText, Activity, Server, Zap, Lock, ChevronRight, BarChart3, Network } from 'lucide-react';

// Types matching the backend
interface AuditLog {
  category: string;
  score: number;
  reasoning: string;
}

interface VerificationFlag {
  type: string;
  severity: string;
  entity: string;
  message: string;
}

interface SensitivityResult {
  scenario_id: string;
  leverage: number;
  coverage: number;
  ebitda: number;
  interest: number;
}

const CreditSentinel: React.FC = () => {
  const [ticker, setTicker] = useState('ZOMB');
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  // State for results
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [flags, setFlags] = useState<VerificationFlag[]>([]);
  const [sensitivity, setSensitivity] = useState<SensitivityResult[]>([]);

  // Mock Data for Input
  const [financials, setFinancials] = useState({
    balance_sheet: {
      total_debt: 400000000,
      cash_equivalents: 5000000,
    },
    income_statement: {
      consolidated_ebitda: 30000000,
      interest_expense: 45000000,
      revenue: 100000000
    },
    quant_analysis: "The company shows strong resilience. Leverage (Gross): 2.00x. EBITDA: 30,000,000.00. We note the Term Loan B is Senior Secured and thus low risk. Parent Company is a strong Operating Company."
  });

  const runAnalysis = async () => {
    setLoading(true);
    try {
      // 1. Run Analyze (Auditor + Symbolic)
      const res1 = await fetch('/api/credit_sentinel/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(financials)
      });
      const data1 = await res1.json();
      setAuditLogs(data1.audit_logs || []);
      setFlags(data1.verification_flags || []);

      // 2. Run Sensitivity
      const res2 = await fetch('/api/credit_sentinel/sensitivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(financials)
      });
      const data2 = await res2.json();
      setSensitivity(data2 || []);

      setAnalyzed(true);
    } catch (e) {
      console.error("Analysis Failed", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen bg-black text-gray-300 font-mono flex flex-col overflow-hidden selection:bg-cyan-500/30">

      {/* Header */}
      <div className="h-14 border-b border-cyan-900/30 flex items-center justify-between px-6 bg-slate-950">
        <div className="flex items-center gap-3">
          <Shield className="text-cyan-400" />
          <h1 className="text-xl font-bold tracking-tighter text-white">
            CREDIT<span className="text-cyan-400">SENTINEL</span>
          </h1>
          <span className="text-[10px] bg-cyan-900/30 text-cyan-400 px-2 py-0.5 rounded border border-cyan-900/50">
            SYSTEM 2 VERIFICATION
          </span>
        </div>
        <div className="flex items-center gap-4">
           <div className="flex items-center gap-2">
             <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
             <span className="text-xs text-green-500 font-bold">ONLINE</span>
           </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">

        {/* Left Control Panel */}
        <div className="w-80 bg-slate-950 border-r border-cyan-900/30 p-4 flex flex-col gap-6 overflow-y-auto">
           <div>
             <label className="text-[10px] text-gray-500 font-bold uppercase mb-1 block">Target Entity</label>
             <div className="flex items-center bg-slate-900 border border-slate-800 rounded px-3 py-2">
               <input
                 type="text"
                 value={ticker}
                 onChange={e => setTicker(e.target.value)}
                 className="bg-transparent border-none outline-none text-white font-bold w-full uppercase"
               />
               <Activity size={14} className="text-gray-600" />
             </div>
           </div>

           <div>
             <label className="text-[10px] text-gray-500 font-bold uppercase mb-1 block">Analysis Context (Simulated Agent Output)</label>
             <textarea
               className="w-full h-40 bg-slate-900 border border-slate-800 rounded p-3 text-xs text-gray-300 font-mono focus:border-cyan-500/50 outline-none resize-none"
               value={financials.quant_analysis}
               onChange={e => setFinancials({...financials, quant_analysis: e.target.value})}
             />
           </div>

           <div>
              <div className="flex justify-between text-[10px] text-gray-500 font-bold uppercase mb-1">
                 <span>Financials (USD)</span>
                 <span>Override</span>
              </div>
              <div className="space-y-2">
                 <div className="flex justify-between items-center bg-slate-900/50 p-2 rounded border border-slate-800">
                    <span className="text-xs">EBITDA</span>
                    <input type="number" value={financials.income_statement.consolidated_ebitda} className="bg-transparent text-right w-24 outline-none text-cyan-400 font-bold text-xs" readOnly />
                 </div>
                 <div className="flex justify-between items-center bg-slate-900/50 p-2 rounded border border-slate-800">
                    <span className="text-xs">Total Debt</span>
                    <input type="number" value={financials.balance_sheet.total_debt} className="bg-transparent text-right w-24 outline-none text-red-400 font-bold text-xs" readOnly />
                 </div>
              </div>
           </div>

           <button
             onClick={runAnalysis}
             disabled={loading}
             className="mt-auto w-full bg-cyan-600 hover:bg-cyan-500 text-black font-bold py-3 rounded flex items-center justify-center gap-2 transition-all"
           >
             {loading ? <Zap className="animate-spin" /> : <Shield size={18} />}
             RUN VERIFICATION
           </button>
        </div>

        {/* Main Dashboard */}
        <div className="flex-1 bg-black p-6 overflow-y-auto">

           {!analyzed && !loading && (
             <div className="h-full flex flex-col items-center justify-center text-slate-800">
                <Shield size={64} className="mb-4 opacity-20" />
                <p className="text-lg font-bold">AWAITING INPUT</p>
                <p className="text-sm font-mono">Initiate 4-Layer Verification Cycle</p>
             </div>
           )}

           {loading && (
             <div className="h-full flex flex-col items-center justify-center text-cyan-500">
                <div className="w-16 h-16 border-4 border-cyan-900 border-t-cyan-400 rounded-full animate-spin mb-6"></div>
                <p className="text-sm font-mono animate-pulse">RUNNING ZOMBIE SIMULATION...</p>
                <p className="text-xs text-slate-600 mt-2">Checking Ontology...</p>
             </div>
           )}

           {analyzed && !loading && (
             <div className="grid grid-cols-12 gap-6 max-w-7xl mx-auto">

                {/* 1. Confidence Score */}
                <div className="col-span-12 lg:col-span-4 bg-slate-950 border border-slate-800 rounded-lg p-6 relative overflow-hidden group">
                   <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-cyan-500 to-blue-600"></div>
                   <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                     <Lock size={14} /> Confidence Score
                   </h2>

                   <div className="flex items-end gap-2 mb-2">
                      <span className="text-5xl font-bold text-white">
                        {auditLogs.find(l => l.category === "Jury Consensus")?.score === 1 ? '12' : '92'}%
                      </span>
                      <span className="text-sm text-red-400 mb-2 font-bold flex items-center">
                         <AlertTriangle size={12} className="mr-1" /> CRITICAL RISK
                      </span>
                   </div>
                   <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden">
                      <div className="h-full bg-red-500 w-[12%]"></div>
                   </div>
                   <p className="mt-4 text-xs text-gray-500 leading-relaxed">
                      The System 2 Jury has detected high-probability hallucinations in the agent's debt structure analysis.
                      Confidence is severely penalized.
                   </p>
                </div>

                {/* 2. Symbolic Verification (Ontology) */}
                <div className="col-span-12 lg:col-span-8 bg-slate-950 border border-slate-800 rounded-lg p-6">
                   <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                     <Network size={14} /> Knowledge Graph Verification
                   </h2>

                   <div className="space-y-3">
                      {flags.length === 0 ? (
                        <div className="text-green-500 text-sm flex items-center gap-2">
                           <Shield size={14} /> No Semantic Contradictions Found.
                        </div>
                      ) : (
                        flags.map((flag, idx) => (
                           <div key={idx} className="bg-red-950/20 border border-red-900/50 p-3 rounded flex items-start gap-3">
                              <AlertTriangle className="text-red-500 shrink-0 mt-0.5" size={16} />
                              <div>
                                 <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs font-bold text-red-400 bg-red-950/50 px-1.5 rounded">{flag.severity}</span>
                                    <span className="text-xs font-bold text-gray-300">{flag.entity}</span>
                                 </div>
                                 <p className="text-xs text-red-200/80">{flag.message}</p>
                              </div>
                           </div>
                        ))
                      )}
                   </div>
                </div>

                {/* 3. Audit Trail (LLM-as-a-Judge) */}
                <div className="col-span-12 lg:col-span-6 bg-slate-950 border border-slate-800 rounded-lg p-6">
                   <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                     <FileText size={14} /> Auditor Jury
                   </h2>
                   <div className="space-y-4">
                      {auditLogs.map((log, idx) => (
                         <div key={idx} className="border-b border-slate-900 pb-3 last:border-0 last:pb-0">
                            <div className="flex justify-between items-center mb-1">
                               <span className="text-xs font-bold text-cyan-400">{log.category}</span>
                               <div className="flex gap-0.5">
                                  {[1,2,3,4,5].map(s => (
                                     <div key={s} className={`w-1.5 h-3 rounded-sm ${s <= log.score ? 'bg-cyan-500' : 'bg-slate-800'}`}></div>
                                  ))}
                               </div>
                            </div>
                            <p className="text-xs text-gray-500">{log.reasoning}</p>
                         </div>
                      ))}
                   </div>
                </div>

                {/* 4. Sensitivity Analysis (Red Team) */}
                <div className="col-span-12 lg:col-span-6 bg-slate-950 border border-slate-800 rounded-lg p-6">
                   <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                     <BarChart3 size={14} /> Sensitivity Analysis (Red Team)
                   </h2>

                   <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                         <thead>
                            <tr className="text-[10px] text-gray-500 border-b border-slate-800">
                               <th className="pb-2 pl-2">SCENARIO</th>
                               <th className="pb-2 text-right">EBITDA</th>
                               <th className="pb-2 text-right">LEVERAGE</th>
                               <th className="pb-2 text-right">COVERAGE</th>
                            </tr>
                         </thead>
                         <tbody>
                            {sensitivity.map((s, idx) => (
                               <tr key={idx} className="text-xs border-b border-slate-900/50 hover:bg-slate-900/30 transition-colors">
                                  <td className="py-3 pl-2 font-mono text-gray-300">{s.scenario_id.replace(/_/g, ' ')}</td>
                                  <td className="py-3 text-right font-mono text-gray-400">${(s.ebitda/1000000).toFixed(1)}M</td>
                                  <td className={`py-3 text-right font-bold font-mono ${s.leverage > 4.5 ? 'text-red-400' : 'text-green-400'}`}>
                                    {s.leverage.toFixed(2)}x
                                  </td>
                                  <td className={`py-3 text-right font-bold font-mono ${s.coverage < 1.5 ? 'text-red-400' : 'text-green-400'}`}>
                                    {s.coverage.toFixed(2)}x
                                  </td>
                               </tr>
                            ))}
                         </tbody>
                      </table>
                   </div>
                </div>

             </div>
           )}
        </div>
      </div>
    </div>
  );
};

export default CreditSentinel;
