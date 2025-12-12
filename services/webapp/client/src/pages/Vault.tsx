import React, { useState, useEffect } from 'react';
import { dataManager } from '../utils/DataManager';
import DeepDiveViewer from '../components/DeepDiveViewer';

const Vault: React.FC = () => {
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState<any | null>(null);
  const [reportContent, setReportContent] = useState<any | null>(null);

  useEffect(() => {
    const fetchReports = async () => {
        const manifest = await dataManager.getManifest();
        setReports(manifest.reports || []);
        setLoading(false);
    };
    fetchReports();
  }, []);

  const handleReportClick = async (report: any) => {
      setSelectedReport(report);
      // Simulate fetching content. In real app: const data = await dataManager.fetchData(report.path);
      // For now we assume a mock deep dive structure for demonstration if type is Deep Dive

      if (report.type === 'Deep Dive') {
          setReportContent({
             v23_knowledge_graph: {
                 meta: { target: report.title },
                 equity_analysis: {
                     valuation_engine: { dcf_model: { intrinsic_value: "$145.20", implied_upside: "+15%" }, key_metrics: { pe_ratio: "25x", ev_ebitda: "18x" } }
                 },
                 market_sentiment: { overall_score: 0.85, news_sentiment: "Positive", social_sentiment: "Mixed" },
                 credit_analysis: { snc_rating_model: { overall_borrower_rating: "Pass", risk_metrics: { pd_1y: "0.5%" } } },
                 synthesis: { conviction: { level: "HIGH", rationale: "Strong fundamental growth coupled with favorable macro tailwinds." }, recommendation: "BUY" }
             }
          });
      } else {
          setReportContent(null); // Use generic view
      }
  };

  return (
    <div style={{ height: 'calc(100vh - 100px)', display: 'flex', gap: '20px', padding: '20px' }}>
        {/* List */}
        <div className="glass-panel" style={{ width: '350px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '15px', borderBottom: '1px solid #333' }}>
                <h2 className="mono-font text-cyan-glow">ARCHIVES & REPORTS</h2>
            </div>
            <div style={{ flexGrow: 1, overflowY: 'auto', padding: '10px' }}>
                {loading ? <div style={{padding: '20px'}}>Loading Archives...</div> :
                 reports.map(r => (
                    <div
                        key={r.id}
                        className="cyber-card"
                        style={{ padding: '10px', marginBottom: '10px', cursor: 'pointer', borderLeft: selectedReport?.id === r.id ? '2px solid var(--primary-color)' : '1px solid transparent' }}
                        onClick={() => handleReportClick(r)}
                    >
                        <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>{r.title}</div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '5px' }}>
                            <span className="cyber-badge badge-blue">{r.type}</span>
                            <span style={{ fontSize: '0.7rem', color: '#888' }}>{r.date}</span>
                        </div>
                    </div>
                 ))
                }
            </div>
        </div>

        {/* Viewer */}
        <div className="glass-panel" style={{ flexGrow: 1, padding: '20px', overflowY: 'auto' }}>
            {selectedReport ? (
                <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '10px', borderBottom: '1px solid #333' }}>
                        <h2 style={{ margin: 0 }}>{selectedReport.title}</h2>
                        <button className="cyber-btn">DOWNLOAD JSON</button>
                    </div>

                    {selectedReport.type === 'Deep Dive' && reportContent ? (
                        <DeepDiveViewer data={reportContent.v23_knowledge_graph} />
                    ) : (
                        <div className="mono-font" style={{ color: '#aaa', lineHeight: '1.6' }}>
                            <p>ACCESSING SECURE VAULT RECORD: {selectedReport.id}</p>
                            <p>SOURCE PATH: {selectedReport.path}</p>
                            <hr style={{ borderColor: '#333', margin: '20px 0' }} />
                            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '4px' }}>
                                <p>[CONTENT PREVIEW]</p>
                                <p>This report contains standard analysis data.</p>
                                <pre style={{ color: 'var(--success-color)' }}>
    {`{
      "id": "${selectedReport.id}",
      "type": "${selectedReport.type}",
      "encrypted": false
    }`}
                                </pre>
                            </div>
                        </div>
                    )}
                </div>
            ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#666' }}>
                    SELECT A REPORT TO DECRYPT
                </div>
            )}
        </div>
    </div>
  );
};

export default Vault;
