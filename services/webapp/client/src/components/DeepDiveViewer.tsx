import React, { useState } from 'react';

interface DeepDiveViewerProps {
    data: any;
}

const DeepDiveViewer: React.FC<DeepDiveViewerProps> = ({ data }) => {
    const [activeTab, setActiveTab] = useState('FINANCIALS');

    const tabs = ['FINANCIALS', 'SENTIMENT', 'RISKS', 'STRATEGY'];

    const renderContent = () => {
        switch (activeTab) {
            case 'FINANCIALS':
                return (
                    <div>
                        <h3 className="text-cyan-glow">Deep Fundamental & Valuation</h3>
                        <div className="cyber-card" style={{ padding: '20px', marginBottom: '20px' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                                <div>
                                    <h4>Valuation</h4>
                                    <div className="mono-font">DCF Intrinsic: {data.equity_analysis?.valuation_engine?.dcf_model?.intrinsic_value || 'N/A'}</div>
                                    <div className="mono-font">Implied Upside: {data.equity_analysis?.valuation_engine?.dcf_model?.implied_upside || 'N/A'}</div>
                                </div>
                                <div>
                                    <h4>Key Ratios</h4>
                                    <div className="mono-font">P/E: {data.equity_analysis?.key_metrics?.pe_ratio || 'N/A'}</div>
                                    <div className="mono-font">EV/EBITDA: {data.equity_analysis?.key_metrics?.ev_ebitda || 'N/A'}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            case 'SENTIMENT':
                return (
                    <div>
                        <h3 className="text-cyan-glow">Market & Sentiment Analysis</h3>
                        <div className="cyber-card" style={{ padding: '20px' }}>
                            <div className="mono-font">Overall Score: {data.market_sentiment?.overall_score || 'N/A'}</div>
                            <div className="mono-font">News Sentiment: {data.market_sentiment?.news_sentiment || 'N/A'}</div>
                            <div className="mono-font">Social Sentiment: {data.market_sentiment?.social_sentiment || 'N/A'}</div>
                        </div>
                    </div>
                );
            case 'RISKS':
                return (
                    <div>
                        <h3 className="text-cyan-glow">Risk & Simulation</h3>
                        <div className="cyber-card" style={{ padding: '20px' }}>
                            <div className="mono-font">SNC Rating: {data.credit_analysis?.snc_rating_model?.overall_borrower_rating || 'N/A'}</div>
                            <div className="mono-font">Probability of Default (1Y): {data.credit_analysis?.risk_metrics?.pd_1y || 'N/A'}</div>
                            <h4 style={{marginTop: '10px'}}>Simulation Scenarios</h4>
                            <ul>
                                {data.simulation_engine?.scenarios?.map((s: any, i: number) => (
                                    <li key={i} className="mono-font">{s.name}: {s.impact}</li>
                                )) || <li>No scenarios run.</li>}
                            </ul>
                        </div>
                    </div>
                );
            case 'STRATEGY':
                return (
                    <div>
                        <h3 className="text-cyan-glow">Strategic Synthesis</h3>
                        <div className="cyber-card" style={{ padding: '20px', borderLeft: '4px solid var(--primary-color)' }}>
                            <h4 style={{margin: 0}}>CONVICTION: {data.synthesis?.conviction?.level || 'N/A'}</h4>
                            <p>{data.synthesis?.conviction?.rationale || 'No rationale provided.'}</p>
                            <div style={{ marginTop: '20px' }}>
                                <strong>Recommendation:</strong> {data.synthesis?.recommendation || 'HOLD'}
                            </div>
                        </div>
                    </div>
                );
            default:
                return <div>Select a tab</div>;
        }
    };

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', borderBottom: '1px solid #333', marginBottom: '20px' }}>
                {tabs.map(tab => (
                    <div
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        style={{
                            padding: '10px 20px',
                            cursor: 'pointer',
                            color: activeTab === tab ? 'var(--primary-color)' : '#666',
                            borderBottom: activeTab === tab ? '2px solid var(--primary-color)' : 'none',
                            fontFamily: 'var(--font-mono)'
                        }}
                    >
                        {tab}
                    </div>
                ))}
            </div>
            <div style={{ flexGrow: 1, overflowY: 'auto' }}>
                {renderContent()}
            </div>
        </div>
    );
};

export default DeepDiveViewer;
