import React, { useState } from 'react';
import './MarketMayhem.css';

const MarketMayhem: React.FC = () => {
    const [viewMode, setViewMode] = useState<'euphoria' | 'credit'>('euphoria');

    return (
        <div className="market-mayhem-container">
            <div className="scanline"></div>

            <header className="mm-header">
                <h1 className="mm-title glitch-text">MARKET MAYHEM_</h1>
                <div className="mm-status-bar">
                    <span>SYSTEM STATUS: {viewMode === 'euphoria' ? 'NOMINAL' : 'CRITICAL'}</span>
                    <span>PROTOCOL: {viewMode === 'euphoria' ? 'RECOVERY' : 'REALITY_CHECK'}</span>
                    <span>DATE: 2026.02.06</span>
                </div>
            </header>

            <div className="mm-controls">
                <button
                    className={`mm-btn ${viewMode === 'euphoria' ? 'active' : ''}`}
                    onClick={() => setViewMode('euphoria')}
                >
                    VIEW: EQUITY EUPHORIA
                </button>
                <button
                    className={`mm-btn ${viewMode === 'credit' ? 'active' : ''}`}
                    onClick={() => setViewMode('credit')}
                >
                    VIEW: CREDIT TRUTH
                </button>
            </div>

            <div className="mm-grid">
                {viewMode === 'euphoria' ? (
                    <>
                        <div className="mm-card">
                            <div className="mm-label">S&P 500</div>
                            <div className="mm-value positive">6,932.30</div>
                            <div className="mm-change positive">▲ +1.97%</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">DOW JONES</div>
                            <div className="mm-value positive">50,115.42</div>
                            <div className="mm-change positive">▲ +0.92%</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">BITCOIN</div>
                            <div className="mm-value positive">$71,510</div>
                            <div className="mm-change positive">▲ +13.0%</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">NVIDIA</div>
                            <div className="mm-value positive">$145.20</div>
                            <div className="mm-change positive">▲ +8.1%</div>
                        </div>
                    </>
                ) : (
                    <>
                         <div className="mm-card">
                            <div className="mm-label">US 10Y YIELD</div>
                            <div className="mm-value negative">4.22%</div>
                            <div className="mm-change negative">▲ +12bps (Retraced)</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">HYG SPREADS</div>
                            <div className="mm-value warning">+335 bps</div>
                            <div className="mm-change warning">▲ +0.35% (Widening)</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">VIX</div>
                            <div className="mm-value warning">17.76</div>
                            <div className="mm-change positive">▼ -18.4% (Short Squeeze)</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">STELLANTIS</div>
                            <div className="mm-value negative">$12.40</div>
                            <div className="mm-change negative">▼ -25.0% (Crash)</div>
                        </div>
                    </>
                )}
            </div>

            <section className="mm-artifact-section">
                <div className="mm-artifact">
                    <h3>ARTIFACT: THE GHOST IN THE MACHINE</h3>
                    <p>
                        The simulation is lagging. While the S&P 500 reclaims territory, the bond market refuses to participate.
                        A 10-Year Yield at 4.22% is incompatible with infinite PE expansion.
                    </p>
                    <p style={{color: '#888'}}>
                        <em>"We moved to a slightly more expensive part of the forest."</em>
                    </p>
                </div>

                <div className="mm-artifact">
                    <h3>ARTIFACT: MICROSTRATEGY (MSTR)</h3>
                    <div className="mm-value positive">+26.1%</div>
                    <p>
                        The ultimate leverage proxy. A quarter-percent surge in a single session.
                        This is no longer equity; it is a high-frequency volatility derivative.
                    </p>
                </div>
            </section>

            <div style={{marginTop: '40px', borderTop: '1px solid #333', paddingTop: '20px', fontSize: '0.8em', color: '#555'}}>
                TERMINAL ID: SYSTEM_NOMINAL_RECOVERY<br/>
                CONNECTION: SECURE (TLS 1.3)<br/>
                LATENCY: 42ms
            </div>
        </div>
    );
};

export default MarketMayhem;
