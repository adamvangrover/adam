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
                    <span>SYSTEM STATUS: {viewMode === 'euphoria' ? 'DIVERGENT' : 'STRESSED'}</span>
                    <span>PROTOCOL: {viewMode === 'euphoria' ? 'FORTRESS' : 'REALITY_CHECK'}</span>
                    <span>DATE: 2026.03.15</span>
                </div>
            </header>

            <div className="mm-controls">
                <button
                    className={`mm-btn ${viewMode === 'euphoria' ? 'active' : ''}`}
                    onClick={() => setViewMode('euphoria')}
                >
                    VIEW: EQUITY LANDSCAPE
                </button>
                <button
                    className={`mm-btn ${viewMode === 'credit' ? 'active' : ''}`}
                    onClick={() => setViewMode('credit')}
                >
                    VIEW: PHYSICAL TRUTH
                </button>
            </div>

            <div className="mm-grid">
                {viewMode === 'euphoria' ? (
                    <>
                        <div className="mm-card">
                            <div className="mm-label">S&P 500</div>
                            <div className="mm-value warning">6,750.10</div>
                            <div className="mm-change negative">▼ -2.6%</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">NASDAQ 100</div>
                            <div className="mm-value negative">19,450.20</div>
                            <div className="mm-change negative">▼ -4.1%</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">BITCOIN</div>
                            <div className="mm-value positive">$78,200</div>
                            <div className="mm-change positive">▲ +9.3%</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">LOCKHEED (LMT)</div>
                            <div className="mm-value positive">$512.40</div>
                            <div className="mm-change positive">▲ +5.5%</div>
                        </div>
                    </>
                ) : (
                    <>
                         <div className="mm-card">
                            <div className="mm-label">US 10Y YIELD</div>
                            <div className="mm-value negative">4.45%</div>
                            <div className="mm-change negative">▲ +23bps (Breakout)</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">COPPER</div>
                            <div className="mm-value positive">$4.85/lb</div>
                            <div className="mm-change positive">▲ +6.2% (Shortage)</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">VIX</div>
                            <div className="mm-value warning">22.40</div>
                            <div className="mm-change positive">▲ +26.1%</div>
                        </div>
                        <div className="mm-card">
                            <div className="mm-label">OIL (BRENT)</div>
                            <div className="mm-value positive">$98.50</div>
                            <div className="mm-change positive">▲ +12.0%</div>
                        </div>
                    </>
                )}
            </div>

            <section className="mm-artifact-section">
                <div className="mm-artifact">
                    <h3>ARTIFACT: THE GREAT BIFURCATION</h3>
                    <p>
                        The "Digital" vs "Physical" spread is widening. While software multiples compress,
                        energy and defense assets are repricing for a world of scarcity and conflict.
                    </p>
                    <p style={{color: '#888'}}>
                        <em>"The screen says one thing, the grocery store says another."</em>
                    </p>
                </div>

                <div className="mm-artifact">
                    <h3>ARTIFACT: SOVEREIGN CLOUD</h3>
                    <div className="mm-value positive">THEME ACTIVE</div>
                    <p>
                        Nations are building "Sovereign AI" stacks. We are long local infrastructure
                        providers and physical security (cyber-defense + kinetic defense).
                    </p>
                </div>
            </section>

            <div style={{marginTop: '40px', borderTop: '1px solid #333', paddingTop: '20px', fontSize: '0.8em', color: '#555'}}>
                TERMINAL ID: ADAM_V24_0<br/>
                CONNECTION: ENCRYPTED (QUANTUM-RESISTANT)<br/>
                LATENCY: 12ms
            </div>
        </div>
    );
};

export default MarketMayhem;
