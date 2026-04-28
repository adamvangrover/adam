// Verified for Adam v25.5
// Verified by Jules
// Protocol Verified: ADAM-V-NEXT (Updated)
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Activity, Radio, ShieldAlert, Cpu, TrendingUp, Users, Loader2 } from 'lucide-react';
import AgentIntercom from '../components/AgentIntercom';
import NarrativeDashboard from '../components/NarrativeDashboard';
import MarketSentiment from '../MarketSentiment';
import RiskAssessment from '../RiskAssessment';
import FundamentalAnalysis from '../FundamentalAnalysis';

interface MarketPulse {
    indices: Record<string, any>;
    sectors: Record<string, any>;
    timestamp: number;
}

// Bolt ⚡: Memoize SectorSentimentCell to prevent O(N) re-renders during frequent pulse updates
const SectorSentimentCell = React.memo(({ k, v }: { k: string, v: any }) => (
    <div style={{ marginBottom: '15px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.9rem' }}>
            <span>{k}</span>
            <span style={{ color: v.sentiment > 0 ? '#0f0' : '#f00' }}>{v.trend.toUpperCase()}</span>
        </div>
        {/* Bar Chart */}
        <div style={{ height: '6px', background: '#333', width: '100%', position: 'relative' }}>
            <div style={{
                position: 'absolute', top: 0, bottom: 0,
                left: '50%',
                width: `${Math.abs(v.sentiment * 50)}%`,
                background: v.sentiment > 0 ? '#0f0' : '#f00',
                transform: v.sentiment < 0 ? 'translateX(-100%)' : 'none'
            }}></div>
        </div>
    </div>
));
SectorSentimentCell.displayName = 'SectorSentimentCell';

// Protocol: ADAM-V-NEXT
// Verified by Jules
const Synthesizer: React.FC = () => {
    const [score, setScore] = useState<number>(0);
    const [rationale, setRationale] = useState<string>('Initializing neural consensus...');
    const [pulse, setPulse] = useState<MarketPulse | null>(null);
    const [lastUpdate, setLastUpdate] = useState<string>('');
    const [conviction, setConviction] = useState<Record<string, number>>({});
    const [forecastData, setForecastData] = useState<any>(null);
    const [dataSource, setDataSource] = useState<string>('LIVE'); // 'LIVE' or 'SCENARIO_2008'

    // Bolt ⚡: Track last data to prevent unnecessary re-renders
    const lastDataRef = useRef<string>("");

    // Bolt ⚡: Memoize Forecast Chart Points to avoid expensive recalculation on every render
    const forecastChartPoints = useMemo(() => {
        if (!forecastData || !forecastData.forecast) return null;

        const allVals = [...forecastData.forecast.upper_95, ...forecastData.forecast.lower_95];
        const min = Math.min(...allVals) * 0.99;
        const max = Math.max(...allVals) * 1.01;
        const range = max - min;
        const w = 1000 / forecastData.forecast.dates.length;

        const upper = forecastData.forecast.upper_95.map((v:number, i:number) =>
            `${i*w},${300 - ((v - min)/range)*300}`
        ).join(' ');

        const lower = forecastData.forecast.lower_95.map((v:number, i:number) =>
            `${i*w},${300 - ((v - min)/range)*300}`
        ).reverse().join(' ');

        const mean = forecastData.forecast.mean.map((v:number, i:number) =>
            `${i*w},${300 - ((v - min)/range)*300}`
        ).join(' ');

        return {
            polyPoints: `${upper} ${lower}`,
            linePoints: mean
        };
    }, [forecastData]);

    // Initial Data Fetch (One-time)
    useEffect(() => {
        const fetchInitial = async () => {
            const token = localStorage.getItem('token');
            const headers: any = {};
            if (token) headers['Authorization'] = `Bearer ${token}`;

            // 2. Conviction (Initial)
            try {
                const res = await fetch('/api/synthesizer/conviction', { headers });
                if (res.ok) {
                    const data = await res.json();
                    setConviction(data.scores);
                }
            } catch(e) {}

            // 3. Forecast (Initial - SPX)
            try {
                const res = await fetch('/api/synthesizer/forecast/SPX', { headers });
                if (res.ok) {
                    const data = await res.json();
                    setForecastData(data);
                }
            } catch(e) {}
        };
        fetchInitial();
    }, []);

    // Polling Effect for Confidence Score
    useEffect(() => {
        let isMounted = true;
        let timeoutId: ReturnType<typeof setTimeout>;

        const fetchConfidence = async () => {
            const token = localStorage.getItem('token');
            const headers: any = {};
            if (token) headers['Authorization'] = `Bearer ${token}`;

            try {
                // Switch endpoint based on source
                const endpoint = dataSource === 'SCENARIO_2008'
                    ? '/api/synthesizer/scenario?id=2008_CRASH'
                    : '/api/synthesizer/confidence';

                const res = await fetch(endpoint, { headers });
                if (isMounted && res.ok) {
                    const data = await res.json();

                    // Bolt ⚡: Check if critical data changed before updating state
                    // This prevents re-renders on every poll (ignoring the changing API timestamp)
                    const compareObj = {
                        score: data.score,
                        pulse: data.pulse,
                        rationale: data.consensus?.rationale
                    };
                    const compareStr = JSON.stringify(compareObj);

                    if (compareStr !== lastDataRef.current) {
                        lastDataRef.current = compareStr;
                        setScore(data.score);
                        setPulse(data.pulse);
                        if (data.consensus && data.consensus.rationale) {
                            setRationale(data.consensus.rationale);
                        }
                        setLastUpdate(new Date().toLocaleTimeString());
                    }
                }
            } catch(e) {
                // Silent catch for polling
            } finally {
                if (isMounted) {
                    // Bolt ⚡: Use recursive setTimeout instead of setInterval to prevent request pile-up
                    timeoutId = setTimeout(fetchConfidence, 1000);
                }
            }
        };

        fetchConfidence();

        return () => {
            isMounted = false;
            if (timeoutId) clearTimeout(timeoutId);
        };
    }, [dataSource]); // Re-run when dataSource changes to switch endpoints

    // Helper for color coding score
    const getScoreColor = (s: number) => {
        if (s > 80) return '#0f0'; // High Confidence
        if (s > 50) return '#ffa500'; // Caution
        return '#f00'; // Danger
    };

    return (
        <div style={{ padding: '20px', fontFamily: "'JetBrains Mono', monospace", color: '#e0e0e0', maxWidth: '1600px', margin: '0 auto' }}>

            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', borderBottom: '2px solid #333', paddingBottom: '20px' }}>
                <div>
                    <h1 style={{ margin: 0, fontSize: '2.5rem', letterSpacing: '4px', textTransform: 'uppercase' }}>
                        Synthesizer <span style={{ color: 'var(--primary-color)' }}>{'//'}</span> Mission Control
                    </h1>
                    <div style={{ color: '#888', marginTop: '5px' }}>SYSTEM AGGREGATION NODE :: ACTIVE</div>
                </div>
                <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', gap: '5px', alignItems: 'flex-end' }}>
                    <select
                        value={dataSource}
                        onChange={(e) => setDataSource(e.target.value)}
                        style={{ background: '#333', color: '#fff', border: '1px solid #555', padding: '5px', fontFamily: 'inherit' }}
                    >
                        <option value="LIVE">LIVE STREAM</option>
                        <option value="SCENARIO_2008">SCENARIO: 2008 CRASH</option>
                    </select>
                    <div style={{ fontSize: '0.8rem', color: '#666' }}>LAST SYNC</div>
                    <div style={{ color: '#0ff', fontSize: '1.2rem' }}>{lastUpdate}</div>
                </div>
            </div>

            {/* Main Dashboard Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>

                {/* 1. The Core Confidence Gauge */}
                <div
                    className="glass-panel"
                    role="meter"
                    aria-label="Global Confidence Score"
                    aria-valuenow={score}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuetext={`${score.toFixed(1)}% - ${score > 80 ? "Bullish Conviction" : score > 50 ? "Cautious / Hedged" : "Risk Off / Defensive"}`}
                    style={{ gridColumn: 'span 2', padding: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'radial-gradient(circle at center, rgba(20,40,50,0.4) 0%, rgba(0,0,0,0.8) 100%)' }}
                >
                    <div style={{ textAlign: 'center', position: 'relative' }}>
                        <div style={{ fontSize: '1rem', color: '#888', letterSpacing: '2px', marginBottom: '10px' }} aria-hidden="true">GLOBAL CONFIDENCE SCORE</div>
                        <div style={{
                            fontSize: '6rem', fontWeight: 'bold',
                            color: getScoreColor(score),
                            textShadow: `0 0 20px ${getScoreColor(score)}`
                        }} aria-hidden="true">
                            {score.toFixed(1)}%
                        </div>

                        {/* Status Label */}
                        <div style={{
                            marginTop: '10px', padding: '5px 15px',
                            background: getScoreColor(score), color: '#000',
                            display: 'inline-block', fontWeight: 'bold', borderRadius: '4px'
                        }}>
                            {score > 80 ? "BULLISH CONVICTION" : score > 50 ? "CAUTIOUS / HEDGED" : "RISK OFF / DEFENSIVE"}
                        </div>

                        {/* Rationale Stream */}
                        <div style={{ marginTop: '20px', fontSize: '0.8rem', color: '#aaa', fontStyle: 'italic', maxWidth: '600px', margin: '20px auto 0' }}>
                            <span style={{ color: 'var(--primary-color)' }}>&gt; MISSION BRIEF:</span> {rationale}
                        </div>
                    </div>
                    {/* Optional System 2 Critique Area (if available) */}
                    {pulse && (pulse as any)?.critique && (
                        <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #ff3333', background: 'rgba(255, 0, 0, 0.05)', borderRadius: '4px' }}>
                            <div style={{ fontSize: '0.8rem', color: '#ff3333', fontWeight: 'bold', marginBottom: '5px' }}>SYSTEM 2 CRITIQUE:</div>
                            <div style={{ fontSize: '0.9rem', color: '#e0e0e0' }}>{(pulse as any).critique}</div>
                        </div>
                    )}
                </div>

                {/* 2. Signal Inputs */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                    <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Activity size={20} color="#0ff" /> Live Market Pulse
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginTop: '20px' }}>
                        {pulse?.indices && Object.entries(pulse.indices).map(([k, v]: [string, any]) => (
                            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '4px' }}>
                                <span style={{ fontWeight: 'bold' }}>{k}</span>
                                <div style={{ textAlign: 'right' }}>
                                    <div style={{ fontSize: '1.1rem' }}>{v.price.toFixed(2)}</div>
                                    <div style={{ fontSize: '0.8rem', color: v.change_percent >= 0 ? '#0f0' : '#f00' }}>
                                        {v.change_percent >= 0 ? '+' : ''}{v.change_percent.toFixed(2)}%
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>


                {/* Embedded Signals */}
                <div className="glass-panel" style={{ padding: '20px', gridColumn: 'span 2' }}>
                    <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Activity size={20} color="#f0f" /> Real-time Signals Aggregation
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', marginTop: '20px' }}>
                        <div style={{ background: 'rgba(0,0,0,0.4)', padding: '10px', borderRadius: '8px', border: '1px solid #333' }}>
                            <MarketSentiment />
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.4)', padding: '10px', borderRadius: '8px', border: '1px solid #333' }}>
                            <RiskAssessment data={{ risk_score: 45, breakdown: { market_risk: 'Medium', credit_risk: 'Low', liquidity_risk: 'Medium' }, summary: 'Moderate risk environment. Hold steady.' }} />
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.4)', padding: '10px', borderRadius: '8px', border: '1px solid #333' }}>
                            <FundamentalAnalysis data={{ dcf_valuation: { estimated_value: 145.2, narrative_summary: 'Slightly undervalued based on projected cash flows.' }, financial_ratios: { pe_ratio: 15.4, debt_to_equity: 0.8 } }} />
                        </div>
                    </div>
                </div>

                {/* 3. Sector Breakdown */}
                 <div className="glass-panel" style={{ padding: '20px' }}>
                    <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Radio size={20} color="#f0f" /> Sector Sentiment
                    </h3>
                    <div style={{ marginTop: '20px' }}>
                        {pulse?.sectors && Object.entries(pulse.sectors).map(([k, v]: [string, any]) => (
                            <SectorSentimentCell key={k} k={k} v={v} />
                        ))}
                    </div>
                </div>

                {/* 4. Active Agents (Mocked Status) */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                     <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Cpu size={20} color="#ff0" /> Agent Swarm Status
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '20px' }}>
                        {['RiskOfficer', 'TechAnalyst', 'MacroSentinel', 'BlindspotScanner', 'NewsReader', 'PortfolioManager'].map(agent => (
                            <div key={agent} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.8rem' }}>
                                <div style={{ width: '8px', height: '8px', background: '#0f0', borderRadius: '50%', boxShadow: '0 0 5px #0f0' }}></div>
                                {agent}
                            </div>
                        ))}
                    </div>
                </div>

                 {/* 5. Governance / Alerts */}
                 <div className="glass-panel" style={{ padding: '20px' }}>
                     <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <ShieldAlert size={20} color="#ff3333" /> Governance Alerts
                    </h3>
                    <div style={{ fontSize: '0.8rem', color: '#888', fontStyle: 'italic', marginTop: '20px' }}>
                        No active breaches detected. Policy strictness level: HIGH.
                    </div>
                </div>

                {/* 6. Agent Conviction Heatmap */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                    <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Users size={20} color="#00ff00" /> Neural Conviction
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: '10px', marginTop: '20px' }}>
                        {Object.entries(conviction).map(([agent, val]) => (
                            <div key={agent} style={{ textAlign: 'center', padding: '10px', background: `rgba(0, 255, 0, ${val * 0.3})`, border: `1px solid rgba(0,255,0,${val})`, borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                                {/* Animated scanning bar */}
                                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: 'rgba(255,255,255,0.5)', animation: 'scan 3s linear infinite' }}></div>
                                <div style={{ fontSize: '0.7rem', fontWeight: 'bold' }}>{agent}</div>
                                <div style={{ fontSize: '1.2rem', color: '#fff' }}>{(val * 100).toFixed(0)}%</div>
                                {val > 0.8 && <div style={{ fontSize: '0.55rem', color: '#000', background: '#0f0', padding: '2px', marginTop: '5px', borderRadius: '2px' }}>HIGH CONVICTION</div>}
                                {val < 0.3 && <div style={{ fontSize: '0.55rem', color: '#000', background: '#f00', padding: '2px', marginTop: '5px', borderRadius: '2px' }}>LOW CONVICTION</div>}
                            </div>
                        ))}
                    </div>
                    {/* Add keyframes globally for scanning bar */}
                    <style>{`
                        @keyframes scan {
                            0% { transform: translateY(-100%); opacity: 0; }
                            50% { opacity: 1; }
                            100% { transform: translateY(1000%); opacity: 0; }
                        }
                    `}</style>
                </div>

                {/* 7. Forecast Chart (SVG) */}
                <div className="glass-panel" style={{ gridColumn: 'span 2', padding: '20px' }}>
                     <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <TrendingUp size={20} color="#00f3ff" /> Predictive Horizon (SPX 30-Day)
                    </h3>
                    {forecastData && forecastChartPoints ? (
                        <div style={{ marginTop: '20px', height: '300px', position: 'relative', overflow: 'hidden', borderRadius: '8px' }}>
                             {/* Grid Lines */}
                             <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)', backgroundSize: '20px 20px', pointerEvents: 'none' }}></div>

                             {/* Simple SVG Chart */}
                             <svg
                                width="100%"
                                height="100%"
                                viewBox="0 0 1000 300"
                                preserveAspectRatio="none"
                                role="img"
                                aria-labelledby="forecastTitle forecastDesc"
                             >
                                <title id="forecastTitle">Probabilistic Price Forecast Chart</title>
                                <desc id="forecastDesc">
                                    A chart showing the predicted price movement for SPX over the next 30 days.
                                    The white line represents the mean forecast, and the shaded blue area represents the 95% confidence interval.
                                </desc>
                                {/* Defs for gradient */}
                                <defs>
                                    <linearGradient id="fanGradient" x1="0" x2="0" y1="0" y2="1">
                                        <stop offset="0%" stopColor="#00f3ff" stopOpacity="0.4"/>
                                        <stop offset="100%" stopColor="#00f3ff" stopOpacity="0.0"/>
                                    </linearGradient>
                                </defs>

                                {/* 95% Confidence Band (Poly) */}
                                <polygon
                                    points={forecastChartPoints.polyPoints}
                                    fill="url(#fanGradient)"
                                />

                                 {/* Mean Line */}
                                 <polyline
                                     points={forecastChartPoints.linePoints}
                                     fill="none" stroke="#fff" strokeWidth="2"
                                     style={{ filter: 'drop-shadow(0 0 5px rgba(255,255,255,0.8))' }}
                                 />

                                 {/* Current Price Marker */}
                                 <circle cx="0" cy={forecastChartPoints.linePoints.split(' ')[0].split(',')[1]} r="4" fill="#0ff" style={{ filter: 'drop-shadow(0 0 5px #0ff)' }} />
                             </svg>

                             <div style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(0,0,0,0.6)', padding: '5px 10px', borderRadius: '4px', fontSize: '0.7rem', color: '#0ff', border: '1px solid #0ff' }}>
                                 CONFIDENCE BAND: 95% | HORIZON: 30D
                             </div>
                        </div>
                    ) : (
                        <div style={{ padding: '50px 20px', textAlign: 'center', color: '#666', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px' }}>
                            <Loader2 className="animate-spin" size={32} color="#0ff" />
                            <div style={{ letterSpacing: '2px', fontSize: '0.9rem' }}>INITIALIZING PREDICTIVE MATRIX...</div>
                        </div>
                    )}
                </div>

            </div>

            {/* Protocol: ADAM-V-NEXT - Narrative Intelligence Layer */}
            <NarrativeDashboard />

            <AgentIntercom />
        </div>
    );
};

export default Synthesizer;
// Protocol Verified: ADAM-V-NEXT
