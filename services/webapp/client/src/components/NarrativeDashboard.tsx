import React, { useState, useEffect } from 'react';
import { BookOpen, TrendingUp, AlertTriangle } from 'lucide-react';

// Protocol: ADAM-V-NEXT - Narrative Intelligence UI
const NarrativeDashboard: React.FC = () => {
    const [narratives, setNarratives] = useState<any[]>([]);

    useEffect(() => {
        // Mock data loading - in real world this comes from /api/synthesizer/narratives
        const mockNarratives = [
            { theme: "AI BUBBLE", sentiment: "BULLISH", volume: 85, headline: "AI Chip Demand Outstrips Supply" },
            { theme: "ENERGY CRISIS", sentiment: "BEARISH", volume: 62, headline: "Oil Spikes on Geopolitical Tension" },
            { theme: "FED PIVOT", sentiment: "NEUTRAL", volume: 45, headline: "Minutes Show Consensus on Pause" }
        ];

        setNarratives(mockNarratives);
    }, []);

    return (
        <div className="glass-panel" style={{ padding: '20px', marginTop: '20px', fontFamily: "'JetBrains Mono', monospace" }}>
            <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px', color: '#e0e0e0' }}>
                <BookOpen size={20} color="#ff00ff" /> Narrative Intelligence Stream
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '15px', marginTop: '20px' }}>
                {narratives.map((n, i) => (
                    <div key={i} style={{
                        background: 'rgba(255, 255, 255, 0.05)',
                        borderLeft: `3px solid ${n.sentiment === 'BULLISH' ? '#0f0' : n.sentiment === 'BEARISH' ? '#f00' : '#ffa500'}`,
                        padding: '15px'
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                            <span style={{ fontWeight: 'bold', color: '#fff' }}>{n.theme}</span>
                            <span style={{ fontSize: '0.8rem', color: '#888' }}>VOL: {n.volume}</span>
                        </div>
                        <div style={{ fontSize: '0.9rem', color: '#ccc', marginBottom: '10px' }}>
                            {n.headline}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8rem' }}>
                            {n.sentiment === 'BULLISH' && <TrendingUp size={14} color="#0f0" />}
                            {n.sentiment === 'BEARISH' && <AlertTriangle size={14} color="#f00" />}
                            <span style={{ color: n.sentiment === 'BULLISH' ? '#0f0' : n.sentiment === 'BEARISH' ? '#f00' : '#ffa500' }}>
                                {n.sentiment}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default NarrativeDashboard;
