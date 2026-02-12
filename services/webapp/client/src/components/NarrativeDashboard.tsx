import React, { useState, useEffect } from 'react';
import { BookOpen, TrendingUp, AlertTriangle } from 'lucide-react';

// Protocol: ADAM-V-NEXT - Narrative Intelligence UI
const NarrativeDashboard: React.FC = () => {
    const [narratives, setNarratives] = useState<any[]>([]);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        const fetchNarratives = async () => {
            try {
                const token = localStorage.getItem('token');
                const headers: any = {};
                if (token) headers['Authorization'] = `Bearer ${token}`;

                const res = await fetch('/api/synthesizer/narratives', { headers });
                if (res.ok) {
                    const data = await res.json();
                    setNarratives(data);
                } else {
                    // Fallback handled by API, but just in case
                    console.error("Failed to fetch narratives");
                }
            } catch (e) {
                console.error("Narrative fetch error", e);
            } finally {
                setLoading(false);
            }
        };

        fetchNarratives();
        // Poll every 10 seconds
        const interval = setInterval(fetchNarratives, 10000);
        return () => clearInterval(interval);
    }, []);

    if (loading && narratives.length === 0) {
        return (
            <div className="glass-panel" style={{ padding: '20px', marginTop: '20px', fontFamily: "'JetBrains Mono', monospace", color: '#666' }}>
                INITIALIZING NARRATIVE INTELLIGENCE...
            </div>
        );
    }

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
