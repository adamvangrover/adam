import React, { useState, useEffect, memo } from 'react';

// --- STYLES (Extracted from component to reduce GC pressure) ---
const containerStyle = {
    width: '100%',
    maxWidth: '1200px',
    margin: '20px auto',
    padding: '20px',
    fontFamily: "'Courier New', monospace",
    fontSize: '0.9rem',
    backgroundColor: '#000',
    color: '#0f0',
    border: '1px solid #004400',
    borderRadius: '4px',
    boxShadow: '0 0 20px rgba(0,255,0,0.1)'
};

const headerStyle = {
    fontSize: '1.2rem',
    marginBottom: '1rem',
    borderBottom: '1px solid #004400',
    paddingBottom: '0.5rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
};

const logContainerStyle = {
    maxHeight: '600px',
    overflowY: 'auto',
    paddingRight: '10px'
};

const entryStyle = {
    marginBottom: '1.5rem',
    borderLeft: '2px solid #005500',
    paddingLeft: '1rem',
    position: 'relative'
};

const entryHeaderStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.8rem',
    color: '#008800',
    marginBottom: '0.5rem'
};

const eventStyle = {
    backgroundColor: 'rgba(0,50,0,0.2)',
    padding: '8px',
    borderRadius: '2px',
    marginBottom: '4px',
    border: '1px solid rgba(0,100,0,0.2)'
};

const footerStyle = {
    marginTop: '1rem',
    fontSize: '0.7rem',
    color: '#004400',
    textAlign: 'center',
    borderTop: '1px solid #002200',
    paddingTop: '0.5rem'
};

const pulseStyle = { fontSize: '0.7rem', animation: 'pulse 2s infinite' };
const eventMetaStyle = { display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' };
const eventDetailStyle = { paddingLeft: '8px', borderLeft: '1px solid #003300', fontSize: '0.8rem', opacity: 0.8, whiteSpace: 'pre-wrap' };
const analysisStyle = { marginTop: '4px', fontStyle: 'italic' };

// --- SUB-COMPONENTS ---

// Bolt: Memoized component for individual trace entry.
// This prevents re-rendering of existing log entries when the list is polled
// but the specific trace data hasn't changed.
const AuditLogEntry = memo(({ trace }) => {
    return (
        <div style={entryStyle}>
            <div style={entryHeaderStyle}>
                <span>SESSION: {trace.session_id}</span>
                <span>{new Date(trace.start_time).toLocaleTimeString()}</span>
            </div>

            <div>
                {trace.events.map((event, eIdx) => {
                    let badgeColor = '#004400';
                    let textColor = '#afa';
                    if (event.component === 'RedTeam') { badgeColor = '#440000'; textColor = '#faa'; }
                    if (event.component === 'Auditor') { badgeColor = '#000044'; textColor = '#aaf'; }

                    const badgeStyle = {
                        fontSize: '0.7rem',
                        padding: '2px 4px',
                        borderRadius: '2px',
                        backgroundColor: badgeColor,
                        color: textColor
                    };

                    return (
                        <div key={eIdx} style={eventStyle}>
                            <div style={eventMetaStyle}>
                                <span style={badgeStyle}>
                                    [{event.component}]
                                </span>
                                <span style={{fontWeight: 'bold', color: '#0f0'}}>{event.event_type}</span>
                            </div>

                            <div style={eventDetailStyle}>
                                    {event.payload.risk_score !== undefined && (
                                        <div>Risk Score: <span style={{color: '#ff0'}}>{event.payload.risk_score}</span></div>
                                    )}
                                    {event.payload.overall_score !== undefined && (
                                        <div>Auditor Score: <span style={{color: '#88f'}}>{event.payload.overall_score}/5.0</span></div>
                                    )}
                                    {event.payload.analysis && (
                                        <div style={analysisStyle}>"{event.payload.analysis.substring(0, 100)}..."</div>
                                    )}
                                    {event.payload.verified !== undefined && (
                                        <div style={{color: event.payload.verified ? '#0f0' : '#f00'}}>
                                            VERIFIED: {event.payload.verified.toString().toUpperCase()}
                                        </div>
                                    )}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}, (prevProps, nextProps) => {
    // Custom comparison function for React.memo
    // Returns true if props are equal (do not re-render)

    // Fast path: Reference equality
    if (prevProps.trace === nextProps.trace) return true;

    // Deep check logic for Trace objects.
    // Since API returns new objects on poll, we check if the content is effectively the same.
    // We assume if trace_id and event count are same, the log entry is unchanged.
    return (
        prevProps.trace.trace_id === nextProps.trace.trace_id &&
        prevProps.trace.events.length === nextProps.trace.events.length
    );
});

const AuditLogViewer = () => {
    const [traces, setTraces] = useState([]);
    const [loading, setLoading] = useState(true);

    const fetchTraces = async () => {
        try {
            const response = await fetch('/api/traces');
            if (response.ok) {
                const data = await response.json();
                setTraces(data.traces || []);
            }
        } catch (error) {
            console.error("Failed to fetch traces:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchTraces();
        const interval = setInterval(fetchTraces, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading && traces.length === 0) return <div style={{padding: '20px', color: '#0f0'}}>Loading Neural Audit Logs...</div>;

    return (
        <div style={containerStyle}>
            <h2 style={headerStyle}>
                <span>SYSTEM_AUDIT_LOG::V26.0</span>
                <span style={pulseStyle}>● LIVE</span>
            </h2>

            <div style={logContainerStyle}>
                {traces.map((trace, idx) => (
                    <AuditLogEntry key={trace.trace_id || idx} trace={trace} />
                ))}
            </div>

            <div style={footerStyle}>
                END_OF_STREAM // SECURITY_LEVEL: ALPHA
            </div>
        </div>
    );
};

export default AuditLogViewer;
