import React, { useEffect, useState } from 'react';

// Interfaces for our data
interface EvalMetrics {
  spread_bps: number;
  inventory: number;
}

interface EvalData {
  iteration: number;
  metrics: EvalMetrics;
  status: string;
  feedback: string;
}

interface WebSocketPayload {
  timestamp: string;
  event_type: string;
  data: EvalData;
}

const EvalDashboard: React.FC = () => {
  const [logs, setLogs] = useState<WebSocketPayload[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<EvalMetrics | null>(null);
  const [wsStatus, setWsStatus] = useState<string>('Connecting...');

  const MAX_INVENTORY_THRESHOLD = 1000; // Example threshold

  useEffect(() => {
    // 1. Fetch Historical State via REST API on boot
    const fetchHistory = async () => {
      try {
        const response = await fetch('/api/v1/eval-logs');
        if (response.ok) {
          const data = await response.json();
          // Assuming data.logs is an array of payloads
          setLogs(data.logs || []);
          if (data.logs && data.logs.length > 0) {
              setCurrentMetrics(data.logs[data.logs.length - 1].data.metrics);
          }
        }
      } catch (error) {
        console.error('Failed to fetch historical logs', error);
      }
    };

    fetchHistory();

    // 2. Connect to WebSocket for Real-Time State
    // Using a dynamic host based on window.location would be better in production
    const ws = new WebSocket('ws://127.0.0.1:8000/ws/stream');

    ws.onopen = () => {
      setWsStatus('Connected');
    };

    ws.onmessage = (event) => {
      try {
        // We might get simple echoes or actual JSON payloads
        if (event.data.startsWith('Echo:')) return;

        const payload: WebSocketPayload = JSON.parse(event.data);
        setLogs(prevLogs => [...prevLogs, payload]);

        if (payload.data && payload.data.metrics) {
            setCurrentMetrics(payload.data.metrics);
        }
      } catch (e) {
        console.warn('Received non-JSON message from WebSocket', event.data);
      }
    };

    ws.onclose = () => {
      setWsStatus('Disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      setWsStatus('Error');
    };

    return () => {
      ws.close();
    };
  }, []);

  // Compute Risk Matrix color based on inventory threshold
  const isInventoryHighRisk = currentMetrics ? Math.abs(currentMetrics.inventory) > MAX_INVENTORY_THRESHOLD : false;

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace', backgroundColor: '#1e1e1e', color: '#00ff00', minHeight: '100vh' }}>
      <h1>Adam v30 Telemetry Dashboard</h1>
      <div style={{ marginBottom: '10px' }}>
        <strong>WebSocket Status:</strong> {wsStatus}
      </div>

      {/* Risk Matrix Indicator */}
      <div style={{
          padding: '15px',
          margin: '20px 0',
          backgroundColor: isInventoryHighRisk ? '#8b0000' : '#004d00',
          color: 'white',
          border: '1px solid white',
          borderRadius: '5px'
        }}>
        <h2>Risk Matrix</h2>
        <p>Current Inventory: {currentMetrics ? currentMetrics.inventory : 'N/A'}</p>
        <p>Status: {isInventoryHighRisk ? 'HIGH RISK - SKEW REQUIRED' : 'NORMAL'}</p>
      </div>

      {/* Live Charting Placeholder */}
      <div style={{ border: '1px solid #00ff00', padding: '10px', marginBottom: '20px', height: '200px' }}>
        <h3>Live Charting (Spread vs Fair Value)</h3>
        {/* In a real implementation, a library like Recharts or Chart.js would go here */}
        <p>Current Spread (BPS): {currentMetrics ? currentMetrics.spread_bps.toFixed(2) : 'N/A'}</p>
      </div>

      {/* Event Stream UI Terminal */}
      <div style={{ border: '1px solid #00ff00', padding: '10px', height: '300px', overflowY: 'auto', backgroundColor: '#000' }}>
        <h3>Event Stream (Terminal)</h3>
        {logs.map((log, index) => (
          <div key={index} style={{ marginBottom: '10px', borderBottom: '1px dashed #333', paddingBottom: '5px' }}>
            <span style={{ color: '#888' }}>[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
            <strong style={{ color: log.data.status === 'PASS' ? '#00ff00' : '#ff4444' }}>{log.event_type}</strong>
            <pre style={{ margin: '5px 0 0 0', whiteSpace: 'pre-wrap', fontSize: '0.9em' }}>
              {JSON.stringify(log.data, null, 2)}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EvalDashboard;
