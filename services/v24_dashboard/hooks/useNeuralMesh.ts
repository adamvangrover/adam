import { useState, useEffect, useRef } from 'react';

export interface NeuralPacket {
  id: string;
  timestamp: string;
  source_agent: string;
  packet_type: 'market_data' | 'risk_alert' | 'thought' | 'system_status';
  payload: any;
  priority: number;
}

export const useNeuralMesh = () => {
  const [status, setStatus] = useState<'Connecting' | 'Connected' | 'Disconnected'>('Connecting');
  const [feed, setFeed] = useState<NeuralPacket[]>([]);
  const [marketData, setMarketData] = useState<Record<string, any>>({});
  const [systemMetrics, setSystemMetrics] = useState<any>({});

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      // Connect to the new v2 endpoint
      const ws = new WebSocket('ws://localhost:8001/ws/mesh');
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('Connected');
        console.log('Neural Mesh: Uplink Established');
      };

      ws.onmessage = (event) => {
        try {
          const packet: NeuralPacket = JSON.parse(event.data);

          // Add to log
          setFeed((prev) => [...prev.slice(-99), packet]); // Keep last 100

          // Update specialized state
          if (packet.packet_type === 'market_data') {
            setMarketData((prev) => ({
              ...prev,
              [packet.payload.symbol]: packet.payload
            }));
          } else if (packet.packet_type === 'system_status') {
            setSystemMetrics(packet.payload);
          }

        } catch (err) {
          console.error('Neural Mesh: Parse Error', err);
        }
      };

      ws.onclose = () => {
        setStatus('Disconnected');
        console.log('Neural Mesh: Uplink Lost. Retrying in 3s...');
        setTimeout(connect, 3000); // Auto-reconnect
      };
    };

    connect();

    return () => {
      wsRef.current?.close();
    };
  }, []);

  return { status, feed, marketData, systemMetrics };
};
