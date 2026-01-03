import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

export default function TerminalWidget() {
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    // In a real implementation, this would connect to the Python backend
    const socket = io('http://localhost:5001');

    socket.on('log', (message: string) => {
      setLogs((prev) => [...prev, message]);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <div className="bg-black text-green-500 font-mono p-4 h-64 overflow-y-auto rounded-lg border border-green-800">
      <div className="text-sm text-gray-500 mb-2">System Terminal</div>
      {logs.map((log, i) => (
        <div key={i} className="whitespace-pre-wrap">
          <span className="text-blue-500">$</span> {log}
        </div>
      ))}
      {logs.length === 0 && (
        <div className="opacity-50">Waiting for system connection...</div>
      )}
    </div>
  );
}
