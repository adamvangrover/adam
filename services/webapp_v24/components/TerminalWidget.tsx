import React from 'react';

const TerminalWidget = () => {
  return (
    <div className="bg-black text-green-500 font-mono p-4 rounded-lg border border-green-800 h-64 overflow-y-auto">
      <div className="mb-2 border-b border-green-800 pb-2 flex justify-between">
        <span>ADAM-v24 TERMINAL</span>
        <span className="animate-pulse">● ONLINE</span>
      </div>
      <div className="space-y-1 text-sm">
        <p>[SYSTEM] Initializing v24 Interface...</p>
        <p>[SYSTEM] Connected to Rust Core Engine.</p>
        <p>[SYSTEM] Listening for market data...</p>
        <p className="text-blue-400">[INFO] Waiting for user input...</p>
      </div>
      <div className="mt-4 flex">
        <span className="mr-2">{'>'}</span>
        <input
          type="text"
          className="bg-transparent outline-none flex-1 text-green-500 placeholder-green-800"
          placeholder="Enter command..."
        />
      </div>
    </div>
  );
};

export default TerminalWidget;
