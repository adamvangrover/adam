// services/webapp/client/src/pages/Dashboard.tsx

import React from 'react';
import '../Dashboard.css';
import AgentRunner from '../AgentRunner'; // Assuming AgentRunner is still JS, allowed by allowJs: true

const Dashboard: React.FC = () => {
  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tight">Dashboard</h2>
          <p className="text-slate-400 mt-1 text-sm">Monitor and control your AI agents</p>
        </div>
      </div>

      <div className="Dashboard">
        <div className="Card">
          {/* @ts-ignore - AgentRunner is JS and types are inferred strictly */}
          <AgentRunner />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
