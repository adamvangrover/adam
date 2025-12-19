// services/webapp/client/src/pages/Dashboard.tsx

import React from 'react';
import '../Dashboard.css';
import AgentRunner from '../AgentRunner'; // Assuming AgentRunner is still JS, allowed by allowJs: true

const Dashboard: React.FC = () => {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Dashboard</h2>
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
