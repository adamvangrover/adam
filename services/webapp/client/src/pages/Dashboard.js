import React from 'react';
import '../Dashboard.css';
import AgentRunner from '../AgentRunner';

function Dashboard() {
  return (
    <div>
      <h2>Dashboard</h2>
      <div className="Dashboard">
        <div className="Card">
          <AgentRunner />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
