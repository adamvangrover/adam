import React from 'react';
import { Link } from 'react-router-dom';

const Sidebar: React.FC = () => {
  return (
    <div style={{ width: '250px', backgroundColor: '#f0f0f0', padding: '20px' }}>
      <h2>Adam v19.2</h2>
      <nav>
        <ul>
          <li><Link to="/">Dashboard</Link></li>
          <li><Link to="/market-data">Market Data</Link></li>
          <li><Link to="/analysis-tools">Analysis Tools</Link></li>
          <li><Link to="/portfolio-management">Portfolio Management</Link></li>
          <li><Link to="/alerts">Alerts</Link></li>
          <li><Link to="/news-and-insights">News and Insights</Link></li>
          <li><Link to="/user-preferences">User Preferences</Link></li>
          <li><Link to="/simulation-tools">Simulation Tools</Link></li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
