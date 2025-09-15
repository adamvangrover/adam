import React, { useState, useEffect } from 'react';
import './App.css';
import Login from './Login';
import Dashboard from './Dashboard';
import MarketData from './MarketData';
import AnalysisTools from './AnalysisTools';
import PortfolioManagement from './PortfolioManagement';
import Simulations from './Simulations';
import KnowledgeGraph from './KnowledgeGraph';
import io from 'socket.io-client';

const socket = io('http://localhost:5001');

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [activeSection, setActiveSection] = useState('Dashboard');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  useEffect(() => {
    socket.on('simulation_complete', (data) => {
      new Notification(`Simulation ${data.simulation_name} complete!`);
    });

    return () => {
      socket.off('simulation_complete');
    };
  }, []);

  const handleLogin = (token) => {
    setToken(token);
    localStorage.setItem('token', token);
  };

  const handleLogout = () => {
    setToken(null);
    localStorage.removeItem('token');
  }

  const renderSection = () => {
    switch (activeSection) {
      case 'Dashboard':
        return <Dashboard />;
      case 'Market Data':
        return <MarketData />;
      case 'Analysis Tools':
        return <AnalysisTools />;
      case 'Portfolio Management':
        return <PortfolioManagement />;
      case 'Simulations':
        return <Simulations />;
      case 'Knowledge Graph':
        return <KnowledgeGraph />;
      default:
        return <Dashboard />;
    }
  };

  if (!token) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="App">
      <header className="App-header">
        <button onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}>
          {isSidebarCollapsed ? '>>' : '<<'}
        </button>
        <h1>Adam v19.2</h1>
        <button onClick={handleLogout}>Logout</button>
      </header>
      <div className="App-body">
        <nav className={`App-sidebar ${isSidebarCollapsed ? 'collapsed' : ''}`}>
          <h2>Menu</h2>
          <ul>
            <li onClick={() => setActiveSection('Dashboard')}>Dashboard</li>
            <li onClick={() => setActiveSection('Market Data')}>Market Data</li>
            <li onClick={() => setActiveSection('Analysis Tools')}>Analysis Tools</li>
            <li onClick={() => setActiveSection('Portfolio Management')}>Portfolio Management</li>
            <li onClick={() => setActiveSection('Simulations')}>Simulations</li>
            <li onClick={() => setActiveSection('Knowledge Graph')}>Knowledge Graph</li>
          </ul>
        </nav>
        <main className="App-content">
          {renderSection()}
        </main>
      </div>
    </div>
  );
}

export default App;
