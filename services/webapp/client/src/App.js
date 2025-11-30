import React, { useState, useEffect, Suspense } from 'react';
import { Routes, Route, Link, Navigate, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import './App.css';
import Login from './Login';
import Dashboard from './Dashboard';
import MarketData from './MarketData';
import AnalysisTools from './AnalysisTools';
import PortfolioManagement from './PortfolioManagement';
import Simulations from './Simulations';
import KnowledgeGraph from './KnowledgeGraph';
import io from 'socket.io-client';
import { getToken, logout } from './utils/auth';

const socket = io('http://localhost:5001');

function App() {
  const { t, i18n } = useTranslation();
  const [isLoggedIn, setIsLoggedIn] = useState(!!getToken());
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    socket.on('simulation_complete', (data) => {
      new Notification(`Simulation ${data.simulation_name} complete!`);
    });

    // Check for token on initial load
    if (getToken()) {
      setIsLoggedIn(true);
    }

    return () => {
      socket.off('simulation_complete');
    };
  }, []);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  const handleLogout = async () => {
    await logout();
    setIsLoggedIn(false);
    navigate('/login');
  }

  const PrivateRoute = ({ children }) => {
    return isLoggedIn ? children : <Navigate to="/login" />;
  };

  return (
    <div className="App">
      {isLoggedIn && (
        <header className="App-header">
          <button onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}>
            {isSidebarCollapsed ? '>>' : '<<'}
          </button>
            <h1>{t('app.title')}</h1>
            <div>
              <button onClick={() => i18n.changeLanguage('en')}>English</button>
              <button onClick={() => i18n.changeLanguage('es')}>Español</button>
            </div>
            <button onClick={handleLogout}>{t('app.logout')}</button>
        </header>
      )}
      <div className="App-body">
        {isLoggedIn && (
          <nav className={`App-sidebar ${isSidebarCollapsed ? 'collapsed' : ''}`}>
              <h2>{t('app.menu')}</h2>
            <ul>
                <li><Link to="/">{t('dashboard.title')}</Link></li>
                <li><Link to="/market-data">{t('marketData.title')}</Link></li>
                <li><Link to="/analysis">{t('analysisTools.title')}</Link></li>
                <li><Link to="/portfolios">{t('portfolioManagement.title')}</Link></li>
                <li><Link to="/simulations">{t('simulations.title')}</Link></li>
                <li><Link to="/knowledge-graph">{t('knowledgeGraph.title')}</Link></li>
            </ul>
          </nav>
        )}
        <main className="App-content">
            <Suspense fallback="loading...">
              <Routes>
                <Route path="/login" element={<Login onLogin={handleLogin} />} />
                <Route path="/" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
                <Route path="/market-data" element={<PrivateRoute><MarketData /></PrivateRoute>} />
                <Route path="/analysis" element={<PrivateRoute><AnalysisTools /></PrivateRoute>} />
                <Route path="/portfolios" element={<PrivateRoute><PortfolioManagement /></PrivateRoute>} />
                <Route path="/simulations"element={<PrivateRoute><Simulations /></PrivateRoute>} />
                <Route path="/knowledge-graph" element={<PrivateRoute><KnowledgeGraph /></PrivateRoute>} />
              </Routes>
            </Suspense>
        </main>
      </div>
    </div>
  );
}

export default App;
