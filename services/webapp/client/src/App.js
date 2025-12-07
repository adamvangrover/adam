import React, { Suspense, useState, useEffect } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import './App.css';
import Login from './Login';
import MissionControl from './components/dashboard/MissionControl';
import AgentRegistry from './components/dashboard/AgentRegistry';
import IntelFeed from './components/dashboard/IntelFeed';
import Layout from './components/layout/Layout';
import { getToken, logout } from './utils/auth';
import { dataManager } from './utils/DataManager';

// Lazy load legacy components
const MarketData = React.lazy(() => import('./MarketData'));
const AnalysisTools = React.lazy(() => import('./AnalysisTools'));
const PortfolioManagement = React.lazy(() => import('./PortfolioManagement'));
const Simulations = React.lazy(() => import('./Simulations'));
// const KnowledgeGraph = React.lazy(() => import('./KnowledgeGraph'));
const NeuralDashboard = React.lazy(() => import('./components/dashboard/NeuralDashboard'));

function App() {
  useTranslation();
  // Simplified auth for now, can be expanded
  const isLoggedIn = !!getToken() || true; // Force login for demo purposes if token missing is annoying
  const navigate = useNavigate();
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    const checkBackend = async () => {
        await dataManager.checkConnection();
        setIsOffline(dataManager.isOfflineMode());
    };
    checkBackend();
  }, []);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  }

  if (!isLoggedIn) {
     return <Login onLogin={() => window.location.reload()} />;
  }

  return (
    <Layout onLogout={handleLogout} isOffline={isOffline}>
        <Suspense fallback={
            <div className="flex items-center justify-center h-full text-cyan-500 font-mono animate-pulse">
                INITIALIZING_SYSTEM_CORE...
            </div>
        }>
            <Routes>
                <Route path="/login" element={<Navigate to="/" />} />
                <Route path="/" element={<MissionControl />} />
                <Route path="/market-data" element={<MarketData />} />
                <Route path="/analysis" element={<AnalysisTools />} />
                <Route path="/portfolios" element={<PortfolioManagement />} />
                <Route path="/simulations" element={<Simulations />} />
                <Route path="/knowledge-graph" element={<NeuralDashboard />} />
                <Route path="/agents" element={<AgentRegistry />} />
                <Route path="/intel" element={<IntelFeed />} />
            </Routes>
        </Suspense>
    </Layout>
  );
}

export default App;
