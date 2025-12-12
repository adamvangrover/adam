import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Terminal from './components/Terminal';
import Vault from './pages/Vault';
import AgentStatus from './pages/AgentStatus';
import KnowledgeGraphPage from './pages/KnowledgeGraphPage';
import MarketData from './pages/MarketData';
import AnalysisTools from './pages/AnalysisTools';
import PortfolioManagement from './pages/PortfolioManagement';
import SimulationTools from './pages/SimulationTools';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="terminal" element={<div style={{height: '80vh', padding: '20px'}}><Terminal /></div>} />
        <Route path="market-data" element={<MarketData />} />
        <Route path="analysis-tools" element={<AnalysisTools />} />
        <Route path="knowledge-graph" element={<KnowledgeGraphPage />} />
        <Route path="agents" element={<AgentStatus />} />
        <Route path="vault" element={<Vault />} />
        <Route path="portfolio-management" element={<PortfolioManagement />} />
        <Route path="simulation-tools" element={<SimulationTools />} />
      </Route>
    </Routes>
  );
};

export default App;
