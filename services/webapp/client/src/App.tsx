import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Terminal from './components/Terminal';
import Vault from './pages/Vault';
import AgentStatus from './pages/AgentStatus';
import KnowledgeGraph from './pages/KnowledgeGraph';
import DeepDive from './pages/DeepDive';

// Placeholder imports for remaining pages
import MarketData from './pages/MarketData';
import AnalysisTools from './pages/AnalysisTools';
import PortfolioManagement from './pages/PortfolioManagement';
import SimulationTools from './pages/SimulationTools';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="terminal" element={<div style={{height: '80vh'}}><Terminal /></div>} />
          <Route path="market-data" element={<MarketData />} />
          <Route path="analysis-tools" element={<AnalysisTools />} />
          <Route path="knowledge-graph" element={<KnowledgeGraph />} />
          <Route path="agents" element={<AgentStatus />} />
          <Route path="vault" element={<Vault />} />
          <Route path="deep-dive/:id" element={<DeepDive />} />
          <Route path="portfolio-management" element={<PortfolioManagement />} />
          <Route path="simulation-tools" element={<SimulationTools />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
