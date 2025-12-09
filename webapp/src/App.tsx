import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MarketData from './pages/MarketData';
import AnalysisTools from './pages/AnalysisTools';
import PortfolioManagement from './pages/PortfolioManagement';
import Alerts from './pages/Alerts';
import NewsAndInsights from './pages/NewsAndInsights';
import UserPreferences from './pages/UserPreferences';
import SimulationTools from './pages/SimulationTools';

import Terminal from './components/Terminal';
import AgentRegistry from './pages/AgentRegistry';
import Reports from './pages/Reports';

// New Pages (Phase 3 & 4) - Mocking them for now until Phase 3/4 execution
const KnowledgeGraph = () => <div className="p-10 text-cyber-cyan font-mono">KNOWLEDGE GRAPH :: CONNECTING TO NEO4J...</div>;
const DeepDive = () => <div className="p-10 text-cyber-cyan font-mono">DEEP DIVE :: SELECT TARGET...</div>;
const Settings = () => <div className="p-10 text-cyber-cyan font-mono">SYSTEM CONFIG :: READ ONLY</div>;

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="terminal" element={<Terminal />} />
          <Route path="market-data" element={<MarketData />} />
          <Route path="agents" element={<AgentRegistry />} />
          <Route path="knowledge-graph" element={<KnowledgeGraph />} />
          <Route path="deep-dive" element={<DeepDive />} />
          <Route path="reports" element={<Reports />} />
          <Route path="settings" element={<Settings />} />

          {/* Legacy Routes Mapped or Kept */}
          <Route path="analysis-tools" element={<AnalysisTools />} />
          <Route path="portfolio-management" element={<PortfolioManagement />} />
          <Route path="alerts" element={<Alerts />} />
          <Route path="news-and-insights" element={<NewsAndInsights />} />
          <Route path="simulation-tools" element={<SimulationTools />} />
          <Route path="user-preferences" element={<UserPreferences />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
