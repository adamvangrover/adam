import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MarketData from './pages/MarketData';
import AnalysisTools from './pages/AnalysisTools';
import PortfolioManagement from './pages/PortfolioManagement';
import Alerts from './pages/Alerts';
import NewsAndInsights from './pages/NewsAndInsights';
import UserPreferences from './pages/UserPreferences';
import SimulationTools from './pages/SimulationTools';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="market-data" element={<MarketData />} />
          <Route path="analysis-tools" element={<AnalysisTools />} />
          <Route path="portfolio-management" element={<PortfolioManagement />} />
          <Route path="alerts" element={<Alerts />} />
          <Route path="news-and-insights" element={<NewsAndInsights />} />
          <Route path="user-preferences" element={<UserPreferences />} />
          <Route path="simulation-tools" element={<SimulationTools />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
