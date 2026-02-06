import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Loading from './components/common/Loading';

// Eager load Dashboard for LCP
import Dashboard from './pages/Dashboard';

// Bolt: Lazy load heavy route components to split bundle and improve initial load time
const Terminal = lazy(() => import('./components/Terminal'));
const Vault = lazy(() => import('./pages/Vault'));
const AgentStatus = lazy(() => import('./pages/AgentStatus'));
const KnowledgeGraph = lazy(() => import('./pages/KnowledgeGraph'));
const DeepDive = lazy(() => import('./pages/DeepDive'));
const SwarmActivity = lazy(() => import('./pages/SwarmActivity'));

// Lazy load placeholders/secondary pages
const MarketData = lazy(() => import('./pages/MarketData'));
const AnalysisTools = lazy(() => import('./pages/AnalysisTools'));
const PortfolioManagement = lazy(() => import('./pages/PortfolioManagement'));
const SimulationTools = lazy(() => import('./pages/SimulationTools'));
const Synthesizer = lazy(() => import('./pages/Synthesizer'));
const PromptAlpha = lazy(() => import('./pages/PromptAlpha'));

const App: React.FC = () => {
  return (
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="synthesizer" element={<Suspense fallback={<Loading />}><Synthesizer /></Suspense>} />
          <Route path="terminal" element={
            <Suspense fallback={<Loading />}>
              <div style={{height: '80vh'}}><Terminal /></div>
            </Suspense>
          } />
          <Route path="prompt-alpha" element={<Suspense fallback={<Loading />}><PromptAlpha /></Suspense>} />
          <Route path="market-data" element={<Suspense fallback={<Loading />}><MarketData /></Suspense>} />
          <Route path="analysis-tools" element={<Suspense fallback={<Loading />}><AnalysisTools /></Suspense>} />
          <Route path="knowledge-graph" element={<Suspense fallback={<Loading />}><KnowledgeGraph /></Suspense>} />
          <Route path="agents" element={<Suspense fallback={<Loading />}><AgentStatus /></Suspense>} />
          <Route path="swarm" element={<Suspense fallback={<Loading />}><SwarmActivity /></Suspense>} />
          <Route path="vault" element={<Suspense fallback={<Loading />}><Vault /></Suspense>} />
          <Route path="deep-dive/:id" element={<Suspense fallback={<Loading />}><DeepDive /></Suspense>} />
          <Route path="portfolio-management" element={<Suspense fallback={<Loading />}><PortfolioManagement /></Suspense>} />
          <Route path="simulation-tools" element={<Suspense fallback={<Loading />}><SimulationTools /></Suspense>} />
        </Route>
      </Routes>
  );
};

export default App;
