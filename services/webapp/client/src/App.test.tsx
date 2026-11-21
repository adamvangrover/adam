import React from 'react';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from './App';

// Mock child components to avoid dependency issues
vi.mock('./components/Layout', () => ({ default: () => <div>Layout</div> }));
vi.mock('./pages/Dashboard', () => ({ default: () => <div>Dashboard</div> }));
vi.mock('./components/Terminal', () => ({ default: () => <div>Terminal</div> }));
vi.mock('./pages/Vault', () => ({ default: () => <div>Vault</div> }));
vi.mock('./pages/AgentStatus', () => ({ default: () => <div>AgentStatus</div> }));
vi.mock('./pages/KnowledgeGraph', () => ({ default: () => <div>KnowledgeGraph</div> }));
vi.mock('./pages/DeepDive', () => ({ default: () => <div>DeepDive</div> }));
vi.mock('./pages/MarketData', () => ({ default: () => <div>MarketData</div> }));
vi.mock('./pages/AnalysisTools', () => ({ default: () => <div>AnalysisTools</div> }));
vi.mock('./pages/PortfolioManagement', () => ({ default: () => <div>PortfolioManagement</div> }));
vi.mock('./pages/SimulationTools', () => ({ default: () => <div>SimulationTools</div> }));

test('renders app without crashing', () => {
  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>
  );
});
