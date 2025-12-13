import React from 'react';
import { render } from '@testing-library/react';
import App from './App';

// Mock child components to avoid dependency issues
jest.mock('./components/Layout', () => () => <div>Layout</div>);
jest.mock('./pages/Dashboard', () => () => <div>Dashboard</div>);
jest.mock('./components/Terminal', () => () => <div>Terminal</div>);
jest.mock('./pages/Vault', () => () => <div>Vault</div>);
jest.mock('./pages/AgentStatus', () => () => <div>AgentStatus</div>);
jest.mock('./pages/KnowledgeGraph', () => () => <div>KnowledgeGraph</div>);
jest.mock('./pages/DeepDive', () => () => <div>DeepDive</div>);
jest.mock('./pages/MarketData', () => () => <div>MarketData</div>);
jest.mock('./pages/AnalysisTools', () => () => <div>AnalysisTools</div>);
jest.mock('./pages/PortfolioManagement', () => () => <div>PortfolioManagement</div>);
jest.mock('./pages/SimulationTools', () => () => <div>SimulationTools</div>);

test('renders app without crashing', () => {
  render(<App />);
});
