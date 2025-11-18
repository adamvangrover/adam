import React from 'react';
import MarketSummary from '../components/dashboard/MarketSummary';
import PortfolioOverview from '../components/dashboard/PortfolioOverview';
import InvestmentIdeas from '../components/dashboard/InvestmentIdeas';
import AlertsSummary from '../components/dashboard/AlertsSummary';
import SimulationResults from '../components/dashboard/SimulationResults';

const Dashboard: React.FC = () => {
  return (
    <div>
      <h1>Dashboard</h1>
      <MarketSummary />
      <PortfolioOverview />
      <InvestmentIdeas />
      <AlertsSummary />
      <SimulationResults />
    </div>
  );
};

export default Dashboard;
