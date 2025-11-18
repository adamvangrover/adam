import React from 'react';
import HoldingsTable from '../components/portfolio-management/HoldingsTable';
import PortfolioEditor from '../components/portfolio-management/PortfolioEditor';
import PerformanceHistory from '../components/portfolio-management/PerformanceHistory';

const PortfolioManagement: React.FC = () => {
  return (
    <div>
      <h1>Portfolio Management</h1>
      <HoldingsTable />
      <PortfolioEditor />
      <PerformanceHistory />
    </div>
  );
};

export default PortfolioManagement;
