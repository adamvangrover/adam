import React from 'react';
import CreditRatingSimulation from '../components/simulation-tools/CreditRatingSimulation';
import InvestmentCommitteeSimulation from '../components/simulation-tools/InvestmentCommitteeSimulation';
import SimulationReports from '../components/simulation-tools/SimulationReports';

const SimulationTools: React.FC = () => {
  return (
    <div>
      <h1>Simulation Tools and Reports</h1>
      <CreditRatingSimulation />
      <InvestmentCommitteeSimulation />
      <SimulationReports />
    </div>
  );
};

export default SimulationTools;
