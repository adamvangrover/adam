import React from 'react';
import FundamentalAnalysis from '../components/analysis-tools/FundamentalAnalysis';
import TechnicalAnalysis from '../components/analysis-tools/TechnicalAnalysis';
import RiskAssessment from '../components/analysis-tools/RiskAssessment';
import FinancialModeling from '../components/analysis-tools/FinancialModeling';
import LegalAnalysis from '../components/analysis-tools/LegalAnalysis';

const AnalysisTools: React.FC = () => {
  return (
    <div>
      <h1>Analysis Tools</h1>
      <FundamentalAnalysis />
      <TechnicalAnalysis />
      <RiskAssessment />
      <FinancialModeling />
      <LegalAnalysis />
    </div>
  );
};

export default AnalysisTools;
