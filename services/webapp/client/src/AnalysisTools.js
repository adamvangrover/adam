import React from 'react';
import FundamentalAnalysis from './FundamentalAnalysis';
import TechnicalAnalysis from './TechnicalAnalysis';
import RiskAssessment from './RiskAssessment';
import MarketSentiment from './MarketSentiment';

function AnalysisTools() {
  return (
    <div>
      <h2>Analysis Tools</h2>
      <div className="Card">
        <FundamentalAnalysis />
      </div>
      <div className="Card">
        <TechnicalAnalysis />
      </div>
      <div className="Card">
        <RiskAssessment />
      </div>
      <div className="Card">
        <MarketSentiment />
      </div>
    </div>
  );
}

export default AnalysisTools;
