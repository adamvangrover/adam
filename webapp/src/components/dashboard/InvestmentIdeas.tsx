import React from 'react';

const InvestmentCard: React.FC<{ name: string; rationale: string; rating: string; risk: string }> = ({ name, rationale, rating, risk }) => (
    <div style={{ border: '1px solid #eee', padding: '10px', borderRadius: '5px', marginBottom: '10px' }}>
        <h4 style={{ margin: '0 0 5px 0' }}>{name}</h4>
        <p style={{ margin: 0 }}><strong>Rationale:</strong> {rationale}</p>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}>
            <span><strong>Conviction:</strong> <span style={{ color: 'green' }}>{rating}</span></span>
            <span><strong>Risk:</strong> <span style={{ color: 'orange' }}>{risk}</span></span>
        </div>
    </div>
);

const InvestmentIdeas: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Investment Ideas</h3>
      <InvestmentCard name="TechCorp (TC)" rationale="Strong earnings growth and new product pipeline." rating="High" risk="Medium" />
      <InvestmentCard name="GreenEnergy Co. (GEC)" rationale="Favorable regulatory environment and increasing demand." rating="Medium" risk="Low" />
    </div>
  );
};

export default InvestmentIdeas;
