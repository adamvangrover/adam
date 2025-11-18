import React from 'react';

const PortfolioOverview: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Portfolio Overview</h3>
      <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <h4>Asset Allocation</h4>
          <div style={{
            width: '150px',
            height: '150px',
            borderRadius: '50%',
            background: 'conic-gradient(blue 0% 40%, green 40% 70%, orange 70% 90%, red 90% 100%)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            color: 'white',
            fontWeight: 'bold'
          }}>
            Pie Chart
          </div>
        </div>
        <div>
          <h4>Key Metrics</h4>
          <p><strong>Total Value:</strong> $1,234,567.89</p>
          <p><strong>Day's Gain:</strong> <span style={{ color: 'green' }}>+$5,432.10 (0.44%)</span></p>
          <p><strong>Overall Return:</strong> <span style={{ color: 'green' }}>+$123,456.78 (11.11%)</span></p>
        </div>
      </div>
      <div style={{ marginTop: '16px' }}>
        <h4>Performance Over Time</h4>
        <div style={{ border: '1px solid #eee', height: '150px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#999' }}>
          [Line Chart Placeholder]
        </div>
      </div>
    </div>
  );
};

export default PortfolioOverview;
