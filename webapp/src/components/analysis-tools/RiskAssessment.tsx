import React from 'react';

const RiskAssessment: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Risk Assessment</h3>
      <div style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <h4>Inputs</h4>
          <label>Portfolio:</label>
          <select style={{ width: '100%', padding: '5px', marginBottom: '10px' }}>
            <option>Current Portfolio</option>
            <option>Test Portfolio A</option>
          </select>
           <label>Risk Model:</label>
          <select style={{ width: '100%', padding: '5px', marginBottom: '10px' }}>
            <option>Value at Risk (VaR)</option>
            <option>Monte Carlo Simulation</option>
          </select>
          <button style={{ padding: '10px', width: '100%' }}>Assess Risk</button>
        </div>
        <div style={{ flex: 2, border: '1px solid #eee', padding: '10px' }}>
          <h4>Output</h4>
          <p><strong>95% VaR:</strong> $15,234.56</p>
           <div style={{ height: '150px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#999' }}>
            [Risk Distribution Chart Placeholder]
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment;
