import React from 'react';

const CreditRatingSimulation: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Credit Rating Simulation</h3>
      <div style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <h4>Inputs</h4>
          <label>Company Ticker:</label>
          <input type="text" defaultValue="TC" style={{ width: '100%', padding: '5px', marginBottom: '10px' }} />
          <label>Macroeconomic Scenario:</label>
          <select style={{ width: '100%', padding: '5px', marginBottom: '10px' }}>
            <option>Baseline</option>
            <option>Recession</option>
            <option>Rapid Growth</option>
          </select>
          <button style={{ padding: '10px', width: '100%' }}>Run Simulation</button>
        </div>
        <div style={{ flex: 2, border: '1px solid #eee', padding: '10px', textAlign: 'center' }}>
          <h4>Predicted Credit Rating</h4>
          <div style={{ fontSize: '2em', fontWeight: 'bold', margin: '20px 0' }}>AA</div>
          <p>Confidence Score: 92%</p>
        </div>
      </div>
    </div>
  );
};

export default CreditRatingSimulation;
