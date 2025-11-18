import React from 'react';

const FundamentalAnalysis: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Fundamental Analysis</h3>
      <div style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <h4>Inputs</h4>
          <label>Company Ticker:</label>
          <input type="text" defaultValue="TC" style={{ width: '100%', padding: '5px', marginBottom: '10px' }} />
          <label>Valuation Model:</label>
          <select style={{ width: '100%', padding: '5px', marginBottom: '10px' }}>
            <option>Discounted Cash Flow (DCF)</option>
            <option>Comparable Company Analysis</option>
          </select>
          <button style={{ padding: '10px', width: '100%' }}>Run Analysis</button>
        </div>
        <div style={{ flex: 2, border: '1px solid #eee', padding: '10px' }}>
          <h4>Output</h4>
          <p><strong>Intrinsic Value:</strong> $165.00</p>
          <div style={{ height: '150px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#999' }}>
            [Valuation Chart Placeholder]
          </div>
        </div>
      </div>
    </div>
  );
};

export default FundamentalAnalysis;
