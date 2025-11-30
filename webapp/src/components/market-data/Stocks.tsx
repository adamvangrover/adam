import React from 'react';

const Stocks: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <h4>Stocks</h4>
      <div style={{ border: '1px solid #eee', height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#999', marginBottom: '16px' }}>
        [Interactive Candlestick Chart Placeholder]
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Symbol</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Change</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Volume</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>TC</td>
            <td style={{ padding: '8px' }}>$152.45</td>
            <td style={{ padding: '8px', color: 'green' }}>+2.10</td>
            <td style={{ padding: '8px' }}>8.5M</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>GEC</td>
            <td style={{ padding: '8px' }}>$78.90</td>
            <td style={{ padding: '8px', color: 'red' }}>-0.55</td>
            <td style={{ padding: '8px' }}>3.2M</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default Stocks;
