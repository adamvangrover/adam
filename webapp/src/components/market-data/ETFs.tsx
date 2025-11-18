import React from 'react';

const ETFs: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <h4>ETFs</h4>
      <div style={{ border: '1px solid #eee', height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#999', marginBottom: '16px' }}>
        [Performance Chart Placeholder]
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Symbol</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Name</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Assets</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>SPY</td>
            <td style={{ padding: '8px' }}>SPDR S&P 500 ETF Trust</td>
            <td style={{ padding: '8px' }}>$543.21</td>
            <td style={{ padding: '8px' }}>$400B</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>QQQ</td>
            <td style={{ padding: '8px' }}>Invesco QQQ Trust</td>
            <td style={{ padding: '8px' }}>$456.78</td>
            <td style={{ padding: '8px' }}>$200B</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default ETFs;
