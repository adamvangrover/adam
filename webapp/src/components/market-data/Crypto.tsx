import React from 'react';

const Crypto: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <h4>Crypto</h4>
      <div style={{ border: '1px solid #eee', height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#999', marginBottom: '16px' }}>
        [Price Chart Placeholder]
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Name</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Market Cap</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>24h Change</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>Bitcoin (BTC)</td>
            <td style={{ padding: '8px' }}>$65,432.10</td>
            <td style={{ padding: '8px' }}>$1.3T</td>
            <td style={{ padding: '8px', color: 'green' }}>+1.2%</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>Ethereum (ETH)</td>
            <td style={{ padding: '8px' }}>$3,456.78</td>
            <td style={{ padding: '8px' }}>$415B</td>
            <td style={{ padding: '8px', color: 'red' }}>-0.5%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default Crypto;
