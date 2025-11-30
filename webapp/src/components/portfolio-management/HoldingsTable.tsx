import React from 'react';

const HoldingsTable: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Current Holdings</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Symbol</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Quantity</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>TC</td>
            <td style={{ padding: '8px' }}>100</td>
            <td style={{ padding: '8px' }}>$152.45</td>
            <td style={{ padding: '8px' }}>$15,245.00</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>GEC</td>
            <td style={{ padding: '8px' }}>200</td>
            <td style={{ padding: '8px' }}>$78.90</td>
            <td style={{ padding: '8px' }}>$15,780.00</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default HoldingsTable;
