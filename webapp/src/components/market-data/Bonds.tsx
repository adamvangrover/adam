import React from 'react';

const Bonds: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <h4>Bonds</h4>
      <div style={{ border: '1px solid #eee', height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#999', marginBottom: '16px' }}>
        [Yield Curve Chart Placeholder]
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Name</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Yield</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Maturity</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>US 10-Year</td>
            <td style={{ padding: '8px' }}>4.5%</td>
            <td style={{ padding: '8px' }}>98.5</td>
            <td style={{ padding: '8px' }}>2034-06-12</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>US 2-Year</td>
            <td style={{ padding: '8px' }}>4.8%</td>
            <td style={{ padding: '8px' }}>101.2</td>
            <td style={{ padding: '8px' }}>2026-06-12</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default Bonds;
