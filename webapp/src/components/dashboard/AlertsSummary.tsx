import React from 'react';

const AlertsSummary: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Active Alerts</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Asset</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Condition</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>TechCorp (TC)</td>
            <td style={{ padding: '8px' }}>Price &gt; $150</td>
            <td style={{ padding: '8px', color: 'green' }}>Active</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>GreenEnergy Co. (GEC)</td>
            <td style={{ padding: '8px' }}>Volume &gt; 10M</td>
            <td style={{ padding: '8px', color: 'orange' }}>Triggered</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default AlertsSummary;
