import React from 'react';

const AlertsDashboard: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Alerts Dashboard</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Asset</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Condition</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Status</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>TechCorp (TC)</td>
            <td style={{ padding: '8px' }}>Price &gt; $150</td>
            <td style={{ padding: '8px', color: 'green' }}>Active</td>
            <td style={{ padding: '8px' }}>
              <button style={{ marginRight: '5px' }}>Edit</button>
              <button style={{ backgroundColor: 'red', color: 'white' }}>Delete</button>
            </td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>GreenEnergy Co. (GEC)</td>
            <td style={{ padding: '8px' }}>Volume &gt; 10M</td>
            <td style={{ padding: '8px', color: 'orange' }}>Triggered</td>
            <td style={{ padding: '8px' }}>
              <button style={{ marginRight: '5px' }}>Edit</button>
              <button style={{ backgroundColor: 'red', color: 'white' }}>Delete</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default AlertsDashboard;
