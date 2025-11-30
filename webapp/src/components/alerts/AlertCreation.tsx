import React from 'react';

const AlertCreation: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Create New Alert</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '10px', alignItems: 'center' }}>
        <label>Alert Type:</label>
        <select style={{ padding: '5px' }}>
          <option>Price</option>
          <option>Volume</option>
          <option>News Sentiment</option>
        </select>

        <label>Asset:</label>
        <input type="text" placeholder="e.g., TC" style={{ padding: '5px' }} />

        <label>Condition:</label>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <select style={{ padding: '5px' }}>
            <option>&gt;</option>
            <option>&lt;</option>
          </select>
          <input type="text" placeholder="Value" style={{ padding: '5px', flex: 1 }} />
        </div>

        <label>Notification:</label>
        <div>
          <input type="checkbox" id="email" defaultChecked />
          <label htmlFor="email" style={{ marginRight: '10px' }}>Email</label>
          <input type="checkbox" id="sms" />
          <label htmlFor="sms">SMS</label>
        </div>

        <div style={{ gridColumn: 'span 2', textAlign: 'right' }}>
          <button style={{ padding: '10px 20px' }}>Create Alert</button>
        </div>
      </div>
    </div>
  );
};

export default AlertCreation;
