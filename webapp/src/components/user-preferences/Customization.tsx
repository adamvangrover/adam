import React from 'react';

const Customization: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Customization</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '10px', alignItems: 'center', maxWidth: '500px' }}>
        <label>Theme:</label>
        <div>
          <input type="radio" name="theme" id="light" defaultChecked />
          <label htmlFor="light" style={{ marginRight: '10px' }}>Light</label>
          <input type="radio" name="theme" id="dark" />
          <label htmlFor="dark">Dark</label>
        </div>

        <label>Risk Tolerance:</label>
        <select style={{ padding: '5px' }}>
          <option>Low</option>
          <option>Medium</option>
          <option>High</option>
        </select>

        <div style={{ gridColumn: 'span 2', textAlign: 'right', paddingTop: '10px' }}>
          <button style={{ padding: '10px 20px' }}>Save Preferences</button>
        </div>
      </div>
    </div>
  );
};

export default Customization;
