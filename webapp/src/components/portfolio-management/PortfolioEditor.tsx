import React from 'react';

const PortfolioEditor: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Edit Portfolio</h3>
      <div style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <h4>Add Holding</h4>
          <label>Symbol:</label>
          <input type="text" style={{ width: '100%', padding: '5px', marginBottom: '10px' }} />
          <label>Quantity:</label>
          <input type="number" style={{ width: '100%', padding: '5px', marginBottom: '10px' }} />
          <button style={{ padding: '10px', width: '100%' }}>Add</button>
        </div>
        <div style={{ flex: 1 }}>
          <h4>Remove Holding</h4>
           <label>Symbol:</label>
          <select style={{ width: '100%', padding: '5px', marginBottom: '10px' }}>
            <option>TC</option>
            <option>GEC</option>
          </select>
          <button style={{ padding: '10px', width: '100%', backgroundColor: 'red', color: 'white' }}>Remove</button>
        </div>
         <div style={{ flex: 1 }}>
          <h4>Rebalance</h4>
          <button style={{ padding: '10px', width: '100%' }}>Rebalance Portfolio</button>
        </div>
      </div>
    </div>
  );
};

export default PortfolioEditor;
