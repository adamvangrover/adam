import React from 'react';

const AdamsInsights: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Adam's Insights</h3>
      <p>Access to Adam v19.2's generated newsletters and reports.</p>
      <ul style={{ margin: 0, paddingLeft: '20px' }}>
        <li>
          <a href="#">Weekly Market Summary - 2024-06-10</a>
        </li>
        <li>
          <a href="#">Deep Dive: The Future of Green Energy</a>
        </li>
      </ul>
    </div>
  );
};

export default AdamsInsights;
