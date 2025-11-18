import React from 'react';

const Card: React.FC<{ title: string; value: string; change: string }> = ({ title, value, change }) => (
  <div style={{ border: '1px solid #eee', padding: '10px', borderRadius: '5px', minWidth: '150px', margin: '0 10px' }}>
    <h4 style={{ margin: '0 0 5px 0' }}>{title}</h4>
    <div style={{ fontSize: '1.2em', fontWeight: 'bold' }}>{value}</div>
    <div style={{ color: change.startsWith('+') ? 'green' : 'red' }}>{change}</div>
  </div>
);

const MarketSummary: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Market Summary</h3>
      <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '16px' }}>
        <Card title="S&P 500" value="5,432.10" change="+12.34 (0.23%)" />
        <Card title="Dow Jones" value="34,567.89" change="-56.78 (0.16%)" />
        <Card title="Nasdaq" value="17,890.12" change="+98.76 (0.55%)" />
      </div>
      <div style={{ borderTop: '1px solid #eee', paddingTop: '10px' }}>
        <h4>News Ticker</h4>
        <p style={{ margin: 0 }}>[Breaking News] Fed hints at potential rate cuts later this year...</p>
      </div>
      <div style={{ borderTop: '1px solid #eee', paddingTop: '10px', marginTop: '10px' }}>
        <h4>Market Sentiment</h4>
        <div style={{ backgroundColor: '#e0e0e0', borderRadius: '5px', height: '20px', width: '100%' }}>
          <div style={{ backgroundColor: 'green', width: '65%', height: '100%', borderRadius: '5px' }}></div>
        </div>
        <p style={{ margin: '5px 0 0 0', textAlign: 'center' }}>Bullish (65%)</p>
      </div>
    </div>
  );
};

export default MarketSummary;
