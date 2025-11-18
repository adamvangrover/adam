import React from 'react';

const NewsItem: React.FC<{ source: string; headline: string; sentiment: string }> = ({ source, headline, sentiment }) => (
    <div style={{ borderBottom: '1px solid #eee', padding: '10px 0' }}>
        <p style={{ margin: 0 }}><strong>[{source}]</strong> {headline}</p>
        <small>Sentiment: <span style={{ color: sentiment === 'Positive' ? 'green' : 'red' }}>{sentiment}</span></small>
    </div>
);

const News: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>News Feed</h3>
      <NewsItem source="Financial Times" headline="Global markets rally on positive economic data." sentiment="Positive" />
      <NewsItem source="Reuters" headline="Tech sector faces new regulatory scrutiny." sentiment="Negative" />
    </div>
  );
};

export default News;
