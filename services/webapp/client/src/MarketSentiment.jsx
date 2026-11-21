import React, { useState } from 'react';

function MarketSentiment() {
  const [query, setQuery] = useState('');
  const [output, setOutput] = useState('');

  const handleRunAnalysis = () => {
    setOutput('Running analysis...');
    fetch(`/api/agents/market_sentiment_agent/invoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    })
      .then(res => res.json())
      .then(data => {
        setOutput(JSON.stringify(data, null, 2));
      })
      .catch(err => {
        setOutput('Error running analysis.');
      });
  };

  return (
    <div>
      <h3>Market Sentiment Agent</h3>
      <div>
        <label>Query:</label>
        <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} />
      </div>
      <button onClick={handleRunAnalysis}>Run Analysis</button>
      <div>
        <h3>Output:</h3>
        <pre>{output}</pre>
      </div>
    </div>
  );
}

export default MarketSentiment;
