import React, { useState } from 'react';

function FundamentalAnalysis() {
  const [ticker, setTicker] = useState('');
  const [output, setOutput] = useState('');

  const handleRunAnalysis = () => {
    setOutput('Running analysis...');
    fetch(`/api/agents/fundamental_analyst_agent/invoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ticker }),
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
      <h3>Fundamental Analyst Agent</h3>
      <div>
        <label>Company Ticker:</label>
        <input type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} />
      </div>
      <button onClick={handleRunAnalysis}>Run Analysis</button>
      <div>
        <h3>Output:</h3>
        <pre>{output}</pre>
      </div>
    </div>
  );
}

export default FundamentalAnalysis;
