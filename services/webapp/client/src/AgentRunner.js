import React, { useState, useEffect } from 'react';

function AgentRunner({ singleAgent }) {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(singleAgent || '');
  const [inputSchema, setInputSchema] = useState(null);
  const [inputValues, setInputValues] = useState({});
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!singleAgent) {
      setLoading(true);
      fetch('/api/agents')
        .then(res => res.json())
        .then(data => {
          setAgents(data);
          if (data.length > 0) {
            setSelectedAgent(data[0]);
          }
          setLoading(false);
        });
    }
  }, [singleAgent]);

  useEffect(() => {
    if (selectedAgent) {
      fetch(`/api/agents/${selectedAgent}/schema`)
        .then(res => res.json())
        .then(data => {
          setInputSchema(data);
          const initialValues = {};
          for (const key in data) {
            initialValues[key] = data[key].default || '';
          }
          setInputValues(initialValues);
        });
    }
  }, [selectedAgent]);

  const handleInputChange = (e) => {
    setInputValues({
      ...inputValues,
      [e.target.name]: e.target.value,
    });
  };

  const handleRunAgent = () => {
    setLoading(true);
    setOutput('');
    fetch(`/api/agents/${selectedAgent}/invoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(inputValues),
    })
      .then(res => {
        if (!res.ok) {
          return res.json().then(err => { throw new Error(err.error) });
        }
        return res.json();
      })
      .then(data => {
        setOutput(JSON.stringify(data, null, 2));
        setLoading(false);
      })
      .catch(err => {
        setOutput(err.message);
        setLoading(false);
      });
  };

  const renderForm = () => {
    if (!inputSchema) {
      return <textarea value={JSON.stringify(inputValues)} onChange={(e) => setInputValues(JSON.parse(e.target.value))} rows="10" cols="50" />;
    }
    return Object.entries(inputSchema).map(([key, value]) => {
      if (value.type === 'select') {
        return (
          <div key={key}>
            <label>{value.description}</label>
            <select name={key} value={inputValues[key]} onChange={handleInputChange}>
              {value.options.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
        );
      }
      return (
        <div key={key}>
          <label>{value.description}</label>
          <input type="text" name={key} value={inputValues[key]} onChange={handleInputChange} />
        </div>
      );
    });
  };

  return (
    <div>
      {!singleAgent && <h2>Agent Runner</h2>}
      {!singleAgent && (
        <div>
          <label>Select Agent:</label>
          <select value={selectedAgent} onChange={(e) => setSelectedAgent(e.target.value)}>
            {agents.map(agent => (
              <option key={agent} value={agent}>{agent}</option>
            ))}
          </select>
        </div>
      )}
      <div>
        <label>Input:</label>
        {renderForm()}
      </div>
      <button onClick={handleRunAgent} disabled={loading}>
        {loading ? 'Running...' : 'Run Agent'}
      </button>
      <div>
        <h3>Output:</h3>
        {loading && <p>Loading...</p>}
        <pre>{output}</pre>
      </div>
    </div>
  );
}

export default AgentRunner;
