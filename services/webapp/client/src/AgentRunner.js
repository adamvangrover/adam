import React, { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';

function AgentRunner({ singleAgent }) {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(singleAgent || '');
  const [inputSchema, setInputSchema] = useState(null);
  const [inputValues, setInputValues] = useState({});
  const [rawInput, setRawInput] = useState('{}');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
        })
        .catch(err => {
          console.error("Failed to fetch agents", err);
          setError("Failed to load agents list");
        })
        .finally(() => setLoading(false));
    }
  }, [singleAgent]);

  useEffect(() => {
    if (selectedAgent) {
      setInputSchema(null);
      setInputValues({});
      setRawInput('{}');
      setError(null);

      fetch(`/api/agents/${selectedAgent}/schema`)
        .then(res => res.json())
        .then(data => {
          setInputSchema(data);
          const initialValues = {};
          if (data) {
            for (const key in data) {
              initialValues[key] = data[key].default || '';
            }
          }
          setInputValues(initialValues);
        })
        .catch(err => console.error("Failed to fetch schema", err));
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
    setError(null);

    let payload = inputValues;
    if (!inputSchema) {
      try {
        payload = JSON.parse(rawInput);
      } catch (e) {
        setError("Invalid JSON in input: " + e.message);
        setLoading(false);
        return;
      }
    }

    fetch(`/api/agents/${selectedAgent}/invoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then(res => {
        if (!res.ok) {
          return res.json().then(err => { throw new Error(err.error || 'Agent execution failed') });
        }
        return res.json();
      })
      .then(data => {
        setOutput(JSON.stringify(data, null, 2));
      })
      .catch(err => {
        setError(err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const renderForm = () => {
    if (!inputSchema) {
      return (
        <div className="flex flex-col gap-2">
          <label htmlFor="raw-input" className="text-sm font-medium text-slate-300">Raw Input (JSON)</label>
          <textarea
            id="raw-input"
            className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-slate-200 font-mono text-sm focus:ring-2 focus:ring-cyan-500 outline-none"
            value={rawInput}
            onChange={(e) => setRawInput(e.target.value)}
            rows="5"
          />
        </div>
      );
    }
    return Object.entries(inputSchema).map(([key, value]) => {
      const inputId = `input-${selectedAgent}-${key}`;
      if (value.type === 'select') {
        return (
          <div key={key} className="flex flex-col gap-1 mb-3">
            <label htmlFor={inputId} className="text-sm font-medium text-slate-300">
              {value.description || key}
            </label>
            <select
              id={inputId}
              name={key}
              value={inputValues[key]}
              onChange={handleInputChange}
              className="bg-slate-800 border border-slate-700 rounded p-2 text-slate-200 focus:ring-2 focus:ring-cyan-500 outline-none"
            >
              {value.options.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
        );
      }
      return (
        <div key={key} className="flex flex-col gap-1 mb-3">
          <label htmlFor={inputId} className="text-sm font-medium text-slate-300">
            {value.description || key}
          </label>
          <input
            id={inputId}
            type="text"
            name={key}
            value={inputValues[key]}
            onChange={handleInputChange}
            className="bg-slate-800 border border-slate-700 rounded p-2 text-slate-200 focus:ring-2 focus:ring-cyan-500 outline-none w-full"
          />
        </div>
      );
    });
  };

  return (
    <div className="space-y-4">
      {!singleAgent && (
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-cyan-400">Agent Runner</h2>
        </div>
      )}

      {!singleAgent && (
        <div className="flex flex-col gap-1">
          <label htmlFor="agent-select" className="text-sm font-medium text-slate-300">Select Agent:</label>
          <select
            id="agent-select"
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded p-2 text-slate-200 w-full focus:ring-2 focus:ring-cyan-500 outline-none"
          >
            {agents.map(agent => (
              <option key={agent} value={agent}>{agent}</option>
            ))}
          </select>
        </div>
      )}

      <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800">
        {renderForm()}
      </div>

      <button
        onClick={handleRunAgent}
        disabled={loading || !selectedAgent}
        className={`flex items-center justify-center gap-2 px-4 py-2 rounded font-medium transition-all w-full
          ${loading || !selectedAgent
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
            : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-900/20 hover:shadow-cyan-900/40'
          }`}
        aria-busy={loading}
      >
        {loading ? <Loader2 className="animate-spin h-4 w-4" /> : null}
        {loading ? 'Executing Agent...' : 'Run Agent'}
      </button>

      <div aria-live="polite" className="mt-4">
        <h3 className="text-lg font-semibold text-slate-200 mb-2">Output</h3>

        {loading && (
          <div className="p-4 bg-slate-900 rounded border border-slate-800 text-cyan-400 font-mono text-sm animate-pulse" role="status">
            > Initializing agent runtime...
            <br />> Waiting for response...
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-900/20 border border-red-800 rounded text-red-400" role="alert">
            <span className="font-bold">Error:</span> {error}
          </div>
        )}

        {output && !loading && (
          <pre className="p-4 bg-black/50 rounded border border-slate-800 text-green-400 font-mono text-sm overflow-auto max-h-[400px] whitespace-pre-wrap">
            {output}
          </pre>
        )}

        {!output && !loading && !error && (
            <div className="p-4 border border-dashed border-slate-800 rounded text-slate-500 text-center text-sm">
                Ready to execute. Select an agent and provide input.
            </div>
        )}
      </div>
    </div>
  );
}

export default AgentRunner;
