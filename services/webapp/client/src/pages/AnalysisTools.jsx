import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { getAuthHeaders } from '../utils/auth';
import FundamentalAnalysis from '../FundamentalAnalysis';
import TechnicalAnalysis from '../TechnicalAnalysis';
import RiskAssessment from '../RiskAssessment';

// --- Components ---

const GenericResultVisualizer = ({ data }) => {
  const { t } = useTranslation();
  return (
    <div className="Card">
      <h3>{t('analysisTools.agentResult')}</h3>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
};

const ResultVisualizer = ({ agent, data }) => {
    if (!data) return null;

    switch (agent) {
        case 'fundamental_analyst_agent':
            return <FundamentalAnalysis data={data} />;
        case 'technical_analyst_agent':
            return <TechnicalAnalysis data={data} />;
        case 'risk_assessment_agent':
            return <RiskAssessment data={data} />;
        default:
            return <GenericResultVisualizer data={data} />;
    }
};

const DynamicForm = ({ schema, onSubmit, onCancel }) => {
  const { t } = useTranslation();
  const [formData, setFormData] = useState({});

  useEffect(() => {
    // Initialize form data with default values from schema
    const initialData = {};
    Object.keys(schema).forEach(key => {
      initialData[key] = schema[key].default || '';
    });
    setFormData(initialData);
  }, [schema]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      {Object.keys(schema).map(key => {
        const field = schema[key];
        return (
          <div key={key}>
            <label>{field.description || key}</label>
            {field.type === 'string' && !field.enum && <input type="text" name={key} value={formData[key] || ''} onChange={handleChange} required />}
            {field.type === 'number' && <input type="number" name={key} value={formData[key] || ''} onChange={handleChange} required />}
            {field.type === 'string' && field.enum && (
              <select name={key} value={formData[key] || ''} onChange={handleChange} required>
                <option value="">Select...</option>
                {field.enum.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            )}
          </div>
        );
      })}
      <button type="submit">{t('analysisTools.runAgent')}</button>
      <button type="button" onClick={onCancel}>{t('analysisTools.clear')}</button>
    </form>
  );
};


// --- Main Component ---

function AnalysisTools() {
  const { t } = useTranslation();
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [schema, setSchema] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchAgents = useCallback(async () => {
    const headers = await getAuthHeaders();
    const response = await fetch('/api/agents', { headers });
    const data = await response.json();
    setAgents(data);
  }, []);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const handleAgentSelect = async (agentName) => {
    if (!agentName) {
        setSelectedAgent(null);
        setSchema(null);
        setResult(null);
        return;
    }
    setSelectedAgent(agentName);
    setResult(null);
    setIsLoading(true);
    const headers = await getAuthHeaders();
    const response = await fetch(`/api/agents/${agentName}/schema`, { headers });
    const data = await response.json();
    setSchema(data);
    setIsLoading(false);
  };

  const handleFormSubmit = async (formData) => {
    setIsLoading(true);
    setResult(null);
    const headers = await getAuthHeaders();
    const response = await fetch(`/api/agents/${selectedAgent}/invoke`, {
      method: 'POST',
      headers,
      body: JSON.stringify(formData),
    });
    const data = await response.json();
    setResult(data);
    setIsLoading(false);
  };

  return (
    <div>
      <h2>{t('analysisTools.title')}</h2>
      <div className="Card">
        <h3>{t('analysisTools.selectAgent')}</h3>
        <select onChange={(e) => handleAgentSelect(e.target.value)} value={selectedAgent || ''}>
          <option value="">-- {t('analysisTools.selectAgent')} --</option>
          {agents.map(agent => (
            <option key={agent} value={agent}>{agent}</option>
          ))}
        </select>
      </div>

      {isLoading && <p>{t('analysisTools.loading')}</p>}

      {selectedAgent && schema && (
        <div className="Card">
            <h3>{selectedAgent}</h3>
            <DynamicForm schema={schema} onSubmit={handleFormSubmit} onCancel={() => handleAgentSelect(null)} />
        </div>
      )}

      {result && <ResultVisualizer agent={selectedAgent} data={result} />}
    </div>
  );
}

export default AnalysisTools;
