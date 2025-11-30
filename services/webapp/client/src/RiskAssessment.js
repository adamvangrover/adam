import React from 'react';
import { useTranslation } from 'react-i18next';

const getRiskColor = (score) => {
  if (score < 30) return 'green';
  if (score < 70) return 'orange';
  return 'red';
};

const RiskAssessment = ({ data }) => {
  const { t } = useTranslation();
  if (!data || !data.risk_score) {
    return <p>Incompatible data format for risk assessment.</p>;
  }

  const { risk_score, breakdown, summary } = data;

  return (
    <div className="Card">
      <h3>{t('riskAssessment.title')}</h3>
      <div style={{
        backgroundColor: getRiskColor(risk_score),
        color: 'white',
        padding: '20px',
        borderRadius: '10px',
        textAlign: 'center'
      }}>
        <h4>{t('riskAssessment.riskScore')}</h4>
        <h1>{risk_score}</h1>
        <p>/ 100</p>
      </div>

      <h4>{t('riskAssessment.riskBreakdown')}</h4>
      <ul>
        {Object.entries(breakdown).map(([key, value]) => (
          <li key={key}>
            <strong>{key.replace(/_/g, ' ')}:</strong> {value}
          </li>
        ))}
      </ul>

      <h4>{t('riskAssessment.summary')}</h4>
      <p>{summary}</p>
    </div>
  );
};

export default RiskAssessment;
