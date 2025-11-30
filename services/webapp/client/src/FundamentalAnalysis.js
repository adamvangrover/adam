import React from 'react';
import { useTranslation } from 'react-i18next';

const FundamentalAnalysis = ({ data }) => {
  const { t } = useTranslation();
  if (!data) {
    return <p>No data to display.</p>;
  }

  const { dcf_valuation, financial_ratios } = data;

  return (
    <div className="Card">
      <h3>{t('fundamentalAnalysis.title')}</h3>

      {dcf_valuation && (
        <div>
          <h4>{t('fundamentalAnalysis.dcfValuation')}</h4>
          <p><strong>{t('fundamentalAnalysis.estimatedValue')}:</strong> ${dcf_valuation.estimated_value?.toFixed(2)}</p>
          <p><em>{dcf_valuation.narrative_summary}</em></p>
        </div>
      )}

      {financial_ratios && (
        <div>
          <h4>{t('fundamentalAnalysis.financialRatios')}</h4>
          <table>
            <thead>
              <tr>
                <th>{t('fundamentalAnalysis.ratio')}</th>
                <th>{t('fundamentalAnalysis.value')}</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(financial_ratios).map(([key, value]) => (
                <tr key={key}>
                  <td>{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                  <td>{typeof value === 'number' ? value.toFixed(2) : value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default FundamentalAnalysis;
