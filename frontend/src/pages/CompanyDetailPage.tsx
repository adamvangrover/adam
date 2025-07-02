import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fetchCompanyExplanation, CompanyExplanationResponse } from '../services/api';
import { Driver } from '../models/core';

// Small component for displaying a single driver
const DriverCard: React.FC<{ driver: Driver }> = ({ driver }) => (
  <div className="card">
    <h3>{driver.name} ({driver.id})</h3>
    <p><strong>Type:</strong> {driver.type}</p>
    <p><strong>Description:</strong> {driver.description}</p>
    {driver.impactPotential && <p><strong>Impact:</strong> {driver.impactPotential}</p>}
    {driver.timeHorizon && <p><strong>Horizon:</strong> {driver.timeHorizon}</p>}
    {driver.metrics && Object.keys(driver.metrics).length > 0 && (
      <div>
        <strong>Metrics:</strong>
        <ul>
          {Object.entries(driver.metrics).map(([key, value]) => (
            <li key={key} style={{paddingLeft: '15px', fontSize:'0.9em', borderBottom:'none', padding: '2px 0'}}><em>{key}:</em> {value}</li>
          ))}
        </ul>
      </div>
    )}
     {driver.relatedMacroFactorIds && driver.relatedMacroFactorIds.length > 0 && (
      <div>
        <strong>Related Macro Factors:</strong>
        <ul>
            {driver.relatedMacroFactorIds.map((factorId) => (
                 <li key={factorId} style={{paddingLeft: '15px', fontSize:'0.9em', borderBottom:'none', padding: '2px 0'}}>{factorId}</li> // Ideally, link to macro factor details
            ))}
        </ul>
      </div>
    )}
  </div>
);


function CompanyDetailPage() {
  const { companyId } = useParams<{ companyId: string }>();
  const [explanation, setExplanation] = useState<CompanyExplanationResponse | null>(null);
  // drivers are part of the explanation object from the backend
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!companyId) {
      setError("No company ID provided.");
      setLoading(false);
      return;
    }

    const getCompanyData = async () => {
      try {
        setLoading(true);
        setError(null);
        const explanationData = await fetchCompanyExplanation(companyId);
        setExplanation(explanationData);
      } catch (err) {
        setError(err instanceof Error ? err.message : `Failed to fetch data for ${companyId}. Ensure backend is running and company ID is correct.`);
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    getCompanyData();
  }, [companyId]);

  if (loading) return <div className="loading">Loading company details for {companyId}...</div>;
  if (error) return <div className="error">Error: {error}</div>;
  if (!explanation) return <p>No data found for company ID: {companyId}.</p>;

  return (
    <div>
      <nav aria-label="breadcrumb" style={{marginBottom: "15px"}}>
        <ol style={{listStyle: 'none', padding: 0, margin:0, display: 'flex', gap: '5px', alignItems: 'center', fontSize: '0.9em'}}>
          <li><Link to="/companies">Companies</Link></li>
          <li>&rsaquo;</li>
          <li aria-current="page">{explanation.company_name}</li>
        </ol>
      </nav>
      <h2>{explanation.company_name} ({explanation.company_id})</h2>

      <div className="card">
        <h3>Narrative Summary</h3>
        <pre>{explanation.narrative_summary}</pre>
      </div>

      <h3>Associated Drivers ({explanation.num_drivers_found})</h3>
      {explanation.drivers && explanation.drivers.length > 0 ? (
        explanation.drivers.map((driver) => (
          <DriverCard key={driver.id} driver={driver} />
        ))
      ) : (
        <p>No specific drivers listed in the explanation.</p>
      )}
    </div>
  );
}

export default CompanyDetailPage;
