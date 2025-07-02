import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fetchCompanies, CompanySummary } from '../services/api';

function CompaniesPage() {
  const [companies, setCompanies] = useState<CompanySummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getCompanies = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchCompanies();
        setCompanies(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch companies');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    getCompanies();
  }, []);

  if (loading) return <div className="loading">Loading companies...</div>;
  if (error) return <div className="error">Error: {error}</div>;

  return (
    <div>
      <h2>Companies</h2>
      {companies.length === 0 ? (
        <p>No companies found. Ensure the backend API is running (e.g., on port 8000) and data is loaded. Check browser console for network errors.</p>
      ) : (
        <ul>
          {companies.map((company) => (
            <li key={company.id}>
              <Link to={`/companies/${company.id}`}>
                {company.name} ({company.id})
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default CompaniesPage;
