import React, { useState, useEffect } from 'react';
import { Pie } from 'react-chartjs-2';

function MarketData() {
  const [companyData, setCompanyData] = useState(null);
  const [knowledgeGraph, setKnowledgeGraph] = useState(null);
  const [sectorData, setSectorData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('/api/data/company_data.json').then(res => res.json()),
      fetch('/api/data/knowledge_graph.json').then(res => res.json()),
    ]).then(([companyData, knowledgeGraphData]) => {
      setCompanyData(companyData);
      setKnowledgeGraph(knowledgeGraphData);
      const sectors = companyData.reduce((acc, company) => {
        acc[company.sector] = (acc[company.sector] || 0) + 1;
        return acc;
      }, {});
      setSectorData({
        labels: Object.keys(sectors),
        datasets: [
          {
            data: Object.values(sectors),
            backgroundColor: [
              '#FF6384',
              '#36A2EB',
              '#FFCE56',
              '#FF6384',
              '#36A2EB',
              '#FFCE56',
            ],
          },
        ],
      });
      setLoading(false);
    });
  }, []);

  return (
    <div>
      <h2>Market Data</h2>
      {loading ? <p>Loading...</p> : (
        <>
          <div className="Card">
            <h3>Company Data</h3>
            <table>
              <thead>
                <tr>
                  <th>Company</th>
                  <th>Ticker</th>
                  <th>Sector</th>
                </tr>
              </thead>
              <tbody>
                {companyData.map(company => (
                  <tr key={company.ticker}>
                    <td>{company.name}</td>
                    <td>{company.ticker}</td>
                    <td>{company.sector}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="Card">
            <h3>Sector Distribution</h3>
            {sectorData && <Pie data={sectorData} />}
          </div>
          <div className="Card">
            <h3>Knowledge Graph</h3>
            <pre>{JSON.stringify(knowledgeGraph, null, 2)}</pre>
          </div>
        </>
      )}
    </div>
  );
}

export default MarketData;
