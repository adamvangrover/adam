// services/webapp/client/src/pages/MarketData.tsx

import React, { useState, useEffect } from 'react';
import { Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend
} from 'chart.js';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend);

interface Company {
  name: string;
  ticker: string;
  sector: string;
}

interface SectorData {
  labels: string[];
  datasets: {
    data: number[];
    backgroundColor: string[];
  }[];
}

const MarketData: React.FC = () => {
  const [companyData, setCompanyData] = useState<Company[] | null>(null);
  const [knowledgeGraph, setKnowledgeGraph] = useState<any>(null);
  const [sectorData, setSectorData] = useState<SectorData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([
      fetch('/api/data/company_data.json').then(res => res.json()),
      fetch('/api/data/knowledge_graph.json').then(res => res.json()),
    ]).then(([companyData, knowledgeGraphData]) => {
      setCompanyData(companyData);
      setKnowledgeGraph(knowledgeGraphData);

      const sectors = companyData.reduce((acc: {[key: string]: number}, company: Company) => {
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
              '#4BC0C0',
              '#9966FF',
              '#FF9F40',
            ],
          },
        ],
      });
      setLoading(false);
    }).catch(err => {
        console.error("Failed to load market data", err);
        setLoading(false);
    });
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Market Data</h2>
      {loading ? <p>Loading...</p> : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="Card p-4 shadow rounded bg-white">
            <h3 className="text-xl mb-2">Company Data</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="p-2 text-left">Company</th>
                    <th className="p-2 text-left">Ticker</th>
                    <th className="p-2 text-left">Sector</th>
                  </tr>
                </thead>
                <tbody>
                  {companyData?.map(company => (
                    <tr key={company.ticker} className="border-b">
                      <td className="p-2">{company.name}</td>
                      <td className="p-2">{company.ticker}</td>
                      <td className="p-2">{company.sector}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="Card p-4 shadow rounded bg-white">
            <h3 className="text-xl mb-2">Sector Distribution</h3>
            {sectorData && <Pie data={sectorData} />}
          </div>
          <div className="Card p-4 shadow rounded bg-white col-span-1 md:col-span-2">
            <h3 className="text-xl mb-2">Knowledge Graph Preview</h3>
            <pre className="bg-gray-50 p-2 text-xs overflow-auto max-h-60">
              {JSON.stringify(knowledgeGraph, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default MarketData;
