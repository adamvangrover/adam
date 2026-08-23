import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { stockData } from '../../utils/historicData';

const Stocks: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <h4>Stocks</h4>
      <div style={{ height: '300px', marginBottom: '16px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={stockData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="date" stroke="#888" tickFormatter={(tick) => tick.substring(5)} />
            <YAxis stroke="#888" domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ backgroundColor: '#111', borderColor: '#333' }} />
            <Legend />
            <Line type="monotone" dataKey="price" stroke="#00f3ff" name="Historical Price" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="projected" stroke="#ff00ff" name="Projected Price" strokeDasharray="5 5" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Symbol</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Change</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Volume</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>TC</td>
            <td style={{ padding: '8px' }}>$152.45</td>
            <td style={{ padding: '8px', color: 'green' }}>+2.10</td>
            <td style={{ padding: '8px' }}>8.5M</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>GEC</td>
            <td style={{ padding: '8px' }}>$78.90</td>
            <td style={{ padding: '8px', color: 'red' }}>-0.55</td>
            <td style={{ padding: '8px' }}>3.2M</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default Stocks;