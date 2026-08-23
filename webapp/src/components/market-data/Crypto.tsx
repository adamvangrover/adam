import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { cryptoData } from '../../utils/historicData';

const Crypto: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <h4>Crypto</h4>
      <div style={{ height: '300px', marginBottom: '16px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={cryptoData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
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
            <th style={{ textAlign: 'left', padding: '8px' }}>Name</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Market Cap</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>24h Change</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>Bitcoin (BTC)</td>
            <td style={{ padding: '8px' }}>$65,432.10</td>
            <td style={{ padding: '8px' }}>$1.3T</td>
            <td style={{ padding: '8px', color: 'green' }}>+1.2%</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>Ethereum (ETH)</td>
            <td style={{ padding: '8px' }}>$3,456.78</td>
            <td style={{ padding: '8px' }}>$415B</td>
            <td style={{ padding: '8px', color: 'red' }}>-0.5%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default Crypto;