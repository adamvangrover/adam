import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { bondData } from '../../utils/historicData';

const Bonds: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <h4>Bonds</h4>
      <div style={{ height: '300px', marginBottom: '16px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={bondData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="date" stroke="#888" tickFormatter={(tick) => tick.substring(5)} />
            <YAxis stroke="#888" domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ backgroundColor: '#111', borderColor: '#333' }} />
            <Legend />
            <Area type="monotone" dataKey="price" stroke="#00f3ff" fill="#00f3ff" fillOpacity={0.3} name="Historical Yield" />
            <Area type="monotone" dataKey="projected" stroke="#ff00ff" fill="#ff00ff" fillOpacity={0.3} name="Projected Yield" strokeDasharray="5 5" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Name</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Yield</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Price</th>
            <th style={{ textAlign: 'left', padding: '8px' }}>Maturity</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '8px' }}>US 10-Year</td>
            <td style={{ padding: '8px' }}>4.5%</td>
            <td style={{ padding: '8px' }}>98.5</td>
            <td style={{ padding: '8px' }}>2034-06-12</td>
          </tr>
          <tr>
            <td style={{ padding: '8px' }}>US 2-Year</td>
            <td style={{ padding: '8px' }}>4.8%</td>
            <td style={{ padding: '8px' }}>101.2</td>
            <td style={{ padding: '8px' }}>2026-06-12</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default Bonds;