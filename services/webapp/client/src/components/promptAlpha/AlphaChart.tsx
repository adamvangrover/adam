import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { usePromptStore } from '../../stores/promptStore';

export const AlphaChart: React.FC = () => {
  const prompts = usePromptStore((state) => state.prompts);

  // Bucket data
  const buckets = [
    { range: '0-20', count: 0 },
    { range: '21-40', count: 0 },
    { range: '41-60', count: 0 },
    { range: '61-80', count: 0 },
    { range: '81-100', count: 0 },
  ];

  prompts.forEach(p => {
    if (p.alphaScore <= 20) buckets[0].count++;
    else if (p.alphaScore <= 40) buckets[1].count++;
    else if (p.alphaScore <= 60) buckets[2].count++;
    else if (p.alphaScore <= 80) buckets[3].count++;
    else buckets[4].count++;
  });

  return (
    <div className="h-40 w-full bg-gray-900 border border-gray-800 rounded p-2">
      <div className="text-[10px] text-gray-500 mb-2 uppercase tracking-widest font-bold">Alpha Distribution</div>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={buckets}>
          <XAxis dataKey="range" tick={{fontSize: 10, fill: '#666'}} interval={0} />
          <YAxis hide />
          <Tooltip
            contentStyle={{backgroundColor: '#000', border: '1px solid #333', fontSize: '10px'}}
            itemStyle={{color: '#4ade80'}}
          />
          <Bar dataKey="count">
            {buckets.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={index > 3 ? '#4ade80' : '#1f2937'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
