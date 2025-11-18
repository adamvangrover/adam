import React from 'react';

const SimulationReports: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Simulation Reports</h3>
      <p>A repository of past simulation reports, with filtering and search capabilities.</p>
       <ul style={{ margin: 0, paddingLeft: '20px' }}>
        <li>
          <a href="#">Credit Rating Assessment: TechCorp (TC) - 2024-06-11</a>
        </li>
        <li>
          <a href="#">Investment Committee Sim: GreenEnergy Co. (GEC) - 2024-06-10</a>
        </li>
      </ul>
    </div>
  );
};

export default SimulationReports;
