import React from 'react';

const SimulationResults: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Recent Simulation Results</h3>
       <ul style={{ margin: 0, paddingLeft: '20px' }}>
        <li>
          <a href="#">Credit Rating Assessment: TechCorp (TC) - Result: AA</a>
        </li>
        <li>
          <a href="#">Investment Committee Sim: GreenEnergy Co. (GEC) - Outcome: Approved</a>
        </li>
      </ul>
    </div>
  );
};

export default SimulationResults;
