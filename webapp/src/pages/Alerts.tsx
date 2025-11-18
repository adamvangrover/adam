import React from 'react';
import AlertsDashboard from '../components/alerts/AlertsDashboard';
import AlertCreation from '../components/alerts/AlertCreation';

const Alerts: React.FC = () => {
  return (
    <div>
      <h1>Alerts</h1>
      <AlertsDashboard />
      <AlertCreation />
    </div>
  );
};

export default Alerts;
