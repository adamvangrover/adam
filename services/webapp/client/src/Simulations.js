import React, { useState, useEffect } from 'react';

function Simulations() {
  const [tasks, setTasks] = useState({});
  const simulations = [
    'Credit_Rating_Assessment_Simulation',
    'Investment_Committee_Simulation',
    'Portfolio_Optimization_Simulation',
  ];

  const handleRunSimulation = (simulationName) => {
    fetch(`/api/simulations/${simulationName}`, {
      method: 'POST',
    })
      .then(res => res.json())
      .then(data => {
        setTasks(prevTasks => ({
          ...prevTasks,
          [data.task_id]: { status: 'Pending', name: simulationName }
        }));
      });
  };

  useEffect(() => {
    const interval = setInterval(() => {
      Object.keys(tasks).forEach(taskId => {
        if (tasks[taskId].status !== 'SUCCESS' && tasks[taskId].status !== 'FAILURE') {
          fetch(`/api/tasks/${taskId}`)
            .then(res => res.json())
            .then(data => {
              setTasks(prevTasks => ({
                ...prevTasks,
                [taskId]: { ...prevTasks[taskId], status: data.state, message: data.status }
              }));
            });
        }
      });
    }, 2000);
    return () => clearInterval(interval);
  }, [tasks]);


  return (
    <div>
      <h2>Simulations</h2>
      {simulations.map(sim => (
        <div key={sim} className="Card">
          <h3>{sim}</h3>
          <button onClick={() => handleRunSimulation(sim)}>Run Simulation</button>
        </div>
      ))}
      <div>
        <h3>Task Status:</h3>
        <ul>
          {Object.entries(tasks).map(([taskId, task]) => (
            <li key={taskId}>
              {task.name}: {task.status} {task.message && `- ${task.message}`}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default Simulations;
