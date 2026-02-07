import { useEffect, useRef } from 'react';
import { useSwarmStore, Agent } from '../stores/swarmStore';
import { generateAgentTask } from '../utils/simulationEngine';

export function useSwarmSimulation() {
  const { initializeSwarm, updateAgent, updateMetrics, agents } = useSwarmStore();
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (agents.length === 0) {
      initializeSwarm(24); // 24 active agents
    }

    intervalRef.current = setInterval(() => {
      // 1. Update random agent
      const randomAgentIndex = Math.floor(Math.random() * agents.length);
      const agent = agents[randomAgentIndex];

      if (!agent) return;

      let updates: Partial<Agent> = {};

      if (agent.status === 'IDLE') {
        if (Math.random() > 0.4) {
          updates = {
            status: 'WORKING',
            currentTask: generateAgentTask(),
            efficiency: Math.min(100, agent.efficiency + 1)
          };
        }
      } else if (agent.status === 'WORKING') {
        if (Math.random() > 0.6) {
          updates = {
            status: 'COMPUTING',
            efficiency: Math.max(0, agent.efficiency - 2) // Strain reduces efficiency
          };
        }
      } else if (agent.status === 'COMPUTING') {
         if (Math.random() > 0.5) {
            updates = {
                status: 'IDLE',
                currentTask: 'AWAITING INSTRUCTION'
            };
         }
      }

      updateAgent(agent.id, updates);

      // 2. Update System Metrics (Brownian motion)
      updateMetrics({
        networkLoad: Math.min(100, Math.max(0, 45 + (Math.random() * 20 - 10))),
        consensusRate: Math.min(100, Math.max(90, 98 + (Math.random() * 2 - 1))),
        activeNodes: agents.filter(a => a.status !== 'IDLE').length
      });

    }, 200); // 5 updates per second for "Live" feel

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [agents.length, initializeSwarm, updateAgent, updateMetrics, agents]);
}
