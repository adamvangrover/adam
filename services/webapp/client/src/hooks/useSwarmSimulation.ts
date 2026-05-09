import { useEffect, useRef } from 'react';
import { useSwarmStore, Agent } from '../stores/swarmStore';
import { generateAgentTask } from '../utils/simulationEngine';

export function useSwarmSimulation() {
  // Bolt Optimization ⚡: Extract only initializeSwarm using a selector to prevent
  // subscribing the parent component (PromptAlpha) to ALL store changes in the simulation loop.
  const initializeSwarm = useSwarmStore((state) => state.initializeSwarm);

  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (useSwarmStore.getState().agents.length === 0) {
      initializeSwarm(24); // 24 active agents
    }

    let isMounted = true;

    // Bolt Optimization ⚡: Replace setInterval with recursive setTimeout to prevent
    // request pile-ups, and use `getState()` to access current store state without re-rendering.
    const runSimulationStep = () => {
      if (!isMounted) return;

      const state = useSwarmStore.getState();
      const agents = state.agents;

      if (agents.length === 0) {
        timeoutRef.current = setTimeout(runSimulationStep, 200);
        return;
      }

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

      state.updateAgent(agent.id, updates);

      // 2. Update System Metrics (Brownian motion)
      state.updateMetrics({
        networkLoad: Math.min(100, Math.max(0, 45 + (Math.random() * 20 - 10))),
        consensusRate: Math.min(100, Math.max(90, 98 + (Math.random() * 2 - 1))),
        activeNodes: agents.filter(a => a.status !== 'IDLE').length
      });

      timeoutRef.current = setTimeout(runSimulationStep, 200); // 5 updates per second for "Live" feel
    };

    runSimulationStep();

    return () => {
      isMounted = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [initializeSwarm]);
}
