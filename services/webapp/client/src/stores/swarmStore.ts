import { create } from 'zustand';

export interface Agent {
  id: string;
  name: string;
  role: 'RESEARCHER' | 'OPTIMIZER' | 'CRITIC' | 'CODER';
  status: 'IDLE' | 'WORKING' | 'COMPUTING' | 'WAITING';
  currentTask: string;
  efficiency: number; // 0-100
}

interface SwarmState {
  agents: Agent[];
  networkLoad: number; // 0-100
  consensusRate: number; // 0-100
  activeNodes: number;
  totalCompute: number; // TFLOPS

  updateAgent: (id: string, updates: Partial<Agent>) => void;
  updateMetrics: (metrics: Partial<Omit<SwarmState, 'agents' | 'updateAgent' | 'updateMetrics' | 'initializeSwarm'>>) => void;
  initializeSwarm: (count: number) => void;
}

const ROLES = ['RESEARCHER', 'OPTIMIZER', 'CRITIC', 'CODER'] as const;

export const useSwarmStore = create<SwarmState>((set) => ({
  agents: [],
  networkLoad: 45,
  consensusRate: 98.2,
  activeNodes: 0,
  totalCompute: 0,

  initializeSwarm: (count) => {
    const agents: Agent[] = Array.from({ length: count }, (_, i) => ({
      id: `AGENT-${(i + 1).toString().padStart(3, '0')}`,
      name: `UNIT-${Math.floor(Math.random() * 9999)}`,
      role: ROLES[Math.floor(Math.random() * ROLES.length)],
      status: 'IDLE',
      currentTask: 'AWAITING INSTRUCTION',
      efficiency: 85 + Math.floor(Math.random() * 15),
    }));
    set({ agents, activeNodes: count, totalCompute: count * 12.5 });
  },

  updateAgent: (id, updates) =>
    set((state) => ({
      agents: state.agents.map((agent) =>
        agent.id === id ? { ...agent, ...updates } : agent
      ),
    })),

  updateMetrics: (metrics) => set((state) => ({ ...state, ...metrics })),
}));
