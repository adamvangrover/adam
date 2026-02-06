import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UserState {
  alphaPoints: number;
  rank: string;
  actionsTaken: number;
  incrementAlpha: (amount: number) => void;
  recordAction: () => void;
}

const calculateRank = (points: number): string => {
  if (points < 100) return 'JUNIOR ANALYST';
  if (points < 500) return 'SENIOR ANALYST';
  if (points < 1500) return 'PROMPT ENGINEER';
  if (points < 5000) return 'QUANT RESEARCHER';
  if (points < 10000) return 'SYSTEM ARCHITECT';
  return 'NEURO-MANCER';
};

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      alphaPoints: 0,
      rank: 'JUNIOR ANALYST',
      actionsTaken: 0,
      incrementAlpha: (amount) =>
        set((state) => {
          const newPoints = state.alphaPoints + amount;
          return {
            alphaPoints: newPoints,
            rank: calculateRank(newPoints),
          };
        }),
      recordAction: () =>
        set((state) => ({ actionsTaken: state.actionsTaken + 1 })),
    }),
    {
      name: 'prompt-alpha-user-storage',
    }
  )
);
