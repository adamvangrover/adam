import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { PromptObject, SourceFeed, UserPreferences } from '../types/promptAlpha';

interface PromptState {
  prompts: PromptObject[];
  feeds: SourceFeed[];
  preferences: UserPreferences;
  selectedPromptId: string | null;

  // Actions
  addPrompts: (newPrompts: PromptObject[]) => void;
  updateFeedStatus: (feedId: string, status: SourceFeed['status']) => void;
  updatePreferences: (prefs: Partial<UserPreferences>) => void;
  selectPrompt: (id: string | null) => void;
  toggleFavorite: (id: string) => void;
  clearHistory: () => void;
}

const DEFAULT_PREFERENCES: UserPreferences = {
  minAlphaThreshold: 20,
  refreshInterval: 60000,
  theme: 'cyberpunk',
  maxItems: 500,
  useSimulation: true,
};

const DEFAULT_FEEDS: SourceFeed[] = [
  {
    id: 'reddit-chatgpt',
    name: 'r/ChatGPT (New)',
    url: 'https://www.reddit.com/r/ChatGPT/new.json?limit=25',
    isActive: true,
    lastFetched: 0,
    status: 'idle',
  },
  {
    id: 'reddit-prompt-engineering',
    name: 'r/PromptEngineering',
    url: 'https://www.reddit.com/r/PromptEngineering/new.json?limit=25',
    isActive: true,
    lastFetched: 0,
    status: 'idle',
  }
];

export const usePromptStore = create<PromptState>()(
  persist(
    (set, get) => ({
      prompts: [],
      feeds: DEFAULT_FEEDS,
      preferences: DEFAULT_PREFERENCES,
      selectedPromptId: null,

      addPrompts: (newPrompts) => set((state) => {
        // Merge and de-duplicate based on ID
        const existingIds = new Set(state.prompts.map(p => p.id));
        const uniqueNewPrompts = newPrompts.filter(p => !existingIds.has(p.id));

        // Sort by timestamp desc and slice to maxItems
        const allPrompts = [...uniqueNewPrompts, ...state.prompts]
          .sort((a, b) => b.timestamp - a.timestamp)
          .slice(0, state.preferences.maxItems);

        return { prompts: allPrompts };
      }),

      updateFeedStatus: (feedId, status) => set((state) => ({
        feeds: state.feeds.map(f => f.id === feedId ? { ...f, status, lastFetched: Date.now() } : f)
      })),

      updatePreferences: (prefs) => set((state) => ({
        preferences: { ...state.preferences, ...prefs }
      })),

      selectPrompt: (id) => set({ selectedPromptId: id }),

      toggleFavorite: (id) => set((state) => ({
        prompts: state.prompts.map(p => p.id === id ? { ...p, isFavorite: !p.isFavorite } : p)
      })),

      clearHistory: () => set({ prompts: [] }),
    }),
    {
      name: 'prompt-alpha-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        preferences: state.preferences,
        prompts: state.prompts.filter(p => p.isFavorite), // Only persist favorites to save space? Or persist all?
        // Let's persist all for now, but limit maxItems in logic
        // Actually, let's persist everything for a "Terminal" feel, up to the limit
        // feeds: state.feeds // persist feed config
      }),
    }
  )
);
