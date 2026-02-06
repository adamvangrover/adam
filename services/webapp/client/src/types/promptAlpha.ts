export interface PromptObject {
  id: string;
  title: string;
  content: string;
  source: string;
  timestamp: number;
  author: string;

  // Scoring Metrics
  alphaScore: number; // 0-100
  metrics: {
    length: number;
    variableDensity: number;
    structuralKeywords: number;
  };

  // Tags/Metadata
  tags: string[];
  isFavorite: boolean;
}

export interface SourceFeed {
  id: string;
  name: string;
  url: string;
  isActive: boolean;
  lastFetched: number;
  status: 'idle' | 'loading' | 'error' | 'success';
}

export interface UserPreferences {
  minAlphaThreshold: number; // Filter out noise below this score
  refreshInterval: number; // ms
  theme: 'cyberpunk' | 'minimal';
  maxItems: number; // History limit
  useSimulation: boolean; // Fallback to synthetic data
}
