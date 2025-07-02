// frontend/src/models/core.ts

export interface Industry {
  id: string;
  name: string;
  description?: string;
  macroDriverIds?: string[];
  industrySpecificDriverIds?: string[];
}

export interface Company {
  id: string; // e.g., ticker symbol
  name: string;
  industryId: string;
  ownershipStructure?: string;
  corporateStructure?: string;
  financials?: Record<string, any>; // Flexible for various metrics
  companySpecificDriverIds?: string[];
  tradingLevels?: TradingLevelData;
}

export interface Driver {
  id: string;
  name: string;
  description: string;
  type: 'Macroeconomic' | 'Fundamental' | 'Technical' | 'Geopolitical' | 'CompanySpecific' | 'IndustrySpecific';
  impactPotential?: 'High' | 'Medium' | 'Low';
  timeHorizon?: 'Short-term' | 'Medium-term' | 'Long-term';
  metrics?: Record<string, string>; // e.g., {"Beta to Oil Price": "1.2"}
  // Could also link to specific MacroEnvironmentFactors if it's a direct proxy
  relatedMacroFactorIds?: string[];
}

export interface MacroEnvironmentFactor {
  id: string;
  name: string;
  currentValue?: string | number;
  trend?: 'Increasing' | 'Decreasing' | 'Stable';
  impactNarrative?: string;
  source?: string; // e.g., "Central Bank Report Q1 2023"
}

export interface NarrativeExplanation {
  id: string;
  generatedForEntityId: string; // Company.id or Industry.id
  entityType: 'Company' | 'Industry';
  driverIds: string[];
  explanationText: string;
  linkedTradingLevelContext?: string; // e.g., "Explains current P/E ratio of 25"
  timestamp: Date;
}

export interface TradingLevelData {
  price?: number;
  volume?: number;
  volatility?: number;
  timestamp?: Date;
  // Other relevant trading metrics
}

// Knowledge Graph Edge/Relationship (Conceptual)
export interface Relationship {
  sourceId: string;
  targetId: string;
  type:
    | 'BELONGS_TO_INDUSTRY'
    | 'AFFECTED_BY_DRIVER'
    | 'HAS_EXPLANATION'
    | 'RELATES_TO_TRADING_LEVEL'; // etc.
  properties?: Record<string, any>;
}
