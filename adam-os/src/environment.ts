import { AdamNode, UUID } from "./core";

// ==============================================================================
// 14. Environment
// ==============================================================================

export interface Environment extends AdamNode {
  name: "Production" | "Simulation" | "Backtest" | "Research";
}

export interface MacroRegime extends AdamNode {
  name: string; // e.g., "High Inflation", "Recession"
  indicators: UUID[]; // Variable IDs
}

export interface SystemState extends AdamNode {
  environmentId: UUID;
  activeRegimeId?: UUID;
  activeScenarioId?: UUID;
}

export interface ExecutionContext extends AdamNode {
  environmentId: UUID;
  userId?: UUID;
  agentId?: UUID;
  effectiveDate: string;
  asOfDate: string;
  observedDate: string;
}
