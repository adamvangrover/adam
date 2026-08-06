import { AdamNode, UUID } from "./core";

// ==============================================================================
// 20. Decision Engine
// ==============================================================================

export enum DecisionType {
  Trade = "TRADE",
  Rebalance = "REBALANCE",
  RiskOverride = "RISK_OVERRIDE",
  ModelApproval = "MODEL_APPROVAL",
  ResearchSummary = "RESEARCH_SUMMARY",
}

export interface Decision extends AdamNode {
  decisionType: DecisionType;
  rationaleIds: UUID[]; // Explanations or Evidence
  confidence: number;
}

export interface TradeProposal extends Decision {
  securityId: UUID;
  action: "BUY" | "SELL" | "HOLD";
  targetWeight: number;
}

export interface RebalanceProposal extends Decision {
  portfolioId: UUID;
  targetAllocations: Record<string, number>; // string mapped UUID to weight
}

export interface RiskOverride extends Decision {
  measureId: UUID;
  adjustedValue: number;
}

export interface ModelApproval extends Decision {
  modelVersionId: UUID;
  status: "Approved" | "Rejected";
}

export interface ResearchSummary extends Decision {
  topic: string;
  summary: string;
}
