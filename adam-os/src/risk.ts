import { AdamNode, UUID } from "./core";

// ==============================================================================
// 13. Risk
// ==============================================================================

export interface Measure extends AdamNode {
  targetId: UUID; // Portfolio or Position ID
  value: number;
}

export interface VaR extends Measure {
  confidenceLevel: number;
  horizonDays: number;
  methodology: "Parametric" | "Historical" | "MonteCarlo";
}

export interface StressResult extends Measure {
  scenarioId: UUID;
}

export interface FactorExposure extends Measure {
  factorId: UUID;
  beta: number;
}

export interface LiquidityRisk extends Measure {
  daysToLiquidate: number;
  bidAskSpread: number;
}
