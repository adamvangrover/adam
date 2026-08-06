import { AdamNode, UUID } from "./core";

// ==============================================================================
// 11. Simulation
// ==============================================================================

export interface Branch extends AdamNode {
  name: string;
  scenarioId: UUID;
}

export interface Timeline extends AdamNode {
  baseDate: string;
  horizon: number;
  intervals: string; // e.g., "Daily", "Monthly"
}

export interface SimulationNode extends AdamNode {
  branchId: UUID;
  parentStateId?: UUID;
  timeIndex: number;
  stateSnapshot: Record<string, number>; // State of variables (keyed by variable UUID as string)
}
