import { AdamNode, UUID } from "./core";

// ==============================================================================
// 24. Evaluation
// ==============================================================================

export interface Metric extends AdamNode {
  name: string;
  description: string;
  targetId: UUID; // Model, Agent, or Portfolio ID
  value: number;
}
