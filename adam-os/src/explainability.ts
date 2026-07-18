import { AdamNode, UUID } from "./core";

// ==============================================================================
// 21. Explainability
// ==============================================================================

export interface Explanation extends AdamNode {
  targetId: UUID; // Decision or Output ID
  narrative: string; // LLM-generated explanation
  evidenceIds: UUID[]; // Traceability to inputs/model state
}
