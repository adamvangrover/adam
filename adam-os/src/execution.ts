import { AdamNode, UUID } from "./core";

// ==============================================================================
// 18. Execution Graph
// ==============================================================================

export interface WorkflowStep extends AdamNode {
  action: "INGEST" | "VERIFY" | "CALCULATE" | "PUBLISH";
  targetToolId?: UUID;
}

export interface WorkflowEdge extends AdamNode {
  fromStepId: UUID;
  toStepId: UUID;
  condition?: string; // Logic condition to proceed
}

export interface Workflow extends AdamNode {
  name: string;
  steps: UUID[]; // References to WorkflowStep
  edges: UUID[]; // References to WorkflowEdge
}
