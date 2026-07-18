import { AdamNode, UUID, ISO8601Date } from "./core";

// ==============================================================================
// 5. Provenance & Dependency
// ==============================================================================

export interface ProvenanceEdge extends AdamNode {
  parentNode: UUID;
  childNode: UUID;
  transformation: "API" | "Interpolation" | "Simulation" | "Inference" | "Manual" | "Aggregation";
  timestamp: ISO8601Date;
}

export interface ModelDependency extends AdamNode {
  parentModelId: UUID;
  childModelId: UUID;
}
