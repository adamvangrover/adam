import { AdamNode, UUID } from "./core";

// ==============================================================================
// 2. Knowledge Graph & Variables
// ==============================================================================

export interface Variable extends AdamNode {
  name: string;
  category: "Macro" | "Rates" | "Credit" | "Equity" | "FX" | "Commodity" | "Alternative";
  unit: string;
  description: string;
  sourcePriority: UUID[]; // List of models/datasets, ordered by priority
  lineage: UUID[]; // Upstream dependencies
  entityId?: UUID; // Relationship to the asset it belongs to
}

export interface GraphEdge extends AdamNode {
  from: UUID;
  to: UUID;
  relationship: string;
}
