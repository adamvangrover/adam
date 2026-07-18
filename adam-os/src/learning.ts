import { AdamNode, UUID } from "./core";

// ==============================================================================
// 25. Learning
// ==============================================================================

export interface FineTuningJob extends AdamNode {
  modelId: UUID;
  datasetId: UUID;
  status: "Pending" | "Running" | "Completed" | "Failed";
}
