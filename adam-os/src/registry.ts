import { AdamNode, UUID } from "./core";

// ==============================================================================
// 9. Registry
// ==============================================================================

export interface ModelVersion extends AdamNode {
  modelId: UUID;
  tag: string; // e.g. "v1.2.0"
  artifactUri: string; // e.g. "s3://models/..."
  status: "Staging" | "Production" | "Archived";
  parentVersionId?: UUID;
}

export interface Registry extends AdamNode {
  name: string;
  models: UUID[];
}
