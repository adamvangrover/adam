import { AdamNode, UUID } from "./core";

// ==============================================================================
// 7. Feature Store
// ==============================================================================

export interface Feature extends AdamNode {
  name: string;
  variableId: UUID; // The underlying raw variable
  dataType: "Numeric" | "Categorical" | "Embedding" | "Text";
}

export interface Transformation extends AdamNode {
  name: string;
  expression: string; // e.g., "log(return)", "z-score"
  inputFeatureIds: UUID[];
  outputFeatureId: UUID;
}

export interface FeatureSet extends AdamNode {
  name: string;
  featureIds: UUID[];
}

export interface Embedding extends AdamNode {
  featureId: UUID;
  vector: number[];
  modelId: UUID;
}
