import { AdamNode, UUID } from "./core";

// ==============================================================================
// 8. Models
// ==============================================================================

export enum ModelClass {
  Stochastic = "STOCHASTIC",
  Heuristic = "HEURISTIC",
  GenerativeLLM = "GENERATIVE",
  Copula = "COPULA",
}

export interface Model extends AdamNode {
  name: string;
  modelClass: ModelClass;
  expectedInputs: UUID[]; // References to Variables / Features
  expectedOutputs: UUID[];
}

export interface ModelRun extends AdamNode {
  modelId: UUID;
  modelVersionId: UUID;
  environmentId: UUID;
  parameters: Record<string, string | number | boolean>;
}

export interface ModelOutput extends AdamNode {
  modelRunId: UUID;
  variableId: UUID;
  value: number;
  confidence: number;
  lowerBound?: number;
  upperBound?: number;
}
